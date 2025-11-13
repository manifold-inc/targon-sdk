import re
import os
import shlex
import contextlib
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path, PurePosixPath

from targon.core.objects import _Object
from targon.core.exceptions import ValidationError
from targon.core.resolver import Resolver


@dataclass
class DockerfileSpec:
    commands: List[str]
    context_files: Dict[str, str]


def _dockerfile_function_rep(rep: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        func.rep = rep
        return func

    return decorator


def _make_pip_install_args(
    find_links: Optional[str] = None,
    index_url: Optional[str] = None,
    extra_index_url: Optional[str] = None,
    pre: bool = False,
    extra_options: str = "",
) -> str:
    flags = [
        ("--find-links", find_links),
        ("--index-url", index_url),
        ("--extra-index-url", extra_index_url),
    ]

    args = " ".join(
        f"{flag} {shlex.quote(value)}" for flag, value in flags if value is not None
    )
    if pre:
        args += " --pre"

    if extra_options:
        if args:
            args += " "
        args += f"{extra_options}"

    return args


def _flatten_str_args(
    function_name: str, arg_name: str, args: Sequence[Union[str, List[str]]]
) -> List[str]:
    """Takes a sequence of strings, or string lists, and flattens it."""
    ret = []
    for x in args:
        if isinstance(x, str):
            ret.append(x)
        elif isinstance(x, list) and all(isinstance(y, str) for y in x):
            ret.extend(x)
        else:
            raise ValidationError(
                f"{function_name}: {arg_name} must only contain strings"
            )
    return ret


def _create_ignore_function(
    ignore: Union[Sequence[str], Callable[[Path], bool]],
) -> Callable[[Path], bool]:
    """Create an ignore function from various input types"""
    if callable(ignore):
        return ignore

    if not ignore:  # Empty sequence
        return lambda p: False

    # Convert patterns to simple matching
    patterns = list(ignore)

    def pattern_matcher(path: Path) -> bool:
        """Check if path matches any ignore pattern."""
        path_str = str(path)
        name = path.name

        for pattern in patterns:
            # Simple pattern matching
            if pattern.startswith("*"):
                # Suffix match: *.pyc
                if name.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("*"):
                # Prefix match: __pycache__*
                if name.startswith(pattern[:-1]):
                    return True
            elif "*" in pattern:
                # Contains match: *test*
                clean_pattern = pattern.replace("*", "")
                if clean_pattern in name:
                    return True
            else:
                # Exact match
                if name == pattern or path_str.endswith(pattern):
                    return True

        return False

    return pattern_matcher


def _collect_directory_files(
    directory: Path, ignore_fn: Callable[[Path], bool]
) -> List[Tuple[Path, Path]]:
    """Recursively collect all files from a directory.

    Args:
        directory: Root directory to scan
        ignore_fn: Function to check if a path should be ignored

    Returns:
        List of tuples (absolute_path, relative_path)
    """
    collected = []

    for root, dirs, files in os.walk(directory):
        root_path = Path(root)

        # Filter out ignored directories (modifies dirs in-place to prevent descent)
        dirs[:] = [d for d in dirs if not ignore_fn(root_path / d)]

        for file in files:
            file_path = root_path / file

            # Skip ignored files
            if ignore_fn(file_path):
                continue

            # Calculate relative path
            try:
                rel_path = file_path.relative_to(directory)
                collected.append((file_path, rel_path))
            except ValueError:
                continue

    return collected


class _Image(_Object, type_prefix="img"):
    def _initialize_from_empty(self) -> None:
        self._registry_ref = None
        self._collected_base_images = {}
        self._collected_dockerfile_function = None
        self._runtime_injected = False

    def _get_metadata(self) -> Optional[Dict]:
        return {"registry_ref": self._registry_ref}

    def _hydrate_metadata(self, metadata: Optional[Dict]) -> None:
        if metadata is not None:
            self._registry_ref = metadata.get("registry_ref")

    def _get_dockerfile_function(self) -> Optional[Callable]:
        return getattr(self, '_collected_dockerfile_function', None)

    @staticmethod
    def _from_args(
        *,
        base_images: Optional[Dict[str, "_Image"]] = None,
        dockerfile_function: Optional[Callable[[], DockerfileSpec]] = None,
        force_build: bool = False,
    ) -> "_Image":

        # This means this is the base Image
        if base_images is None:
            base_images = {}

        def _deps() -> List[_Image]:
            return []

        def _collect_all_dockerfile_parts(
            current_base_images: Dict[str, "_Image"],
            current_dockerfile_function: Optional[Callable],
        ) -> Tuple[List[str], Dict[str, str]]:
            """Recursively collect all Dockerfile commands and context files from the image chain."""
            all_commands = []
            all_context_files = {}

            for _, base_img in current_base_images.items():
                base_df_func = (
                    base_img._get_dockerfile_function()
                    if hasattr(base_img, '_get_dockerfile_function')
                    else None
                )
                base_base_images = getattr(base_img, '_collected_base_images', {})

                base_commands, base_context = _collect_all_dockerfile_parts(
                    base_base_images, base_df_func
                )
                all_commands.extend(base_commands)
                all_context_files.update(base_context)

            if current_dockerfile_function:
                spec = current_dockerfile_function()

                all_commands.extend(spec.commands)
                all_context_files.update(spec.context_files)

            return all_commands, all_context_files

        async def _load(self, resolver, existing_object_id: Optional[str]):
            """
            This function executes during hydration.
            """

            client = resolver.client
            console_instance = resolver.console_instance

            # Auto-inject runtime if we have app context
            if resolver._app_file and not self._runtime_injected:
                # Create a new image with runtime injected
                runtime_image = self._inject_runtime(
                    resolver._app_file, resolver._app_module_name
                )
                # Use the runtime image's collected parts instead
                all_commands, all_context_files = _collect_all_dockerfile_parts(
                    runtime_image._collected_base_images,
                    runtime_image._collected_dockerfile_function,
                )
            else:
                # Collect all Dockerfile commands and context files from the entire chain
                # base_images and dockerfile_function are already stored on self during _from_args
                all_commands, all_context_files = _collect_all_dockerfile_parts(
                    self._collected_base_images, self._collected_dockerfile_function
                )

            if not all_commands:
                raise ValidationError(
                    "No commands provided for image", field="commands"
                )

            # Read file contents for context files
            # context_files maps filename -> filepath, we need to read the content
            hydrated_context_files = {}
            for filename, filepath in all_context_files.items():
                with open(filepath, 'r') as f:
                    hydrated_context_files[filename] = f.read()

            # Start the build step
            if console_instance:
                console_instance.step("Building image")

            # Use Heim client to build the image with NDJSON streaming
            try:
                registry_ref = await client.async_heim.build_image(
                    app_id=resolver.app_id,
                    dockerfile_commands=all_commands,
                    context_files=hydrated_context_files,
                    console_instance=console_instance,
                )

                # Create a stable local ID from the registry reference
                # Format: "185-209-179-245.nip.io/img-ci1tzd1ba079:20251027213343"
                safe_ref = (
                    registry_ref.split("/")[1].replace("/", "-").replace(":", "-")
                )
                object_id = f"{safe_ref}"

            except Exception as e:
                if console_instance:
                    console_instance.error("Image build failed", str(e))
                raise ValidationError(f"Image build failed: {e}", field="build")

            self._hydrate(object_id, client, {"registry_ref": registry_ref})

        # Create representation for debugging
        rep = f"Image({dockerfile_function.rep if dockerfile_function else None})"

        img = _Image._from_loader(_load, rep, deps=_deps)

        # Store base_images and dockerfile_function so they can be accessed during collection
        # These are set immediately upon creation, not just during _load
        img._collected_base_images = base_images
        img._collected_dockerfile_function = dockerfile_function

        # Ensure _runtime_injected flag is initialized if not already set
        if not hasattr(img, '_runtime_injected'):
            img._runtime_injected = False

        return img

    def apt_install(
        self,
        *packages: Union[str, List[str]],
    ) -> "_Image":
        pkgs = _flatten_str_args("apt_install", "packages", list(packages))
        if not pkgs:
            return self

        package_args = " ".join(pkgs)

        @_dockerfile_function_rep("apt_install")
        def build_dockerfile():
            commands = [
                "RUN apt-get update",
                f"RUN apt-get install -y {package_args}",
            ]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={'base': self},
            dockerfile_function=build_dockerfile,
        )

    def pip_install(
        self,
        *packages: Union[str, List[str]],
        find_links: Optional[str] = None,
        index_url: Optional[str] = None,
        extra_index_url: Optional[str] = None,
        pre: bool = False,
        extra_options: str = "",
    ) -> "_Image":

        pkgs = _flatten_str_args("pip_install", "packages", list(packages))
        if not pkgs:
            return self

        if any(pkg.startswith("-") for pkg in pkgs):
            raise ValidationError(
                "Package list for `Image.pip_install` cannot contain other arguments; try the `extra_options` parameter instead.",
                field="packages",
            )

        @_dockerfile_function_rep("pip_install")
        def build_dockerfile():
            package_args = " ".join(pkgs)
            extra_args = _make_pip_install_args(
                find_links, index_url, extra_index_url, pre, extra_options
            )
            commands = [f"RUN python -m pip install {package_args} {extra_args}"]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def pip_install_from_requirements(
        self,
        requirements_txt: str,
        find_links: Optional[str] = None,
        *,
        index_url: Optional[str] = None,
        extra_index_url: Optional[str] = None,
        pre: bool = False,
        extra_options: str = "",
    ) -> "_Image":
        # Expand user path and validate file exists
        requirements_txt_path = os.path.expanduser(requirements_txt)
        requirements_path = Path(requirements_txt_path).resolve()

        if not requirements_path.exists():
            raise ValidationError(f"Requirements file not found: {requirements_txt}")

        if not requirements_path.is_file():
            raise ValidationError(f"Path is not a file: {requirements_txt}")

        context_filename = ".requirements.txt"

        @_dockerfile_function_rep("pip_install_from_requirements")
        def build_dockerfile():
            context_files = {context_filename: str(requirements_path)}
            extra_args = _make_pip_install_args(
                find_links, index_url, extra_index_url, pre, extra_options
            )

            commands = [
                f"COPY {context_filename} /.requirements.txt",
                f"RUN python -m pip install -r /.requirements.txt {extra_args}".strip(),
            ]

            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def dockerfile_commands(
        self,
        *dockerfile_commands: Union[str, List[str]],
        context_files: Dict[str, str] = {},
    ) -> "_Image":
        if not isinstance(context_files, dict):
            raise ValidationError(
                "context_files must be a dictionary",
                field="context_files",
                value=type(context_files).__name__,
            )

        cmds = _flatten_str_args(
            "dockerfile_commands", "dockerfile_commands", list(dockerfile_commands)
        )
        if not cmds:
            return self

        @_dockerfile_function_rep("dockerfile_commands")
        def build_dockerfile():
            return DockerfileSpec(commands=[*cmds], context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def entrypoint(
        self,
        entrypoint_commands: List[str],
    ) -> "_Image":
        if not isinstance(entrypoint_commands, list) or not all(
            isinstance(x, str) for x in entrypoint_commands
        ):
            raise ValidationError("entrypoint_commands must be a list of string")

        args_str = _flatten_str_args(
            "entrypoint", "entrypoint_commands", entrypoint_commands
        )
        args_str = '"' + '", "'.join(args_str) + '"' if args_str else ""

        @_dockerfile_function_rep("entrypoint")
        def build_dockerfile():
            dockerfile_cmd = f"ENTRYPOINT [{args_str}]"
            return DockerfileSpec(commands=[dockerfile_cmd], context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def run_commands(
        self,
        *commands: Union[str, List[str]],
    ) -> "_Image":
        cmds = _flatten_str_args("run_commands", "commands", list(commands))
        if not cmds:
            return self

        @_dockerfile_function_rep("run_commands")
        def build_dockerfile() -> DockerfileSpec:
            return DockerfileSpec(
                commands=[f"RUN {cmd}" for cmd in cmds], context_files={}
            )

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def env(self, vars: Dict[str, str]) -> "_Image":
        if not isinstance(vars, dict):
            raise ValidationError(
                "env() expects a dictionary", field="vars", value=type(vars).__name__
            )

        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in vars.items()
        ):
            raise ValidationError("ENV keys and values must be strings", field="vars")

        @_dockerfile_function_rep("env")
        def build_dockerfile() -> DockerfileSpec:
            env_commands = [
                f"ENV {key}={shlex.quote(val)}" for (key, val) in vars.items()
            ]
            return DockerfileSpec(commands=env_commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def workdir(self, path: Union[str, PurePosixPath]) -> "_Image":
        @_dockerfile_function_rep("workdir")
        def build_dockerfile() -> DockerfileSpec:
            commands = [f"WORKDIR {shlex.quote(str(path))}"]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def cmd(self, cmd: List[str]) -> "_Image":
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise ValidationError("Image CMD must be a list of strings", field="cmd")

        cmd_str = _flatten_str_args("cmd", "cmd", cmd)
        cmd_str = '"' + '", "'.join(cmd_str) + '"' if cmd_str else ""

        @_dockerfile_function_rep("cmd")
        def build_dockerfile() -> DockerfileSpec:
            dockerfile_cmd = f"CMD [{cmd_str}]"
            return DockerfileSpec(commands=[dockerfile_cmd], context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    @contextlib.contextmanager
    def imports(self):
        is_remote = (
            os.environ.get("TARGON_ENVIRONMENT") == "remote"
            or os.environ.get("APP_MODULE") is not None
        )

        try:
            yield
        except Exception as exc:
            # If we're in remote environment, always raise
            if is_remote:
                raise

            if isinstance(exc, ImportError):
                pass
            else:
                warnings.warn(
                    f"Warning: Caught a non-ImportError exception in an `image.imports()` block: {repr(exc)}\n"
                    f"This might indicate a problem in your import code beyond just missing packages."
                )

    @staticmethod
    def from_dockerfile(
        path: Union[str, Path],
        *,
        context_dir: Optional[Union[Path, str]] = None,
        ignore: Union[Sequence[str], Callable[[Path], bool]] = [],
    ) -> "_Image":

        # Expand and validate Dockerfile path
        dockerfile_path = Path(os.path.expanduser(path)).resolve()

        if not dockerfile_path.exists():
            raise ValidationError(f"Dockerfile not found: {path}")

        if not dockerfile_path.is_file():
            raise ValidationError(f"Path is not a file: {path}")

        # Determine context directory
        if context_dir is None:
            # Default to directory containing the Dockerfile
            resolved_context_dir = dockerfile_path.parent
        else:
            resolved_context_dir = Path(os.path.expanduser(context_dir)).resolve()

            if not resolved_context_dir.exists():
                raise ValidationError(f"Context directory not found: {context_dir}")

            if not resolved_context_dir.is_dir():
                raise ValidationError(f"Context path is not a directory: {context_dir}")

        # Create ignore function
        ignore_fn = _create_ignore_function(ignore)

        # Collect context files
        collected_files = _collect_directory_files(resolved_context_dir, ignore_fn)

        # Create context files mapping
        context_files = {}
        for local_file_path, rel_path in collected_files:
            # Use relative path as-is for context files to match Dockerfile COPY commands
            context_files[str(rel_path)] = str(local_file_path)

        @_dockerfile_function_rep("from_dockerfile")
        def build_dockerfile():
            # Read Dockerfile contents
            with open(os.path.expanduser(path), 'r') as f:
                dockerfile_contents = f.read()

            # Split into commands (lines)
            commands = [line for line in dockerfile_contents.split('\n')]

            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
        )

    @staticmethod
    def debian_slim(python_version: Optional[str] = None) -> "_Image":
        """Create a Debian-based slim image."""
        if python_version is None:
            python_version = "3.11"

        # Handle float inputs (common mistake)
        if isinstance(python_version, float):
            python_version = str(python_version)

        # Create base image tag
        base_image = f"python:{python_version}-slim"

        @_dockerfile_function_rep("debain_slim")
        def build_dockerfile():
            commands = [
                f"FROM {base_image}",
                "RUN apt-get update",
                "RUN apt-get install -y gcc gfortran build-essential",
                "RUN pip install --upgrade pip",
                "RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections",
            ]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
        )

    @staticmethod
    def _registry_setup_commands(
        tag: str,
        setup_commands: List[str],
        add_python: Optional[str] = None,
    ) -> List[str]:
        """Generate Dockerfile commands for registry images with optional Python installation"""
        if not add_python:
            return [f"FROM {tag}", *setup_commands]

        # Validate Python version format (major.minor only)
        if not re.match(r'^3\.\d{1,2}$', add_python):
            raise ValidationError(
                f"Invalid Python version: {add_python}. "
                "Expected format like '3.11', '3.12', etc."
            )

        major, minor = add_python.split('.')
        minor_int = int(minor)

        # Build package list - distutils is deprecated in 3.12+
        python_packages = [
            f"python{add_python}",
            f"python{add_python}-dev",
            f"python{add_python}-venv",
        ]

        # Only add distutils for Python < 3.12
        if minor_int < 12:
            python_packages.append(f"python{add_python}-distutils")

        packages_str = " ".join(python_packages)

        commands = [
            f"FROM {tag}",
            # Run any user-provided setup commands first
            *setup_commands,
            # Install Python using deadsnakes PPA (for Ubuntu/Debian)
            # Use DEBIAN_FRONTEND=noninteractive to avoid prompts
            "ENV DEBIAN_FRONTEND=noninteractive",
            "RUN apt-get update",
            "RUN apt-get install -y software-properties-common",
            "RUN add-apt-repository -y ppa:deadsnakes/ppa",
            "RUN apt-get update",
            f"RUN apt-get install -y {packages_str}",
            # Install pip using get-pip.py (most reliable method)
            "RUN apt-get install -y curl ca-certificates",
            "RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py",
            f"RUN python{add_python} /tmp/get-pip.py",
            "RUN rm /tmp/get-pip.py",
            # Create symlinks so 'python' and 'pip' work without version suffix
            f"RUN ln -sf /usr/bin/python{add_python} /usr/local/bin/python",
            f"RUN ln -sf /usr/bin/python{add_python} /usr/local/bin/python3",
            f"RUN ln -sf /usr/local/bin/pip{major} /usr/local/bin/pip || true",
            # Set environment
            "ENV PATH=/usr/local/bin:$PATH",
            # Verify installation
            "RUN python --version && pip --version",
            # Clean up
            "RUN apt-get clean && rm -rf /var/lib/apt/lists/*",
        ]

        return commands

    @staticmethod
    def from_registry(
        tag: str,
        *,
        setup_dockerfile_commands: List[str] = [],
        force_build: bool = False,
        add_python: Optional[str] = None,
    ) -> "_Image":
        """Build a Targon Image from a public or private image registry"""
        if not tag or not isinstance(tag, str) or not tag.strip():
            raise ValidationError(
                "tag must be a non-empty string", field="tag", value=tag
            )

        if not isinstance(setup_dockerfile_commands, list):
            raise ValidationError(
                "setup_dockerfile_commands must be a list",
                field="setup_dockerfile_commands",
                value=type(setup_dockerfile_commands).__name__,
            )

        @_dockerfile_function_rep("from_registry")
        def build_dockerfile():
            commands = _Image._registry_setup_commands(
                tag, setup_dockerfile_commands, add_python
            )
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
            force_build=force_build,
        )

    def add_local_file(
        self, local_path: Union[str, Path], remote_path: str, *, copy: bool = True
    ) -> "_Image":
        """Adds a local file to the image at `remote_path` within the container"""
        # Validate that copy=True (Targon doesn't support lazy mounting yet)
        if not copy:
            raise ValidationError(
                "Targon SDK currently only supports copy=True for add_local_file(). "
                "Files are always copied into the image layer at build time."
            )

        # Validate remote_path is absolute
        if not PurePosixPath(remote_path).is_absolute():
            raise ValidationError(
                "image.add_local_file() currently only supports absolute remote_path values"
            )

        # Convert local_path to Path object
        local_file = Path(local_path).expanduser().resolve()

        # Validate local file exists
        if not local_file.exists():
            raise ValidationError(f"Local file not found: {local_path}")

        if not local_file.is_file():
            raise ValidationError(f"Path is not a file: {local_path}")

        # If remote_path ends with "/", append the filename
        if remote_path.endswith("/"):
            remote_path = remote_path + local_file.name

        # Create a unique context file name to avoid collisions
        context_filename = f".local_file_{local_file.name}"

        @_dockerfile_function_rep("add_local_file")
        def build_dockerfile():
            commands = [
                f"COPY {context_filename} {remote_path}",
            ]
            context_files = {context_filename: str(local_file)}
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def add_local_dir(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        *,
        copy: bool = True,
        ignore: Union[Sequence[str], Callable[[Path], bool]] = [],
    ) -> "_Image":
        """Adds a local directory's content to the image at `remote_path` within the container."""
        # Validate that copy=True (Targon doesn't support lazy mounting yet)
        if not copy:
            raise ValidationError(
                "Targon SDK currently only supports copy=True for add_local_dir(). "
                "Files are always copied into the image layer at build time."
            )

        # Validate remote_path is absolute
        if not PurePosixPath(remote_path).is_absolute():
            raise ValidationError(
                "image.add_local_dir() currently only supports absolute remote_path values"
            )

        # Convert local_path to Path object
        local_dir = Path(local_path).expanduser().resolve()

        # Validate local directory exists
        if not local_dir.exists():
            raise ValidationError(f"Local directory not found: {local_path}")

        if not local_dir.is_dir():
            raise ValidationError(f"Path is not a directory: {local_path}")

        # Create ignore function
        ignore_fn = _create_ignore_function(ignore)

        # Collect all files from directory
        collected_files = _collect_directory_files(local_dir, ignore_fn)

        if not collected_files:
            raise ValidationError(
                f"No files found in directory (or all ignored): {local_path}"
            )

        # Create context files mapping with unique names
        context_files = {}
        copy_commands = []
        parent_dirs = set()

        for local_file_path, rel_path in collected_files:
            # Create a unique context filename to avoid collisions
            # Use a safe version of the relative path
            safe_name = str(rel_path).replace("/", "_").replace("\\", "_")
            context_filename = f".local_dir_{safe_name}"

            # Map context filename to actual file path
            context_files[context_filename] = str(local_file_path)

            # Create remote path preserving directory structure
            remote_file_path = PurePosixPath(remote_path) / rel_path.as_posix()

            # Collect parent directory
            parent_dirs.add(str(remote_file_path.parent))

            # Store copy command (we'll add mkdir commands before these)
            copy_commands.append(f"COPY {context_filename} {remote_file_path}")

        # Create all parent directories at once before any COPY commands
        if parent_dirs:
            # Sort for consistent ordering and remove duplicates
            sorted_dirs = sorted(parent_dirs)
            mkdir_cmd = f"RUN mkdir -p {' '.join(sorted_dirs)}"
            copy_commands.insert(0, mkdir_cmd)

        @_dockerfile_function_rep("add_local_dir")
        def build_dockerfile():
            return DockerfileSpec(commands=copy_commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def _inject_runtime(
        self, app_file: Union[str, Path], app_module_name: str = "app"
    ) -> "_Image":
        """
        Internal method to inject runtime into an image.
        Called automatically during image hydration when used with an App.
        """
        # Get path to targon runtime and proto files
        targon_root = Path(__file__).parent.parent  # src/targon/
        runtime_dir = targon_root / "runtime"
        proto_dir = targon_root / "proto"

        # Validate files exist
        runtime_file = runtime_dir / "universal_runtime.py"
        asgi_file = runtime_dir / "asgi.py"
        runtime_init_file = runtime_dir / "__init__.py"

        if not runtime_file.exists():
            raise ValidationError(f"Runtime file not found: {runtime_file}")
        if not asgi_file.exists():
            raise ValidationError(f"ASGI file not found: {asgi_file}")
        if not proto_dir.exists():
            raise ValidationError(f"Proto directory not found: {proto_dir}")

        app_file_path = Path(app_file).resolve()
        if not app_file_path.exists():
            raise ValidationError(f"App file not found: {app_file}")

        @_dockerfile_function_rep("universal_runtime")
        def build_dockerfile():
            commands = [
                # Install all dependencies in one layer for efficiency
                "RUN pip install --no-cache-dir \\\n"
                "    'grpcio>=1.60.0' \\\n"
                "    'grpcio-health-checking>=1.60.0' \\\n"
                "    'fastapi>=0.100.0' \\\n"
                "    'uvicorn[standard]>=0.23.0' \\\n"
                "    'cloudpickle>=2.0.0' \\\n"
                "    'protobuf>=4.25.0' \\\n"
                "    'a2wsgi>=1.10.0' \\\n"
                "    'httpx>=0.24.0'",
                # Download grpc_health_probe for Knative
                "RUN apt-get update && apt-get install -y wget && \\",
                "    wget -qO /bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.19/grpc_health_probe-linux-amd64 && \\",
                "    chmod +x /bin/grpc_health_probe && \\",
                "    apt-get clean && rm -rf /var/lib/apt/lists/*",
                # Create app directory structure
                "RUN mkdir -p /app/targon/proto /app/targon/runtime",
                # Copy runtime files
                "COPY universal_runtime.py /app/targon/runtime/universal_runtime.py",
                "COPY asgi.py /app/targon/runtime/asgi.py",
                "COPY runtime_init.py /app/targon/runtime/__init__.py",
                # Copy proto files
                "COPY function_execution_pb2.py /app/targon/proto/function_execution_pb2.py",
                "COPY function_execution_pb2_grpc.py /app/targon/proto/function_execution_pb2_grpc.py",
                "COPY proto_init.py /app/targon/proto/__init__.py",
                # Copy targon package files
                "COPY targon_init.py /app/targon/__init__.py",
                "COPY version.py /app/targon/version.py",
                # Copy user's app file
                f"COPY {app_file_path.name} /app/{app_module_name}.py",
                # Set up environment
                "WORKDIR /app",
                "ENV PYTHONPATH=/app",
                f"ENV APP_MODULE={app_module_name}",
                # Expose both gRPC and web ports
                "EXPOSE 50051",
                "EXPOSE 8080",
                # Set entrypoint - TARGET_FUNCTION will be set by Knative at deployment time
                'ENTRYPOINT ["python", "-m", "targon.runtime.universal_runtime"]',
            ]

            # Map context files
            context_files = {
                # Runtime files
                "universal_runtime.py": str(runtime_file),
                "asgi.py": str(asgi_file),
                "runtime_init.py": str(runtime_init_file),
                # Proto files
                "function_execution_pb2.py": str(
                    proto_dir / "function_execution_pb2.py"
                ),
                "function_execution_pb2_grpc.py": str(
                    proto_dir / "function_execution_pb2_grpc.py"
                ),
                "proto_init.py": str(proto_dir / "__init__.py"),
                # Targon package files
                "targon_init.py": str(targon_root / "runtime" / "runtime_init.py"),
                "version.py": str(targon_root / "version.py"),
                # User app
                app_file_path.name: str(app_file_path),
            }

            return DockerfileSpec(commands=commands, context_files=context_files)

        img = _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )
        img._runtime_injected = True
        return img

    def with_runtime(
        self, app_file: Union[str, Path], app_module_name: str = "app"
    ) -> "_Image":
        return self._inject_runtime(app_file, app_module_name)


Image = _Image
