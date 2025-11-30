from __future__ import annotations
import abc
import asyncio
import base64
import concurrent.futures
from contextlib import AbstractAsyncContextManager, aclosing
import dataclasses
import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import time
from typing import (
    Any,
    AsyncGenerator,
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Union,
)
from targon.core.config import logger
from targon.core.exceptions import TargonError, ValidationError
from targon.core.objects import _Object
from targon.utils.async_utils import async_map
from targon.core.resolver import Resolver

ROOT_DIR: PurePosixPath = PurePosixPath("/root")


class NonLocalMountError(Exception):
    """Raised when attempting to inspect entries on a non-local mount."""


class _MountEntry(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def description(self) -> str:
        ...

    @abc.abstractmethod
    def get_files_to_upload(self) -> Iterator[tuple[Path, PurePosixPath]]:
        ...

    @abc.abstractmethod
    def top_level_paths(self) -> list[tuple[Path, PurePosixPath]]:
        ...


@dataclasses.dataclass(slots=True)
class _MountFile(_MountEntry):
    local_file: Path
    remote_path: PurePosixPath

    def description(self) -> str:
        return str(self.local_file)

    def get_files_to_upload(self) -> Iterator[tuple[Path, PurePosixPath]]:
        local_file = self.local_file.resolve()
        if not local_file.exists():
            raise FileNotFoundError(f"Local file {local_file} does not exist")
        if not local_file.is_file():
            raise ValidationError(f"Path is not a file: {local_file}")
        yield local_file, self.remote_path

    def top_level_paths(self) -> list[tuple[Path, PurePosixPath]]:
        return [(self.local_file, self.remote_path)]


@dataclasses.dataclass(slots=True)
class _MountDir(_MountEntry):
    local_dir: Path
    remote_path: PurePosixPath
    ignore: Callable[[Path], bool]
    recursive: bool

    def description(self) -> str:
        return str(self.local_dir.expanduser().resolve())

    def get_files_to_upload(self) -> Iterator[tuple[Path, PurePosixPath]]:
        base_dir = self.local_dir.expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"Local directory {base_dir} does not exist")
        if not base_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {base_dir}")

        if self.recursive:
            walker: Iterable[tuple[str, list[str], list[str]]] = os.walk(base_dir)
        else:
            file_names: list[str] = []
            with os.scandir(base_dir) as scan_iter:
                for entry in scan_iter:
                    if entry.is_file():
                        file_names.append(entry.name)
            walker = [(str(base_dir), [], file_names)]

        for root, _, files in walker:
            root_path = Path(root)
            for filename in files:
                local_path = root_path / filename
                rel_path = local_path.relative_to(base_dir)
                if self.ignore(rel_path):
                    continue
                remote_path = self.remote_path / rel_path.as_posix()
                yield local_path, remote_path

    def top_level_paths(self) -> list[tuple[Path, PurePosixPath]]:
        return [(self.local_dir, self.remote_path)]


def _select_files(entries: list[_MountEntry]) -> list[tuple[Path, PurePosixPath]]:
    files: set[tuple[Path, PurePosixPath]] = set()
    for entry in entries:
        files |= set(entry.get_files_to_upload())
    return list(files)


@dataclasses.dataclass(slots=True)
class _FileUploadSpec:
    source: Callable[[], Union[AbstractAsyncContextManager, BinaryIO]]
    source_description: Any
    source_is_path: bool
    mount_filename: str

    use_blob: bool
    content: Optional[str]
    sha256_hex: str
    mode: int
    size: int


def _build_ignore_condition(
    ignore: Union[Sequence[str], Callable[[Path], bool], None],
) -> Callable[[Path], bool]:
    if callable(ignore):
        return ignore

    patterns = list(ignore or [])

    def _predicate(path: Path) -> bool:
        path_str = str(path)
        name = path.name
        for pattern in patterns:
            if pattern.startswith("*") and name.endswith(pattern[1:]):
                return True
            if pattern.endswith("*") and name.startswith(pattern[:-1]):
                return True
            if "*" in pattern and pattern.strip("*") in name:
                return True
            if pattern == name or path_str.endswith(pattern):
                return True
        return False

    return _predicate


HASH_CHUNK_SIZE = 65536
@dataclasses.dataclass
class Sha256:
    digest: bytes

    def sha256_base64(self) -> str:
        return base64.b64encode(self.digest).decode("ascii")

    def sha256_hex(self) -> str:
        return self.digest.hex()


def calculate_sha256_b64(data: Union[str, bytes, BinaryIO]) -> Sha256:
    t0 = time.monotonic()
    sha256 = hashlib.sha256()
    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(data, bytes):
        sha256.update(data)
    else:
        # Streamed reading for file-like objects
        assert not isinstance(data, (bytearray, memoryview))
        pos = data.tell()
        while True:
            chunk = data.read(HASH_CHUNK_SIZE)
            if not isinstance(chunk, bytes):
                raise ValueError(f"Expected bytes, got {type(chunk)}")
            if not chunk:
                break
            sha256.update(chunk)
        data.seek(pos)

    ret = Sha256(sha256.digest())
    logger.debug("calculate_sha256_b64 took %.3fs", time.monotonic() - t0)
    return ret

LARGE_FILE_LIMIT = 4 * 1024 * 1024  # 4 MiB

def _get_file_upload_spec(source_description: Path, mount_filename: PurePosixPath) -> _FileUploadSpec:
    source = lambda: open(source_description, "rb")
    mode = os.stat(source_description).st_mode & (0o7777 if platform.system() != "Windows" else 0o7755)

    with source() as fp:
        # Current position is ignored - we always upload from position 0
        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        fp.seek(0)

        if size >= LARGE_FILE_LIMIT:
            use_blob = True
            content = None
            hash = calculate_sha256_b64(fp)
        else:
            use_blob = False
            content = base64.b64encode(fp.read()).decode("ascii")
            hash = calculate_sha256_b64(content)

    return _FileUploadSpec(
        source=source,
        source_description=source_description,
        source_is_path=isinstance(source_description, Path),
        mount_filename=mount_filename.as_posix(),
        use_blob=use_blob,
        content=content,
        sha256_hex=hash.sha256_hex(),
        mode=mode & 0o7777,
        size=size,
    )

@dataclasses.dataclass
class MountManifestEntry:
    filename: str
    sha256_hex: str
    mode: int


class _Mount(_Object, type_prefix="mnt"):
    """Mount encapsulates files to be synchronized alongside an image without rebuilding it."""

    _entries: Optional[list[_MountEntry]] = None
    _content_checksum_sha256_hex: Optional[str] = None
    _manifest: Sequence[MountManifestEntry] = ()

    @staticmethod
    def _new(entries: Optional[list[_MountEntry]] = []) -> "_Mount":
        rep = f"Mount({', '.join(entry.description() for entry in entries)})"
        obj = _Mount._from_loader(_Mount._load_mount, rep)
        obj._entries = entries
        obj._is_local = True
        return obj

    def _extend(self, entry: _MountEntry) -> "_Mount":
        return _Mount._new(self.entries + [entry])

    @property
    def entries(self) -> list[_MountEntry]:
        if self._entries is None:
            raise NonLocalMountError()
        return self._entries

    def is_local(self) -> bool:
        return getattr(self, "_is_local", False)

    def _hydrate_metadata(self, metadata: Optional[dict]):
        if not metadata:
            return
        self._content_checksum_sha256_hex = metadata.get("checksum")
        manifest = metadata.get("manifest", [])
        self._manifest = tuple(
            MountManifestEntry(filename=entry["filename"], sha256_hex=entry["sha256_hex"], size=entry["size"])
            for entry in manifest
        )

    def _get_metadata(self) -> Optional[dict]:
        if not self._content_checksum_sha256_hex:
            return None
        return {
            "checksum": self._content_checksum_sha256_hex,
            "manifest": [dataclasses.asdict(entry) for entry in self._manifest],
        }
    @staticmethod
    def _description(entries: list[_MountEntry]) -> str:
        local_contents = [e.description() for e in entries]
        return ", ".join(local_contents)

    @staticmethod
    async def _get_files(entries: list[_MountEntry]) -> AsyncGenerator[_FileUploadSpec, None]:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_files = await loop.run_in_executor(executor, _select_files, entries)
            futures = [
                loop.run_in_executor(executor, _get_file_upload_spec, local_path, remote_path)
                for local_path, remote_path in all_files
            ]
            logger.debug(f"Computing checksums for {len(futures)} files using {executor._max_workers} worker threads")
            for future in asyncio.as_completed(futures):
                try:
                    yield await future
                except FileNotFoundError:
                    logger.info(f"Ignoring file not found: {executor}")

    async def _load_mount(self, resolver: Resolver, existing_object_id: Optional[str]):
        n_concurrent_uploads = 512
        files: list[MountManifestEntry] = []

        n_seen, n_finished = 0, 0
        total_uploads, total_bytes = 0, 0
        accounted_hashes: set[str] = set()
        message_label = _Mount._description(self._entries)

        async def _put_file(file_spec: _FileUploadSpec) -> MountManifestEntry:
            nonlocal n_seen, n_finished, total_uploads, total_bytes
            n_seen += 1
            logger.debug(f"Creating mount {message_label}: Uploaded {n_finished}/{n_seen} files")
            mount_client = resolver.client.async_mount

            mount_file_manifest = MountManifestEntry(
                filename=file_spec.mount_filename,
                sha256_hex=file_spec.sha256_hex,
                mode=file_spec.mode,
            )

            if file_spec.sha256_hex in accounted_hashes:
                n_finished += 1
                return mount_file_manifest

            exists = await mount_client.mount_put_file(sha256_hex=file_spec.sha256_hex)
            if exists:
                n_finished += 1
                return mount_file_manifest

            total_uploads += 1
            total_bytes += file_spec.size
            # Uploading to tha
            if file_spec.use_blob:
                raise TargonError("The use of file more than 4MB is not implemented.")
            else:
                start_time = time.monotonic()
                while time.monotonic() - start_time < 10 * 60:
                    exists = await mount_client.mount_put_file(
                        sha256_hex=file_spec.sha256_hex, data=file_spec.content
                    )
                    if exists:
                        n_finished += 1
                        return mount_file_manifest

                logger.debug(
                    "Prepared inline upload for %s (%s bytes)",
                    file_spec.source_description,
                    file_spec.size,
                )

        async with aclosing(
            async_map(
                _Mount._get_files(self.entries),
                _put_file,
                concurrency=n_concurrent_uploads,
            )
        ) as stream:
            async for file in stream:
                files.append(file)
                
        if not files:
            logger.warning(f"Mount of '{message_label}' is empty.")


        logger.info("Now regesiter all the checksum to tha and get the mount id !")
        manifest = [
            MountManifestEntry(
                filename=file_spec.mount_filename,
                sha256_hex=file_spec.sha256_hex,
                size=file_spec.size,
            )
            for file_spec in sorted(files, key=lambda spec: spec.mount_filename)
        ]

        object_id = f"mnt-xxxxxx"
        metadata = {"checksum": "xxxxxx", "manifest": [dataclasses.asdict(entry) for entry in manifest]}
        self._hydrate(object_id, resolver.client, metadata)

    @staticmethod
    def _ensure_remote_path(remote_path: Union[str, PurePosixPath, None], fallback: str) -> PurePosixPath:
        remote = PurePosixPath("/", remote_path or fallback)
        if not remote.is_absolute():
            raise ValidationError("Remote path for mounts must be absolute")
        return remote

    @staticmethod
    def _from_local_file(local_path: Union[str, Path], remote_path: Union[str, PurePosixPath, None] = None) -> "_Mount":
        return _Mount._new().add_local_file(local_path,remote_path)

    @staticmethod
    def _from_local_dir(
        local_path: Union[str, Path],
        *,
        remote_path: Union[str, PurePosixPath, None] = None,
        ignore: Union[Sequence[str], Callable[[Path], bool], None] = None,
        recursive: bool = True,
    ) -> "_Mount":
        return _Mount._new().add_local_dir(
            local_path, remote_path=remote_path, ignore=ignore, recursive=recursive
        )

    def add_local_file(
        self, local_path: Union[str, Path], remote_path: Union[str, PurePosixPath, None] = None
    ) -> "_Mount":
        path = Path(local_path).expanduser()
        remote = _Mount._ensure_remote_path(remote_path, path.name)
        return self._extend(
            _MountFile(
                local_file=path,
                remote_path=remote,
            )
        )

    def add_local_dir(
        self,
        local_path: Union[str, Path],
        *,
        remote_path: Union[str, PurePosixPath, None] = None,
        ignore: Union[Sequence[str], Callable[[Path], bool], None] = None,
        recursive: bool = True,
    ) -> "_Mount":
        path = Path(local_path).expanduser()
        remote = _Mount._ensure_remote_path(remote_path, path.name)
        ignore_func = _build_ignore_condition(ignore)
        return self._extend(
            _MountDir(
                local_dir=path,
                remote_path=remote,
                ignore=ignore_func,
                recursive=recursive,
            )
        )


Mount = _Mount

