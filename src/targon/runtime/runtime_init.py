"""
Minimal runtime __init__.py for targon containers.

This module provides stub classes for Image and App that allow user app modules
to import and execute module-level code without errors, even though the actual
functionality isn't needed at runtime (image already built, app already configured).
"""

import contextlib


class _StubImage:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        """Return a method that returns self for any attribute access."""

        def method(*args, **kwargs):
            return self

        return method

    @contextlib.contextmanager
    def imports(self):
        yield

    @classmethod
    def debian_slim(cls, python_version=None):
        """Stub debian_slim class method."""
        return cls()

    @classmethod
    def from_registry(cls, *args, **kwargs):
        """Stub from_registry class method."""
        return cls()

    @classmethod
    def from_dockerfile(cls, *args, **kwargs):
        """Stub from_dockerfile class method."""
        return cls()

    def pip_install(self, *args, **kwargs):
        """Stub pip_install method."""
        return self

    def apt_install(self, *args, **kwargs):
        """Stub apt_install method."""
        return self

    def pip_install_from_requirements(self, *args, **kwargs):
        """Stub pip_install_from_requirements method."""
        return self

    def add_local_file(self, *args, **kwargs):
        """Stub add_local_file method."""
        return self

    def add_local_dir(self, *args, **kwargs):
        """Stub add_local_dir method."""
        return self

    def env(self, *args, **kwargs):
        """Stub env method."""
        return self

    def workdir(self, *args, **kwargs):
        """Stub workdir method."""
        return self

    def with_runtime(self, *args, **kwargs):
        """Stub with_runtime method."""
        return self

    def run_commands(self, *args, **kwargs):
        """Stub run_commands method."""
        return self


class _StubApp:
    def __init__(self, *args, **kwargs):
        """Accept any arguments to match the real App signature."""
        pass

    def function(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def local_entrypoint(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


class _StubCompute:
    """Stub Compute class with tier name constants."""
    # CPU Tiers
    CPU_SMALL = "cpu-small"
    CPU_MEDIUM = "cpu-medium"
    CPU_LARGE = "cpu-large"
    CPU_XL = "cpu-xl"

    # GPU Tiers
    H200_SMALL = "h200-small"
    H200_MEDIUM = "h200-medium"
    H200_LARGE = "h200-large"
    H200_XL = "h200-xl"

def _stub_decorator(*args, **kwargs):
    def decorator(func):
        return func

    # Support both @decorator and @decorator()
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return decorator


fastapi_endpoint = _stub_decorator
asgi_app = _stub_decorator
wsgi_app = _stub_decorator
web_server = _stub_decorator

Image = _StubImage
App = _StubApp
Compute = _StubCompute

__version__ = "0.1.0"
__all__ = [
    "Image",
    "App",
    "Compute",
    "fastapi_endpoint",
    "asgi_app",
    "wsgi_app",
    "web_server",
    "__version__",
]
