from typing import Any, Optional, Callable, Dict, List, AsyncIterator
from contextlib import asynccontextmanager

from targon.core.exceptions import ValidationError
from targon.core.resources import Compute
from targon.core.objects import BaseApp, _LocalEntrypoint
from targon.core.partial_function import _PartialFunction
from targon.core.function import _Function
from targon.core.image import _Image


class _App(BaseApp):
    _name: Optional[str]
    _description: Optional[str]
    _tags: Dict[str, str]
    _functions: Dict[str, _Function]
    _classes: Dict[str, Any]
    _image: _Image
    _local_entrypoints: Dict[str, _LocalEntrypoint]
    _web_endpoints: List[str]
    _app_id: Optional[str]
    _project_name: str

    def __init__(self, name: str, image: _Image, project_name: str = "default") -> None:
        if not name or not isinstance(name, str) or not name.strip():
            raise ValidationError(
                "name must be a non-empty string", field="name", value=name
            )

        if not isinstance(image, _Image):
            raise ValidationError(
                "image must be a valid _Image instance",
                field="image",
                value=type(image).__name__,
            )

        if (
            not project_name
            or not isinstance(project_name, str)
            or not project_name.strip()
        ):
            raise ValidationError(
                "project_name must be a non-empty string",
                field="project_name",
                value=project_name,
            )

        self._name = name.strip()
        self._image = image
        self._project_name = project_name.strip()
        self._functions = {}
        self._classes = {}
        self._local_entrypoints = {}
        self._web_endpoints = []
        self._app_id = None

    def function(
        self,
        *,
        image: Optional[_Image] = None,
        resource: str = Compute.CPU_SMALL,
        min_replicas: int = 1,
        max_replicas: int = 3,
        timeout: int = 300,
        **kwargs: Any,
    ) -> Callable[[Any], _Function]:
        if min_replicas < 0:
            raise ValidationError(
                "min_replicas must be non-negative",
                field="min_replicas",
                value=min_replicas,
            )

        if max_replicas < 1:
            raise ValidationError(
                "max_replicas must be at least 1",
                field="max_replicas",
                value=max_replicas,
            )

        if min_replicas > max_replicas:
            raise ValidationError(
                f"min_replicas ({min_replicas}) cannot exceed max_replicas ({max_replicas})",
                field="min_replicas",
            )

        if timeout < 0:
            raise ValidationError(
                "timeout must be non-negative", field="timeout", value=timeout
            )

        if not resource or not isinstance(resource, str):
            raise ValidationError(
                "resource must be a non-empty string", field="resource", value=resource
            )

        def wrapper(f: Any) -> _Function:
            if isinstance(f, _PartialFunction):
                raw_func = f.raw_f
                if f.is_web_endpoint and raw_func:
                    self._web_endpoints.append(raw_func.__name__)
                    webhook_config = f.webhook_config
            else:
                raw_func = f
                webhook_config = None

            if raw_func is None:
                raise ValidationError(
                    "Function is None - invalid PartialFunction state", field="function"
                )

            if not callable(raw_func):
                raise ValidationError(
                    "Decorated object must be callable",
                    field="function",
                    value=type(raw_func).__name__,
                )

            tag = getattr(raw_func, "__qualname__", raw_func.__name__)

            fn_obj = _Function.from_local(
                func=raw_func,
                app=self,
                image=image or self._image,
                name=tag,
                webhook_config=webhook_config,
                user_cls=None,
                resource_name=resource,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                timeout=timeout,
                **kwargs,
            )

            self._functions[tag] = fn_obj

            return fn_obj

        return wrapper

    def local_entrypoint(
        self, _warn_parentheses_missing: Any = None, *, name: Optional[str] = None
    ) -> Callable[[Callable[..., Any]], _LocalEntrypoint]:
        if _warn_parentheses_missing is not None:
            raise ValidationError(
                "Did you forget parentheses? Suggestion: `@app.local_entrypoint()`.",
                field="decorator_usage",
            )

        if name is not None:
            if not isinstance(name, str):
                raise ValidationError(
                    "name must be a string", field="name", value=type(name).__name__
                )
            if not name.strip():
                raise ValidationError(
                    "name cannot be an empty string", field="name", value=name
                )

        def wrapped(raw_f: Callable[..., Any]) -> _LocalEntrypoint:
            if not callable(raw_f):
                raise ValidationError(
                    "Decorated object must be callable",
                    field="function",
                    value=type(raw_f).__name__,
                )

            tag = name.strip() if name is not None else raw_f.__qualname__

            if tag in self._local_entrypoints:
                raise ValidationError(
                    f"Duplicate local entrypoint name: {tag}. Local entrypoint names must be unique.",
                    field="name",
                    value=tag,
                )

            entrypoint = self._local_entrypoints[tag] = _LocalEntrypoint(raw_f, self)
            return entrypoint

        return wrapped

    def deploy(self) -> "_App":
        return self

    def hydrate(self, app_id: str) -> None:
        if not app_id or not isinstance(app_id, str) or not app_id.strip():
            raise ValidationError(
                "app_id must be a non-empty string", field="app_id", value=app_id
            )
        self._app_id = app_id.strip()

    @asynccontextmanager
    async def run(self) -> AsyncIterator["_App"]:
        import inspect
        from targon.core.executor import run_app
        from targon.core.console import Console

        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            app_file_path = caller_frame.f_globals.get('__file__')
        else:
            app_file_path = None
        
        c = Console(self.name)
        with c:
            async with run_app(app=self, console_instance=c, app_file_path=app_file_path):
                yield self


App = _App
