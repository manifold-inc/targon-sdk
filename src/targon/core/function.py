from __future__ import annotations

import asyncio
import cloudpickle
import functools
import inspect
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING, Tuple, TypeVar, Union
from urllib.parse import urlparse

from targon.client.functions import AutoscalerSettings, FunctionMetadata
from targon.client.invocation import GRPCFunctionClient
from targon.core.exceptions import (
    HydrationError,
    NetworkError,
    ServerlessError,
    ValidationError,
)
from targon.core.objects import _Object, BaseApp, live_method
from targon.core.resources import Compute

if TYPE_CHECKING:
    from targon.core.image import _Image
    from targon.core.partial_function import WebhookConfig
    from targon.core.resolver import Resolver


class DefinitionType(Enum):
    FILE = "FILE"
    SERIALIZED = "SERIALIZED"


def is_notebook_function(func: Callable) -> bool:
    """Detect if a function is defined in a Jupyter notebook."""
    try:
        module = inspect.getmodule(func)
        if module is None or not hasattr(module, "__file__"):
            return True

        if module.__name__ == "__main__":
            import sys

            return "ipykernel" in sys.modules

        return False
    except Exception:
        return True


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Metadata for a function invocation."""

    function_id: str
    function_name: str
    request_id: str
    args: bytes
    kwargs: bytes
    grpc_endpoint: str
    timeout: int = 21600
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_coefficient: float = 2.0


class _Invocation:
    __slots__ = ("invocation_context",)

    def __init__(self, invocation_context: InvocationContext) -> None:
        self.invocation_context = invocation_context

    @classmethod
    async def create(
        cls,
        function: _Function,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout: int = 21600,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_coefficient: float = 2.0,
    ) -> _Invocation:
        """Create invocation from function object and arguments."""
        if not function._name:
            raise HydrationError(
                "Function name is not set. Function may not be properly initialized.",
                object_type="Function",
                object_id=getattr(function, "_object_id", None),
            )
        if not function._grpc_endpoint:
            raise HydrationError(
                f"gRPC endpoint not available for function '{function._name}'. "
                "Function may not be properly hydrated or deployed.",
                object_type="Function",
                object_id=getattr(function, "_object_id", None),
            )

        kwargs = kwargs or {}

        try:
            args_bytes = cloudpickle.dumps(args) if args else b""
            kwargs_bytes = cloudpickle.dumps(kwargs) if kwargs else b""
        except Exception as e:
            raise ServerlessError(
                f"Failed to serialize function arguments: {e}",
                function_id=getattr(function, "_object_id", None),
            ) from e

        grpc_endpoint = (
            function._grpc_endpoint
            if isinstance(function._grpc_endpoint, str)
            else f"{function._grpc_endpoint[0]}:{function._grpc_endpoint[1]}"
        )

        request_id = str(uuid.uuid4())

        return cls(
            InvocationContext(
                function_id=function.object_id,
                function_name=function._name,
                request_id=request_id,
                args=args_bytes,
                kwargs=kwargs_bytes,
                grpc_endpoint=grpc_endpoint,
                timeout=timeout,
                max_retries=max_retries,
                initial_delay=initial_delay,
                backoff_coefficient=backoff_coefficient,
            )
        )

    async def run_function(self) -> Any:
        """Execute the function remotely with retry logic."""
        ctx = self.invocation_context
        parsed_url = urlparse(ctx.grpc_endpoint)
        grpc_host = parsed_url.netloc or ctx.grpc_endpoint

        client = GRPCFunctionClient(host=grpc_host, port=443, timeout=ctx.timeout)

        try:
            args = cloudpickle.loads(ctx.args) if ctx.args else ()
            kwargs = cloudpickle.loads(ctx.kwargs) if ctx.kwargs else {}
        except Exception as e:
            raise ServerlessError(
                f"Failed to deserialize function arguments: {e}",
                function_id=ctx.function_id,
                execution_id=ctx.request_id,
            ) from e

        delay = ctx.initial_delay
        last_error: Exception | None = None

        for attempt in range(ctx.max_retries + 1):
            try:
                return await client.invoke(
                    *args, timeout=ctx.timeout, request_id=ctx.request_id, **kwargs
                )
            except Exception as e:
                last_error = e
                is_retryable = self._is_retryable_error(e)

                if attempt < ctx.max_retries and is_retryable:
                    await asyncio.sleep(delay)
                    delay = min(delay * ctx.backoff_coefficient, 60.0)
                else:
                    if is_retryable:
                        raise NetworkError(
                            f"Function invocation failed after {ctx.max_retries + 1} attempts: {e}",
                            cause=e,
                        ) from e
                    else:
                        raise ServerlessError(
                            f"Function invocation failed: {e}",
                            function_id=ctx.function_id,
                            execution_id=ctx.request_id,
                        ) from e

        if last_error:
            raise ServerlessError(
                f"Function invocation failed unexpectedly: {last_error}",
                function_id=ctx.function_id,
                execution_id=ctx.request_id,
            ) from last_error
        raise ServerlessError(
            "Function invocation failed with no error captured",
            function_id=ctx.function_id,
            execution_id=ctx.request_id,
        )

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Check if error is retryable (network/connection issues)."""
        error_msg = str(error).lower()
        return any(
            pattern in error_msg
            for pattern in (
                "connection",
                "network",
                "unavailable",
                "refused",
                "timeout",
                "grpc error",
            )
        )


class _Function(_Object, type_prefix="fnc"):
    """Functions are the basic units of serverless execution on Targon."""

    # Function metadata
    _web_url: str | None = None
    _grpc_endpoint: Union[Tuple[str, int], str] | None = None
    _revision: int | None = None
    _webhook_config: WebhookConfig | None = None
    _raw_f: Callable | None = None
    # App/Object context
    _app: BaseApp | None = None
    _obj: _Object | None = None
    # Additional attributes
    _module: str | None = None
    _qualname: str | None = None
    _image: _Image | None = None
    _resource_name: str | None = None
    _timeout: int = 21600

    def _initialize_from_empty(self) -> None:
        """Initialize an empty function object."""
        self._web_url = None
        self._grpc_endpoint = None
        self._name = None
        self._webhook_config = None
        self._raw_f = None
        self._app = None
        self._obj = None
        self._module = None
        self._qualname = None
        self._image = None
        self._resource_name = None
        self._timeout = 21600

    def _hydrate_metadata(self, metadata: dict[str, Any]) -> None:
        """Extract function-specific metadata from backend."""
        if not metadata:
            return
        self._name = metadata.get("name")
        self._revision = metadata.get("revision")
        self._web_url = metadata.get("web_url")
        self._grpc_endpoint = metadata.get("grpc_endpoint")

    @staticmethod
    def from_local(
        func: Callable | None,
        app: BaseApp,
        image: _Image,
        *,
        name: str = "",
        webhook_config: WebhookConfig | None = None,
        user_cls: type | None = None,
        resource_name: str = Compute.CPU_SMALL,
        min_replicas: int = 1,
        max_replicas: int = 3,
        initial_replicas: int = 0,
        container_concurrency: int | None = None,
        target_concurrency: int | None = None,
        scale_up_delay: str | None = None,
        scale_down_delay: str | None = None,
        zero_grace_period: str | None = None,
        scaling_metric: str | None = None,
        target_value: float | None = None,
        timeout: int = 1500,
        startup_timeout: int = 1500
    ) -> _Function:
        if user_cls is not None:
            func_name = name or user_cls.__name__
            qualname = user_cls.__name__
            module = user_cls.__module__
        else:
            if func is None:
                raise ValidationError("Either func or user_cls must be provided")
            func_name = name or getattr(func, "__name__", "anonymous")
            qualname = getattr(func, "__qualname__", func_name)
            module = getattr(func, "__module__", "__main__")

        if max_replicas < min_replicas:
            raise ValidationError(
                f"`min_replicas` ({min_replicas}) cannot be greater than `max_replicas` ({max_replicas})",
                field="min_replicas",
            )

        def _deps():
            """Collect all object dependencies for this function."""
            deps: list[_Object] = []
            if image is not None:
                deps.append(image)
            return deps

        async def _load(
            self: _Object, resolver: Resolver, existing_object_id: str | None
        ) -> None:
            """Load function into backend, creating remote representation."""
            assert isinstance(self, _Function)
            object_dependencies = []
            for dep in _deps():
                if not dep.object_id:
                    raise HydrationError(
                        f"Dependency {dep} is not hydrated",
                        object_type=type(dep).__name__,
                    )
                object_dependencies.append(dep.object_id)

            definition_type = (
                DefinitionType.SERIALIZED
                if func and is_notebook_function(func)
                else DefinitionType.FILE
            )

            function_serialized = None
            class_serialized = None

            if definition_type == DefinitionType.SERIALIZED:
                MAX_SIZE = 16 << 20  # 16 MiB

                target = user_cls if user_cls is not None else func
                serialized = cloudpickle.dumps(target)

                if len(serialized) > MAX_SIZE:
                    raise ValidationError(
                        f"Function '{func_name}' has size {len(serialized)} bytes when packaged. "
                        f"This exceeds the maximum limit of {MAX_SIZE // (1 << 20)} MiB. "
                        "Reduce the size of the closure by using parameters or mounts.",
                        field="function_size",
                        value=len(serialized),
                    )

                if user_cls is not None:
                    class_serialized = serialized
                else:
                    function_serialized = serialized

            # Enforce the limits of below in tha
            autoscaler_settings = AutoscalerSettings(
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                initial_replicas=initial_replicas,
                container_concurrency=container_concurrency,
                target_concurrency=target_concurrency,
                scale_up_delay=scale_up_delay,
                scale_down_delay=scale_down_delay,
                zero_grace_period=zero_grace_period,
                scaling_metric=scaling_metric,
                target_value=target_value,
            )

            func_data = {
                # Core identity fields
                "app_id": resolver.app_id,
                "name": func_name,
                "module": module,
                "qualname": qualname,
                # Function code delivery
                "definition_type": definition_type.value,
                "function_serialized": (
                    function_serialized.hex() if function_serialized else ""
                ),
                "class_serialized": class_serialized.hex() if class_serialized else "",
                # Runtime environment
                "image_id": image._registry_ref,
                "webhook_config": webhook_config,
                "timeout_secs": timeout,
                "startup_timeout": startup_timeout,
                "object_dependencies": object_dependencies,
                "autoscaler_settings": autoscaler_settings,
                "resource_name": resource_name,
            }

            response = await resolver.client.async_functions.register_function(
                **func_data
            )
            self._hydrate(
                response.uid, resolver.client, metadata=asdict(response.metadata)
            )

        obj = _Function._from_loader(
            _load, f"Function({func_name})", deps=_deps, name=func_name
        )
        obj._raw_f = func
        obj._name = func_name
        obj._module = module
        obj._qualname = qualname
        obj._app = app
        obj._image = image
        obj._webhook_config = webhook_config

        # Preserve function metadata if we have a function
        if func is not None:
            functools.update_wrapper(obj, func)

        return obj

    def __repr__(self) -> str:
        return f"Function({getattr(self, '_name', '<anonymous>')})"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.remote(*args, **kwargs)

    def local(self, *args: Any, **kwargs: Any) -> Any:
        if not self._raw_f:
            raise ServerlessError(
                "The definition for this Function is missing, so it is not possible to invoke it locally. "
                "If this function was retrieved via `Function.from_name`, "
                "you need to use one of the remote invocation methods instead.",
                function_id=self._object_id,
            )
        return self._raw_f(*args, **kwargs)

    async def _call_function(
        self, *args: Any, timeout: int | None = None, **kwargs: Any
    ) -> Any:
        timeout = timeout or 21600
        invocation = await _Invocation.create(
            function=self,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
        )
        return await invocation.run_function()

    @live_method
    async def remote(
        self, *args: Any, timeout: int | None = None, **kwargs: Any
    ) -> Any:
        """Execute the function remotely on Targon infrastructure."""
        self._check_no_web_url("remote")
        return await self._call_function(*args, timeout=timeout, **kwargs)

    def map(self, iterable: Any) -> Any:
        """Map the function over an iterable, executing remotely for each item."""
        self._check_no_web_url("map")
        items = list(iterable) if not isinstance(iterable, list) else iterable

        try:
            asyncio.get_running_loop()

            async def _async_map():
                for item in items:
                    yield await self.remote(item)

            return _async_map()
        except RuntimeError:

            def _sync_map():
                for item in items:
                    yield self.remote(item)

            return _sync_map()

    def serialize(self, func: Any) -> bytes:
        return cloudpickle.dumps(func)

    def hydrate(self, object_id: str) -> None:
        self._object_id = object_id
        self._is_hydrated = True

    def _is_web_endpoint(self) -> bool:
        return self._webhook_config is not None

    def _check_no_web_url(self, fn_name: str) -> None:
        if self._web_url:
            raise ValidationError(
                f"A webhook function cannot be invoked for remote execution with `.{fn_name}()`. "
                f"Invoke this function via its web url '{self._web_url}' "
                f"or call it locally: {self._name}.local()"
            )

    def get_raw_f(self) -> Callable:
        if self._raw_f is None:
            raise ServerlessError(
                f"Function '{self._name}' does not have a raw function definition",
                function_id=self._object_id,
            )
        return self._raw_f

    @property
    def object_id(self) -> str:
        if self._object_id is None:
            raise HydrationError(
                f"Attempting to get object_id of unhydrated Function '{self._name}'",
                object_type="Function",
            )
        return self._object_id

    @property
    def web_url(self) -> str | None:
        return self._web_url
