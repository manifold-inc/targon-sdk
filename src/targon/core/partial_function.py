import enum
import inspect
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Type, Union, TYPE_CHECKING
from targon.core.exceptions import ValidationError


class _PartialFunctionFlags(enum.IntFlag):
    # Interface flags
    CALLABLE_INTERFACE = 1
    WEB_INTERFACE = 2
    # Lifecycle method flags
    ENTER = 4
    EXIT = 8
    # Service behavior flags
    CONCURRENT = 16

    @staticmethod
    def all() -> int:
        return ~_PartialFunctionFlags(0)

    @staticmethod
    def lifecycle_flags() -> int:
        return _PartialFunctionFlags.ENTER | _PartialFunctionFlags.EXIT

    @staticmethod
    def interface_flags() -> int:
        return (
            _PartialFunctionFlags.CALLABLE_INTERFACE
            | _PartialFunctionFlags.WEB_INTERFACE
        )


class WebhookType(str, Enum):
    UNSPECIFIED = "unspecified"
    ASGI_APP = "asgi_app"
    FUNCTION = "function"
    WSGI_APP = "wsgi_app"
    WEB_SERVER = "web_server"


@dataclass
class WebhookConfig:
    type: str
    method: str = "GET"
    docs: bool = False
    label: str = ""
    requires_auth: bool = False
    port: Optional[int] = None
    startup_timeout: Optional[int] = None


@dataclass
class _PartialFunctionParams:
    webhook_config: Optional[WebhookConfig] = None
    max_concurrent_inputs: Optional[int] = None
    target_concurrent_inputs: Optional[int] = None

    def update(self, other: "_PartialFunctionParams") -> None:
        if other.webhook_config is not None:
            if self.webhook_config is not None:
                raise ValidationError("Cannot set `webhook_config` twice.")
            self.webhook_config = other.webhook_config
        if other.max_concurrent_inputs is not None:
            if self.max_concurrent_inputs is not None:
                raise ValidationError("Cannot set `max_concurrent_inputs` twice.")
            self.max_concurrent_inputs = other.max_concurrent_inputs
        if other.target_concurrent_inputs is not None:
            if self.target_concurrent_inputs is not None:
                raise ValidationError("Cannot set `target_concurrent_inputs` twice.")
            self.target_concurrent_inputs = other.target_concurrent_inputs


class _PartialFunction:
    _raw_f: Optional[Callable[..., Any]]
    _user_cls: Optional[Type[Any]]
    _flags: _PartialFunctionFlags
    _params: _PartialFunctionParams

    def __init__(
        self,
        obj: Union[Callable[..., Any], Type[Any]],
        flags: _PartialFunctionFlags,
        params: _PartialFunctionParams,
    ) -> None:
        if isinstance(obj, type):
            self._user_cls = obj
            self._raw_f = None
        else:
            self._raw_f = obj
            self._user_cls = None

        self._flags = flags
        self._params = params
        self.validate_flag_composition()

    def stack(
        self, flags: _PartialFunctionFlags, params: _PartialFunctionParams
    ) -> "_PartialFunction":
        """Implement decorator composition by combining the flags and params."""
        self._flags |= flags
        self._params.update(params)
        self.validate_flag_composition()
        return self

    def validate_flag_composition(self) -> None:
        uses_interface_flags = self._flags & _PartialFunctionFlags.interface_flags()
        uses_lifecycle_flags = self._flags & _PartialFunctionFlags.lifecycle_flags()

        if uses_interface_flags and uses_lifecycle_flags:
            raise ValidationError(
                "Interface decorators cannot be combined with lifecycle decorators",
                field="decorator_composition",
            )

        has_web_interface = self._flags & _PartialFunctionFlags.WEB_INTERFACE
        has_callable_interface = self._flags & _PartialFunctionFlags.CALLABLE_INTERFACE
        if has_web_interface and has_callable_interface:
            raise ValidationError(
                "Callable decorators cannot be combined with web interface decorators",
                field="decorator_composition",
            )

    def validate_obj_compatibility(
        self,
        decorator_name: str,
        require_sync: bool = False,
        require_nullary: bool = False,
    ) -> None:
        uses_interface_flags = self._flags & _PartialFunctionFlags.interface_flags()
        uses_lifecycle_flags = self._flags & _PartialFunctionFlags.lifecycle_flags()

        # For interface and lifecycle decorators we don't allow decorating classes directly.
        # Non-interface service decorators (like `concurrent`) may choose to support classes.
        if self._user_cls is not None and (
            uses_interface_flags or uses_lifecycle_flags
        ):
            raise ValidationError(
                f"Cannot apply `@targon.{decorator_name}` to a class. Consider applying to a method instead.",
                field="decorator_target",
            )

        wrapped_object = self._raw_f
        if wrapped_object is None:
            return
        try:
            from targon.core.function import _Function

            if isinstance(wrapped_object, _Function):
                raise ValidationError(
                    f"Cannot stack `@targon.{decorator_name}` on top of `@app.function`. Swap the order of the decorators.",
                    field="decorator_stacking",
                )
        except ImportError:
            pass

        if self._raw_f is not None:
            if not callable(self._raw_f):
                raise ValidationError(
                    f"The object wrapped by `@targon.{decorator_name}` must be callable",
                    field="callable",
                )

            if require_sync and inspect.iscoroutinefunction(self._raw_f):
                raise ValidationError(
                    f"The `@targon.{decorator_name}` decorator can't be applied to an async function",
                    field="function_type",
                )

            if require_nullary and _callable_has_non_self_params(self.raw_f):
                raise ValidationError(
                    f"Functions decorated by `@targon.{decorator_name}` can't have parameters",
                    field="function_parameters",
                )

    @property
    def raw_f(self) -> Callable[..., Any]:
        assert self._raw_f is not None
        return self._raw_f

    @property
    def webhook_config(self) -> Optional[WebhookConfig]:
        return self._params.webhook_config

    @property
    def is_web_endpoint(self) -> bool:
        return (
            self._params.webhook_config is not None
            and self._params.webhook_config.type != ""
        )


def _callable_has_non_self_params(f: Callable[..., Any]) -> bool:
    return any(
        param.name != "self" for param in inspect.signature(f).parameters.values()
    )
