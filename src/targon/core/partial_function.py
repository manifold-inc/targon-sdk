import enum
import inspect
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING

from targon.core.exceptions import ValidationError


if TYPE_CHECKING:
    from targon.core.function import _Function


class PartialFunctionFlags(enum.IntFlag):
    CALLABLE_INTERFACE = 1
    WEB_INTERFACE = 2


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


class _PartialFunction:

    _raw_f: Optional[Callable[..., Any]]
    _user_cls: Optional[Type[Any]]
    _flags: PartialFunctionFlags
    _webhook_config: (
        WebhookConfig  # This attribute is used in universal_runtime as well
    )

    def __init__(
        self,
        obj: Union[Callable[..., Any], Type[Any]],
        flags: PartialFunctionFlags,
        webhook_config: WebhookConfig,
    ) -> None:
        if isinstance(obj, type):
            self._user_cls = obj
            self._raw_f = None
        else:
            self._raw_f = obj
            self._user_cls = None

        self._flags = flags
        self._webhook_config = webhook_config
        self.validate_flag_composition()

    def stack(self, flags: PartialFunctionFlags) -> "_PartialFunction":
        """Implement decorator composition by combining the flags and params."""
        self._flags |= flags
        self.validate_flag_composition()
        return self

    def validate_flag_composition(self) -> None:
        if (
            self._flags & PartialFunctionFlags.WEB_INTERFACE
            and self._flags & PartialFunctionFlags.CALLABLE_INTERFACE
        ):
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
        if self._user_cls is not None:
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
        return self._webhook_config

    @property
    def is_web_endpoint(self) -> bool:
        return self._webhook_config is not None and self._webhook_config.type != ""


def _callable_has_non_self_params(f: Callable[..., Any]) -> bool:
    return any(
        param.name != "self" for param in inspect.signature(f).parameters.values()
    )
