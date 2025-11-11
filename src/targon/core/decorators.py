from typing import Callable, Optional, Union
from targon.core.exceptions import ValidationError
from targon.core.partial_function import (
    WebhookType,
    _PartialFunction,
    PartialFunctionFlags,
    WebhookConfig,
)


def fastapi_endpoint(
    _warn_parentheses_missing: Optional[Callable] = None,
    *,
    method: str = "GET",
    label: Optional[str] = None,
    docs: bool = False,
    requires_auth: bool = False,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:
    if _warn_parentheses_missing is not None:
        if isinstance(_warn_parentheses_missing, str):
            raise ValidationError(
                f'Positional arguments are not allowed. Use `@targon.fastapi_endpoint(method="{method}")`',
                field="decorator_usage",
            )
        raise ValidationError(
            "Missing parentheses. Use `@targon.fastapi_endpoint()`",
            field="decorator_usage",
        )

    if not isinstance(method, str) or not method:
        raise ValidationError(
            "Method must be a non-empty string", field="method", value=method
        )

    if not isinstance(docs, bool):
        raise ValidationError("Docs must be a boolean", field="docs", value=docs)

    if not isinstance(requires_auth, bool):
        raise ValidationError(
            "Requires_auth must be a boolean",
            field="requires_auth",
            value=requires_auth,
        )

    webhook_config = WebhookConfig(
        type=WebhookType.FUNCTION,
        method=method.upper(),
        docs=docs,
        label=label or "",
        requires_auth=requires_auth,
    )

    flags = PartialFunctionFlags.WEB_INTERFACE

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags)
        else:
            pf = _PartialFunction(obj, flags, webhook_config)

        pf.validate_obj_compatibility("fastapi_endpoint")
        return pf

    return wrapper


def asgi_app(
    _warn_parentheses_missing: Optional[Callable] = None,
    *,
    label: Optional[str] = None,
    requires_auth: bool = False,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:
    if _warn_parentheses_missing is not None:
        raise ValidationError(
            "Missing parentheses. Use `@targon.asgi_app()`", field="decorator_usage"
        )

    if not isinstance(requires_auth, bool):
        raise ValidationError(
            "Requires_auth must be a boolean",
            field="requires_auth",
            value=requires_auth,
        )

    webhook_config = WebhookConfig(
        type=WebhookType.ASGI_APP,
        label=label or "",
        requires_auth=requires_auth,
    )

    flags = PartialFunctionFlags.WEB_INTERFACE

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags)
        else:
            pf = _PartialFunction(obj, flags, webhook_config)
        pf.validate_obj_compatibility(
            "asgi_app", require_sync=True, require_nullary=True
        )
        return pf

    return wrapper


def wsgi_app(
    _warn_parentheses_missing: Optional[Callable] = None,
    *,
    label: Optional[str] = None,
    requires_auth: bool = False,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:
    if _warn_parentheses_missing is not None:
        raise ValidationError(
            "Missing parentheses. Use `@targon.wsgi_app()`", field="decorator_usage"
        )

    if not isinstance(requires_auth, bool):
        raise ValidationError(
            "Requires_auth must be a boolean",
            field="requires_auth",
            value=requires_auth,
        )

    webhook_config = WebhookConfig(
        type=WebhookType.WSGI_APP,
        label=label or "",
        requires_auth=requires_auth,
    )

    flags = PartialFunctionFlags.WEB_INTERFACE

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags)
        else:
            pf = _PartialFunction(obj, flags, webhook_config)
        pf.validate_obj_compatibility(
            "wsgi_app", require_sync=True, require_nullary=True
        )
        return pf

    return wrapper


def web_server(
    _warn_parentheses_missing: Optional[Callable] = None,
    *,
    port: int,
    startup_timeout: int = 300,
    label: Optional[str] = None,
    requires_auth: bool = False,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:
    if _warn_parentheses_missing is not None:
        raise ValidationError(
            "Missing parentheses. Use `@targon.web_server(port=8000)`",
            field="decorator_usage",
        )

    if not isinstance(port, int) or port < 1 or port > 65535:
        raise ValidationError(
            "Port must be a valid port number (1-65535)", field="port", value=port
        )

    if not isinstance(startup_timeout, int) or startup_timeout <= 0:
        raise ValidationError(
            "Startup_timeout must be a positive integer",
            field="startup_timeout",
            value=startup_timeout,
        )

    if not isinstance(requires_auth, bool):
        raise ValidationError(
            "Requires_auth must be a boolean",
            field="requires_auth",
            value=requires_auth,
        )

    webhook_config = WebhookConfig(
        type=WebhookType.WEB_SERVER,
        label=label or "",
        requires_auth=requires_auth,
        port=port,
        startup_timeout=startup_timeout,
    )

    flags = PartialFunctionFlags.WEB_INTERFACE

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags)
        else:
            pf = _PartialFunction(obj, flags, webhook_config)
        pf.validate_obj_compatibility(
            "web_server", require_sync=True, require_nullary=True
        )
        return pf

    return wrapper
