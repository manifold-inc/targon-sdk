from typing import Callable, Optional, Union
from targon.core.exceptions import ValidationError
from targon.core.partial_function import (
    WebhookType,
    _PartialFunction,
    _PartialFunctionFlags,
    _PartialFunctionParams,
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

    flags = _PartialFunctionFlags.WEB_INTERFACE
    pf_params = _PartialFunctionParams(webhook_config=webhook_config)

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)

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

    flags = _PartialFunctionFlags.WEB_INTERFACE
    pf_params = _PartialFunctionParams(webhook_config=webhook_config)

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)

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

    flags = _PartialFunctionFlags.WEB_INTERFACE
    pf_params = _PartialFunctionParams(webhook_config=webhook_config)

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)

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

    flags = _PartialFunctionFlags.WEB_INTERFACE
    pf_params = _PartialFunctionParams(webhook_config=webhook_config)

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)

        pf.validate_obj_compatibility(
            "web_server", require_sync=True, require_nullary=True
        )
        return pf

    return wrapper


def enter(
    _warn_parentheses_missing: Optional[Callable] = None,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:
    if _warn_parentheses_missing is not None:
        raise ValidationError(
            "Missing parentheses. Use `@targon.enter()`",
            field="decorator_usage",
        )

    flags = _PartialFunctionFlags.ENTER
    pf_params = _PartialFunctionParams()

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)
        pf.validate_obj_compatibility("enter", require_nullary=False)
        return pf

    return wrapper


def exit(
    _warn_parentheses_missing: Optional[Callable] = None,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:

    if _warn_parentheses_missing is not None:
        raise ValidationError(
            "Missing parentheses. Use `@targon.exit()`",
            field="decorator_usage",
        )

    flags = _PartialFunctionFlags.EXIT
    pf_params = _PartialFunctionParams()

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)
        pf.validate_obj_compatibility("exit", require_nullary=False)
        return pf

    return wrapper


def method(
    _warn_parentheses_missing: Optional[Callable] = None,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:
    if _warn_parentheses_missing is not None:
        raise ValidationError(
            "Missing parentheses. Use `@targon.method()`",
            field="decorator_usage",
        )

    flags = _PartialFunctionFlags.CALLABLE_INTERFACE
    pf_params = _PartialFunctionParams()

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)
        pf.validate_obj_compatibility("method")
        return pf

    return wrapper


def concurrent(
    _warn_parentheses_missing: Optional[Callable] = None,
    *,
    max_concurrency: int,
    target_concurrency: Optional[int] = None,
) -> Callable[[Union[_PartialFunction, Callable]], _PartialFunction]:
    if _warn_parentheses_missing is not None:
        raise ValidationError(
            "Positional arguments are not allowed. "
            "Did you forget parentheses? Use `@targon.concurrent(max_concurrency=...)`.",
            field="decorator_usage",
        )

    if not isinstance(max_concurrency, int) or max_concurrency <= 0:
        raise ValidationError(
            "max_concurrency must be a positive integer",
            field="max_concurrency",
            value=max_concurrency,
        )

    if target_concurrency is not None:
        if not isinstance(target_concurrency, int) or target_concurrency <= 0:
            raise ValidationError(
                "target_concurrency must be a positive integer when provided",
                field="target_concurrency",
                value=target_concurrency,
            )
        if target_concurrency > max_concurrency:
            raise ValidationError(
                "`target_concurrency` parameter cannot be greater than `max_concurrency`.",
                field="target_concurrency",
                value=target_concurrency,
            )

    flags = _PartialFunctionFlags.CONCURRENT
    pf_params = _PartialFunctionParams(
        max_concurrent_inputs=max_concurrency,
        target_concurrent_inputs=target_concurrency,
    )

    def wrapper(obj: Union[_PartialFunction, Callable]) -> _PartialFunction:
        if isinstance(obj, _PartialFunction):
            pf = obj.stack(flags, pf_params)
        else:
            pf = _PartialFunction(obj, flags, pf_params)

        pf.validate_obj_compatibility("concurrent")
        return pf

    return wrapper
