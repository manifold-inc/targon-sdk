from typing import Any, Dict, Optional


class TargonError(Exception):
    __slots__ = ("message", "details", "cause")

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


class APIError(TargonError):
    __slots__ = ("status_code", "response", "request_id")

    def __init__(
        self,
        status_code: int,
        message: str,
        response: Optional[Any] = None,
        request_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        details: Dict[str, Any] = {"status_code": status_code}
        if request_id:
            details["request_id"] = request_id

        super().__init__(message, details, cause)
        self.status_code = status_code
        self.response = response
        self.request_id = request_id

    @property
    def reason(self) -> Optional[str]:
        if isinstance(self.response, dict):
            return self.response.get("reason")
        return None

    @property
    def is_client_error(self) -> bool:
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        return 500 <= self.status_code < 600

    @property
    def is_retryable(self) -> bool:
        return self.is_server_error or self.status_code == 429

    @property
    def is_rate_limit(self) -> bool:
        return self.status_code == 429

    @property
    def is_not_found(self) -> bool:
        return self.status_code == 404

    @property
    def is_unauthorized(self) -> bool:
        return self.status_code == 401

    @property
    def is_forbidden(self) -> bool:
        return self.status_code == 403


class ValidationError(TargonError):
    __slots__ = ("field", "value")

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)

        super().__init__(message, details)
        self.field = field
        self.value = value


class ConfigurationError(TargonError):
    __slots__ = ("config_key",)

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if config_key:
            details["config_key"] = config_key

        super().__init__(message, details)
        self.config_key = config_key


class ResourceNotFoundError(APIError):
    __slots__ = ("resource_type", "resource_id")

    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if message is None:
            message = (
                f"{resource_type} '{resource_id}' not found"
                if resource_id
                else f"{resource_type} not found"
            )

        super().__init__(404, message)
        self.resource_type = resource_type
        self.resource_id = resource_id


class RateLimitError(APIError):
    __slots__ = ("retry_after",)

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(429, message)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class AuthenticationError(APIError):
    __slots__ = ()

    def __init__(self, message: Optional[str] = None) -> None:
        if message is None:
            message = (
                "Authentication failed. Check your API key is valid. "
                "Set TARGON_API_KEY environment variable or pass api_key to Client()."
            )
        super().__init__(401, message)


class AuthorizationError(APIError):
    __slots__ = ("resource",)

    def __init__(
        self,
        message: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> None:
        if message is None:
            message = (
                f"Permission denied for resource: {resource}"
                if resource
                else "Permission denied"
            )
        super().__init__(403, message)
        self.resource = resource


class TimeoutError(TargonError):
    __slots__ = ("timeout",)

    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if timeout:
            details["timeout"] = timeout

        super().__init__(message, details)
        self.timeout = timeout

    @property
    def is_retryable(self) -> bool:
        return True


class NetworkError(TargonError):
    __slots__ = ()

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, cause=cause)

    @property
    def is_retryable(self) -> bool:
        return True


__all__ = [
    "TargonError",
    "APIError",
    "ValidationError",
    "HydrationError",
    "ConfigurationError",
    "ResourceNotFoundError",
    "RateLimitError",
    "AuthenticationError",
    "AuthorizationError",
    "TimeoutError",
    "NetworkError",
]
