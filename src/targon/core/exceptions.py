from typing import Optional, Any, Dict


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


class HydrationError(TargonError):
    __slots__ = ("object_type", "object_id")

    def __init__(
        self,
        message: str,
        object_type: Optional[str] = None,
        object_id: Optional[str] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if object_type:
            details["object_type"] = object_type
        if object_id:
            details["object_id"] = object_id

        super().__init__(message, details)
        self.object_type = object_type
        self.object_id = object_id


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


class DeploymentError(TargonError):
    __slots__ = ("deployment_id", "status")

    def __init__(
        self,
        message: str,
        deployment_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if deployment_id:
            details["deployment_id"] = deployment_id
        if status:
            details["status"] = status

        super().__init__(message, details)
        self.deployment_id = deployment_id
        self.status = status


class ServerlessError(TargonError):
    __slots__ = ("function_id", "execution_id")

    def __init__(
        self,
        message: str,
        function_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if function_id:
            details["function_id"] = function_id
        if execution_id:
            details["execution_id"] = execution_id

        super().__init__(message, details)
        self.function_id = function_id
        self.execution_id = execution_id


class TemplateError(TargonError):
    __slots__ = ("template_id",)

    def __init__(
        self,
        message: str,
        template_id: Optional[str] = None,
    ) -> None:
        details: Dict[str, Any] = {}
        if template_id:
            details["template_id"] = template_id

        super().__init__(message, details)
        self.template_id = template_id


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
    "DeploymentError",
    "ServerlessError",
    "TemplateError",
]
