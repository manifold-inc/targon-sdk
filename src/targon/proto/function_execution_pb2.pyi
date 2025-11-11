from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExecuteRequest(_message.Message):
    __slots__ = (
        "request_id",
        "args",
        "kwargs",
        "input_id",
        "function_call_id",
        "timeout_ms",
        "metadata",
    )

    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...

    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    INPUT_ID_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    args: bytes
    kwargs: bytes
    input_id: str
    function_call_id: str
    timeout_ms: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(
        self,
        request_id: _Optional[str] = ...,
        args: _Optional[bytes] = ...,
        kwargs: _Optional[bytes] = ...,
        input_id: _Optional[str] = ...,
        function_call_id: _Optional[str] = ...,
        timeout_ms: _Optional[int] = ...,
        metadata: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class ExecuteResponse(_message.Message):
    __slots__ = (
        "request_id",
        "status",
        "result",
        "error",
        "traceback",
        "execution_time_ms",
        "metadata",
    )

    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SUCCESS: _ClassVar[ExecuteResponse.Status]
        ERROR: _ClassVar[ExecuteResponse.Status]
        TIMEOUT: _ClassVar[ExecuteResponse.Status]
        CANCELLED: _ClassVar[ExecuteResponse.Status]
        RATE_LIMITED: _ClassVar[ExecuteResponse.Status]

    SUCCESS: ExecuteResponse.Status
    ERROR: ExecuteResponse.Status
    TIMEOUT: ExecuteResponse.Status
    CANCELLED: ExecuteResponse.Status
    RATE_LIMITED: ExecuteResponse.Status

    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...

    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    TRACEBACK_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    status: ExecuteResponse.Status
    result: bytes
    error: str
    traceback: str
    execution_time_ms: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(
        self,
        request_id: _Optional[str] = ...,
        status: _Optional[_Union[ExecuteResponse.Status, str]] = ...,
        result: _Optional[bytes] = ...,
        error: _Optional[str] = ...,
        traceback: _Optional[str] = ...,
        execution_time_ms: _Optional[int] = ...,
        metadata: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("service",)
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    service: str
    def __init__(self, service: _Optional[str] = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status",)

    class ServingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[HealthCheckResponse.ServingStatus]
        SERVING: _ClassVar[HealthCheckResponse.ServingStatus]
        NOT_SERVING: _ClassVar[HealthCheckResponse.ServingStatus]
        SERVICE_UNKNOWN: _ClassVar[HealthCheckResponse.ServingStatus]

    UNKNOWN: HealthCheckResponse.ServingStatus
    SERVING: HealthCheckResponse.ServingStatus
    NOT_SERVING: HealthCheckResponse.ServingStatus
    SERVICE_UNKNOWN: HealthCheckResponse.ServingStatus
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: HealthCheckResponse.ServingStatus
    def __init__(
        self, status: _Optional[_Union[HealthCheckResponse.ServingStatus, str]] = ...
    ) -> None: ...
