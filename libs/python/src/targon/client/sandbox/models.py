from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Sequence, TypeVar

from targon.client.workload import PortConfig
from targon.core.exceptions import ValidationError

if TYPE_CHECKING:
    from targon.client.sandbox.resources import SandboxTemplate


def defined(**values: Any) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


class SandboxTemplateKind(str, Enum):
    FRESH = "FRESH"
    USER = "USER"


class SandboxTemplateStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"


class SandboxStatus(str, Enum):
    REGISTERED = "registered"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    FROZEN = "frozen"
    ERROR = "error"
    SUSPENDED = "suspended"
    DELETED = "deleted"
    PENDING = "pending"
    POWERING_ON = "powering_on"
    POWERING_OFF = "powering_off"
    REBOOTING = "rebooting"
    STOPPED = "stopped"


class PortProtocol(str, Enum):
    TCP = "TCP"
    UDP = "UDP"


@dataclass(frozen=True)
class SandboxConfigInput:
    ttl_sec: Optional[int] = None
    idle_timeout_sec: Optional[int] = None

    def to_payload(self) -> Dict[str, int]:
        validate_timeouts(self.ttl_sec, self.idle_timeout_sec)
        return defined(ttl_sec=self.ttl_sec, idle_timeout_sec=self.idle_timeout_sec)


@dataclass(frozen=True)
class SandboxConfig:
    template_uid: str = ""
    parent_workload_uid: Optional[str] = None
    ttl_sec: Optional[int] = None
    idle_timeout_sec: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Any) -> Optional["SandboxConfig"]:
        if not isinstance(data, dict):
            return None
        return cls(
            data.get("template_uid", ""),
            data.get("parent_workload_uid"),
            data.get("ttl_sec"),
            data.get("idle_timeout_sec"),
        )


@dataclass(frozen=True)
class SandboxPort:
    port: int
    protocol: PortProtocol
    routing: str = "PROXIED"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxPort":
        return cls(
            data.get("port", 0),
            PortProtocol(data.get("protocol", "TCP")),
            data.get("routing", "PROXIED"),
        )


@dataclass(frozen=True)
class SandboxSshKey:
    uid: str
    name: str = ""
    public_key: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxSshKey":
        return cls(
            data.get("uid", ""),
            data.get("name", ""),
            data.get("public_key_raw", ""),
        )


@dataclass(frozen=True)
class SandboxResource:
    name: str = ""
    display_name: str = ""
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    vcpu: int = 0
    memory: int = 0
    disk_size_mib: Optional[int] = None
    network_mode: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> Optional["SandboxResource"]:
        if not isinstance(data, dict):
            return None
        return cls(
            data.get("name", ""),
            data.get("display_name", ""),
            data.get("gpu_type"),
            data.get("gpu_count"),
            data.get("vcpu", 0),
            data.get("memory", 0),
            data.get("disk_size_mib"),
            data.get("network_mode"),
        )


@dataclass(frozen=True)
class SandboxState:
    status: SandboxStatus
    message: str = ""
    ready_replicas: int = 0
    total_replicas: int = 0

    @classmethod
    def from_dict(cls, data: Any) -> Optional["SandboxState"]:
        if not isinstance(data, dict) or not data.get("status"):
            return None
        return cls(
            SandboxStatus(data["status"].lower()),
            data.get("message", ""),
            data.get("ready_replicas", 0),
            data.get("total_replicas", 0),
        )


@dataclass(frozen=True)
class SandboxSummary:
    uid: str
    name: str
    type: str = "SANDBOX"
    state: Optional[SandboxState] = None
    resource: Optional[SandboxResource] = None
    cost_per_hour: Optional[float] = None
    frozen_cost_per_hour: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxSummary":
        if not isinstance(data, dict) or not data.get("uid"):
            raise ValidationError("sandbox summary is missing uid", field="uid")
        if data.get("type") != "SANDBOX":
            raise ValidationError(
                "workload summary is not a SANDBOX",
                field="type",
                value=data.get("type"),
            )
        return cls(
            uid=data["uid"],
            name=data.get("name", ""),
            type=data["type"],
            state=SandboxState.from_dict(data.get("state")),
            resource=SandboxResource.from_dict(data.get("resource")),
            cost_per_hour=data.get("cost_per_hour"),
            frozen_cost_per_hour=data.get("frozen_cost_per_hour"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


T = TypeVar("T")


@dataclass(frozen=True)
class ListPage(Generic[T]):
    items: Sequence[T] = field(default_factory=tuple)
    next_cursor: Optional[str] = None


@dataclass(frozen=True)
class SandboxCreateParams:
    name: str
    template: "SandboxTemplate"
    project_id: Optional[str] = None
    ssh_keys: Sequence[str] = field(default_factory=tuple)
    ports: Sequence[PortConfig] = field(default_factory=tuple)
    ttl_sec: Optional[int] = None
    idle_timeout_sec: Optional[int] = None
    wait_until_running: bool = True


@dataclass(frozen=True)
class SandboxUpdateParams:
    name: Optional[str] = None
    project_id: Optional[str] = None
    ssh_keys: Optional[Sequence[str]] = None
    ports: Optional[Sequence[PortConfig]] = None
    sandbox_config: Optional[SandboxConfigInput] = None


@dataclass(frozen=True)
class ForkRequest:
    name: Optional[str] = None
    project_id: Optional[str] = None
    sandbox_config: Optional[SandboxConfigInput] = None


@dataclass(frozen=True)
class PublishRequest:
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    code: int
    timed_out: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecResult":
        return cls(
            data.get("stdout", ""),
            data.get("stderr", ""),
            data.get("code", 0),
            bool(data.get("timed_out", False)),
        )


@dataclass(frozen=True)
class AccessTicket:
    ticket: str
    expires_at: str


@dataclass(frozen=True)
class TerminalSession:
    id: str
    pid: int
    started_at: str
    exited: bool
    exit_code: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TerminalSession":
        return cls(
            data["id"],
            data.get("pid", 0),
            data.get("started_at", ""),
            bool(data.get("exited", False)),
            data.get("exit_code"),
        )


@dataclass(frozen=True)
class DesktopInfo:
    available: bool
    port: Optional[int] = None
    listening: Optional[bool] = None
    ws_url: Optional[str] = None


MAX_LIST_LIMIT = 1000
MAX_COMMAND_BYTES = 64 << 10
MAX_FILE_BYTES = 256 << 20
MAX_TERMINAL_DIMENSION = 1000

NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")
TEMPLATE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TERMINAL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def invalid(message: str, field: str, value: Any = None) -> ValidationError:
    return ValidationError(message, field=field, value=value)


def non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise invalid(f"{field_name} must be a non-empty string", field_name, value)
    return value.strip()


def bounded_int(
    value: int,
    field_name: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise invalid(
            f"{field_name} must be between {minimum} and {maximum}", field_name, value
        )
    return value


def validate_name(value: str, field_name: str = "name") -> str:
    value = non_empty(value, field_name)
    if not NAME_RE.fullmatch(value):
        raise invalid(
            f"{field_name} must be 1-32 lowercase alphanumeric or hyphen characters "
            "and cannot start or end with a hyphen",
            field_name,
            value,
        )
    return value


def _nonnegative(value: Optional[int], field: str) -> None:
    if value is not None and (not isinstance(value, int) or value < 0):
        raise invalid(f"{field} must be zero or positive", field, value)


def validate_timeouts(ttl_sec: Optional[int], idle_timeout_sec: Optional[int]) -> None:
    _nonnegative(ttl_sec, "ttl_sec")
    _nonnegative(idle_timeout_sec, "idle_timeout_sec")
    if (
        ttl_sec is not None
        and idle_timeout_sec is not None
        and ttl_sec > 0
        and idle_timeout_sec > 0
        and idle_timeout_sec >= ttl_sec
    ):
        raise invalid(
            "idle_timeout_sec must be less than ttl_sec",
            "idle_timeout_sec",
            idle_timeout_sec,
        )


def validate_update_timeouts(
    ttl_sec: Optional[int], idle_timeout_sec: Optional[int]
) -> None:
    validate_timeouts(ttl_sec, idle_timeout_sec)
    if idle_timeout_sec is not None and idle_timeout_sec <= 0:
        raise invalid(
            "idle_timeout_sec must be positive when updating a sandbox",
            "idle_timeout_sec",
            idle_timeout_sec,
        )


def ports_payload(ports: Sequence[PortConfig]) -> List[Dict[str, Any]]:
    payload = []
    for item in ports:
        if not isinstance(item, PortConfig):
            raise invalid("ports must contain PortConfig instances", "ports", item)
        if item.port == 22:
            raise invalid("port 22 cannot be exposed", "ports", 22)
        if not 1 <= item.port <= 65535:
            raise invalid("port must be between 1 and 65535", "ports", item.port)
        try:
            protocol = PortProtocol(item.protocol.upper())
        except (AttributeError, ValueError) as exc:
            raise invalid(
                "sandbox port protocol must be TCP or UDP", "ports", item.protocol
            ) from exc
        value = item.to_payload()
        value["protocol"] = protocol.value
        payload.append(value)
    return payload


def validate_path(path: str) -> str:
    path = non_empty(path, "path")
    if not path.startswith("/") or "\x00" in path:
        raise invalid("path must be absolute and contain no NUL bytes", "path", path)
    return path


def validate_terminal_id(terminal_id: str) -> str:
    terminal_id = non_empty(terminal_id, "terminal_id")
    if not TERMINAL_ID_RE.fullmatch(terminal_id):
        raise invalid(
            "terminal_id contains invalid characters", "terminal_id", terminal_id
        )
    return terminal_id
