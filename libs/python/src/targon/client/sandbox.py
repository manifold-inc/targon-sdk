from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
from urllib.parse import urlencode

from targon.client.constants import org_path
from targon.client.workload import PortConfig
from targon.core.exceptions import (
    AccessTicketError,
    SandboxStateError,
    SandboxTemplateError,
    TimeoutError,
    ValidationError,
)
from targon.core.objects import BaseHTTPClient

if TYPE_CHECKING:
    from targon.client.client import Client

MAX_LIST_LIMIT = 1000
MAX_COMMAND_BYTES = 64 << 10
MAX_FILE_BYTES = 256 << 20
MAX_TERMINAL_DIMENSION = 1000
NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")
TEMPLATE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TERMINAL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


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
        _validate_timeouts(self.ttl_sec, self.idle_timeout_sec)
        payload: Dict[str, int] = {}
        if self.ttl_sec is not None:
            payload["ttl_sec"] = self.ttl_sec
        if self.idle_timeout_sec is not None:
            payload["idle_timeout_sec"] = self.idle_timeout_sec
        return payload


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
            template_uid=data.get("template_uid", ""),
            parent_workload_uid=data.get("parent_workload_uid"),
            ttl_sec=data.get("ttl_sec"),
            idle_timeout_sec=data.get("idle_timeout_sec"),
        )


@dataclass(frozen=True)
class SandboxPort:
    port: int
    protocol: PortProtocol
    routing: str = "PROXIED"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxPort":
        return cls(
            port=data.get("port", 0),
            protocol=PortProtocol(data.get("protocol", "TCP")),
            routing=data.get("routing", "PROXIED"),
        )


@dataclass(frozen=True)
class SandboxSshKey:
    uid: str
    name: str = ""
    public_key: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxSshKey":
        return cls(
            uid=data.get("uid", ""),
            name=data.get("name", ""),
            public_key=data.get("public_key_raw", ""),
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
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            gpu_type=data.get("gpu_type"),
            gpu_count=data.get("gpu_count"),
            vcpu=data.get("vcpu", 0),
            memory=data.get("memory", 0),
            disk_size_mib=data.get("disk_size_mib"),
            network_mode=data.get("network_mode"),
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
            status=SandboxStatus(data["status"].lower()),
            message=data.get("message", ""),
            ready_replicas=data.get("ready_replicas", 0),
            total_replicas=data.get("total_replicas", 0),
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


@dataclass(frozen=True)
class _SandboxData:
    uid: str
    name: str = ""
    type: str = "SANDBOX"
    image: str = ""
    resource_name: str = ""
    project_id: Optional[str] = None
    ports: Sequence[SandboxPort] = field(default_factory=tuple)
    ssh_keys: Sequence[SandboxSshKey] = field(default_factory=tuple)
    sandbox_config: Optional[SandboxConfig] = None
    state: Optional[SandboxState] = None
    resource: Optional[SandboxResource] = None
    cost_per_hour: Optional[float] = None
    frozen_cost_per_hour: Optional[float] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_SandboxData":
        if not isinstance(data, dict) or not data.get("uid"):
            raise ValidationError("sandbox response is missing uid", field="uid")
        workload_type = data.get("type")
        if workload_type != "SANDBOX":
            raise ValidationError(
                "workload is not a SANDBOX", field="type", value=workload_type
            )
        return cls(
            uid=data["uid"],
            name=data.get("name", ""),
            type=workload_type,
            image=data.get("image", ""),
            resource_name=data.get("resource_name", ""),
            project_id=data.get("project_id"),
            ports=tuple(
                SandboxPort.from_dict(item)
                for item in data.get("ports") or []
                if isinstance(item, dict)
            ),
            ssh_keys=tuple(
                SandboxSshKey.from_dict(item)
                for item in data.get("ssh_keys") or []
                if isinstance(item, dict)
            ),
            sandbox_config=SandboxConfig.from_dict(data.get("sandbox_config")),
            state=SandboxState.from_dict(data.get("state")),
            resource=SandboxResource.from_dict(data.get("resource")),
            cost_per_hour=data.get("cost_per_hour"),
            frozen_cost_per_hour=data.get("frozen_cost_per_hour"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(frozen=True)
class Sandbox:
    """Hydrated sandbox resource with read-only fields and thin API methods."""

    _client: "Client" = field(repr=False, compare=False)
    _data: _SandboxData = field(repr=False)

    @classmethod
    def _from_dict(cls, client: "Client", data: Dict[str, Any]) -> "Sandbox":
        return cls(client, _SandboxData.from_dict(data))

    @property
    def uid(self) -> str:
        return self._data.uid

    @property
    def name(self) -> str:
        return self._data.name

    @property
    def type(self) -> str:
        return self._data.type

    @property
    def image(self) -> str:
        return self._data.image

    @property
    def resource_name(self) -> str:
        return self._data.resource_name

    @property
    def project_id(self) -> Optional[str]:
        return self._data.project_id

    @property
    def ports(self) -> Sequence[SandboxPort]:
        return self._data.ports

    @property
    def ssh_keys(self) -> Sequence[SandboxSshKey]:
        return self._data.ssh_keys

    @property
    def sandbox_config(self) -> Optional[SandboxConfig]:
        return self._data.sandbox_config

    @property
    def state(self) -> Optional[SandboxState]:
        return self._data.state

    @property
    def resource(self) -> Optional[SandboxResource]:
        return self._data.resource

    @property
    def cost_per_hour(self) -> Optional[float]:
        return self._data.cost_per_hour

    @property
    def frozen_cost_per_hour(self) -> Optional[float]:
        return self._data.frozen_cost_per_hour

    @property
    def created_at(self) -> str:
        return self._data.created_at

    @property
    def updated_at(self) -> str:
        return self._data.updated_at

    def __repr__(self) -> str:
        return f"Sandbox(uid={self.uid!r}, name={self.name!r}, state={self.state!r})"

    def refresh(self) -> "Sandbox":
        return self._client.sandboxes.get(self.uid)

    def get_state(self) -> SandboxState:
        return self._client.sandboxes.get_state(self.uid)

    def update(self, request: "SandboxUpdateParams") -> "Sandbox":
        return self._client.sandboxes.update(self.uid, request)

    def freeze(
        self,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "Sandbox":
        return self._client.sandboxes.freeze(
            self.uid,
            wait=wait,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def thaw(
        self,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "Sandbox":
        return self._client.sandboxes.thaw(
            self.uid,
            wait=wait,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def delete(self) -> None:
        self._client.sandboxes.delete(self.uid)

    def fork(
        self,
        request: Optional["ForkRequest"] = None,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "Sandbox":
        return self._client.sandboxes.fork(
            self.uid,
            request,
            wait=wait,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def publish(
        self,
        request: "PublishRequest",
        *,
        wait: bool = True,
        timeout: float = 600,
        poll_interval: float = 5,
    ) -> "SandboxTemplate":
        return self._client.sandboxes.publish(
            self.uid,
            request,
            wait=wait,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def exec(self, cmd: str, *, timeout_sec: int = 60) -> "ExecResult":
        return self._client.sandboxes.exec(self.uid, cmd, timeout_sec=timeout_sec)

    def read_file(self, path: str, *, as_text: bool = False):
        return self._client.sandboxes.files.read(self.uid, path, as_text=as_text)

    def write_file(self, path: str, data: Union[bytes, str]) -> None:
        self._client.sandboxes.files.write(self.uid, path, data)

    def mint_access_ticket(self, *, ttl_sec: int = 60) -> "AccessTicket":
        return self._client.sandboxes.mint_access_ticket(self.uid, ttl_sec=ttl_sec)

    def get_desktop(self) -> "DesktopInfo":
        return self._client.sandboxes.get_desktop(self.uid)

    def list_terminals(self) -> Sequence["TerminalSession"]:
        return self._client.sandboxes.terminals.list(self.uid)

    def create_terminal(self, *, cols: int = 80, rows: int = 24) -> "TerminalSession":
        return self._client.sandboxes.terminals.create(self.uid, cols=cols, rows=rows)

    def delete_terminal(self, terminal_id: str) -> None:
        self._client.sandboxes.terminals.delete(self.uid, terminal_id)

    def connect_terminal(
        self,
        terminal_id: str,
        *,
        ticket: Optional[str] = None,
        ticket_ttl_sec: int = 60,
        use_bearer: bool = False,
        open_timeout: float = 10,
    ) -> "TerminalConnection":
        return self._client.sandboxes.terminals.connect(
            self.uid,
            terminal_id,
            ticket=ticket,
            ticket_ttl_sec=ticket_ttl_sec,
            use_bearer=use_bearer,
            open_timeout=open_timeout,
        )


@dataclass(frozen=True)
class _SandboxTemplateData:
    uid: str
    name: str
    kind: SandboxTemplateKind
    status: SandboxTemplateStatus
    display_name: Optional[str] = None
    description: Optional[str] = None
    status_message: Optional[str] = None
    resource_name: str = ""
    cost_per_hour: Optional[float] = None
    frozen_cost_per_hour: Optional[float] = None
    source_workload_uid: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_SandboxTemplateData":
        if not isinstance(data, dict) or not data.get("uid"):
            raise ValidationError("sandbox template is missing uid", field="uid")
        return cls(
            uid=data["uid"],
            name=data["name"],
            display_name=data.get("display_name"),
            description=data.get("description"),
            kind=SandboxTemplateKind(data["kind"]),
            status=SandboxTemplateStatus(data["status"]),
            status_message=data.get("status_message"),
            resource_name=data.get("resource_name", ""),
            cost_per_hour=data.get("cost_per_hour"),
            frozen_cost_per_hour=data.get("frozen_cost_per_hour"),
            source_workload_uid=data.get("source_workload_uid"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(frozen=True)
class SandboxTemplate:
    """Hydrated sandbox template with read-only fields and thin API methods."""

    _client: Optional["Client"] = field(repr=False, compare=False)
    _data: _SandboxTemplateData = field(repr=False)

    @classmethod
    def _from_dict(cls, client: "Client", data: Dict[str, Any]) -> "SandboxTemplate":
        return cls(client, _SandboxTemplateData.from_dict(data))

    @property
    def uid(self) -> str:
        return self._data.uid

    @property
    def name(self) -> str:
        return self._data.name

    @property
    def kind(self) -> SandboxTemplateKind:
        return self._data.kind

    @property
    def status(self) -> SandboxTemplateStatus:
        return self._data.status

    @property
    def display_name(self) -> Optional[str]:
        return self._data.display_name

    @property
    def description(self) -> Optional[str]:
        return self._data.description

    @property
    def status_message(self) -> Optional[str]:
        return self._data.status_message

    @property
    def resource_name(self) -> str:
        return self._data.resource_name

    @property
    def cost_per_hour(self) -> Optional[float]:
        return self._data.cost_per_hour

    @property
    def frozen_cost_per_hour(self) -> Optional[float]:
        return self._data.frozen_cost_per_hour

    @property
    def source_workload_uid(self) -> Optional[str]:
        return self._data.source_workload_uid

    @property
    def created_at(self) -> str:
        return self._data.created_at

    @property
    def updated_at(self) -> str:
        return self._data.updated_at

    def __repr__(self) -> str:
        return (
            f"SandboxTemplate(uid={self.uid!r}, name={self.name!r}, "
            f"status={self.status!r})"
        )

    def _require_client(self) -> "Client":
        if self._client is None:
            raise ValidationError(
                "sandbox template is not bound to a client", field="template"
            )
        return self._client

    def refresh(self) -> "SandboxTemplate":
        client = self._require_client()
        return client.sandboxes.templates.get(_non_empty(self.uid, "template.uid"))

    def update(
        self,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "SandboxTemplate":
        client = self._require_client()
        return client.sandboxes.templates.update(
            _non_empty(self.uid, "template.uid"),
            display_name=display_name,
            description=description,
        )

    def delete(self) -> None:
        client = self._require_client()
        client.sandboxes.templates.delete(_non_empty(self.uid, "template.uid"))

    def create_sandbox(
        self,
        request: Optional[Union["SandboxCreateParams", str]] = None,
        *,
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        ssh_keys: Sequence[str] = (),
        ports: Sequence[PortConfig] = (),
        ttl_sec: Optional[int] = None,
        idle_timeout_sec: Optional[int] = None,
        wait_until_running: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "Sandbox":
        client = self._require_client()
        if isinstance(request, SandboxCreateParams):
            if (
                name is not None
                or any(
                    value is not None
                    for value in (project_id, ttl_sec, idle_timeout_sec)
                )
                or ssh_keys
                or ports
                or not wait_until_running
            ):
                raise ValidationError(
                    "sandbox options cannot be combined with SandboxCreateParams",
                    field="request",
                )
            if request.template is not self:
                raise ValidationError(
                    "SandboxCreateParams.template must be this template",
                    field="template",
                )
            params = request
        else:
            if request is not None and not isinstance(request, str):
                raise ValidationError(
                    "request must be SandboxCreateParams or a sandbox name",
                    field="request",
                    value=request,
                )
            if request is not None and name is not None:
                raise ValidationError(
                    "sandbox name must be provided once", field="name"
                )
            sandbox_name = request if isinstance(request, str) else name
            if sandbox_name is None:
                raise ValidationError("sandbox name is required", field="name")
            params = SandboxCreateParams(
                name=sandbox_name,
                template=self,
                project_id=project_id,
                ssh_keys=ssh_keys,
                ports=ports,
                ttl_sec=ttl_sec,
                idle_timeout_sec=idle_timeout_sec,
                wait_until_running=wait_until_running,
            )
        return client.sandboxes.create(
            params, timeout=timeout, poll_interval=poll_interval
        )


T = TypeVar("T")


@dataclass(frozen=True)
class ListPage(Generic[T]):
    items: Sequence[T] = field(default_factory=tuple)
    next_cursor: Optional[str] = None


@dataclass(frozen=True)
class SandboxCreateParams:
    name: str
    template: SandboxTemplate
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
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            code=data.get("code", 0),
            timed_out=bool(data.get("timed_out", False)),
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
            id=data["id"],
            pid=data.get("pid", 0),
            started_at=data.get("started_at", ""),
            exited=bool(data.get("exited", False)),
            exit_code=data.get("exit_code"),
        )


@dataclass(frozen=True)
class DesktopInfo:
    available: bool
    port: Optional[int] = None
    listening: Optional[bool] = None
    ws_url: Optional[str] = None


class TerminalConnection:
    """A synchronous raw-binary PTY WebSocket connection."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def send(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("terminal data must be bytes")
        self._connection.send(data)

    def recv(self, timeout: Optional[float] = None) -> bytes:
        data = self._connection.recv(timeout=timeout)
        if not isinstance(data, bytes):
            raise RuntimeError("terminal server sent a non-binary WebSocket frame")
        return data

    def __iter__(self) -> Iterator[bytes]:
        for data in self._connection:
            if not isinstance(data, bytes):
                raise RuntimeError("terminal server sent a non-binary WebSocket frame")
            yield data

    def close(self, code: int = 1000, reason: str = "") -> None:
        self._connection.close(code=code, reason=reason)

    def __enter__(self) -> "TerminalConnection":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{field_name} must be a non-empty string", field=field_name, value=value
        )
    return value.strip()


def _validate_name(value: str, field_name: str = "name") -> str:
    value = _non_empty(value, field_name)
    if not NAME_RE.fullmatch(value):
        raise ValidationError(
            f"{field_name} must be 1-32 lowercase alphanumeric or hyphen characters "
            "and cannot start or end with a hyphen",
            field=field_name,
            value=value,
        )
    return value


def _validate_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValidationError(
            "limit must be between 1 and 1000", field="limit", value=limit
        )
    return limit


def _validate_timeouts(ttl_sec: Optional[int], idle_timeout_sec: Optional[int]) -> None:
    if ttl_sec is not None and (not isinstance(ttl_sec, int) or ttl_sec < 0):
        raise ValidationError(
            "ttl_sec must be zero or positive", field="ttl_sec", value=ttl_sec
        )
    if idle_timeout_sec is not None and (
        not isinstance(idle_timeout_sec, int) or idle_timeout_sec < 0
    ):
        raise ValidationError(
            "idle_timeout_sec must be zero or positive",
            field="idle_timeout_sec",
            value=idle_timeout_sec,
        )
    if (
        ttl_sec is not None
        and idle_timeout_sec is not None
        and ttl_sec > 0
        and idle_timeout_sec > 0
        and idle_timeout_sec >= ttl_sec
    ):
        raise ValidationError(
            "idle_timeout_sec must be less than ttl_sec",
            field="idle_timeout_sec",
            value=idle_timeout_sec,
        )


def _validate_update_timeouts(
    ttl_sec: Optional[int], idle_timeout_sec: Optional[int]
) -> None:
    _validate_timeouts(ttl_sec, idle_timeout_sec)
    if idle_timeout_sec is not None and idle_timeout_sec <= 0:
        raise ValidationError(
            "idle_timeout_sec must be positive when updating a sandbox",
            field="idle_timeout_sec",
            value=idle_timeout_sec,
        )


def _ports_payload(ports: Sequence[PortConfig]) -> List[Dict[str, Any]]:
    payload = []
    for item in ports:
        if not isinstance(item, PortConfig):
            raise ValidationError(
                "ports must contain PortConfig instances", field="ports", value=item
            )
        if item.port == 22:
            raise ValidationError("port 22 cannot be exposed", field="ports", value=22)
        if not 1 <= item.port <= 65535:
            raise ValidationError(
                "port must be between 1 and 65535", field="ports", value=item.port
            )
        try:
            protocol = PortProtocol(item.protocol.upper())
        except (AttributeError, ValueError) as exc:
            raise ValidationError(
                "sandbox port protocol must be TCP or UDP",
                field="ports",
                value=item.protocol,
            ) from exc
        value = item.to_payload()
        value["protocol"] = protocol.value
        payload.append(value)
    return payload


class _SandboxHTTPClient(BaseHTTPClient):
    def _workloads_path(self, *parts: str) -> str:
        path = org_path(self.client.require_org(), "workloads")
        if parts:
            path = f"{path}/{'/'.join(parts)}"
        return path

    def _request_for(self, workload_uid: str, method: str, path: str, **kwargs: Any):
        return self._request(method, path, _workload_uid=workload_uid, **kwargs)

    def _request_no_retry(
        self,
        method: str,
        path: str,
        *,
        workload_uid: Optional[str] = None,
        **kwargs: Any,
    ):
        """Use the sandbox-only transport with retries disabled."""
        kwargs.setdefault("timeout", self.client.config.timeout)
        kwargs.setdefault("verify", self.client.config.verify_ssl)
        response = self.client.sandbox_no_retry_session.request(
            method,
            f"{self.base_url}{path}",
            **kwargs,
        )
        return self._handle_response(response, workload_uid=workload_uid)


class SandboxTemplatesClient(_SandboxHTTPClient):
    def _path(self, uid: Optional[str] = None) -> str:
        path = org_path(self.client.require_org(), "sandbox-templates")
        return f"{path}/{uid}" if uid else path

    def _hydrate(self, data: Dict[str, Any]) -> SandboxTemplate:
        return SandboxTemplate._from_dict(self.client, data)

    def list(
        self,
        *,
        kind: Optional[SandboxTemplateKind] = None,
        status: Optional[SandboxTemplateStatus] = None,
        limit: int = MAX_LIST_LIMIT,
        cursor: Optional[str] = None,
    ) -> ListPage[SandboxTemplate]:
        params: Dict[str, Any] = {"limit": _validate_limit(limit)}
        if kind is not None:
            params["kind"] = SandboxTemplateKind(kind).value
        if status is not None:
            params["status"] = SandboxTemplateStatus(status).value
        if cursor is not None:
            params["cursor"] = cursor
        data = self._get(self._path(), params=params)
        return ListPage(
            items=tuple(self._hydrate(item) for item in data["items"]),
            next_cursor=data.get("next_cursor"),
        )

    def get(self, uid: str) -> SandboxTemplate:
        return self._hydrate(self._get(self._path(_non_empty(uid, "uid"))))

    def update(
        self,
        uid: str,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SandboxTemplate:
        if display_name is None and description is None:
            raise ValidationError(
                "display_name or description is required", field="display_name"
            )
        if display_name is not None and len(display_name.strip()) > 128:
            raise ValidationError(
                "display_name must be at most 128 characters", field="display_name"
            )
        payload: Dict[str, str] = {}
        if display_name is not None:
            payload["display_name"] = display_name.strip()
        if description is not None:
            payload["description"] = description.strip()
        return self._hydrate(
            self._request_no_retry(
                "PATCH",
                self._path(_non_empty(uid, "uid")),
                json=payload,
            )
        )

    def delete(self, uid: str) -> None:
        self._request_no_retry("DELETE", self._path(_non_empty(uid, "uid")))


class SandboxFilesClient(_SandboxHTTPClient):
    def read(self, workload_uid: str, path: str, *, as_text: bool = False):
        workload_uid = _non_empty(workload_uid, "workload_uid")
        path = _validate_path(path)
        data = self._request_for(
            workload_uid,
            "GET",
            self._workloads_path(workload_uid, "files"),
            params={"path": path},
        )
        raw = base64.b64decode(data["content_b64"], validate=True)
        return raw.decode("utf-8") if as_text else raw

    def write(self, workload_uid: str, path: str, data: Union[bytes, str]) -> None:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        path = _validate_path(path)
        if isinstance(data, str):
            raw = data.encode("utf-8")
        elif isinstance(data, bytes):
            raw = data
        else:
            raise ValidationError("data must be bytes or str", field="data", value=data)
        if len(raw) > MAX_FILE_BYTES:
            raise ValidationError(
                "file content exceeds 256 MiB", field="data", value=len(raw)
            )
        self._request_no_retry(
            "PUT",
            self._workloads_path(workload_uid, "files"),
            workload_uid=workload_uid,
            json={
                "path": path,
                "content_b64": base64.b64encode(raw).decode("ascii"),
            },
        )


def _validate_path(path: str) -> str:
    path = _non_empty(path, "path")
    if not path.startswith("/") or "\x00" in path:
        raise ValidationError(
            "path must be absolute and contain no NUL bytes", field="path", value=path
        )
    return path


class SandboxTerminalsClient(_SandboxHTTPClient):
    def list(self, workload_uid: str) -> Sequence[TerminalSession]:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        data = self._request_for(
            workload_uid,
            "GET",
            self._workloads_path(workload_uid, "terminals"),
        )
        return tuple(TerminalSession.from_dict(item) for item in data)

    def create(
        self, workload_uid: str, *, cols: int = 80, rows: int = 24
    ) -> TerminalSession:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        for field_name, value in (("cols", cols), ("rows", rows)):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= MAX_TERMINAL_DIMENSION
            ):
                raise ValidationError(
                    f"{field_name} must be between 1 and {MAX_TERMINAL_DIMENSION}",
                    field=field_name,
                    value=value,
                )
        data = self._request_no_retry(
            "POST",
            self._workloads_path(workload_uid, "terminals"),
            workload_uid=workload_uid,
            json={"cols": cols, "rows": rows},
        )
        return TerminalSession.from_dict(data)

    def delete(self, workload_uid: str, terminal_id: str) -> None:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        terminal_id = _non_empty(terminal_id, "terminal_id")
        if not TERMINAL_ID_RE.fullmatch(terminal_id):
            raise ValidationError(
                "terminal_id contains invalid characters",
                field="terminal_id",
                value=terminal_id,
            )
        self._request_no_retry(
            "DELETE",
            self._workloads_path(workload_uid, "terminals", terminal_id),
            workload_uid=workload_uid,
        )

    def connect(
        self,
        workload_uid: str,
        terminal_id: str,
        *,
        ticket: Optional[str] = None,
        ticket_ttl_sec: int = 60,
        use_bearer: bool = False,
        open_timeout: float = 10,
    ) -> TerminalConnection:
        """Connect to a terminal using raw binary WebSocket frames.

        By default a fresh, single-use access ticket is minted immediately
        before dialing. Set ``use_bearer=True`` for Authorization-header auth.
        Terminal dimensions are fixed by :meth:`create`; live resize is not
        supported by the backend protocol.
        """
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "Terminal WebSocket support requires the optional dependency; "
                "install with `pip install 'targon-sdk[sandbox]'`."
            ) from exc

        workload_uid = _non_empty(workload_uid, "workload_uid")
        terminal_id = _non_empty(terminal_id, "terminal_id")
        headers = None
        query = ""
        if use_bearer:
            if ticket is not None:
                raise ValidationError(
                    "ticket and use_bearer cannot be used together", field="ticket"
                )
            headers = {"Authorization": f"Bearer {self.client.config.api_key}"}
        else:
            if ticket is None:
                ticket = self.client.sandboxes.mint_access_ticket(
                    workload_uid, ttl_sec=ticket_ttl_sec
                ).ticket
            query = "?" + urlencode({"ticket": _non_empty(ticket, "ticket")})

        base_url = self.base_url
        if base_url.startswith("https://"):
            base_url = "wss://" + base_url[8:]
        elif base_url.startswith("http://"):
            base_url = "ws://" + base_url[7:]
        else:
            raise ValidationError(
                "base_url must use http or https", field="base_url", value=base_url
            )
        ws_path = self._workloads_path(workload_uid, "terminals", terminal_id, "ws")
        url = f"{base_url}{ws_path}{query}"
        try:
            connection = connect(
                url, additional_headers=headers, open_timeout=open_timeout
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code is None:
                status_code = getattr(exc, "status_code", None)
            if status_code == 401:
                raise AccessTicketError(
                    401,
                    "Terminal access ticket is invalid, expired, or already used",
                    reason="WORKLOAD_ACCESS_TICKET_INVALID",
                    workload_uid=workload_uid,
                    cause=exc,
                ) from exc
            raise
        return TerminalConnection(connection)


class SandboxesClient(_SandboxHTTPClient):
    def __init__(self, client: "Client") -> None:
        super().__init__(client)
        self.templates = SandboxTemplatesClient(client)
        self.files = SandboxFilesClient(client)
        self.terminals = SandboxTerminalsClient(client)

    def _hydrate(self, data: Dict[str, Any]) -> Sandbox:
        return Sandbox._from_dict(self.client, data)

    @staticmethod
    def _operation_uid(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict) or not data.get("uid"):
            raise ValidationError("sandbox operation is missing uid", field="uid")
        if data.get("type") not in (None, "", "SANDBOX"):
            raise ValidationError(
                "workload operation is not a SANDBOX",
                field="type",
                value=data.get("type"),
            )
        return data["uid"]

    def create(
        self,
        request: SandboxCreateParams,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        name = _validate_name(request.name)
        if not isinstance(request.template, SandboxTemplate):
            raise ValidationError(
                "template must be a SandboxTemplate resource",
                field="template",
                value=request.template,
            )
        request.template._require_client()
        template_uid = _non_empty(request.template.uid, "template.uid")
        if request.template.status is not SandboxTemplateStatus.READY:
            raise ValidationError(
                "template must be READY before creating a sandbox",
                field="template.status",
                value=request.template.status.value,
            )
        _validate_timeouts(request.ttl_sec, request.idle_timeout_sec)
        config = SandboxConfigInput(
            ttl_sec=request.ttl_sec, idle_timeout_sec=request.idle_timeout_sec
        ).to_payload()
        payload: Dict[str, Any] = {
            "type": "SANDBOX",
            "name": name,
            "image": template_uid,
        }
        if request.project_id is not None:
            payload["project_id"] = request.project_id
        if request.ssh_keys:
            payload["ssh_keys"] = list(request.ssh_keys)
        if request.ports:
            payload["ports"] = _ports_payload(request.ports)
        if config:
            payload["sandbox_config"] = config
        created_uid = self._operation_uid(
            self._request_no_retry("POST", self._workloads_path(), json=payload)
        )
        self._request_no_retry(
            "POST",
            self._workloads_path(created_uid, "deploy"),
            workload_uid=created_uid,
        )
        if not request.wait_until_running:
            return self.get(created_uid)
        return self.wait_for_status(
            created_uid,
            SandboxStatus.RUNNING,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def get(self, workload_uid: str) -> Sandbox:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        return self._hydrate(
            self._request_for(workload_uid, "GET", self._workloads_path(workload_uid))
        )

    def list(
        self,
        *,
        status: Optional[SandboxStatus] = None,
        project_id: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = MAX_LIST_LIMIT,
        cursor: Optional[str] = None,
    ) -> ListPage[SandboxSummary]:
        params: Dict[str, Any] = {"type": "SANDBOX", "limit": _validate_limit(limit)}
        if status is not None:
            params["status"] = SandboxStatus(status).value
        if project_id is not None:
            params["project_id"] = project_id
        if name is not None:
            params["name"] = name
        if cursor is not None:
            params["cursor"] = cursor
        data = self._get(self._workloads_path(), params=params)
        return ListPage(
            items=tuple(SandboxSummary.from_dict(item) for item in data["items"]),
            next_cursor=data.get("next_cursor"),
        )

    def get_state(self, workload_uid: str) -> SandboxState:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        data = self._request_for(
            workload_uid,
            "GET",
            self._workloads_path(workload_uid, "state"),
        )
        if not isinstance(data, dict) or data.get("workload_type") != "SANDBOX":
            value = data.get("workload_type") if isinstance(data, dict) else None
            raise ValidationError(
                "workload state is not for a SANDBOX",
                field="workload_type",
                value=value,
            )
        state = SandboxState.from_dict(data)
        if state is None:
            raise SandboxStateError(
                502,
                "Sandbox state response is missing status",
                workload_uid=workload_uid,
            )
        return state

    def update(self, workload_uid: str, request: SandboxUpdateParams) -> Sandbox:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        payload: Dict[str, Any] = {}
        if request.name is not None:
            payload["name"] = _validate_name(request.name)
        if request.project_id is not None:
            payload["project_id"] = request.project_id
        if request.ssh_keys is not None:
            payload["ssh_keys"] = list(request.ssh_keys)
        if request.ports is not None:
            payload["ports"] = _ports_payload(request.ports)
        if request.sandbox_config is not None:
            _validate_update_timeouts(
                request.sandbox_config.ttl_sec,
                request.sandbox_config.idle_timeout_sec,
            )
            payload["sandbox_config"] = request.sandbox_config.to_payload()
        if not payload:
            raise ValidationError("update requires at least one field", field="request")
        data = self._request_no_retry(
            "PATCH",
            self._workloads_path(workload_uid),
            workload_uid=workload_uid,
            json=payload,
        )
        return self._hydrate(data)

    def delete(self, workload_uid: str) -> None:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        self._request_no_retry(
            "DELETE",
            self._workloads_path(workload_uid),
            workload_uid=workload_uid,
        )

    def attach_ssh_key(self, workload_uid: str, key_uid: str) -> None:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        key_uid = _non_empty(key_uid, "key_uid")
        self._request_no_retry(
            "PUT",
            self._workloads_path(workload_uid, "ssh-keys", key_uid),
            workload_uid=workload_uid,
        )

    def detach_ssh_key(self, workload_uid: str, key_uid: str) -> None:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        key_uid = _non_empty(key_uid, "key_uid")
        self._request_no_retry(
            "DELETE",
            self._workloads_path(workload_uid, "ssh-keys", key_uid),
            workload_uid=workload_uid,
        )

    def freeze(
        self,
        workload_uid: str,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        self._request_no_retry(
            "POST",
            self._workloads_path(workload_uid, "freeze"),
            workload_uid=workload_uid,
        )
        if not wait:
            return self.get(workload_uid)
        return self.wait_for_status(
            workload_uid,
            SandboxStatus.FROZEN,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def thaw(
        self,
        workload_uid: str,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        self._request_no_retry(
            "POST",
            self._workloads_path(workload_uid, "thaw"),
            workload_uid=workload_uid,
        )
        if not wait:
            return self.get(workload_uid)
        return self.wait_for_status(
            workload_uid,
            SandboxStatus.RUNNING,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def fork(
        self,
        workload_uid: str,
        request: Optional[ForkRequest] = None,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        request = request or ForkRequest()
        payload: Dict[str, Any] = {}
        if request.name is not None:
            payload["name"] = _validate_name(request.name)
        if request.project_id is not None:
            payload["project_id"] = request.project_id
        if request.sandbox_config is not None:
            payload["sandbox_config"] = request.sandbox_config.to_payload()
        child_uid = self._operation_uid(
            self._request_no_retry(
                "POST",
                self._workloads_path(workload_uid, "fork"),
                workload_uid=workload_uid,
                json=payload,
            )
        )
        if not wait:
            return self.get(child_uid)
        return self.wait_for_status(
            child_uid,
            SandboxStatus.RUNNING,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def publish(
        self,
        workload_uid: str,
        request: PublishRequest,
        *,
        wait: bool = True,
        timeout: float = 600,
        poll_interval: float = 5,
    ) -> SandboxTemplate:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        name = _non_empty(request.name, "name")
        if not TEMPLATE_NAME_RE.fullmatch(name):
            raise ValidationError(
                "name must be 1-64 letters, digits, '.', '_' or '-', starting "
                "with a letter or digit",
                field="name",
                value=name,
            )
        payload: Dict[str, str] = {"name": name}
        if request.display_name is not None:
            if len(request.display_name.strip()) > 128:
                raise ValidationError(
                    "display_name must be at most 128 characters",
                    field="display_name",
                )
            payload["display_name"] = request.display_name.strip()
        if request.description is not None:
            payload["description"] = request.description.strip()
        template = self.templates._hydrate(
            self._request_no_retry(
                "POST",
                self._workloads_path(workload_uid, "publish"),
                workload_uid=workload_uid,
                json=payload,
            )
        )
        if not wait:
            return template
        deadline = time.monotonic() + timeout
        while True:
            template = self.templates.get(template.uid)
            if template.status is SandboxTemplateStatus.READY:
                return template
            if template.status is SandboxTemplateStatus.FAILED:
                raise SandboxTemplateError(
                    409,
                    template.status_message or "Sandbox template publishing failed",
                    reason="WORKLOAD_SANDBOX_TEMPLATE_FAILED",
                    workload_uid=workload_uid,
                )
            if time.monotonic() >= deadline:
                message = (
                    f"Sandbox template {template.uid} was not ready within "
                    f"{timeout:.0f}s"
                )
                raise TimeoutError(
                    message,
                    timeout=timeout,
                )
            time.sleep(poll_interval)

    def wait_for_status(
        self,
        workload_uid: str,
        status: SandboxStatus,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        target = SandboxStatus(status)
        deadline = time.monotonic() + timeout
        terminal = {
            SandboxStatus.ERROR,
            SandboxStatus.SUSPENDED,
            SandboxStatus.DELETED,
        }
        while True:
            state = self.get_state(workload_uid)
            if state.status is target:
                return self.get(workload_uid)
            if state.status in terminal and state.status is not target:
                raise SandboxStateError(
                    409,
                    f"Sandbox entered {state.status.value} while waiting for "
                    f"{target.value}: {state.message}",
                    reason="WORKLOAD_SANDBOX_INVALID_STATE",
                    workload_uid=workload_uid,
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Sandbox {workload_uid} did not reach {target.value} within "
                    f"{timeout:.0f}s (last status: {state.status.value})",
                    timeout=timeout,
                )
            time.sleep(poll_interval)

    def exec(self, workload_uid: str, cmd: str, *, timeout_sec: int = 60) -> ExecResult:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        _non_empty(cmd, "cmd")
        if len(cmd.encode("utf-8")) > MAX_COMMAND_BYTES:
            raise ValidationError(
                "cmd exceeds 64 KiB", field="cmd", value=len(cmd.encode("utf-8"))
            )
        if (
            not isinstance(timeout_sec, int)
            or isinstance(timeout_sec, bool)
            or not 1 <= timeout_sec <= 600
        ):
            raise ValidationError(
                "timeout_sec must be between 1 and 600",
                field="timeout_sec",
                value=timeout_sec,
            )
        data = self._request_no_retry(
            "POST",
            self._workloads_path(workload_uid, "exec"),
            workload_uid=workload_uid,
            json={"cmd": cmd, "timeout_sec": timeout_sec},
        )
        return ExecResult.from_dict(data)

    def mint_access_ticket(
        self, workload_uid: str, *, ttl_sec: int = 60
    ) -> AccessTicket:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        if (
            not isinstance(ttl_sec, int)
            or isinstance(ttl_sec, bool)
            or not 1 <= ttl_sec <= 300
        ):
            raise ValidationError(
                "ttl_sec must be between 1 and 300", field="ttl_sec", value=ttl_sec
            )
        data = self._request_no_retry(
            "POST",
            self._workloads_path(workload_uid, "access-tickets"),
            workload_uid=workload_uid,
            json={"ttl_sec": ttl_sec},
        )
        return AccessTicket(ticket=data["ticket"], expires_at=data["expires_at"])

    def get_desktop(self, workload_uid: str) -> DesktopInfo:
        workload_uid = _non_empty(workload_uid, "workload_uid")
        data = self._request_for(
            workload_uid,
            "GET",
            self._workloads_path(workload_uid, "desktop"),
        )
        return DesktopInfo(
            available=bool(data.get("available", False)),
            port=data.get("port"),
            listening=data.get("listening"),
            ws_url=data.get("ws_url"),
        )


__all__ = [
    "AccessTicket",
    "DesktopInfo",
    "ExecResult",
    "ForkRequest",
    "ListPage",
    "PortProtocol",
    "PublishRequest",
    "Sandbox",
    "SandboxConfig",
    "SandboxConfigInput",
    "SandboxCreateParams",
    "SandboxFilesClient",
    "SandboxPort",
    "SandboxResource",
    "SandboxSshKey",
    "SandboxState",
    "SandboxStatus",
    "SandboxSummary",
    "SandboxTemplate",
    "SandboxTemplateKind",
    "SandboxTemplateStatus",
    "SandboxTemplatesClient",
    "SandboxTerminalsClient",
    "SandboxUpdateParams",
    "SandboxesClient",
    "TerminalConnection",
    "TerminalSession",
]
