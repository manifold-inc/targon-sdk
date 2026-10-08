from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import targon.client.sandbox.capabilities as capabilities
from targon.client.sandbox.models import (
    AccessTicket,
    DesktopInfo,
    ExecResult,
    ForkRequest,
    PublishRequest,
    SandboxConfig,
    SandboxCreateParams,
    SandboxPort,
    SandboxResource,
    SandboxSshKey,
    SandboxState,
    SandboxTemplateKind,
    SandboxTemplateStatus,
    SandboxUpdateParams,
    invalid,
    non_empty,
)
from targon.client.workload import PortConfig
from targon.core.exceptions import ValidationError

if TYPE_CHECKING:
    from targon.client.client import Client


def _items(data: Dict[str, Any], key: str, model: Any) -> Sequence[Any]:
    return tuple(
        model.from_dict(item) for item in data.get(key) or [] if isinstance(item, dict)
    )


def _wait_options(wait: bool, timeout: float, poll_interval: float) -> Dict[str, Any]:
    return {"wait": wait, "timeout": timeout, "poll_interval": poll_interval}


@dataclass(frozen=True)
class Sandbox:
    _client: "Client" = field(repr=False, compare=False)
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
    def _from_dict(cls, client: "Client", data: Dict[str, Any]) -> "Sandbox":
        if not isinstance(data, dict) or not data.get("uid"):
            raise ValidationError("sandbox response is missing uid", field="uid")
        workload_type = data.get("type")
        if workload_type != "SANDBOX":
            raise invalid("workload is not a SANDBOX", "type", workload_type)
        return cls(
            _client=client,
            uid=data["uid"],
            name=data.get("name", ""),
            type=workload_type,
            image=data.get("image", ""),
            resource_name=data.get("resource_name", ""),
            project_id=data.get("project_id"),
            ports=_items(data, "ports", SandboxPort),
            ssh_keys=_items(data, "ssh_keys", SandboxSshKey),
            sandbox_config=SandboxConfig.from_dict(data.get("sandbox_config")),
            state=SandboxState.from_dict(data.get("state")),
            resource=SandboxResource.from_dict(data.get("resource")),
            cost_per_hour=data.get("cost_per_hour"),
            frozen_cost_per_hour=data.get("frozen_cost_per_hour"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def __repr__(self) -> str:
        return f"Sandbox(uid={self.uid!r}, name={self.name!r}, state={self.state!r})"

    @property
    def files(self) -> capabilities.SandboxFiles:
        return capabilities.SandboxFiles(self._client, self.uid)

    @property
    def terminals(self) -> capabilities.SandboxTerminals:
        return capabilities.SandboxTerminals(self._client, self.uid)

    def refresh(self) -> "Sandbox":
        return self._client.sandboxes.get(self.uid)

    def get_state(self) -> SandboxState:
        return self._client.sandboxes.get_state(self.uid)

    def update(self, request: SandboxUpdateParams) -> "Sandbox":
        return self._client.sandboxes.update(self.uid, request)

    def freeze(
        self,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "Sandbox":
        return self._client.sandboxes.freeze(
            self.uid, **_wait_options(wait, timeout, poll_interval)
        )

    def thaw(
        self,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "Sandbox":
        return self._client.sandboxes.thaw(
            self.uid, **_wait_options(wait, timeout, poll_interval)
        )

    def delete(self) -> None:
        self._client.sandboxes.delete(self.uid)

    def fork(
        self,
        request: Optional[ForkRequest] = None,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "Sandbox":
        return self._client.sandboxes.fork(
            self.uid, request, **_wait_options(wait, timeout, poll_interval)
        )

    def publish(
        self,
        request: PublishRequest,
        *,
        wait: bool = True,
        timeout: float = 600,
        poll_interval: float = 5,
    ) -> "SandboxTemplate":
        return self._client.sandboxes.publish(
            self.uid, request, **_wait_options(wait, timeout, poll_interval)
        )

    def exec(self, cmd: str, *, timeout_sec: int = 60) -> ExecResult:
        return self._client.sandboxes.exec(self.uid, cmd, timeout_sec=timeout_sec)

    def mint_access_ticket(self, *, ttl_sec: int = 60) -> AccessTicket:
        return self._client.sandboxes.mint_access_ticket(self.uid, ttl_sec=ttl_sec)

    def get_desktop(self) -> DesktopInfo:
        return self._client.sandboxes.get_desktop(self.uid)


@dataclass(frozen=True)
class SandboxTemplate:
    _client: Optional["Client"] = field(repr=False, compare=False)
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
    def _from_dict(cls, client: "Client", data: Dict[str, Any]) -> "SandboxTemplate":
        if not isinstance(data, dict) or not data.get("uid"):
            raise ValidationError("sandbox template is missing uid", field="uid")
        return cls(
            _client=client,
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

    def __repr__(self) -> str:
        values = self.uid, self.name, self.status
        return "SandboxTemplate(uid=%r, name=%r, status=%r)" % values

    def _require_client(self) -> "Client":
        if self._client is None:
            raise invalid("sandbox template is not bound to a client", "template")
        return self._client

    def refresh(self) -> "SandboxTemplate":
        client = self._require_client()
        return client.sandboxes.templates.get(non_empty(self.uid, "template.uid"))

    def update(
        self,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "SandboxTemplate":
        client = self._require_client()
        return client.sandboxes.templates.update(
            non_empty(self.uid, "template.uid"),
            display_name=display_name,
            description=description,
        )

    def delete(self) -> None:
        client = self._require_client()
        client.sandboxes.templates.delete(non_empty(self.uid, "template.uid"))

    def create_sandbox(
        self,
        name: str,
        *,
        project_id: Optional[str] = None,
        ssh_keys: Sequence[str] = (),
        ports: Sequence[PortConfig] = (),
        ttl_sec: Optional[int] = None,
        idle_timeout_sec: Optional[int] = None,
        wait_until_running: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        client = self._require_client()
        return client.sandboxes.create(
            SandboxCreateParams(
                name=name,
                template=self,
                project_id=project_id,
                ssh_keys=ssh_keys,
                ports=ports,
                ttl_sec=ttl_sec,
                idle_timeout_sec=idle_timeout_sec,
                wait_until_running=wait_until_running,
            ),
            timeout=timeout,
            poll_interval=poll_interval,
        )
