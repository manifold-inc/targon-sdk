from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import org_path
from targon.core.exceptions import HydrationError, ValidationError
from targon.core.objects import BaseHTTPClient


def _validate_non_empty(value: Optional[str], field_name: str) -> str:
    if not value or not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{field_name} must be a non-empty string",
            field=field_name,
            value=value,
        )
    return value.strip()


def _require_dict(data: Any, *, source: str, object_type: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise HydrationError(
            f"Expected dict from {source}, got {type(data).__name__}",
            object_type=object_type,
        )
    return data


@dataclass
class PortConfig:
    port: int
    protocol: str = "TCP"
    routing: str = "PROXIED"

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"port": self.port}
        if self.protocol:
            payload["protocol"] = self.protocol
        if self.routing:
            payload["routing"] = self.routing
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PortConfig:
        return cls(
            port=data.get("port", 0),
            protocol=data.get("protocol", "TCP"),
            routing=data.get("routing", "PROXIED"),
        )


@dataclass
class WorkloadSshKey:
    uid: str
    name: str = ""
    public_key: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadSshKey:
        return cls(
            uid=data.get("uid", ""),
            name=data.get("name", ""),
            public_key=data.get("public_key_raw", ""),
        )


@dataclass
class WorkloadURL:
    port: int
    url: str


@dataclass
class WorkloadState:
    status: str = ""
    message: str = ""
    ready_replicas: int = 0
    total_replicas: int = 0
    urls: List[WorkloadURL] = field(default_factory=list)
    public_ip: Optional[str] = None
    ssh_port: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> WorkloadState:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            status=data.get("status", ""),
            message=data.get("message", ""),
            ready_replicas=data.get("ready_replicas", 0),
            total_replicas=data.get("total_replicas", 0),
            urls=[
                WorkloadURL(port=item.get("port", 0), url=item.get("url", ""))
                for item in data.get("urls") or []
                if isinstance(item, dict)
            ],
            public_ip=data.get("public_ip"),
            ssh_port=data.get("ssh_port"),
        )


@dataclass
class WorkloadResource:
    name: str = ""
    display_name: str = ""
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    vcpu: int = 0
    memory: int = 0
    disk_size_mib: Optional[int] = None
    network_mode: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional[WorkloadResource]:
        if not data or not isinstance(data, dict):
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


@dataclass
class WorkloadResponse:
    uid: str
    name: str = ""
    image: str = ""
    type: str = ""
    resource_name: str = ""
    region: str = ""
    hostname: str = ""
    project_id: Optional[str] = None
    ports: List[PortConfig] = field(default_factory=list)
    ssh_keys: List[WorkloadSshKey] = field(default_factory=list)
    state: Optional[WorkloadState] = None
    resource: Optional[WorkloadResource] = None
    cost_per_hour: Optional[float] = None
    frozen_cost_per_hour: Optional[float] = None
    revision: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadResponse:
        data = _require_dict(
            data, source="workload response", object_type="WorkloadResponse"
        )
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in workload response", object_type="WorkloadResponse"
            )
        return cls(
            uid=uid,
            name=data.get("name", ""),
            image=data.get("image", ""),
            type=data.get("type", ""),
            resource_name=data.get("resource_name", ""),
            region=data.get("region", ""),
            hostname=data.get("hostname", ""),
            project_id=data.get("project_id"),
            ports=[
                PortConfig.from_dict(item)
                for item in data.get("ports") or []
                if isinstance(item, dict)
            ],
            ssh_keys=[
                WorkloadSshKey.from_dict(item)
                for item in data.get("ssh_keys") or []
                if isinstance(item, dict)
            ],
            state=WorkloadState.from_dict(data.get("state")),
            resource=WorkloadResource.from_dict(data.get("resource")),
            cost_per_hour=data.get("cost_per_hour"),
            frozen_cost_per_hour=data.get("frozen_cost_per_hour"),
            revision=data.get("revision", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class WorkloadListItem:
    uid: str
    name: str = ""
    image: str = ""
    type: str = ""
    state: Optional[WorkloadState] = None
    resource: Optional[WorkloadResource] = None
    cost_per_hour: Optional[float] = None
    frozen_cost_per_hour: Optional[float] = None
    revision: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadListItem:
        data = _require_dict(
            data, source="workload list item", object_type="WorkloadListItem"
        )
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in workload list item", object_type="WorkloadListItem"
            )
        return cls(
            uid=uid,
            name=data.get("name", ""),
            image=data.get("image", ""),
            type=data.get("type", ""),
            state=WorkloadState.from_dict(data.get("state")),
            resource=WorkloadResource.from_dict(data.get("resource")),
            cost_per_hour=data.get("cost_per_hour"),
            frozen_cost_per_hour=data.get("frozen_cost_per_hour"),
            revision=data.get("revision", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class WorkloadListResponse:
    items: List[WorkloadListItem] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadListResponse:
        data = _require_dict(
            data, source="workload list", object_type="WorkloadListResponse"
        )
        items = data.get("items", [])
        if not isinstance(items, list):
            raise HydrationError(
                f"Expected list for items, got {type(items).__name__}",
                object_type="WorkloadListResponse",
            )
        return cls(
            items=[
                WorkloadListItem.from_dict(item)
                for item in items
                if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class WorkloadStateResponse:
    uid: str
    workload_type: str = ""
    status: str = ""
    message: str = ""
    ready_replicas: int = 0
    total_replicas: int = 0
    urls: List[WorkloadURL] = field(default_factory=list)
    public_ip: Optional[str] = None
    ssh_port: Optional[int] = None
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadStateResponse:
        data = _require_dict(
            data, source="workload state", object_type="WorkloadStateResponse"
        )
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in state response", object_type="WorkloadStateResponse"
            )
        return cls(
            uid=uid,
            workload_type=data.get("workload_type", ""),
            status=data.get("status", ""),
            message=data.get("message", ""),
            ready_replicas=data.get("ready_replicas", 0),
            total_replicas=data.get("total_replicas", 0),
            urls=[
                WorkloadURL(port=item.get("port", 0), url=item.get("url", ""))
                for item in data.get("urls") or []
                if isinstance(item, dict)
            ],
            public_ip=data.get("public_ip"),
            ssh_port=data.get("ssh_port"),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class WorkloadEvent:
    workload_uid: str
    workload_type: str = ""
    event_type: str = ""
    new_status: Optional[str] = None
    message: Optional[str] = None
    display_message: Optional[str] = None
    reason: Optional[str] = None
    exit_code: Optional[int] = None
    replica_count: Optional[int] = None
    old_replica_count: Optional[int] = None
    resource_name: Optional[str] = None
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadEvent:
        data = _require_dict(data, source="workload event", object_type="WorkloadEvent")
        return cls(
            workload_uid=data.get("workload_uid", ""),
            workload_type=data.get("workload_type", ""),
            event_type=data.get("event_type", ""),
            new_status=data.get("new_status"),
            message=data.get("message"),
            display_message=data.get("display_message"),
            reason=data.get("reason"),
            exit_code=data.get("exit_code"),
            replica_count=data.get("replica_count"),
            old_replica_count=data.get("old_replica_count"),
            resource_name=data.get("resource_name"),
            created_at=data.get("created_at", ""),
        )


@dataclass
class WorkloadEventsResponse:
    items: List[WorkloadEvent] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadEventsResponse:
        data = _require_dict(
            data, source="workload events", object_type="WorkloadEventsResponse"
        )
        items = data.get("items", [])
        if not isinstance(items, list):
            raise HydrationError(
                f"Expected list for event items, got {type(items).__name__}",
                object_type="WorkloadEventsResponse",
            )
        return cls(
            items=[
                WorkloadEvent.from_dict(item)
                for item in items
                if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


class WorkloadClient(BaseHTTPClient):
    def _path(self, *parts: str) -> str:
        path = org_path(self.client.require_org(), "workloads")
        if parts:
            path = f"{path}/{'/'.join(parts)}"
        return path

    def _request_no_retry(self, method: str, path: str, **kwargs: Any):
        """Send a non-idempotent workload request without automatic retries."""
        workload_uid = kwargs.pop("_workload_uid", None)
        kwargs.setdefault("timeout", self.client.config.timeout)
        kwargs.setdefault("verify", self.client.config.verify_ssl)
        response = self.client.sandbox_no_retry_session.request(
            method,
            f"{self.base_url}{path}",
            **kwargs,
        )
        return self._handle_response(response, workload_uid=workload_uid)

    def get(self, workload_uid: str) -> WorkloadResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = self._get(self._path(workload_uid))
        return WorkloadResponse.from_dict(result)

    def list(
        self,
        *,
        type: Optional[str] = None,
        status: Optional[str] = None,
        project_id: Optional[str] = None,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> WorkloadListResponse:
        params: Dict[str, Any] = {}
        if type is not None:
            params["type"] = type
        if status is not None:
            params["status"] = status
        if project_id is not None:
            params["project_id"] = project_id
        if name is not None:
            params["name"] = name
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        result = self._get(self._path(), params=params or None)
        return WorkloadListResponse.from_dict(result)

    def delete(self, workload_uid: str) -> None:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        self._delete(self._path(workload_uid))

    def get_state(self, workload_uid: str) -> WorkloadStateResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = self._get(self._path(workload_uid, "state"))
        return WorkloadStateResponse.from_dict(result)

    def get_events(
        self,
        workload_uid: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> WorkloadEventsResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(
            self._path(workload_uid, "events"),
            params=params or None,
        )
        return WorkloadEventsResponse.from_dict(result)
