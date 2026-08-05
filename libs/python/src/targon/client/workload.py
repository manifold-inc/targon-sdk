from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import requests
from requests.adapters import HTTPAdapter

from targon.client.constants import org_path
from targon.core.exceptions import (
    APIError,
    HydrationError,
    TargonError,
    TimeoutError,
    ValidationError,
)
from targon.core.objects import BaseHTTPClient

# Workload states from which a workload will not become ready on its own.
TERMINAL_WORKLOAD_STATES = frozenset({"error", "suspended", "deleted"})
VALID_WORKLOAD_TYPES = frozenset({"RENTAL", "VM"})
DEPRECATED_WORKLOAD_TYPES = frozenset({"SERVERLESS", "INFERENCE", "FUNCTION"})
VALID_LOG_TYPES = frozenset({"serial", "qemu"})

# Maximum total byte length of exec command arguments. The Linux ARG_MAX is
# typically 2**17; we keep some headroom to avoid "Argument list too long".
ARG_MAX_BYTES = 2**16


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


def _validate_absolute_path(value: Optional[str], field_name: str) -> str:
    path = _validate_non_empty(value, field_name)
    if not path.startswith("/"):
        raise ValidationError(
            f"{field_name} must be an absolute path starting with '/'",
            field=field_name,
            value=value,
        )
    return path


def _normalize_log_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = _validate_non_empty(value, "log_type").lower()
    if normalized not in VALID_LOG_TYPES:
        raise ValidationError(
            "log_type must be one of qemu, serial",
            field="log_type",
            value=value,
        )
    return normalized


@dataclass
class EnvVar:
    name: str
    value: str

    def to_payload(self) -> Dict[str, str]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvVar:
        return cls(name=data.get("name", ""), value=data.get("value", ""))


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
class RegistryAuth:
    server: str
    username: str
    password: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "server": _validate_non_empty(self.server, "server"),
            "username": _validate_non_empty(self.username, "username"),
            "password": _validate_non_empty(self.password, "password"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegistryAuth:
        return cls(
            server=data.get("server", ""),
            username=data.get("username", ""),
            password=data.get("password", ""),
        )


@dataclass
class VmConfig:
    password: str

    def to_payload(self) -> Dict[str, str]:
        password = _validate_non_empty(self.password, "vm_config.password")
        if len(password) < 4:
            raise ValidationError(
                "vm_config.password must be at least 4 characters",
                field="vm_config.password",
            )
        return {"password": password}


@dataclass
class VolumeMount:
    uid: str
    mount_path: str
    read_only: bool = False

    def to_payload(self) -> Dict[str, Any]:
        return {
            "uid": _validate_non_empty(self.uid, "volume.uid"),
            "mount_path": _validate_absolute_path(
                self.mount_path,
                "volume.mount_path",
            ),
            "read_only": self.read_only,
        }


@dataclass
class WorkloadVolumeMount:
    uid: str
    name: str = ""
    mount_path: str = ""
    read_only: bool = False
    last_backup_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadVolumeMount:
        return cls(
            uid=data.get("uid", ""),
            name=data.get("name", ""),
            mount_path=data.get("mount_path", ""),
            read_only=data.get("read_only", False),
            last_backup_at=data.get("last_backup_at"),
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


# ---------------------------------------------------------------------------
# Env coercion helper (dict or list of EnvVar)
# ---------------------------------------------------------------------------


def _coerce_env(
    env: Optional[Union[Dict[str, str], Sequence[EnvVar]]],
) -> Optional[List[EnvVar]]:
    if env is None:
        return None
    if isinstance(env, dict):
        return [EnvVar(name=str(k), value=str(v)) for k, v in env.items()]
    if isinstance(env, Sequence):
        out: List[EnvVar] = []
        for item in env:
            if not isinstance(item, EnvVar):
                raise ValidationError(
                    "env sequence must contain EnvVar instances",
                    field="env",
                    value=item,
                )
            out.append(item)
        return out
    raise ValidationError("env must be a dict or sequence of EnvVar", field="env")


@dataclass
class CreateWorkloadRequest:
    name: str
    image: str
    resource_name: str
    type: str = "RENTAL"
    project_id: Optional[str] = None
    ports: Optional[List[PortConfig]] = None
    envs: Optional[Union[Dict[str, str], List[EnvVar]]] = None
    commands: Optional[List[str]] = None
    args: Optional[List[str]] = None
    registry_auth: Optional[RegistryAuth] = None
    volumes: Optional[List[VolumeMount]] = None
    ssh_keys: Optional[List[str]] = None
    vm_config: Optional[VmConfig] = None

    def to_payload(self) -> Dict[str, Any]:
        workload_type = _validate_non_empty(self.type, "type").upper()
        if workload_type in DEPRECATED_WORKLOAD_TYPES:
            raise ValidationError(
                f"workload type {workload_type} has been deprecated",
                field="type",
                value=self.type,
            )
        if workload_type not in VALID_WORKLOAD_TYPES:
            raise ValidationError(
                "type must be one of RENTAL, VM",
                field="type",
                value=self.type,
            )
        if workload_type == "VM":
            if self.vm_config is None:
                raise ValidationError(
                    "vm_config is required for VM workloads",
                    field="vm_config",
                )
            unsupported = {
                "commands": self.commands,
                "args": self.args,
                "envs": self.envs,
                "volumes": self.volumes,
                "registry_auth": self.registry_auth,
            }
            for field_name, value in unsupported.items():
                if value:
                    raise ValidationError(
                        f"{field_name} is not supported for VM workloads",
                        field=field_name,
                    )
        elif self.vm_config is not None:
            raise ValidationError(
                "vm_config is only valid for VM workloads",
                field="vm_config",
            )

        payload: Dict[str, Any] = {
            "name": _validate_non_empty(self.name, "name"),
            "image": _validate_non_empty(self.image, "image"),
            "resource_name": _validate_non_empty(self.resource_name, "resource_name"),
            "type": workload_type,
        }
        if self.project_id:
            payload["project_id"] = self.project_id
        if self.ports:
            payload["ports"] = [p.to_payload() for p in self.ports]
        env_vars = _coerce_env(self.envs)
        if env_vars:
            payload["envs"] = [e.to_payload() for e in env_vars]
        if self.commands:
            payload["commands"] = self.commands
        if self.args:
            payload["args"] = self.args
        if self.registry_auth:
            payload["registry_auth"] = self.registry_auth.to_payload()
        if self.volumes:
            payload["volumes"] = [v.to_payload() for v in self.volumes]
        if self.ssh_keys:
            payload["ssh_keys"] = self.ssh_keys
        if self.vm_config:
            payload["vm_config"] = self.vm_config.to_payload()

        return payload


@dataclass
class UpdateWorkloadRequest:
    name: Optional[str] = None
    image: Optional[str] = None
    project_id: Optional[str] = None
    ports: Optional[List[PortConfig]] = None
    envs: Optional[Union[Dict[str, str], List[EnvVar]]] = None
    commands: Optional[List[str]] = None
    args: Optional[List[str]] = None
    volumes: Optional[List[VolumeMount]] = None
    ssh_keys: Optional[List[str]] = None
    registry_auth: Optional[RegistryAuth] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.name is not None:
            payload["name"] = self.name
        if self.image is not None:
            payload["image"] = self.image
        if self.project_id is not None:
            payload["project_id"] = self.project_id
        if self.ports is not None:
            payload["ports"] = [p.to_payload() for p in self.ports]
        env_vars = _coerce_env(self.envs)
        if env_vars is not None:
            payload["envs"] = [e.to_payload() for e in env_vars]
        if self.commands is not None:
            payload["commands"] = self.commands
        if self.args is not None:
            payload["args"] = self.args
        if self.volumes is not None:
            payload["volumes"] = [v.to_payload() for v in self.volumes]
        if self.ssh_keys is not None:
            payload["ssh_keys"] = self.ssh_keys
        if self.registry_auth is not None:
            payload["registry_auth"] = self.registry_auth.to_payload()
        return payload


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
        urls_raw = data.get("urls") or []
        urls = [
            WorkloadURL(port=item.get("port", 0), url=item.get("url", ""))
            for item in urls_raw
            if isinstance(item, dict)
        ]
        return cls(
            status=data.get("status", ""),
            message=data.get("message", ""),
            ready_replicas=data.get("ready_replicas", 0),
            total_replicas=data.get("total_replicas", 0),
            urls=urls,
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
    project_id: Optional[str] = None
    ports: List[PortConfig] = field(default_factory=list)
    envs: List[EnvVar] = field(default_factory=list)
    commands: Optional[List[str]] = None
    args: Optional[List[str]] = None
    volumes: List[WorkloadVolumeMount] = field(default_factory=list)
    ssh_keys: List[WorkloadSshKey] = field(default_factory=list)
    registry_auth: Optional[RegistryAuth] = None
    state: Optional[WorkloadState] = None
    resource: Optional[WorkloadResource] = None
    cost_per_hour: Optional[float] = None
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
            project_id=data.get("project_id"),
            ports=[
                PortConfig.from_dict(item)
                for item in data.get("ports") or []
                if isinstance(item, dict)
            ],
            envs=[
                EnvVar.from_dict(item)
                for item in data.get("envs") or []
                if isinstance(item, dict)
            ],
            commands=data.get("commands"),
            args=data.get("args"),
            volumes=[
                WorkloadVolumeMount.from_dict(item)
                for item in data.get("volumes") or []
                if isinstance(item, dict)
            ],
            ssh_keys=[
                WorkloadSshKey.from_dict(item)
                for item in data.get("ssh_keys") or []
                if isinstance(item, dict)
            ],
            registry_auth=(
                RegistryAuth.from_dict(data["registry_auth"])
                if isinstance(data.get("registry_auth"), dict)
                else None
            ),
            state=WorkloadState.from_dict(data.get("state")),
            resource=WorkloadResource.from_dict(data.get("resource")),
            cost_per_hour=data.get("cost_per_hour"),
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
    revision: str = ""
    volumes: List[WorkloadVolumeMount] = field(default_factory=list)
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
            revision=data.get("revision", ""),
            volumes=[
                WorkloadVolumeMount.from_dict(item)
                for item in data.get("volumes") or []
                if isinstance(item, dict)
            ],
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
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for items, got {type(items_raw).__name__}",
                object_type="WorkloadListResponse",
            )
        return cls(
            items=[
                WorkloadListItem.from_dict(item)
                for item in items_raw
                if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class WorkloadDeployResponse:
    uid: str
    name: str = ""
    image: str = ""
    type: str = ""
    state: Optional[WorkloadState] = None
    resource: Optional[WorkloadResource] = None
    cost_per_hour: Optional[float] = None
    revision: str = ""
    volumes: List[WorkloadVolumeMount] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkloadDeployResponse:
        data = _require_dict(
            data, source="workload deploy", object_type="WorkloadDeployResponse"
        )
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in deploy response", object_type="WorkloadDeployResponse"
            )
        return cls(
            uid=uid,
            name=data.get("name", ""),
            image=data.get("image", ""),
            type=data.get("type", ""),
            state=WorkloadState.from_dict(data.get("state")),
            resource=WorkloadResource.from_dict(data.get("resource")),
            cost_per_hour=data.get("cost_per_hour"),
            revision=data.get("revision", ""),
            volumes=[
                WorkloadVolumeMount.from_dict(item)
                for item in data.get("volumes") or []
                if isinstance(item, dict)
            ],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


WorkloadOperationResponse = WorkloadDeployResponse


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
        urls_raw = data.get("urls") or []
        urls = [
            WorkloadURL(port=item.get("port", 0), url=item.get("url", ""))
            for item in urls_raw
            if isinstance(item, dict)
        ]
        return cls(
            uid=uid,
            workload_type=data.get("workload_type", ""),
            status=data.get("status", ""),
            message=data.get("message", ""),
            ready_replicas=data.get("ready_replicas", 0),
            total_replicas=data.get("total_replicas", 0),
            urls=urls,
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
    pod_name: Optional[str] = None
    container_name: Optional[str] = None
    container_image: Optional[str] = None
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
            pod_name=data.get("pod_name"),
            container_name=data.get("container_name"),
            container_image=data.get("container_image"),
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
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for event items, got {type(items_raw).__name__}",
                object_type="WorkloadEventsResponse",
            )
        return cls(
            items=[WorkloadEvent.from_dict(item) for item in items_raw],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class VolumeMountResponse:
    workload_uid: str
    uid: str
    mount_path: str = ""
    read_only: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeMountResponse:
        data = _require_dict(
            data, source="volume mount", object_type="VolumeMountResponse"
        )
        return cls(
            workload_uid=data.get("workload_uid", ""),
            uid=data.get("uid", ""),
            mount_path=data.get("mount_path", ""),
            read_only=data.get("read_only", False),
        )


@dataclass
class SshKeyAttachResponse:
    workload_uid: str
    ssh_key_uid: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SshKeyAttachResponse:
        data = _require_dict(
            data, source="ssh key attach", object_type="SshKeyAttachResponse"
        )
        return cls(
            workload_uid=data.get("workload_uid", ""),
            ssh_key_uid=data.get("ssh_key_uid", ""),
        )


@dataclass
class ExecResponse:
    """Result of running a command inside a workload via ``exec``.

    ``result`` holds the merged stdout/stderr text. ``exit_code`` is the exit
    status of the command (recovered by shell-wrapping the command, since the
    exec endpoint does not surface it directly).
    """

    exit_code: int
    result: str

    @property
    def output(self) -> str:
        """Alias for :attr:`result`."""
        return self.result


@dataclass
class VmImage:
    name: str
    display_name: str
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VmImage:
        data = _require_dict(data, source="VM image", object_type="VmImage")
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
        )


class WorkloadClient(BaseHTTPClient):
    def _path(self, *parts: str) -> str:
        path = org_path(self.client.require_org(), "workloads")
        if parts:
            path = f"{path}/{'/'.join(parts)}"
        return path

    def create(self, request: CreateWorkloadRequest) -> WorkloadResponse:
        payload = request.to_payload()
        result = self._post(self._path(), json=payload)
        return WorkloadResponse.from_dict(result)

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

    def update(
        self, workload_uid: str, request: UpdateWorkloadRequest
    ) -> WorkloadResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = self._patch(
            self._path(workload_uid),
            json=request.to_payload(),
        )
        return WorkloadResponse.from_dict(result)

    def delete(self, workload_uid: str) -> None:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        self._delete(self._path(workload_uid))

    def deploy(self, workload_uid: str) -> WorkloadDeployResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = self._post(self._path(workload_uid, "deploy"))
        return WorkloadDeployResponse.from_dict(result)

    def suspend(self, workload_uid: str) -> WorkloadDeployResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = self._post(self._path(workload_uid, "suspend"))
        return WorkloadDeployResponse.from_dict(result)

    def reboot(self, workload_uid: str) -> WorkloadDeployResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = self._post(self._path(workload_uid, "reboot"))
        return WorkloadDeployResponse.from_dict(result)

    def get_state(self, workload_uid: str) -> WorkloadStateResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = self._get(self._path(workload_uid, "state"))
        return WorkloadStateResponse.from_dict(result)

    def wait_until_ready(
        self,
        workload_uid: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 5.0,
    ) -> WorkloadStateResponse:
        """Poll the workload state until its status is ``running``.

        Returns the final :class:`WorkloadStateResponse` once the workload is
        running.

        Raises:
            TargonError: if the workload reaches an error, suspended, or
                deleted state before becoming ready.
            TimeoutError: if the workload is not ready within ``timeout``
                seconds.
        """
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        deadline = time.monotonic() + timeout
        while True:
            state = self.get_state(workload_uid)
            if state.status.lower() == "running":
                return state
            if state.status.lower() in TERMINAL_WORKLOAD_STATES:
                raise TargonError(
                    f"Workload {workload_uid} entered terminal state "
                    f"'{state.status}' before becoming ready"
                    + (f": {state.message}" if state.message else "")
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Workload {workload_uid} was not ready within "
                    f"{timeout:.0f}s (last status: '{state.status}')",
                    timeout=timeout,
                )
            time.sleep(poll_interval)

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

    def get_logs(
        self,
        workload_uid: str,
        *,
        since: Optional[str] = None,
        tail: Optional[int] = None,
        previous: bool = False,
        log_type: Optional[str] = None,
    ) -> str:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        params: Dict[str, Any] = {}
        if since:
            params["since"] = since
        if tail is not None:
            params["tail"] = tail
        if previous:
            params["previous"] = "true"
        normalized_log_type = _normalize_log_type(log_type)
        if normalized_log_type is not None:
            params["type"] = normalized_log_type
        result = self._get(
            self._path(workload_uid, "logs"),
            params=params or None,
        )
        if isinstance(result, str):
            return result
        return str(result)

    def _stream_text(
        self,
        method: str,
        endpoint: str,
        *,
        params: Any = None,
        timeout: Any = (10, None),
        error_label: str = "endpoint",
    ) -> Iterator[str]:
        """Stream a text/plain response line by line, without retries.

        The shared session retries 5xx responses, which turns a transient
        error (e.g. an endpoint returning 500 while the workload is still
        provisioning) into an opaque urllib3 RetryError. Sending through a
        dedicated adapter with retries disabled lets us raise a clean APIError
        carrying the server's message and reason instead.
        """
        url = f"{self.base_url}{endpoint}"
        request = requests.Request(
            method, url, params=params, headers={"Accept": "text/plain"}
        )
        prepared = self.session.prepare_request(request)
        verify = self.session.verify
        if verify is None:
            verify = True
        adapter = HTTPAdapter(max_retries=0)
        try:
            try:
                response = adapter.send(
                    prepared,
                    stream=True,
                    timeout=timeout,
                    verify=verify,
                )
            except requests.RequestException as e:
                raise APIError(500, f"Failed to connect to {error_label}: {e}", cause=e)

            with response:
                if response.status_code >= 400:
                    error_text = response.text
                    message = error_text
                    reason = None
                    try:
                        body = json.loads(error_text)
                        if isinstance(body, dict):
                            message = body.get("error", error_text)
                            reason = body.get("reason")
                    except (ValueError, json.JSONDecodeError):
                        pass
                    raise APIError(
                        response.status_code,
                        message,
                        response={"reason": reason} if reason else None,
                    )
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        decoded = line.rstrip("\n\r")
                        if decoded:
                            yield decoded
        finally:
            adapter.close()

    def stream_logs(
        self,
        workload_uid: str,
        *,
        follow: bool = True,
        since: Optional[str] = None,
        tail: Optional[int] = None,
        previous: bool = False,
        log_type: Optional[str] = None,
    ) -> Iterator[str]:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        params: Dict[str, str] = {"follow": str(follow).lower()}
        if since is not None:
            params["since"] = since
        if tail is not None:
            params["tail"] = str(tail)
        if previous:
            params["previous"] = "true"
        normalized_log_type = _normalize_log_type(log_type)
        if normalized_log_type is not None:
            params["type"] = normalized_log_type
        yield from self._stream_text(
            "GET",
            self._path(workload_uid, "logs"),
            params=params,
            error_label="logs endpoint",
        )

    # -- Exec ----------------------------------------------------------------

    def _exec_raw(
        self,
        workload_uid: str,
        command: Sequence[str],
        *,
        timeout: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream a raw argv exec against the workload exec endpoint."""
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        argv = list(command)
        if not argv:
            raise ValidationError("command must not be empty", field="command")
        if not all(isinstance(arg, str) for arg in argv):
            raise ValidationError(
                "all command arguments must be strings", field="command"
            )
        total = sum(len(arg) for arg in argv)
        if total > ARG_MAX_BYTES:
            raise ValidationError(
                f"command arguments exceed {ARG_MAX_BYTES} bytes (ARG_MAX); "
                f"got {total}",
                field="command",
            )
        params = [("command", arg) for arg in argv]
        yield from self._stream_text(
            "POST",
            self._path(workload_uid, "exec"),
            params=params,
            timeout=(10, timeout),
            error_label="exec endpoint",
        )

    def exec_stream(
        self,
        workload_uid: str,
        command: str,
        *,
        timeout: Optional[float] = None,
    ) -> Iterator[str]:
        """Run a shell command and stream its merged stdout/stderr lines.

        No exit code is returned; use :meth:`exec` if you need the exit code.
        """
        if not isinstance(command, str) or not command.strip():
            raise ValidationError("command must be a non-empty string", field="command")
        yield from self._exec_raw(workload_uid, ["sh", "-c", command], timeout=timeout)

    def exec(
        self,
        workload_uid: str,
        command: str,
        *,
        timeout: Optional[float] = None,
    ) -> ExecResponse:
        """Run a shell command inside the workload and capture the result.

        The command is wrapped in ``sh -c`` with a sentinel that echoes the
        exit status, since the exec endpoint merges stdout/stderr and does not
        surface an exit code on its own. Requires a shell (``sh``) in the
        image.
        """
        if not isinstance(command, str) or not command.strip():
            raise ValidationError("command must be a non-empty string", field="command")
        sentinel = f"__TARGON_EXIT_{uuid.uuid4().hex}__"
        wrapped = f"{command}\nprintf '\\n%s:%s' '{sentinel}' \"$?\""
        chunks = list(
            self._exec_raw(workload_uid, ["sh", "-c", wrapped], timeout=timeout)
        )
        output = "\n".join(chunks)

        exit_code = 0
        marker = f"{sentinel}:"
        idx = output.rfind(marker)
        if idx != -1:
            result = output[:idx].rstrip("\n")
            tail = output[idx + len(marker) :].strip()
            try:
                exit_code = int(tail)
            except ValueError:
                exit_code = 0
        else:
            result = output
        return ExecResponse(exit_code=exit_code, result=result)

    def verify(self, uid: str, digest: str) -> bool:
        uid = _validate_non_empty(uid, "uid")
        digest = _validate_non_empty(digest, "digest")
        result = self._post(
            self._path("verify"),
            json={"uid": uid, "digest": digest},
        )
        data = _require_dict(result, source="verify", object_type="VerifyResponse")
        return bool(data.get("verified", False))

    def vm_images(self) -> List[VmImage]:
        result = self._get(self._path("vm-images"))
        if not isinstance(result, list):
            raise HydrationError(
                f"Expected list for VM images, got {type(result).__name__}",
                object_type="VmImage",
            )
        return [VmImage.from_dict(item) for item in result if isinstance(item, dict)]

    def attach_volume(
        self,
        workload_uid: str,
        volume_uid: str,
        mount_path: str,
        read_only: bool = False,
    ) -> VolumeMountResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        mount_path = _validate_absolute_path(mount_path, "mount_path")
        result = self._put(
            self._path(workload_uid, "volumes", volume_uid),
            json={"mount_path": mount_path, "read_only": read_only},
        )
        return VolumeMountResponse.from_dict(result)

    def detach_volume(self, workload_uid: str, volume_uid: str) -> None:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        self._delete(self._path(workload_uid, "volumes", volume_uid))

    def attach_ssh_key(
        self, workload_uid: str, ssh_key_uid: str
    ) -> SshKeyAttachResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        ssh_key_uid = _validate_non_empty(ssh_key_uid, "ssh_key_uid")
        result = self._put(
            self._path(workload_uid, "ssh-keys", ssh_key_uid),
        )
        return SshKeyAttachResponse.from_dict(result)

    def detach_ssh_key(self, workload_uid: str, ssh_key_uid: str) -> None:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        ssh_key_uid = _validate_non_empty(ssh_key_uid, "ssh_key_uid")
        self._delete(self._path(workload_uid, "ssh-keys", ssh_key_uid))
