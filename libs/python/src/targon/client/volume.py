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
class VolumeState:
    status: str = ""
    message: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> VolumeState:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            status=data.get("status", ""),
            message=data.get("message", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class Volume:
    uid: str
    name: str = ""
    size: int = 0
    resource_name: str = ""
    state: Optional[VolumeState] = None
    cost_per_hour: Optional[float] = None
    mount_path: Optional[str] = None
    workload_uid: Optional[str] = None
    pvc_name: Optional[str] = None
    last_backup_at: Optional[str] = None
    deleted_by: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Volume:
        data = _require_dict(data, source="volume", object_type="Volume")
        uid = data.get("uid")
        if not uid:
            raise HydrationError("Missing uid in volume response", object_type="Volume")
        return cls(
            uid=uid,
            name=data.get("name", ""),
            size=data.get("size", 0),
            resource_name=data.get("resource_name", ""),
            state=VolumeState.from_dict(data.get("state")),
            cost_per_hour=data.get("cost_per_hour"),
            mount_path=data.get("mount_path"),
            workload_uid=data.get("workload_uid"),
            pvc_name=data.get("pvc_name"),
            last_backup_at=data.get("last_backup_at"),
            deleted_by=data.get("deleted_by"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class VolumeListResponse:
    items: List[Volume] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeListResponse:
        data = _require_dict(
            data, source="volume list", object_type="VolumeListResponse"
        )
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for volume items, got {type(items_raw).__name__}",
                object_type="VolumeListResponse",
            )
        return cls(
            items=[
                Volume.from_dict(item) for item in items_raw if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class VolumeStateResponse:
    uid: str
    status: str = ""
    message: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeStateResponse:
        data = _require_dict(
            data, source="volume state", object_type="VolumeStateResponse"
        )
        return cls(
            uid=data.get("uid", ""),
            status=data.get("status", ""),
            message=data.get("message", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class VolumeEvent:
    volume_uid: str = ""
    event_type: str = ""
    billing_processed_at: Optional[str] = None
    billing_status: Optional[str] = None
    cost_per_second: Optional[int] = None
    k8s_resource_version: Optional[str] = None
    namespace: Optional[str] = None
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    reason: Optional[str] = None
    resource_name: Optional[str] = None
    pvc_name: Optional[str] = None
    requested_size: Optional[str] = None
    storage_class: Optional[str] = None
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeEvent:
        data = _require_dict(data, source="volume event", object_type="VolumeEvent")
        return cls(
            volume_uid=data.get("volume_uid", ""),
            event_type=data.get("event_type", ""),
            billing_processed_at=data.get("billing_processed_at"),
            billing_status=data.get("billing_status"),
            cost_per_second=data.get("cost_per_second"),
            k8s_resource_version=data.get("k8s_resource_version"),
            namespace=data.get("namespace"),
            old_status=data.get("old_status"),
            new_status=data.get("new_status"),
            reason=data.get("reason"),
            resource_name=data.get("resource_name"),
            pvc_name=data.get("pvc_name"),
            requested_size=data.get("requested_size"),
            storage_class=data.get("storage_class"),
            created_at=data.get("created_at", ""),
        )


@dataclass
class VolumeEventsResponse:
    items: List[VolumeEvent] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeEventsResponse:
        data = _require_dict(
            data, source="volume events", object_type="VolumeEventsResponse"
        )
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for volume event items, got {type(items_raw).__name__}",
                object_type="VolumeEventsResponse",
            )
        return cls(
            items=[VolumeEvent.from_dict(item) for item in items_raw],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class VolumeOperationResponse:
    uid: str
    state: Optional[VolumeState] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeOperationResponse:
        data = _require_dict(
            data, source="volume operation", object_type="VolumeOperationResponse"
        )
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in volume create response",
                object_type="VolumeOperationResponse",
            )
        return cls(
            uid=uid,
            state=VolumeState.from_dict(data.get("state")),
        )


VolumeCreateResponse = VolumeOperationResponse


class VolumeClient(BaseHTTPClient):
    def _path(
        self, volume_uid: Optional[str] = None, suffix: Optional[str] = None
    ) -> str:
        path = org_path(self.client.require_org(), "volumes")
        if volume_uid is not None:
            path = f"{path}/{volume_uid}"
        if suffix is not None:
            path = f"{path}/{suffix}"
        return path

    def create(
        self, name: str, size_in_mb: int, resource_name: str
    ) -> VolumeOperationResponse:
        name = _validate_non_empty(name, "name")
        resource_name = _validate_non_empty(resource_name, "resource_name")
        result = self._post(
            self._path(),
            json={
                "name": name,
                "size_in_mb": size_in_mb,
                "resource_name": resource_name,
            },
        )
        return VolumeOperationResponse.from_dict(result)

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        workload_uid: Optional[str] = None,
    ) -> VolumeListResponse:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        if workload_uid is not None:
            params["workload_uid"] = workload_uid
        result = self._get(self._path(), params=params or None)
        return VolumeListResponse.from_dict(result)

    def get(self, volume_uid: str) -> Volume:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        result = self._get(self._path(volume_uid))
        return Volume.from_dict(result)

    def get_state(self, volume_uid: str) -> VolumeStateResponse:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        result = self._get(self._path(volume_uid, "state"))
        return VolumeStateResponse.from_dict(result)

    def get_events(
        self,
        volume_uid: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> VolumeEventsResponse:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(
            self._path(volume_uid, "events"),
            params=params or None,
        )
        return VolumeEventsResponse.from_dict(result)

    def update(self, volume_uid: str, name: str) -> Volume:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        name = _validate_non_empty(name, "name")
        result = self._patch(
            self._path(volume_uid),
            json={"name": name},
        )
        return Volume.from_dict(result)

    def delete(self, volume_uid: str) -> None:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        self._delete(self._path(volume_uid))
