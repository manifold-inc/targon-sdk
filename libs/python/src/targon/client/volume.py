from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import (
    DEFAULT_BASE_URL,
    VOLUME_DELETE_DEPLOYMENT_ENDPOINT,
    VOLUME_DETAIL_ENDPOINT,
    VOLUME_EVENTS_ENDPOINT,
    VOLUME_STATE_ENDPOINT,
    VOLUMES_ENDPOINT,
)
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


@dataclass(slots=True)
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


@dataclass(slots=True)
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
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(slots=True)
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


@dataclass(slots=True)
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


@dataclass(slots=True)
class VolumeEvent:
    volume_uid: str = ""
    event_type: str = ""
    old_status: str = ""
    new_status: str = ""
    reason: str = ""
    resource_name: str = ""
    pvc_name: str = ""
    requested_size: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeEvent:
        data = _require_dict(data, source="volume event", object_type="VolumeEvent")
        return cls(
            volume_uid=data.get("volume_uid", ""),
            event_type=data.get("event_type", ""),
            old_status=data.get("old_status", ""),
            new_status=data.get("new_status", ""),
            reason=data.get("reason", ""),
            resource_name=data.get("resource_name", ""),
            pvc_name=data.get("pvc_name", ""),
            requested_size=data.get("requested_size", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass(slots=True)
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


@dataclass(slots=True)
class VolumeCreateResponse:
    uid: str
    state: Optional[VolumeState] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeCreateResponse:
        data = _require_dict(
            data, source="volume create", object_type="VolumeCreateResponse"
        )
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in volume create response",
                object_type="VolumeCreateResponse",
            )
        return cls(
            uid=uid,
            state=VolumeState.from_dict(data.get("state")),
        )


@dataclass(slots=True)
class VolumeDeleteDeploymentResponse:
    uid: str
    state: Optional[VolumeState] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VolumeDeleteDeploymentResponse:
        data = _require_dict(
            data,
            source="volume delete deployment",
            object_type="VolumeDeleteDeploymentResponse",
        )
        return cls(
            uid=data.get("uid", ""),
            state=VolumeState.from_dict(data.get("state")),
        )


class VolumeClient(BaseHTTPClient):
    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    def create(
        self, name: str, size_in_mb: int, resource_name: str
    ) -> VolumeCreateResponse:
        name = _validate_non_empty(name, "name")
        resource_name = _validate_non_empty(resource_name, "resource_name")
        result = self._post(
            VOLUMES_ENDPOINT,
            json={
                "name": name,
                "size_in_mb": size_in_mb,
                "resource_name": resource_name,
            },
        )
        return VolumeCreateResponse.from_dict(result)

    def list(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> VolumeListResponse:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(VOLUMES_ENDPOINT, params=params or None)
        return VolumeListResponse.from_dict(result)

    def get(self, volume_uid: str) -> Volume:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        result = self._get(VOLUME_DETAIL_ENDPOINT.format(volume_uid=volume_uid))
        return Volume.from_dict(result)

    def get_state(self, volume_uid: str) -> VolumeStateResponse:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        result = self._get(VOLUME_STATE_ENDPOINT.format(volume_uid=volume_uid))
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
            VOLUME_EVENTS_ENDPOINT.format(volume_uid=volume_uid),
            params=params or None,
        )
        return VolumeEventsResponse.from_dict(result)

    def update(self, volume_uid: str, name: str) -> Volume:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        name = _validate_non_empty(name, "name")
        result = self._patch(
            VOLUME_DETAIL_ENDPOINT.format(volume_uid=volume_uid),
            json={"name": name},
        )
        return Volume.from_dict(result)

    def delete(self, volume_uid: str) -> None:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        self._delete(VOLUME_DETAIL_ENDPOINT.format(volume_uid=volume_uid))

    def delete_deployment(self, volume_uid: str) -> VolumeDeleteDeploymentResponse:
        volume_uid = _validate_non_empty(volume_uid, "volume_uid")
        result = self._post(
            VOLUME_DELETE_DEPLOYMENT_ENDPOINT.format(volume_uid=volume_uid)
        )
        return VolumeDeleteDeploymentResponse.from_dict(result)
