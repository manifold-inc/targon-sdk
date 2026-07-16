from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import (
    DEFAULT_BASE_URL,
    SSH_KEY_DETAIL_ENDPOINT,
    SSH_KEYS_ENDPOINT,
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
class SshKey:
    uid: str
    name: str = ""
    public_key_raw: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SshKey:
        data = _require_dict(data, source="ssh key", object_type="SshKey")
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in SSH key response", object_type="SshKey"
            )
        return cls(
            uid=uid,
            name=data.get("name", ""),
            public_key_raw=data.get("public_key_raw", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(slots=True)
class SshKeyListResponse:
    items: List[SshKey] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SshKeyListResponse:
        data = _require_dict(
            data, source="ssh key list", object_type="SshKeyListResponse"
        )
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for SSH key items, got {type(items_raw).__name__}",
                object_type="SshKeyListResponse",
            )
        return cls(
            items=[
                SshKey.from_dict(item) for item in items_raw if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


class SshKeyClient(BaseHTTPClient):
    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    def create(self, name: str, ssh_key: str) -> SshKey:
        name = _validate_non_empty(name, "name")
        ssh_key = _validate_non_empty(ssh_key, "ssh_key")
        result = self._post(
            SSH_KEYS_ENDPOINT,
            json={"name": name, "ssh_key": ssh_key},
        )
        return SshKey.from_dict(result)

    def list(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> SshKeyListResponse:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(SSH_KEYS_ENDPOINT, params=params or None)
        return SshKeyListResponse.from_dict(result)

    def get(self, ssh_key_uid: str) -> SshKey:
        ssh_key_uid = _validate_non_empty(ssh_key_uid, "ssh_key_uid")
        result = self._get(SSH_KEY_DETAIL_ENDPOINT.format(ssh_key_uid=ssh_key_uid))
        return SshKey.from_dict(result)

    def update(self, ssh_key_uid: str, name: str) -> SshKey:
        ssh_key_uid = _validate_non_empty(ssh_key_uid, "ssh_key_uid")
        name = _validate_non_empty(name, "name")
        result = self._patch(
            SSH_KEY_DETAIL_ENDPOINT.format(ssh_key_uid=ssh_key_uid),
            json={"name": name},
        )
        return SshKey.from_dict(result)

    def delete(self, ssh_key_uid: str) -> None:
        ssh_key_uid = _validate_non_empty(ssh_key_uid, "ssh_key_uid")
        self._delete(SSH_KEY_DETAIL_ENDPOINT.format(ssh_key_uid=ssh_key_uid))
