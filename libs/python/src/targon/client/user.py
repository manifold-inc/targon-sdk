from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import (
    DEFAULT_BASE_URL,
    USER_API_KEY_DETAIL_ENDPOINT,
    USER_API_KEY_ROTATE_ENDPOINT,
    USER_API_KEYS_ENDPOINT,
    USER_CREDITS_ENDPOINT,
    USER_WALLET_ENDPOINT,
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
class Wallet:
    address: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Wallet:
        data = _require_dict(data, source="wallet", object_type="Wallet")
        return cls(address=data.get("address", ""))


@dataclass(slots=True)
class Credits:
    credits: int = 0
    currency: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Credits:
        data = _require_dict(data, source="credits", object_type="Credits")
        return cls(
            credits=data.get("credits", 0),
            currency=data.get("currency", ""),
        )


@dataclass(slots=True)
class ApiKey:
    uid: str
    name: str = ""
    key_raw: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApiKey:
        data = _require_dict(data, source="api key", object_type="ApiKey")
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in API key response", object_type="ApiKey"
            )
        return cls(
            uid=uid,
            name=data.get("name", ""),
            key_raw=data.get("key_raw", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(slots=True)
class ApiKeyListResponse:
    items: List[ApiKey] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApiKeyListResponse:
        data = _require_dict(
            data, source="api key list", object_type="ApiKeyListResponse"
        )
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for API key items, got {type(items_raw).__name__}",
                object_type="ApiKeyListResponse",
            )
        return cls(
            items=[
                ApiKey.from_dict(item) for item in items_raw if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


class UserClient(BaseHTTPClient):
    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    def get_wallet(self) -> Wallet:
        result = self._get(USER_WALLET_ENDPOINT)
        return Wallet.from_dict(result)

    def get_credits(self) -> Credits:
        result = self._get(USER_CREDITS_ENDPOINT)
        return Credits.from_dict(result)

    def list_api_keys(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> ApiKeyListResponse:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(USER_API_KEYS_ENDPOINT, params=params or None)
        return ApiKeyListResponse.from_dict(result)

    def create_api_key(self, name: str) -> ApiKey:
        name = _validate_non_empty(name, "name")
        result = self._post(
            USER_API_KEYS_ENDPOINT,
            json={"name": name},
        )
        return ApiKey.from_dict(result)

    def update_api_key(self, key_uid: str, name: str) -> ApiKey:
        key_uid = _validate_non_empty(key_uid, "key_uid")
        name = _validate_non_empty(name, "name")
        result = self._patch(
            USER_API_KEY_DETAIL_ENDPOINT.format(key_uid=key_uid),
            json={"name": name},
        )
        return ApiKey.from_dict(result)

    def delete_api_key(self, key_uid: str) -> None:
        key_uid = _validate_non_empty(key_uid, "key_uid")
        self._delete(USER_API_KEY_DETAIL_ENDPOINT.format(key_uid=key_uid))

    def rotate_api_key(self, key_uid: str) -> ApiKey:
        key_uid = _validate_non_empty(key_uid, "key_uid")
        result = self._post(USER_API_KEY_ROTATE_ENDPOINT.format(key_uid=key_uid))
        return ApiKey.from_dict(result)
