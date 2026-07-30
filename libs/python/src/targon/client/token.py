from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import (
    PERSONAL_API_TOKEN_DETAIL_ENDPOINT,
    PERSONAL_API_TOKENS_ENDPOINT,
    SERVICE_TOKEN_DETAIL_ENDPOINT,
    SERVICE_TOKENS_ENDPOINT,
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


@dataclass
class ApiToken:
    uid: str
    name: str = ""
    token: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApiToken:
        if not isinstance(data, dict) or not data.get("uid"):
            raise HydrationError(
                "Missing uid in API token response",
                object_type="ApiToken",
            )
        return cls(
            uid=data["uid"],
            name=data.get("name", ""),
            token=data.get("token"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class ApiTokenListResponse:
    items: List[ApiToken] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApiTokenListResponse:
        if not isinstance(data, dict):
            raise HydrationError(
                "Expected dict for API token list",
                object_type="ApiTokenListResponse",
            )
        items = data.get("items", [])
        if not isinstance(items, list):
            raise HydrationError(
                "Expected list for API token items",
                object_type="ApiTokenListResponse",
            )
        return cls(
            items=[
                ApiToken.from_dict(item) for item in items if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class TokenCreator:
    username: str = ""
    email: str = ""
    first_name: str = ""
    last_name: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenCreator:
        return cls(
            username=data.get("username", ""),
            email=data.get("email", ""),
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
        )


@dataclass
class ServiceToken:
    uid: str
    name: str = ""
    token: Optional[str] = None
    created_by: Optional[TokenCreator] = None
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ServiceToken:
        if not isinstance(data, dict) or not data.get("uid"):
            raise HydrationError(
                "Missing uid in service token response",
                object_type="ServiceToken",
            )
        creator = data.get("created_by")
        return cls(
            uid=data["uid"],
            name=data.get("name", ""),
            token=data.get("token"),
            created_by=(
                TokenCreator.from_dict(creator) if isinstance(creator, dict) else None
            ),
            created_at=data.get("created_at", ""),
        )


@dataclass
class ServiceTokenListResponse:
    items: List[ServiceToken] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ServiceTokenListResponse:
        if not isinstance(data, dict):
            raise HydrationError(
                "Expected dict for service token list",
                object_type="ServiceTokenListResponse",
            )
        items = data.get("items", [])
        if not isinstance(items, list):
            raise HydrationError(
                "Expected list for service token items",
                object_type="ServiceTokenListResponse",
            )
        return cls(
            items=[
                ServiceToken.from_dict(item) for item in items if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


class ApiTokenClient(BaseHTTPClient):
    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> ApiTokenListResponse:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(PERSONAL_API_TOKENS_ENDPOINT, params=params or None)
        return ApiTokenListResponse.from_dict(result)

    def create(self, name: str) -> ApiToken:
        result = self._post(
            PERSONAL_API_TOKENS_ENDPOINT,
            json={"name": _validate_non_empty(name, "name")},
        )
        return ApiToken.from_dict(result)

    def update(self, token_uid: str, *, name: str) -> ApiToken:
        token_uid = _validate_non_empty(token_uid, "token_uid")
        result = self._patch(
            PERSONAL_API_TOKEN_DETAIL_ENDPOINT.format(token_uid=token_uid),
            json={"name": _validate_non_empty(name, "name")},
        )
        return ApiToken.from_dict(result)

    def delete(self, token_uid: str) -> None:
        token_uid = _validate_non_empty(token_uid, "token_uid")
        self._delete(PERSONAL_API_TOKEN_DETAIL_ENDPOINT.format(token_uid=token_uid))


class ServiceTokenClient(BaseHTTPClient):
    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> ServiceTokenListResponse:
        org = self.client.require_org()
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(
            SERVICE_TOKENS_ENDPOINT.format(org_slug=org),
            params=params or None,
        )
        return ServiceTokenListResponse.from_dict(result)

    def create(self, name: str) -> ServiceToken:
        org = self.client.require_org()
        result = self._post(
            SERVICE_TOKENS_ENDPOINT.format(org_slug=org),
            json={"name": _validate_non_empty(name, "name")},
        )
        return ServiceToken.from_dict(result)

    def delete(self, token_uid: str) -> None:
        org = self.client.require_org()
        token_uid = _validate_non_empty(token_uid, "token_uid")
        self._delete(
            SERVICE_TOKEN_DETAIL_ENDPOINT.format(
                org_slug=org,
                token_uid=token_uid,
            )
        )
