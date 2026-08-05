from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import (
    ORG_CREDITS_ENDPOINT,
    ORG_DETAIL_ENDPOINT,
    ORG_WALLET_ENDPOINT,
    ORGS_ENDPOINT,
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


@dataclass
class Org:
    uid: str
    slug: str
    name: str = ""
    org_type: str = ""
    role: str = ""
    billing_email: str = ""
    credits: float = 0.0
    overage: int = 0
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Org:
        data = _require_dict(data, source="organization", object_type="Org")
        uid = data.get("uid")
        slug = data.get("slug")
        if not uid or not slug:
            raise HydrationError(
                "Missing uid or slug in organization response",
                object_type="Org",
            )
        return cls(
            uid=uid,
            slug=slug,
            name=data.get("name", ""),
            org_type=data.get("org_type", ""),
            role=data.get("role", ""),
            billing_email=data.get("billing_email", ""),
            credits=data.get("credits", 0.0),
            overage=data.get("overage", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class OrgListResponse:
    items: List[Org] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrgListResponse:
        data = _require_dict(
            data,
            source="organization list",
            object_type="OrgListResponse",
        )
        items = data.get("items", [])
        if not isinstance(items, list):
            raise HydrationError(
                "Expected list for organization items",
                object_type="OrgListResponse",
            )
        return cls(
            items=[Org.from_dict(item) for item in items if isinstance(item, dict)],
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class Wallet:
    address: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Wallet:
        data = _require_dict(data, source="wallet", object_type="Wallet")
        return cls(address=data.get("address", ""))


@dataclass
class Credits:
    credits: float = 0.0
    currency: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Credits:
        data = _require_dict(data, source="credits", object_type="Credits")
        return cls(
            credits=data.get("credits", 0.0),
            currency=data.get("currency", ""),
        )


class OrgClient(BaseHTTPClient):
    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> OrgListResponse:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(ORGS_ENDPOINT, params=params or None)
        return OrgListResponse.from_dict(result)

    def create(self, *, name: str, slug: str) -> Org:
        result = self._post(
            ORGS_ENDPOINT,
            json={
                "name": _validate_non_empty(name, "name"),
                "slug": _validate_non_empty(slug, "slug"),
            },
        )
        return Org.from_dict(result)

    def get(self, slug: str) -> Org:
        slug = _validate_non_empty(slug, "slug")
        return Org.from_dict(self._get(ORG_DETAIL_ENDPOINT.format(org_slug=slug)))

    def update(
        self,
        slug: str,
        *,
        name: Optional[str] = None,
        new_slug: Optional[str] = None,
        billing_email: Optional[str] = None,
    ) -> Org:
        slug = _validate_non_empty(slug, "slug")
        payload: Dict[str, str] = {}
        if name is not None:
            payload["name"] = _validate_non_empty(name, "name")
        if new_slug is not None:
            payload["slug"] = _validate_non_empty(new_slug, "new_slug")
        if billing_email is not None:
            payload["billing_email"] = billing_email
        if not payload:
            raise ValidationError("At least one organization field is required")
        result = self._patch(
            ORG_DETAIL_ENDPOINT.format(org_slug=slug),
            json=payload,
        )
        return Org.from_dict(result)

    def delete(self, slug: str) -> None:
        slug = _validate_non_empty(slug, "slug")
        self._delete(ORG_DETAIL_ENDPOINT.format(org_slug=slug))


class WalletClient(BaseHTTPClient):
    def get(self) -> Wallet:
        org = self.client.require_org()
        result = self._get(ORG_WALLET_ENDPOINT.format(org_slug=org))
        return Wallet.from_dict(result)


class CreditsClient(BaseHTTPClient):
    def get(self) -> Credits:
        org = self.client.require_org()
        result = self._get(ORG_CREDITS_ENDPOINT.format(org_slug=org))
        return Credits.from_dict(result)
