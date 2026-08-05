from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import MEMBER_DETAIL_ENDPOINT, MEMBERS_ENDPOINT
from targon.core.exceptions import HydrationError, ValidationError
from targon.core.objects import BaseHTTPClient

VALID_ROLES = {"OWNER", "ADMIN", "MEMBER"}
VALID_STATUSES = {"ACTIVE", "INVITED", "SUSPENDED"}


def _validate_non_empty(value: Optional[str], field_name: str) -> str:
    if not value or not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{field_name} must be a non-empty string",
            field=field_name,
            value=value,
        )
    return value.strip()


def _enum_value(value: Optional[str], field_name: str, allowed: set) -> Optional[str]:
    if value is None:
        return None
    normalized = _validate_non_empty(value, field_name).upper()
    if normalized not in allowed:
        raise ValidationError(
            f"{field_name} must be one of {', '.join(sorted(allowed))}",
            field=field_name,
            value=value,
        )
    return normalized


@dataclass
class MemberUser:
    username: str = ""
    email: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemberUser:
        return cls(
            username=data.get("username", ""),
            email=data.get("email", ""),
        )


@dataclass
class Member:
    uid: str
    user: MemberUser
    role: str = ""
    status: str = ""
    invited_by_user: Optional[MemberUser] = None
    joined_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Member:
        if not isinstance(data, dict):
            raise HydrationError(
                "Expected dict for member response",
                object_type="Member",
            )
        uid = data.get("uid")
        user = data.get("user")
        if not uid or not isinstance(user, dict):
            raise HydrationError(
                "Missing uid or user in member response",
                object_type="Member",
            )
        invited_by = data.get("invited_by_user")
        return cls(
            uid=uid,
            user=MemberUser.from_dict(user),
            role=data.get("role", ""),
            status=data.get("status", ""),
            invited_by_user=(
                MemberUser.from_dict(invited_by)
                if isinstance(invited_by, dict)
                else None
            ),
            joined_at=data.get("joined_at"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class MemberListResponse:
    items: List[Member] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemberListResponse:
        if not isinstance(data, dict):
            raise HydrationError(
                "Expected dict for member list",
                object_type="MemberListResponse",
            )
        items = data.get("items", [])
        if not isinstance(items, list):
            raise HydrationError(
                "Expected list for member items",
                object_type="MemberListResponse",
            )
        return cls(
            items=[Member.from_dict(item) for item in items if isinstance(item, dict)],
            next_cursor=data.get("next_cursor"),
        )


class MemberClient(BaseHTTPClient):
    def list(
        self,
        *,
        role: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> MemberListResponse:
        org = self.client.require_org()
        params: Dict[str, Any] = {}
        role = _enum_value(role, "role", VALID_ROLES)
        status = _enum_value(status, "status", VALID_STATUSES)
        if role:
            params["role"] = role
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(
            MEMBERS_ENDPOINT.format(org_slug=org),
            params=params or None,
        )
        return MemberListResponse.from_dict(result)

    def get(self, username: str) -> Member:
        org = self.client.require_org()
        username = _validate_non_empty(username, "username")
        result = self._get(
            MEMBER_DETAIL_ENDPOINT.format(org_slug=org, username=username)
        )
        return Member.from_dict(result)

    def update(self, username: str, *, role: str) -> Member:
        org = self.client.require_org()
        username = _validate_non_empty(username, "username")
        normalized_role = _enum_value(role, "role", VALID_ROLES)
        result = self._patch(
            MEMBER_DETAIL_ENDPOINT.format(org_slug=org, username=username),
            json={"role": normalized_role},
        )
        return Member.from_dict(result)

    def delete(self, username: str) -> None:
        org = self.client.require_org()
        username = _validate_non_empty(username, "username")
        self._delete(MEMBER_DETAIL_ENDPOINT.format(org_slug=org, username=username))
