from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.client.constants import (
    DEFAULT_BASE_URL,
    PROJECT_DETAIL_ENDPOINT,
    PROJECTS_ENDPOINT,
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
class Project:
    uid: str
    name: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Project:
        data = _require_dict(data, source="project", object_type="Project")
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in project response", object_type="Project"
            )
        return cls(
            uid=uid,
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(slots=True)
class ProjectListResponse:
    items: List[Project] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProjectListResponse:
        data = _require_dict(
            data, source="project list", object_type="ProjectListResponse"
        )
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for project items, got {type(items_raw).__name__}",
                object_type="ProjectListResponse",
            )
        return cls(
            items=[
                Project.from_dict(item) for item in items_raw if isinstance(item, dict)
            ],
            next_cursor=data.get("next_cursor"),
        )


class ProjectClient(BaseHTTPClient):
    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    def create(self, name: str) -> Project:
        name = _validate_non_empty(name, "name")
        result = self._post(
            PROJECTS_ENDPOINT,
            json={"name": name},
        )
        return Project.from_dict(result)

    def list(
        self, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> ProjectListResponse:
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = self._get(PROJECTS_ENDPOINT, params=params or None)
        return ProjectListResponse.from_dict(result)

    def get(self, project_uid: str) -> Project:
        project_uid = _validate_non_empty(project_uid, "project_uid")
        result = self._get(PROJECT_DETAIL_ENDPOINT.format(project_uid=project_uid))
        return Project.from_dict(result)

    def update(self, project_uid: str, name: str) -> Project:
        project_uid = _validate_non_empty(project_uid, "project_uid")
        name = _validate_non_empty(name, "name")
        result = self._patch(
            PROJECT_DETAIL_ENDPOINT.format(project_uid=project_uid),
            json={"name": name},
        )
        return Project.from_dict(result)

    def delete(self, project_uid: str) -> None:
        project_uid = _validate_non_empty(project_uid, "project_uid")
        self._delete(PROJECT_DETAIL_ENDPOINT.format(project_uid=project_uid))
