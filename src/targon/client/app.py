from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from targon.core.objects import AsyncBaseHTTPClient
from targon.core.exceptions import TargonError, ValidationError
from targon.client.constants import (
    DEFAULT_BASE_URL,
    GET_APP_ENDPOINT,
    GET_APP_STATUS_ENDPOINT,
    LIST_APPS_ENDPOINT,
    DELETE_APP_ENDPOINT,
    LIST_FUNCTIONS_ENDPOINT,
)


@dataclass(slots=True)
class AppRequest:
    Name: str
    ProjectName: str


@dataclass(slots=True)
class AppResponse:
    AppID: str
    Name: str
    ProjectID: str
    ProjectName: str
    Created: bool
    WebURL: str


@dataclass(slots=True)
class FunctionStatus:
    uid: str
    url: str
    status: str


@dataclass(slots=True)
class AppStatusResponse:
    app_id: str
    name: str
    project_id: str
    project_name: str
    function_count: int
    functions: Dict[str, FunctionStatus]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class AppListItem:
    app_id: str
    name: str
    project_id: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ListAppsResponse:
    apps: List[AppListItem]
    total: int


@dataclass(slots=True)
class FunctionItem:
    uid: str
    name: str
    module: Optional[str]
    qualname: Optional[str]
    image_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ListFunctionsResponse:
    app_id: str
    functions: List[FunctionItem]
    total: int


class AsyncAppClient(AsyncBaseHTTPClient):
    """Async client for app registration and management."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    async def get_app(self, name: str, project_name: str = "default") -> AppResponse:
        if not name or not name.strip():
            raise ValidationError("App name cannot be empty", field="name")

        if not project_name or not project_name.strip():
            raise ValidationError("Project name cannot be empty", field="project_name")

        payload: Dict[str, str] = {"name": name, "project_name": project_name}

        result = await self._async_post(GET_APP_ENDPOINT, json=payload)
        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        return AppResponse(
            AppID=result.get("app_id", ""),
            Name=result.get("name", name),
            ProjectID=result.get("project_id", ""),
            ProjectName=result.get("project_name", project_name),
            Created=result.get("created", False),
            WebURL=result.get("web_url", ""),
        )

    async def get_app_status(self, app_id: str) -> AppStatusResponse:
        if not app_id or not app_id.strip():
            raise ValidationError("App ID cannot be empty", field="app_id")

        endpoint = GET_APP_STATUS_ENDPOINT.format(app_id=app_id)
        result = await self._async_get(endpoint)
        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        functions_data = result.get("functions", {})
        if not isinstance(functions_data, dict):
            raise TargonError(
                f"Expected functions to be dict, got {type(functions_data).__name__}"
            )

        functions = {
            name: FunctionStatus(
                uid=func.get("uid", ""),
                url=func.get("url", ""),
                status=func.get("status", ""),
            )
            for name, func in functions_data.items()
        }

        return AppStatusResponse(
            app_id=result.get("app_id", app_id),
            name=result.get("name", ""),
            project_id=result.get("project_id", ""),
            project_name=result.get("project_name", ""),
            function_count=result.get("function_count", 0),
            functions=functions,
            created_at=result.get("created_at", ""),
            updated_at=result.get("updated_at", ""),
        )

    async def list_apps(self) -> ListAppsResponse:
        """List all apps that are currently deployed/running or recently stopped."""
        result = await self._async_get(LIST_APPS_ENDPOINT)

        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        apps_data = result.get("apps", [])
        if not isinstance(apps_data, list):
            raise TargonError(
                f"Expected apps to be list, got {type(apps_data).__name__}"
            )

        apps = [
            AppListItem(
                app_id=app.get("app_id", ""),
                name=app.get("name", ""),
                project_id=app.get("project_id", ""),
                created_at=app.get("created_at", ""),
                updated_at=app.get("updated_at", ""),
            )
            for app in apps_data
        ]

        return ListAppsResponse(
            apps=apps,
            total=result.get("total", len(apps)),
        )

    async def delete_app(self, app_id: str) -> Dict[str, Any]:
        """Delete an app and all corresponding deployments."""
        if not app_id or not app_id.strip():
            raise ValidationError("App ID cannot be empty", field="app_id")

        endpoint = DELETE_APP_ENDPOINT.format(app_id=app_id)
        result = await self._async_delete(endpoint)

        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        return result

    async def list_functions(self, app_id: str) -> ListFunctionsResponse:
        """List all functions for a given app."""
        if not app_id or not app_id.strip():
            raise ValidationError("App ID cannot be empty", field="app_id")

        endpoint = LIST_FUNCTIONS_ENDPOINT.format(app_id=app_id)
        result = await self._async_get(endpoint)

        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        functions_data = result.get("functions", [])
        if not isinstance(functions_data, list):
            raise TargonError(
                f"Expected functions to be list, got {type(functions_data).__name__}"
            )

        functions = [
            FunctionItem(
                uid=func.get("uid", ""),
                name=func.get("name", ""),
                module=func.get("module"),
                qualname=func.get("qualname"),
                image_id=func.get("image_id"),
                created_at=func.get("created_at", ""),
                updated_at=func.get("updated_at", ""),
            )
            for func in functions_data
        ]

        return ListFunctionsResponse(
            app_id=result.get("app_id", app_id),
            functions=functions,
            total=result.get("total", len(functions)),
        )
