from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import asyncio

from targon.core.objects import AsyncBaseHTTPClient
from targon.core.exceptions import TargonError, ValidationError
from targon.client.constants import (
    DEFAULT_BASE_URL,
    CREATE_APP_ENDPOINT,
    GET_APP_ENDPOINT,
    LIST_APPS_ENDPOINT,
    DELETE_APP_ENDPOINT,
)


@dataclass(slots=True)
class AppRequest:
    Name: str
    ProjectName: str


@dataclass(slots=True)
class AppResponse:
    uid: str
    project_id: Optional[str]
    name: str
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class FunctionStatus:
    name: str
    uid: str
    url: str
    status: str


@dataclass(slots=True)
class FunctionResponse:
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
    uid: str
    name: str
    project_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ListAppsResponse:
    apps: List[AppListItem]
    total: int


@dataclass(slots=True)
class WorkloadState:
    status: str
    message: str
    ready_replicas: int
    total_replicas: int


@dataclass(slots=True)
class WorkloadResource:
    name: str
    display_name: str
    gpu_type: Optional[str]
    gpu_count: Optional[int]
    vcpu: int
    memory: int


@dataclass(slots=True)
class FunctionItem:
    uid: str
    name: str
    image: str
    cost_per_hour: Optional[float]
    resource: Optional[WorkloadResource]
    state: Optional[WorkloadState]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ListFunctionsResponse:
    app_id: str
    app_name: str
    functions: List[FunctionItem]
    total: int


@dataclass(slots=True)
class ServerlessConfig:
    module: Optional[str]
    qualname: Optional[str]
    definition_type: Optional[str]
    min_replicas: Optional[int]
    max_replicas: Optional[int]
    initial_replicas: Optional[int]
    container_concurrency: Optional[int]
    target_concurrency: Optional[int]
    timeout_seconds: Optional[int]
    startup_timeout: Optional[int]
    webhook_config: Optional[Dict[str, Any]]


@dataclass(slots=True)
class FunctionDetailResponse:
    uid: str
    app_id: Optional[str]
    name: str
    image: str
    resource_name: str
    cost_per_hour: Optional[float]
    resource: Optional[WorkloadResource]
    serverless_config: Optional[ServerlessConfig]
    state: Optional[WorkloadState]
    created_at: str
    updated_at: str


class AsyncAppClient(AsyncBaseHTTPClient):
    """Async client for app registration and management."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL


    async def list_apps(self) -> ListAppsResponse:
        """List all apps."""
        result = await self._async_get(LIST_APPS_ENDPOINT)

        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        apps_data = result.get("items", [])
        if not isinstance(apps_data, list):
            raise TargonError(
                f"Expected apps items to be list, got {type(apps_data).__name__}"
            )

        apps = [
            AppListItem(
                uid=app.get("uid", ""),
                name=app.get("name", ""),
                project_id=app.get("project_id"),
                created_at=app.get("created_at", ""),
                updated_at=app.get("updated_at", ""),
            )
            for app in apps_data
            if isinstance(app, dict)
        ]

        return ListAppsResponse(
            apps=apps,
            total=len(apps),
        )
   
    async def create_app(self, name: str, project_id: Optional[str] = None) -> AppResponse:
        if not name or not name.strip():
            raise ValidationError("App name cannot be empty", field="name")

        if project_id is not None:
            project_id = project_id.strip() or None

        payload: Dict[str, str] = {"name": name}
        if project_id is not None:
            payload["project_id"] = project_id

        result = await self._async_post(CREATE_APP_ENDPOINT, json=payload)
        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        return AppResponse(
            uid=result.get("uid", result.get("app_id", "")),
            project_id=result.get("project_id"),
            name=result.get("name", name),
            created_at=result.get("created_at", ""),
            updated_at=result.get("updated_at", ""),
        )

    async def get_app(self, app_uid: Optional[str] = None, name: Optional[str] = None) -> AppResponse:
        resolved_app_uid = app_uid or name
        if not resolved_app_uid or not resolved_app_uid.strip():
            raise ValidationError("App UID cannot be empty", field="app_uid")

        endpoint = GET_APP_ENDPOINT.format(app_uid=resolved_app_uid)
        result = await self._async_get(endpoint)
        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        return AppResponse(
            uid=result.get("uid", resolved_app_uid),
            project_id=result.get("project_id"),
            name=result.get("name", ""),
            created_at=result.get("created_at", ""),
            updated_at=result.get("updated_at", ""),
        )

    async def list_functions(self, app_id: str) -> ListFunctionsResponse:
        if not app_id or not app_id.strip():
            raise ValidationError("App ID cannot be empty", field="app_id")

        app_result: AppResponse = await self.get_app(app_id)

        workloads_result = await self._async_get(
            "/tha/v2/workloads",
            params={"type": "function", "app_id": app_id},
        )
        if not isinstance(workloads_result, dict):
            raise TargonError(
                f"Unexpected workloads response format: {type(workloads_result).__name__}"
            )

        items = workloads_result.get("items", [])
        if not isinstance(items, list):
            raise TargonError(
                f"Expected workloads items to be list, got {type(items).__name__}"
            )

        functions: List[FunctionItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            state_data = item.get("state")
            state = (
                WorkloadState(
                    status=state_data.get("status", ""),
                    message=state_data.get("message", ""),
                    ready_replicas=state_data.get("ready_replicas", 0),
                    total_replicas=state_data.get("total_replicas", 0),
                )
                if isinstance(state_data, dict)
                else None
            )

            functions.append(
                FunctionItem(
                    uid=item.get("uid", ""),
                    name=item.get("name", ""),
                    image=item.get("image", ""),
                    cost_per_hour=None,
                    resource=None,
                    state=state,
                    created_at=item.get("created_at", ""),
                    updated_at=item.get("updated_at", ""),
                )
            )

        return ListFunctionsResponse(
            app_id=app_result.uid or app_id,
            app_name=app_result.name,
            functions=functions,
            total=len(functions),
        )

    async def get_function(self, workload_uid: str) -> FunctionDetailResponse:
        """Get detailed information about a specific function workload by its UID."""
        if not workload_uid or not workload_uid.strip():
            raise ValidationError("Workload UID cannot be empty", field="workload_uid")

        result = await self._async_get(f"/tha/v2/workloads/{workload_uid}")

        if not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        state_data = result.get("state")
        state = (
            WorkloadState(
                status=state_data.get("status", ""),
                message=state_data.get("message", ""),
                ready_replicas=state_data.get("ready_replicas", 0),
                total_replicas=state_data.get("total_replicas", 0),
            )
            if isinstance(state_data, dict)
            else None
        )

        resource_data = result.get("resource")
        resource = (
            WorkloadResource(
                name=resource_data.get("name", ""),
                display_name=resource_data.get("display_name", ""),
                gpu_type=resource_data.get("gpu_type"),
                gpu_count=resource_data.get("gpu_count"),
                vcpu=resource_data.get("vcpu", 0),
                memory=resource_data.get("memory", 0),
            )
            if isinstance(resource_data, dict)
            else None
        )

        sc_data = result.get("serverless_config")
        serverless_config = (
            ServerlessConfig(
                module=sc_data.get("module"),
                qualname=sc_data.get("qualname"),
                definition_type=sc_data.get("definition_type"),
                min_replicas=sc_data.get("min_replicas"),
                max_replicas=sc_data.get("max_replicas"),
                initial_replicas=sc_data.get("initial_replicas"),
                container_concurrency=sc_data.get("container_concurrency"),
                target_concurrency=sc_data.get("target_concurrency"),
                timeout_seconds=sc_data.get("timeout_seconds"),
                startup_timeout=sc_data.get("startup_timeout"),
                webhook_config=sc_data.get("webhook_config"),
            )
            if isinstance(sc_data, dict)
            else None
        )

        return FunctionDetailResponse(
            uid=result.get("uid", workload_uid),
            app_id=result.get("app_id"),
            name=result.get("name", ""),
            image=result.get("image", ""),
            resource_name=result.get("resource_name", ""),
            cost_per_hour=result.get("cost_per_hour"),
            resource=resource,
            serverless_config=serverless_config,
            state=state,
            created_at=result.get("created_at", ""),
            updated_at=result.get("updated_at", ""),
        )
        
    # @TODO
    async def delete_app(self, app_id: str) -> Dict[str, Any]:
        """Delete an app and all corresponding deployments."""
        if not app_id or not app_id.strip():
            raise ValidationError("App ID cannot be empty", field="app_id")

        endpoint = DELETE_APP_ENDPOINT.format(app_uid=app_id)
        result = await self._async_delete(endpoint)

        if result and not isinstance(result, dict):
            raise TargonError(f"Unexpected response format: {type(result).__name__}")

        return result or {}

    async def list_apps_with_details(
        self,
    ) -> Tuple[ListAppsResponse, List[Optional[ListFunctionsResponse]]]:
        apps_response = await self.list_apps()

        if not apps_response.apps:
            return apps_response, []

        tasks = [self.list_functions(app.uid) for app in apps_response.apps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        detailed_apps = []
        for res in results:
            if isinstance(res, Exception):
                detailed_apps.append(None)
            else:
                detailed_apps.append(res)

        return apps_response, detailed_apps
