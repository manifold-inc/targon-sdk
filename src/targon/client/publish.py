import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from targon.core.objects import AsyncBaseHTTPClient
from targon.core.exceptions import ValidationError, HydrationError
from targon.client.constants import (
    DEFAULT_BASE_URL,
    WORKLOAD_DEPLOY_ENDPOINT,
    WORKLOAD_DETAIL_ENDPOINT,
)


@dataclass
class DeploymentSummary:
    total_functions: int
    deployed: int
    failed: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentSummary":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Invalid data type for DeploymentSummary: expected dict, got {type(data).__name__}",
                object_type="DeploymentSummary",
            )
        return cls(
            total_functions=data.get("total_functions", 0),
            deployed=data.get("deployed", 0),
            failed=data.get("failed", 0),
        )


@dataclass
class FunctionDeploymentStatus:
    id: str
    name: str
    status: str
    cost_per_hour: Optional[float] = None
    revision: str = ""
    message: str = ""
    invoke_url: Optional[str] = None
    ready_replicas: int = 0
    total_replicas: int = 0
    created_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionDeploymentStatus":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Invalid data type for FunctionDeploymentStatus: expected dict, got {type(data).__name__}",
                object_type="FunctionDeploymentStatus",
            )
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            revision=data.get("revision", ""),
            status=data.get("status", "unknown"),
            cost_per_hour=data.get("cost_per_hour"),
            message=data.get("message", ""),
            invoke_url=data.get("invoke_url"),
            ready_replicas=data.get("ready_replicas", 0),
            total_replicas=data.get("total_replicas", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            error=data.get("error"),
        )


@dataclass
class PublishResponse:
    app_id: str
    name: str
    status: str
    web_url: Optional[str]
    total_cost_per_hour: float
    summary: Optional[DeploymentSummary]
    functions: List[FunctionDeploymentStatus] = field(default_factory=list)

class AsyncPublishClient(AsyncBaseHTTPClient):
    """Async client for publishing apps to serverless infrastructure."""

    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    @staticmethod
    def _extract_invoke_url(urls: Any) -> Optional[str]:
        if not isinstance(urls, list):
            return None
        for item in urls:
            if isinstance(item, dict) and item.get("url"):
                return item["url"]
        return None

    async def _deploy_workload(self, workload_uid: str) -> FunctionDeploymentStatus:
        detail = await self._async_get(
            WORKLOAD_DETAIL_ENDPOINT.format(workload_uid=workload_uid)
        )
        if not isinstance(detail, dict):
            raise HydrationError(
                f"Invalid workload detail format: expected dict, got {type(detail).__name__}",
                object_type="FunctionDeploymentStatus",
            )

        name = detail.get("name", workload_uid)

        deploy_result = await self._async_post(
            WORKLOAD_DEPLOY_ENDPOINT.format(workload_uid=workload_uid)
        )
        if not isinstance(deploy_result, dict):
            raise HydrationError(
                f"Invalid workload deploy format: expected dict, got {type(deploy_result).__name__}",
                object_type="FunctionDeploymentStatus",
            )

        state = deploy_result.get("state", {})
        if state and not isinstance(state, dict):
            raise HydrationError(
                f"Invalid workload state format: expected dict, got {type(state).__name__}",
                object_type="FunctionDeploymentStatus",
            )

        return FunctionDeploymentStatus(
            id=deploy_result.get("uid", workload_uid),
            name=deploy_result.get("name", name),
            cost_per_hour=deploy_result.get("cost_per_hour"),
            revision=deploy_result.get("revision", ""),
            status=(state or {}).get("status", "unknown"),
            message=(state or {}).get("message", ""),
            invoke_url=self._extract_invoke_url((state or {}).get("urls")),
            ready_replicas=(state or {}).get("ready_replicas", 0),
            total_replicas=(state or {}).get("total_replicas", 0),
            created_at=deploy_result.get("created_at", ""),
            updated_at=deploy_result.get("updated_at", ""),
            error=None,
        )

    async def publish_serverless(
        self, app_id: str, app_name: str, functions: List[str]
    ) -> PublishResponse:
        if not app_id or not app_id.strip():
            raise ValidationError("app_id is required", field="app_id", value=app_id)

        if not app_name or not app_name.strip():
            raise ValidationError(
                "app_name is required", field="app_name", value=app_name
            )

        if not functions:
            raise ValidationError(
                "functions list cannot be empty", field="functions", value=functions
            )

        if not isinstance(functions, list):
            raise ValidationError(
                "functions must be a list of function UIDs",
                field="functions",
                value=type(functions).__name__,
            )

        for i, func_uid in enumerate(functions):
            if not isinstance(func_uid, str) or not func_uid.strip():
                raise ValidationError(
                    f"Invalid function UID at index {i}: must be a non-empty string",
                    field=f"functions[{i}]",
                    value=func_uid,
                )

        responses = await asyncio.gather(
            *(self._deploy_workload(func_uid.strip()) for func_uid in functions),
            return_exceptions=True,
        )

        deployment_statuses: List[FunctionDeploymentStatus] = []
        failed = 0
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                failed += 1
                deployment_statuses.append(
                    FunctionDeploymentStatus(
                        id=functions[i].strip(),
                        name=functions[i].strip(),
                        status="error",
                        message=str(response),
                        error=str(response),
                    )
                )
            else:
                deployment_statuses.append(response)

        deployed = len(deployment_statuses) - failed
        web_url = None
        overall_status = "error"
        total_cost_per_hour = 0.0
        for item in deployment_statuses:
            if item.invoke_url:
                web_url = item.invoke_url
            if item.cost_per_hour is not None:
                total_cost_per_hour += item.cost_per_hour
            if item.status and item.status != "error" and overall_status == "error":
                overall_status = item.status

        return PublishResponse(
            app_id=app_id.strip(),
            name=app_name.strip(),
            status=overall_status,
            web_url=web_url,
            total_cost_per_hour=total_cost_per_hour,
            summary=DeploymentSummary(
                total_functions=len(deployment_statuses),
                deployed=deployed,
                failed=failed,
            ),
            functions=deployment_statuses,
        )
