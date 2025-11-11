from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from targon.core.objects import AsyncBaseHTTPClient
from targon.core.exceptions import ValidationError, HydrationError
from targon.client.constants import DEFAULT_BASE_URL, PUBLISH_ENDPOINT


@dataclass
class PublishRequest:
    """Request payload for publishing an app."""

    app_id: str
    app_name: str
    functions: List[str]


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
    invoke_url: Optional[str] = None
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
            status=data.get("status", "unknown"),
            invoke_url=data.get("invoke_url"),
            error=data.get("error"),
        )


@dataclass
class PublishResponse:
    app_id: str
    name: str
    status: str
    web_url: Optional[str]
    summary: Optional[DeploymentSummary]
    functions: List[FunctionDeploymentStatus] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublishResponse":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Invalid data type for PublishResponse: expected dict, got {type(data).__name__}",
                object_type="PublishResponse",
            )

        summary = None
        if "summary" in data and data["summary"]:
            summary = DeploymentSummary.from_dict(data["summary"])

        functions = []
        if "functions" in data and data["functions"]:
            if not isinstance(data["functions"], list):
                raise HydrationError(
                    "Invalid 'functions' field: expected list",
                    object_type="PublishResponse",
                )
            functions = [
                FunctionDeploymentStatus.from_dict(func) for func in data["functions"]
            ]

        return cls(
            app_id=data.get("app_id", ""),
            name=data.get("name", ""),
            status=data.get("status", "unknown"),
            web_url=data.get("web_url"),
            summary=summary,
            functions=functions,
        )


class AsyncPublishClient(AsyncBaseHTTPClient):
    """Async client for publishing apps to serverless infrastructure."""

    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

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

        payload = PublishRequest(
            app_id=app_id.strip(),
            app_name=app_name.strip(),
            functions=functions,
        )

        result = await self._async_post(PUBLISH_ENDPOINT, json=asdict(payload))

        if not isinstance(result, dict):
            raise HydrationError(
                f"Invalid response format: expected dict, got {type(result).__name__}",
                object_type="PublishResponse",
            )

        return PublishResponse.from_dict(result)
