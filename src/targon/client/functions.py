from dataclasses import dataclass, asdict
from typing import Optional

from targon.core.partial_function import WebhookConfig
from targon.core.objects import AsyncBaseHTTPClient
from targon.core.exceptions import ValidationError, HydrationError
from targon.client.constants import FUNC_REG_ENDPOINT, DEFAULT_BASE_URL


@dataclass
class AutoscalerSettings:
    min_replicas: int
    max_replicas: int
    initial_replicas: int
    container_concurrency: Optional[int] = None
    target_concurrency: Optional[int] = None
    scale_up_delay: Optional[str] = None
    scale_down_delay: Optional[str] = None
    zero_grace_period: Optional[str] = None
    scaling_metric: Optional[str] = None
    target_value: Optional[float] = None


@dataclass
class FunctionRequest:
    """Matches backend FunctionRequest struct"""

    app_id: str
    name: str
    module: str = ""
    qualname: str = ""
    definition_type: str = ""
    function_serialized: str = ""
    class_serialized: str = ""
    image_id: str = ""
    webhook_config: Optional[WebhookConfig] = None
    timeout_secs: int = 0
    startup_timeout: int = 0
    object_dependencies: str = ""
    autoscaler_settings: Optional[AutoscalerSettings] = None
    resource_name: str = ""


@dataclass
class FunctionMetadata:
    name: str
    app_id: str
    created: bool
    revision: int
    web_url: str
    grpc_endpoint: str


@dataclass
class FunctionResponse:
    """Matches backend FunctionResponse struct"""

    uid: str
    metadata: FunctionMetadata


class AsyncFunctionsClient(AsyncBaseHTTPClient):
    """Async client for function registration and management."""

    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    async def register_function(
        self,
        app_id: str,
        name: str,
        module: str = "",
        qualname: str = "",
        definition_type: str = "",
        function_serialized: str = "",
        class_serialized: str = "",
        image_id: str = "",
        webhook_config: Optional[WebhookConfig] = None,
        timeout_secs: int = 300,
        startup_timeout: int = 300,
        object_dependencies: Optional[str] = None,
        autoscaler_settings: Optional[AutoscalerSettings] = None,
        resource_name: str = "",
    ) -> FunctionResponse:
        if not app_id or not app_id.strip():
            raise ValidationError("app_id is required", field="app_id", value=app_id)
        if not name or not name.strip():
            raise ValidationError("name is required", field="name", value=name)

        if timeout_secs < 0:
            raise ValidationError(
                "timeout_secs must be non-negative",
                field="timeout_secs",
                value=timeout_secs,
            )
        if startup_timeout < 0:
            raise ValidationError(
                "startup_timeout must be non-negative",
                field="startup_timeout",
                value=startup_timeout,
            )

        payload = FunctionRequest(
            app_id=app_id,
            name=name,
            module=module,
            qualname=qualname,
            definition_type=definition_type,
            function_serialized=function_serialized or "",
            class_serialized=class_serialized or "",
            image_id=image_id,
            webhook_config=webhook_config,
            timeout_secs=timeout_secs,
            startup_timeout=startup_timeout,
            object_dependencies=object_dependencies or "",
            autoscaler_settings=autoscaler_settings,
            resource_name=resource_name,
        )

        result = await self._async_post(
            FUNC_REG_ENDPOINT.format(app_id=app_id), json=asdict(payload)
        )

        if not isinstance(result, dict):
            raise HydrationError(
                f"Invalid response format from function registration: expected dict, got {type(result).__name__}",
                object_type="FunctionResponse",
            )

        uid = result.get("uid")
        if not uid:
            raise HydrationError(
                "Missing required field 'uid' in function registration response",
                object_type="FunctionResponse",
            )

        return FunctionResponse(
            uid=uid,
            metadata=FunctionMetadata(
                name=result.get("name", name),
                app_id=result.get("app_id", app_id),
                created=result.get("created", False),
                revision=result.get("revision", 0),
                web_url=result.get("web_url", ""),
                grpc_endpoint=result.get("grpc_url", ""),
            ),
        )
