from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from targon.core.partial_function import WebhookConfig
from targon.core.objects import AsyncBaseHTTPClient
from targon.core.exceptions import ValidationError, HydrationError
from targon.client.constants import (
    WORKLOADS_ENDPOINT,
    WORKLOAD_DETAIL_ENDPOINT,
    DEFAULT_BASE_URL,
)


@dataclass
class AutoscalerSettings:
    min_replicas: int
    max_replicas: int
    initial_replicas: Optional[int] = None
    container_concurrency: Optional[int] = None
    target_concurrency: Optional[int] = None
    scale_up_delay: Optional[str] = None
    scale_down_delay: Optional[str] = None
    zero_grace_period: Optional[str] = None
    scaling_metric: Optional[str] = None
    target_value: Optional[float] = None

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}

@dataclass
class WorkloadPort:
    port: int
    protocol: str = "TCP"
    routing: str = "PROXIED"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "port": self.port,
            "protocol": self.protocol,
            "routing": self.routing,
        }

@dataclass
class ServerlessWebhookConfig:
    type: str
    method: Optional[str] = None
    docs: Optional[bool] = None
    label: Optional[str] = None
    requires_auth: Optional[bool] = None
    port: Optional[int] = None
    startup_timeout: Optional[int] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": self.type}
        if self.method:
            payload["method"] = self.method
        if self.docs is not None:
            payload["docs"] = self.docs
        if self.label:
            payload["label"] = self.label
        if self.requires_auth is not None:
            payload["requires_auth"] = self.requires_auth
        if self.port is not None:
            payload["port"] = self.port
        if self.startup_timeout is not None:
            payload["startup_timeout"] = self.startup_timeout
        return payload

@dataclass
class CreateWorkloadServerlessConfig:
    definition_type: str
    module: str = ""
    qualname: str = ""
    function_serialized: str = ""
    class_serialized: str = ""
    webhook_config: Optional[ServerlessWebhookConfig] = None
    timeout_seconds: Optional[int] = None
    startup_timeout: Optional[int] = None
    object_dependencies: Optional[List[str]] = None
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    initial_replicas: Optional[int] = None
    container_concurrency: Optional[int] = None
    target_concurrency: Optional[int] = None
    scale_up_delay: Optional[str] = None
    scale_down_delay: Optional[str] = None
    zero_grace_period: Optional[str] = None
    scaling_metric: Optional[str] = None
    target_value: Optional[float] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"definition_type": self.definition_type}
        if self.module:
            payload["module"] = self.module
        if self.qualname:
            payload["qualname"] = self.qualname
        if self.function_serialized:
            payload["function_serialized"] = self.function_serialized
        if self.class_serialized:
            payload["class_serialized"] = self.class_serialized
        if self.webhook_config:
            payload["webhook_config"] = self.webhook_config.to_payload()
        if self.timeout_seconds is not None:
            payload["timeout_seconds"] = self.timeout_seconds
        if self.startup_timeout is not None:
            payload["startup_timeout"] = self.startup_timeout
        if self.object_dependencies:
            payload["object_dependencies"] = self.object_dependencies
        if self.min_replicas is not None:
            payload["min_replicas"] = self.min_replicas
        if self.max_replicas is not None:
            payload["max_replicas"] = self.max_replicas
        if self.initial_replicas is not None and self.initial_replicas > 0:
            payload["initial_replicas"] = self.initial_replicas
        if self.container_concurrency is not None:
            payload["container_concurrency"] = self.container_concurrency
        if self.target_concurrency is not None:
            payload["target_concurrency"] = self.target_concurrency
        if self.scale_up_delay:
            payload["scale_up_delay"] = self.scale_up_delay
        if self.scale_down_delay:
            payload["scale_down_delay"] = self.scale_down_delay
        if self.zero_grace_period:
            payload["zero_grace_period"] = self.zero_grace_period
        if self.scaling_metric:
            payload["scaling_metric"] = self.scaling_metric
        if self.target_value is not None:
            payload["target_value"] = self.target_value
        return payload

@dataclass
class CreateWorkloadRequest:
    app_id: str
    image: str
    name: str
    ports: List[WorkloadPort]
    resource_name: str
    serverless_config: CreateWorkloadServerlessConfig
    type: str = "FUNCTION"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "app_id": self.app_id,
            "image": self.image,
            "name": self.name,
            "ports": [port.to_payload() for port in self.ports],
            "resource_name": self.resource_name,
            "serverless_config": self.serverless_config.to_payload(),
            "type": self.type,
        }

@dataclass
class UpdateWorkloadRequest:
    name: Optional[str] = None
    image: Optional[str] = None
    app_id: Optional[str] = None
    ports: Optional[List[WorkloadPort]] = None
    serverless_config: Optional[CreateWorkloadServerlessConfig] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.name is not None:
            payload["name"] = self.name
        if self.image is not None:
            payload["image"] = self.image
        if self.app_id is not None:
            payload["app_id"] = self.app_id
        if self.ports is not None:
            payload["ports"] = [port.to_payload() for port in self.ports]
        if self.serverless_config is not None:
            payload["serverless_config"] = self.serverless_config.to_payload()
        return payload


@dataclass
class FunctionMetadata:
    name: str
    app_id: str
    created: bool
    revision: str
    web_url: str
    grpc_endpoint: str


@dataclass
class FunctionResponse:
    """Matches backend FunctionResponse struct"""
    uid: str
    metadata: FunctionMetadata


def _build_function_response(
    *,
    uid: str,
    name: str,
    app_id: str,
    created: bool,
    result: Optional[Dict[str, Any]] = None,
    grpc: bool,
) -> FunctionResponse:
    payload = result or {}
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    urls = state.get("urls") if isinstance(state.get("urls"), list) else []
    primary_url = ""
    for item in urls:
        if isinstance(item, dict) and item.get("url"):
            primary_url = item["url"]
            break

    web_url = payload.get("web_url", "")
    grpc_endpoint = payload.get("grpc_url", "")
    if not web_url and not grpc_endpoint and primary_url:
        if grpc:
            grpc_endpoint = primary_url
        else:
            web_url = primary_url

    return FunctionResponse(
        uid=uid,
        metadata=FunctionMetadata(
            name=payload.get("name", name),
            app_id=payload.get("app_id", app_id),
            created=created,
            revision=payload.get("revision", ""),
            web_url=web_url,
            grpc_endpoint=grpc_endpoint,
        ),
    )


def _require_workload_response(
    result: Any,
    *,
    source: str,
    fallback_uid: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    if not isinstance(result, dict):
        raise HydrationError(
            f"Invalid response format from {source}: expected dict, got {type(result).__name__}",
            object_type="FunctionResponse",
        )

    uid = result.get("uid") or fallback_uid
    if not uid:
        raise HydrationError(
            f"Missing required field 'uid' in {source} response",
            object_type="FunctionResponse",
        )

    return uid, result


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
        object_dependencies: Optional[List[str]] = None,
        autoscaler_settings: Optional[AutoscalerSettings] = None,
        resource_name: str = "",
    ) -> FunctionResponse:
        if not app_id or not app_id.strip():
            raise ValidationError("app_id is required", field="app_id", value=app_id)
        if not name or not name.strip():
            raise ValidationError("name is required", field="name", value=name)
        if not image_id or not image_id.strip():
            raise ValidationError("image is required", field="image_id", value=image_id)
        if not resource_name or not resource_name.strip():
            raise ValidationError(
                "resource_name is required",
                field="resource_name",
                value=resource_name,
            )

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
        if object_dependencies is not None and not isinstance(object_dependencies, list):
            raise ValidationError(
                "object_dependencies must be a list of workload IDs",
                field="object_dependencies",
                value=object_dependencies,
            )

        if webhook_config is None:
            workload_port = 8080
        else:
            workload_port = webhook_config.port or 50051
        workload_webhook = None
        if webhook_config:
            workload_webhook = ServerlessWebhookConfig(
                type=webhook_config.type,
                method=webhook_config.method,
                docs=webhook_config.docs,
                label=webhook_config.label,
                requires_auth=webhook_config.requires_auth,
                port=webhook_config.port,
                startup_timeout=webhook_config.startup_timeout,
            )

        autoscaler_payload = autoscaler_settings.to_payload() if autoscaler_settings else {}
        serverless_config = CreateWorkloadServerlessConfig(
            # definition_type=definition_type,
            module=module,
            qualname=qualname,
            function_serialized=function_serialized or "",
            class_serialized=class_serialized or "",
            webhook_config=workload_webhook,
            timeout_seconds=timeout_secs,
            startup_timeout=startup_timeout,
            object_dependencies=object_dependencies or None,
            min_replicas=autoscaler_payload.get("min_replicas"),
            max_replicas=autoscaler_payload.get("max_replicas"),
            initial_replicas=autoscaler_payload.get("initial_replicas"),
            container_concurrency=autoscaler_payload.get("container_concurrency"),
            target_concurrency=autoscaler_payload.get("target_concurrency"),
            scale_up_delay=autoscaler_payload.get("scale_up_delay"),
            scale_down_delay=autoscaler_payload.get("scale_down_delay"),
            zero_grace_period=autoscaler_payload.get("zero_grace_period"),
            scaling_metric=autoscaler_payload.get("scaling_metric"),
            target_value=autoscaler_payload.get("target_value"),
        )

        payload = CreateWorkloadRequest(
            app_id=app_id,
            image=image_id,
            name=name,
            resource_name=resource_name,
            ports=[WorkloadPort(port=workload_port)],
            type="FUNCTION",
            serverless_config=serverless_config,
        )
        update_payload = UpdateWorkloadRequest(
            name=name,
            image=image_id,
            app_id=app_id,
            ports=[WorkloadPort(port=workload_port)],
            serverless_config=serverless_config,
        )

        existing_result = await self._async_get(
            WORKLOADS_ENDPOINT,
            params={"app_id": app_id, "name": name, "type": "function"},
        )
        if not isinstance(existing_result, dict):
            raise HydrationError(
                f"Invalid response format from workload lookup: expected dict, got {type(existing_result).__name__}",
                object_type="FunctionResponse",
            )

        existing_items = existing_result.get("items", [])
        if not isinstance(existing_items, list):
            raise HydrationError(
                f"Invalid 'items' field from workload lookup: expected list, got {type(existing_items).__name__}",
                object_type="FunctionResponse",
            )

        existing_item = existing_items[0] if existing_items else None
        existing_uid = (
            existing_item.get("uid") if isinstance(existing_item, dict) else None
        )

        if existing_uid:
            patched_result = await self._async_patch(
                WORKLOAD_DETAIL_ENDPOINT.format(workload_uid=existing_uid),
                json=update_payload.to_payload(),
            )
            uid, result_payload = _require_workload_response(
                patched_result,
                source="workload update",
                fallback_uid=existing_uid,
            )
            return _build_function_response(
                uid=uid,
                name=name,
                app_id=app_id,
                created=False,
                result=result_payload,
                grpc=webhook_config is None,
            )

        created_result = await self._async_post(
            WORKLOADS_ENDPOINT, json=payload.to_payload()
        )
        uid, result_payload = _require_workload_response(
            created_result,
            source="function registration",
        )
        return _build_function_response(
            uid=uid,
            name=name,
            app_id=app_id,
            created=True,
            result=result_payload,
            grpc=webhook_config is None,
        )

    async def delete_function(self, workload_uid: str) -> Dict[str, Any]:
        """Delete a function workload by workload UID."""
        if not workload_uid or not workload_uid.strip():
            raise ValidationError(
                "workload_uid is required",
                field="workload_uid",
                value=workload_uid,
            )

        result = await self._async_delete(
            WORKLOAD_DETAIL_ENDPOINT.format(workload_uid=workload_uid.strip())
        )
        if result and not isinstance(result, dict):
            raise HydrationError(
                f"Invalid response format from function deletion: expected empty response, got {type(result).__name__}",
                object_type="FunctionResponse",
            )   
        return result or {}
