from dataclasses import dataclass, replace, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Union

from targon.client.constants import (
    DEFAULT_BASE_URL,
    WORKLOADS_ENDPOINT,
    WORKLOAD_DETAIL_ENDPOINT,
    WORKLOAD_DEPLOY_ENDPOINT,
    WORKLOAD_EVENTS_ENDPOINT,
    WORKLOAD_STATE_ENDPOINT,
)
from targon.core.exceptions import HydrationError, ValidationError
from targon.core.objects import AsyncBaseHTTPClient
from targon.core.resources import Compute


def _validate_non_empty(value: Optional[str], field_name: str) -> str:
    if not value or not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{field_name} must be a non-empty string", field=field_name, value=value
        )
    return value.strip()


@dataclass(slots=True)
class EnvVar:
    name: str
    value: str

    def to_payload(self) -> Dict[str, str]:
        return {"name": self.name, "value": self.value}


@dataclass(slots=True)
class RegistryCredentials:
    server: str
    username: str
    password: str

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "server": _validate_non_empty(self.server, "server"),
            "username": _validate_non_empty(self.username, "username"),
            "password": _validate_non_empty(self.password, "password"),
        }
        return payload


def _coerce_env(
    env: Optional[Union[Dict[str, str], Sequence[EnvVar]]],
) -> Optional[List[EnvVar]]:
    if env is None:
        return None
    if isinstance(env, dict):
        return [EnvVar(name=str(k), value=str(v)) for k, v in env.items()]
    if isinstance(env, Sequence):
        env_list: List[EnvVar] = []
        for item in env:
            if not isinstance(item, EnvVar):
                raise ValidationError(
                    "env sequence must contain EnvVar instances",
                    field="env",
                    value=item,
                )
            env_list.append(item)
        return env_list
    raise ValidationError(
        "env must be a dict or sequence of EnvVar objects", field="env"
    )


@dataclass(slots=True)
class ContainerConfig:
    image: str
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env: Optional[Union[Dict[str, str], Sequence[EnvVar]]] = None
    env_from: Optional[List[Dict[str, Any]]] = None
    working_dir: Optional[str] = None
    security_context: Optional[Dict[str, Any]] = None
    registry_credentials: Optional[RegistryCredentials] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"image": _validate_non_empty(self.image, "image")}
        if self.command:
            payload["commands"] = self.command
        if self.args:
            payload["args"] = self.args
        env_vars = _coerce_env(self.env)
        if env_vars:
            payload["envs"] = [env_var.to_payload() for env_var in env_vars]
        if self.security_context:
            payload["security_context"] = self.security_context
        if self.registry_credentials:
            payload["registry_auth"] = self.registry_credentials.to_payload()
        return payload


@dataclass(slots=True)
class AutoScalingConfig:
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    container_concurrency: Optional[int] = None
    target_concurrency: Optional[int] = None
    initial_replicas: Optional[int] = None
    scale_up_delay: Optional[str] = None
    scale_down_delay: Optional[str] = None
    zero_grace_period: Optional[str] = None
    scaling_metric: Optional[str] = None
    target_value: Optional[float] = None

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v is not None}


@dataclass(slots=True)
class PortConfig:
    port: int
    name: Optional[str] = None
    protocol: Optional[str] = "TCP"
    routing: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"port": self.port}
        if self.protocol:
            payload["protocol"] = self.protocol
        if self.routing:
            payload["routing"] = self.routing
        return payload


@dataclass(slots=True)
class NetworkConfig:
    port: Optional[Union[PortConfig, Dict[str, Any]]] = None
    visibility: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.port:
            if isinstance(self.port, PortConfig):
                payload["port"] = self.port.to_payload()
            elif isinstance(self.port, dict):
                payload["port"] = self.port
            else:
                raise ValidationError(
                    "port must be a PortConfig or mapping",
                    field="port",
                    value=self.port,
                )
        if self.visibility:
            payload["visibility"] = self.visibility
        return payload


@dataclass(slots=True)
class CreateServerlessResourceRequest:
    name: str
    container: ContainerConfig
    resource_name: Optional[str] = None
    scaling: Optional[AutoScalingConfig] = None
    network: Optional[NetworkConfig] = None
    app_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        if not self.name or not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError(
                "name must be provided", field="name", value=self.name
            )
        container_payload = self.container.to_payload()
        payload: Dict[str, Any] = {
            "type": "SERVERLESS",
            "name": self.name.strip(),
            "image": container_payload["image"],
            "resource_name": self.resource_name or "cpu-small",
        }
        if self.app_id:
            payload["app_id"] = _validate_non_empty(self.app_id, "app_id")
        if "envs" in container_payload:
            payload["envs"] = container_payload["envs"]
        if "commands" in container_payload:
            payload["commands"] = container_payload["commands"]
        if "args" in container_payload:
            payload["args"] = container_payload["args"]
        if "registry_auth" in container_payload:
            payload["registry_auth"] = container_payload["registry_auth"]

        serverless_config: Dict[str, Any] = {}
        if self.scaling:
            serverless_config.update(self.scaling.to_payload())
        if self.network:
            network_payload = self.network.to_payload()
            port_payload = network_payload.get("port")
            if port_payload:
                payload["ports"] = [port_payload]
            if network_payload.get("visibility"):
                serverless_config["visibility"] = network_payload["visibility"]
        if container_payload.get("security_context"):
            serverless_config["security_context"] = container_payload["security_context"]
        payload["serverless_config"] = serverless_config
        return payload

    def with_app(self, app_id: str) -> "CreateServerlessResourceRequest":
        return replace(self, app_id=_validate_non_empty(app_id, "app_id"))


@dataclass(slots=True)
class CreateServerlessResponse:
    uid: str
    name: str
    url: str
    status: str = ""
    message: str = ""
    cost_per_hour: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreateServerlessResponse":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Expected dict for CreateServerlessResponse, got {type(data).__name__}",
                object_type="CreateServerlessResponse",
            )

        uid = data.get("uid")
        name = data.get("name", "")
        if not uid:
            raise HydrationError(
                "Missing uid in CreateServerlessResponse response",
                object_type="CreateServerlessResponse",
            )
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        urls = state.get("urls") if isinstance(state.get("urls"), list) else []
        url = ""
        for item in urls:
            if isinstance(item, dict) and item.get("url"):
                url = item["url"]
                break
        return cls(
            uid=uid,
            name=name,
            url=url,
            status=state.get("status", ""),
            message=state.get("message", ""),
            cost_per_hour=data.get("cost_per_hour"),
        )


@dataclass(slots=True)
class WorkloadURL:
    port: int
    url: str


@dataclass(slots=True)
class WorkloadStateResponse:
    uid: str
    workload_type: str
    status: str
    message: str
    urls: List[WorkloadURL] = field(default_factory=list)
    ready_replicas: int = 0
    total_replicas: int = 0
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadStateResponse":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Expected dict for WorkloadStateResponse, got {type(data).__name__}",
                object_type="WorkloadStateResponse",
            )

        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in WorkloadStateResponse response",
                object_type="WorkloadStateResponse",
            )

        urls_raw = data.get("urls", [])
        if urls_raw is None:
            urls_raw = []
        if not isinstance(urls_raw, list):
            raise HydrationError(
                f"Expected list for WorkloadStateResponse.urls, got {type(urls_raw).__name__}",
                object_type="WorkloadStateResponse",
            )

        urls = [
            WorkloadURL(port=item.get("port", 0), url=item.get("url", ""))
            for item in urls_raw
            if isinstance(item, dict)
        ]

        return cls(
            uid=uid,
            workload_type=data.get("workload_type", ""),
            status=data.get("status", ""),
            message=data.get("message", ""),
            urls=urls,
            ready_replicas=data.get("ready_replicas", 0),
            total_replicas=data.get("total_replicas", 0),
            updated_at=data.get("updated_at", ""),
        )


@dataclass(slots=True)
class WorkloadEvent:
    workload_uid: str
    workload_type: str
    resource_name: Optional[str] = None
    event_type: str = ""
    pod_name: Optional[str] = None
    container_name: Optional[str] = None
    container_image: Optional[str] = None
    new_status: Optional[str] = None
    replica_count: Optional[int] = None
    old_replica_count: Optional[int] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    display_message: Optional[str] = None
    exit_code: Optional[int] = None
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadEvent":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Expected dict for WorkloadEvent, got {type(data).__name__}",
                object_type="WorkloadEvent",
            )
        return cls(
            workload_uid=data.get("workload_uid", ""),
            workload_type=data.get("workload_type", ""),
            resource_name=data.get("resource_name"),
            event_type=data.get("event_type", ""),
            pod_name=data.get("pod_name"),
            container_name=data.get("container_name"),
            container_image=data.get("container_image"),
            new_status=data.get("new_status"),
            replica_count=data.get("replica_count"),
            old_replica_count=data.get("old_replica_count"),
            reason=data.get("reason"),
            message=data.get("message"),
            display_message=data.get("display_message"),
            exit_code=data.get("exit_code"),
            created_at=data.get("created_at", ""),
        )


@dataclass(slots=True)
class WorkloadEventsResponse:
    items: List[WorkloadEvent] = field(default_factory=list)
    next_cursor: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadEventsResponse":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Expected dict for WorkloadEventsResponse, got {type(data).__name__}",
                object_type="WorkloadEventsResponse",
            )
        items_raw = data.get("items", [])
        if not isinstance(items_raw, list):
            raise HydrationError(
                f"Expected list for WorkloadEventsResponse.items, got {type(items_raw).__name__}",
                object_type="WorkloadEventsResponse",
            )
        return cls(
            items=[WorkloadEvent.from_dict(item) for item in items_raw],
            next_cursor=data.get("next_cursor"),
        )


@dataclass(slots=True)
class ServerlessResourceListItem:
    uid: str
    name: Optional[str] = None
    url: Optional[str] = None
    cost: Optional[float] = None
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerlessResourceListItem":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Expected dict for ServerlessResourceListItem, got {type(data).__name__}",
                object_type="ServerlessResource",
            )
        uid = data.get("uid")
        if not uid:
            raise HydrationError(
                "Missing uid in ServerlessResourceListItem response",
                object_type="ServerlessResource",
            )
        created_at = data.get("created_at")
        created_at_str = created_at if isinstance(created_at, str) else (str(created_at) if created_at is not None else None)
        urls = data.get("urls") if isinstance(data.get("urls"), list) else []
        if not urls and isinstance(data.get("state"), dict):
            state_urls = data["state"].get("urls")
            if isinstance(state_urls, list):
                urls = state_urls
        url = None
        for item in urls:
            if isinstance(item, dict) and item.get("url"):
                url = item["url"]
                break
        return cls(
            uid=uid,
            name=data.get("name"),
            url=url,
            cost=data.get("cost_per_hour"),
            created_at=created_at_str,
        )


@dataclass
class ReplicasConfig:
    min: int = 1
    max: int = 2
    container_concurrency: int = 100
    target_concurrency: int = 100
    scale_to_zero: bool = False

    def __post_init__(self):
        if self.min < 0:
            raise ValueError('min must be >= 0')
        if self.max < 1:
            raise ValueError('max must be >= 1')
        if self.max < self.min:
            raise ValueError('max must be >= min')
        if self.container_concurrency < 1:
            raise ValueError('target_concurrency must be >= 1')
        if self.target_concurrency < 1:
            raise ValueError('target_concurrency must be >= 1')


@dataclass
class RegistryConfig:
    server: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


class AsyncServerlessClient(AsyncBaseHTTPClient):
    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    async def deploy_container(
        self,
        request: Optional[CreateServerlessResourceRequest] = None,
        *,
        name: Optional[str] = None,
        image: Optional[str] = None,
        resource: Optional[str] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        port: Optional[int] = None,
        internal: bool = False,
        env: Optional[Dict[str, str]] = None,
        replicas: Optional[ReplicasConfig] = None,
        registry: Optional[RegistryConfig] = None,
    ) -> CreateServerlessResponse:

        if request is None:
            registry_creds = None
            if registry:
                registry_creds = RegistryCredentials(
                    server=registry.server or "https://index.docker.io/v1/",
                    username=registry.username or "",
                    password=registry.password or "",
                )

            scaling = None
            if replicas:
                scaling = AutoScalingConfig(
                    min_replicas=replicas.min,
                    max_replicas=replicas.max,
                    container_concurrency=replicas.container_concurrency,
                    target_concurrency=replicas.target_concurrency,
                )

            network = None
            if port:
                port_config = PortConfig(port=port)
                visibility = "cluster-local" if internal else "external"
                network = NetworkConfig(port=port_config, visibility=visibility)

            container_config = ContainerConfig(
                image=image,
                command=command,
                args=args,
                env=env,
                working_dir=working_dir,
                registry_credentials=registry_creds,
            )

            request = CreateServerlessResourceRequest(
                name=name,
                resource_name=resource,
                container=container_config,
                scaling=scaling,
                network=network,
            )

        payload = request.to_payload()
        result = await self._async_post(WORKLOADS_ENDPOINT, json=payload)
        if not isinstance(result, dict):
            raise HydrationError(
                f"Invalid response for deploy_resource: {type(result).__name__}",
                object_type="ServerlessResource",
            )
        workload_uid = result.get("uid")
        if not workload_uid:
            raise HydrationError(
                "Missing uid in workload registration response",
                object_type="CreateServerlessResponse",
            )

        deploy_result = await self._async_post(
            WORKLOAD_DEPLOY_ENDPOINT.format(workload_uid=workload_uid)
        )
        if not isinstance(deploy_result, dict):
            raise HydrationError(
                f"Invalid response for workload deploy: {type(deploy_result).__name__}",
                object_type="CreateServerlessResponse",
            )
        return CreateServerlessResponse.from_dict(deploy_result)

    async def list_container(self) -> List[ServerlessResourceListItem]:
        result = await self._async_get(WORKLOADS_ENDPOINT, params={"type": "SERVERLESS"})

        if not isinstance(result, dict):
            raise HydrationError(
                f"Expected dict from list_resources, got {type(result).__name__}",
                object_type="ServerlessResourceList",
            )
        items = result.get("items")
        if not isinstance(items, list):
            raise HydrationError(
                f"'items' in serverless list response must be a list, got {type(items).__name__}",
                object_type="ServerlessResourceList",
            )
        return [
            ServerlessResourceListItem.from_dict(item)
            for item in items
            if isinstance(item, dict)
            and (
                not isinstance(item.get("state"), dict)
                or item["state"].get("status") != "deleted"
            )
        ]

    async def delete_container(self, resource_id: str) -> Dict[str, Any]:
        resource_id = _validate_non_empty(resource_id, "resource_id")
        result = await self._async_delete(
            WORKLOAD_DETAIL_ENDPOINT.format(workload_uid=resource_id)
        )
        if result and not isinstance(result, dict):
            raise HydrationError(
                f"Invalid response for delete_resource: {type(result).__name__}",
                object_type="ServerlessDeleteResponse",
            )
        return result or {}

    async def get_state(self, workload_uid: str) -> WorkloadStateResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        result = await self._async_get(
            WORKLOAD_STATE_ENDPOINT.format(workload_uid=workload_uid)
        )
        return WorkloadStateResponse.from_dict(result)

    async def get_events(
        self,
        workload_uid: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> WorkloadEventsResponse:
        workload_uid = _validate_non_empty(workload_uid, "workload_uid")
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        result = await self._async_get(
            WORKLOAD_EVENTS_ENDPOINT.format(workload_uid=workload_uid),
            params=params or None,
        )
        return WorkloadEventsResponse.from_dict(result)
