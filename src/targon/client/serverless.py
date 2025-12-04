from dataclasses import dataclass, replace, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Union

from targon.client.constants import (
    DEFAULT_BASE_URL,
    SERVERLESS_ENDPOINT,
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
    email: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "server": _validate_non_empty(self.server, "server"),
            "username": _validate_non_empty(self.username, "username"),
            "password": _validate_non_empty(self.password, "password"),
        }
        if self.email:
            payload["email"] = self.email
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
            payload["command"] = self.command
        if self.args:
            payload["args"] = self.args
        env_vars = _coerce_env(self.env)
        if env_vars:
            payload["env"] = [env_var.to_payload() for env_var in env_vars]
        if self.env_from:
            payload["env_from"] = self.env_from
        if self.working_dir:
            payload["working_dir"] = self.working_dir
        if self.security_context:
            payload["security_context"] = self.security_context
        if self.registry_credentials:
            payload["registry_credentials"] = self.registry_credentials.to_payload()
        return payload


@dataclass(slots=True)
class AutoScalingConfig:
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    container_concurrency: Optional[int] = None
    target_concurrency: Optional[int] = None

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v is not None}


@dataclass(slots=True)
class PortConfig:
    port: int
    name: Optional[str] = None
    protocol: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"port": self.port}
        if self.name:
            payload["name"] = self.name
        if self.protocol:
            payload["protocol"] = self.protocol
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
        payload: Dict[str, Any] = {
            "name": self.name.strip(),
            "container": self.container.to_payload(),
        }
        payload["resource_name"] = self.resource_name or "cpu-small"
        if self.scaling:
            payload["scaling"] = self.scaling.to_payload()
        if self.network:
            network_payload = self.network.to_payload()
            if network_payload:
                payload["network"] = network_payload
        if self.app_id:
            payload["app_id"] = _validate_non_empty(self.app_id, "app_id")
        return payload

    def with_app(self, app_id: str) -> "CreateServerlessResourceRequest":
        return replace(self, app_id=_validate_non_empty(app_id, "app_id"))


@dataclass(slots=True)
class CreateServerlessResponse:
    uid: str
    name: str
    url: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreateServerlessResponse":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Expected dict for CreateServerlessResponse, got {type(data).__name__}",
                object_type="CreateServerlessResponse",
            )

        uid = data.get("serverless_uid")
        name = data.get("name")
        if not uid:
            raise HydrationError(
                "Missing serverless_uid in CreateServerlessResponse response",
                object_type="CreateServerlessResponse",
            )
        url = f"https://{uid}.serverless.targon.com"
        return cls(uid=uid, name=name, url=url)


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
        created_at_str = None
        if created_at is not None:
            created_at_str = (
                created_at if isinstance(created_at, str) else str(created_at)
            )
        return cls(
            uid=uid,
            name=data.get("name"),
            url=data.get("url", f"https://{uid}.serverless.targon.com"),
            cost=float(data.get("cost")) / 27777.5,
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
    email: Optional[str] = None


class AsyncServerlessClient(AsyncBaseHTTPClient):
    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    def _resource_path(self, suffix: Optional[str] = None) -> str:
        if suffix:
            return f"{SERVERLESS_ENDPOINT}/{suffix.lstrip('/')}"
        return SERVERLESS_ENDPOINT

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
                    email=registry.email,
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
        result = await self._async_post(self._resource_path(), json=payload)
        if not isinstance(result, dict):
            raise HydrationError(
                f"Invalid response for deploy_resource: {type(result).__name__}",
                object_type="ServerlessResource",
            )
        return CreateServerlessResponse.from_dict(result)

    async def list_container(self) -> List[ServerlessResourceListItem]:
        payload: Dict[str, Any] = {"active_only": True}
        result = await self._async_post(self._resource_path("list"), json=payload)

        if isinstance(result, dict):
            items_raw = result.get("resources")
            if items_raw is None:
                raise HydrationError(
                    "Serverless list response missing 'resources' array",
                    object_type="ServerlessResourceList",
                )
            if not isinstance(items_raw, list):
                raise HydrationError(
                    f"'resources' in serverless list response must be a list, got {type(items_raw).__name__}",
                    object_type="ServerlessResourceList",
                )
            items = items_raw
        else:
            raise HydrationError(
                f"Expected dict or list from list_resources, got {type(result).__name__}",
                object_type="ServerlessResourceList",
            )
        return [
            ServerlessResourceListItem.from_dict(item)
            for item in items
            if "deleted" not in item
        ]

    async def delete_container(self, resource_id: str) -> Dict[str, Any]:
        resource_id = _validate_non_empty(resource_id, "resource_id")
        result = await self._async_delete(self._resource_path(resource_id))
        if not isinstance(result, dict):
            raise HydrationError(
                f"Invalid response for delete_resource: {type(result).__name__}",
                object_type="ServerlessDeleteResponse",
            )
        return result
