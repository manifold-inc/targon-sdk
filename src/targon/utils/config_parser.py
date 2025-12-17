import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import yaml

from targon.core.exceptions import ValidationError
from targon.core.resources import Compute
from targon.client.serverless import (
    CreateServerlessResourceRequest,
    ContainerConfig as ServerlessContainerConfig,
    RegistryConfig,
    RegistryCredentials,
    AutoScalingConfig,
    NetworkConfig,
    PortConfig,
    EnvVar,
    ReplicasConfig,
)


def _get_valid_compute_resources() -> List[str]:
    return [
        value
        for name, value in vars(Compute).items()
        if not name.startswith('_') and isinstance(value, str)
    ]


def _validate_compute_resource(resource: Optional[str]) -> Optional[str]:
    if resource is None:
        return None

    valid_resources = _get_valid_compute_resources()
    if resource not in valid_resources:
        raise ValueError(
            f"Invalid resource '{resource}'. Must be one of: {', '.join(valid_resources)}"
        )
    return resource


@dataclass
class _ContainerConfig:
    name: str
    image: str
    resource: Optional[str] = None
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    working_dir: Optional[str] = None
    port: Optional[int] = None
    internal: bool = False
    env: Dict[str, str] = field(default_factory=dict)
    replicas: Optional[ReplicasConfig] = None
    registry: Optional[RegistryConfig] = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError('Container name cannot be empty')
        from targon.core.utils import check_object_name

        check_object_name(self.name)

        if self.resource is not None:
            self.resource = _validate_compute_resource(self.resource)

        self.env = self._resolve_env_vars(self.env)

        if self.replicas is None:
            self.replicas = ReplicasConfig()

    @staticmethod
    def _resolve_env_vars(env_dict: Dict[str, str]) -> Dict[str, str]:
        resolved = {}
        for key, value in env_dict.items():
            # Resolve ${ENV_VAR} references in environment variables
            if (
                isinstance(value, str)
                and value.startswith('${')
                and value.endswith('}')
            ):
                env_name = value[2:-1]
                resolved[key] = os.getenv(env_name, value)
            else:
                resolved[key] = str(value)
        return resolved


@dataclass
class _DeployConfig:
    app_name: str
    containers: List[_ContainerConfig]
    version: str = "1.0"
    registry: Optional[RegistryConfig] = None

    def __post_init__(self):
        if not self.app_name or not self.app_name.strip():
            raise ValueError('App name cannot be empty')
        from targon.core.utils import check_object_name

        check_object_name(self.app_name)

        if not self.containers:
            raise ValueError('At least one container must be defined')

        names = [c.name for c in self.containers]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f'Duplicate container names: {", ".join(set(duplicates))}')


def _parse_replicas(data: Optional[Dict[str, Any]]) -> Optional[ReplicasConfig]:
    if not data:
        return None
    return ReplicasConfig(
        min=data.get('min', 1),
        max=data.get('max', 10),
        container_concurrency=data.get('container_concurrency', 100),
        target_concurrency=data.get('target_concurrency', 100),
        scale_to_zero=data.get('scale_to_zero', False),
    )


def _parse_registry(data: Optional[Dict[str, Any]]) -> Optional[RegistryConfig]:
    if not data:
        return None
    return RegistryConfig(
        server=data.get('server', 'https://index.docker.io/v1/'),
        username=data.get('username'),
        password=data.get('password'),
        email=data.get('email'),
    )


def _parse_container(
    data: Dict[str, Any], global_registry: Optional[RegistryConfig]
) -> _ContainerConfig:
    registry = _parse_registry(data.get('registry'))
    if registry is None and global_registry is not None:
        registry = global_registry

    return _ContainerConfig(
        name=data['name'],
        image=data['image'],
        resource=data.get('resource'),
        command=data.get('command'),
        args=data.get('args'),
        working_dir=data.get('working_dir'),
        port=data.get('port'),
        internal=data.get('internal', False),
        env=data.get('env', {}),
        replicas=_parse_replicas(data.get('replicas')),
        registry=registry,
    )


def load_config(filepath: Union[str, Path]) -> _DeployConfig:
    path = Path(filepath)

    if not path.exists():
        raise ValidationError(
            f"Configuration file not found: {path}",
            field="filepath",
            value=str(filepath),
        )

    try:
        with open(path) as f:
            content = f.read()

        # Determine format and parse
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(content)
        elif path.suffix == '.json':
            data = json.loads(content)
        else:
            # Auto-detect format
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    data = yaml.safe_load(content)
                except yaml.YAMLError as e:
                    raise ValidationError(
                        f"Could not parse file as YAML or JSON: {e}",
                        field="format",
                        value=path.suffix,
                    )

        if not isinstance(data, dict):
            raise ValidationError(
                "Configuration must be a YAML/JSON object",
                field="config",
                value=type(data).__name__,
            )

        global_registry = _parse_registry(data.get('registry'))

        containers_data = data.get('containers', [])
        if not containers_data:
            raise ValidationError(
                "At least one container must be defined",
                field="containers",
                value=[],
            )

        containers = [_parse_container(c, global_registry) for c in containers_data]

        # Create config
        return _DeployConfig(
            app_name=data['app_name'],
            containers=containers,
            version=data.get('version', '1.0'),
            registry=global_registry,
        )

    except yaml.YAMLError as e:
        raise ValidationError(
            f"Invalid YAML syntax: {e}",
            field="yaml",
            value=str(filepath),
        )
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON syntax: {e}",
            field="json",
            value=str(filepath),
        )
    except KeyError as e:
        raise ValidationError(
            f"Missing required field: {e}",
            field=str(e),
            value=None,
        )
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"Invalid configuration: {e}",
            field="config",
            value=str(filepath),
        )
    except Exception as e:
        raise ValidationError(
            f"Failed to load configuration: {e}",
            field="config",
            value=str(filepath),
        )


def _config_to_dict(config: _DeployConfig) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'version': config.version,
        'app_name': config.app_name,
    }

    if config.registry:
        reg_dict = {}
        if config.registry.server:
            reg_dict['server'] = config.registry.server
        if config.registry.username:
            reg_dict['username'] = config.registry.username
        if config.registry.password:
            reg_dict['password'] = config.registry.password
        if config.registry.email:
            reg_dict['email'] = config.registry.email
        if reg_dict:
            result['registry'] = reg_dict

    containers = []
    for c in config.containers:
        cont_dict: Dict[str, Any] = {
            'name': c.name,
            'image': c.image,
        }
        if c.resource:
            cont_dict['resource'] = c.resource
        if c.command:
            cont_dict['command'] = c.command
        if c.args:
            cont_dict['args'] = c.args
        if c.working_dir:
            cont_dict['working_dir'] = c.working_dir
        if c.port:
            cont_dict['port'] = c.port
        if c.internal:
            cont_dict['internal'] = c.internal
        if c.env:
            cont_dict['env'] = c.env
        if c.replicas:
            cont_dict['replicas'] = {
                'min': c.replicas.min,
                'max': c.replicas.max,
                'target_concurrency': c.replicas.target_concurrency,
                'scale_to_zero': c.replicas.scale_to_zero,
            }
        containers.append(cont_dict)

    result['containers'] = containers
    return result


def convert_config_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:
    config = load_config(input_path)
    output = Path(output_path)

    config_dict = _config_to_dict(config)

    with open(output, 'w') as f:
        if output.suffix in ['.yaml', '.yml']:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        else:
            json.dump(
                config_dict,
                f,
                indent=2,
            )


def config_to_serverless_requests(
    config: _DeployConfig,
    app_id: Optional[str] = None,
) -> List[CreateServerlessResourceRequest]:
    requests = []

    for container in config.containers:
        # Build environment variables
        env_vars = (
            [EnvVar(name=k, value=v) for k, v in container.env.items()]
            if container.env
            else None
        )

        registry_creds = None
        if container.registry:
            registry_creds = RegistryCredentials(
                server=container.registry.server,
                username=container.registry.username or "",
                password=container.registry.password or "",
                email=container.registry.email,
            )

        serverless_container = ServerlessContainerConfig(
            image=container.image,
            command=container.command,
            args=container.args,
            env=env_vars,
            working_dir=container.working_dir,
            registry_credentials=registry_creds,
        )

        scaling = None
        if container.replicas:
            scaling = AutoScalingConfig(
                min_replicas=container.replicas.min,
                max_replicas=container.replicas.max,
                target_concurrency=container.replicas.target_concurrency,
            )

        network = None
        if container.port:
            port_config = PortConfig(port=container.port)
            visibility = "cluster-local" if container.internal else "external"
            network = NetworkConfig(port=port_config, visibility=visibility)

        # Create request
        request = CreateServerlessResourceRequest(
            name=container.name,
            resource_name=container.resource,  # Map resource to resource_name for API
            container=serverless_container,
            scaling=scaling,
            network=network,
            app_id=app_id,
        )

        requests.append(request)

    return requests
