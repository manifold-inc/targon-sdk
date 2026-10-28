from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from targon.client._workload_lifecycle import (
    deploy_typed,
    get_typed,
    register_typed,
    register_workload,
    require_type,
)
from targon.client.workload import (
    PortConfig,
    WorkloadResponse,
    _validate_non_empty,
)
from targon.core.exceptions import ValidationError

if TYPE_CHECKING:
    from targon.client.client import Client


def _ports_payload(ports: Sequence[PortConfig]) -> list:
    payload = []
    for port in ports:
        if not isinstance(port, PortConfig):
            raise ValidationError(
                "ports must contain PortConfig instances",
                field="ports",
                value=port,
            )
        payload.append(port.to_payload())
    return payload


@dataclass(frozen=True)
class VMCreateParams:
    name: str
    image: str
    resource_name: str
    password: str
    project_id: Optional[str] = None
    reservation_uid: Optional[str] = None
    ssh_keys: Sequence[str] = ()
    ports: Sequence[PortConfig] = ()
    wait_until_running: bool = True


@dataclass(frozen=True)
class VMUpdateParams:
    name: Optional[str] = None
    ports: Optional[Sequence[PortConfig]] = None


@dataclass(frozen=True)
class VM:
    _client: "Client" = field(repr=False, compare=False)
    workload: WorkloadResponse

    def __getattr__(self, name: str) -> Any:
        return getattr(self.workload, name)

    def refresh(self) -> "VM":
        return self._client.vms.get(self.workload.uid)

    def deploy(
        self,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> "VM":
        return self._client.vms.deploy(
            self.workload.uid,
            wait=wait,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def update(self, request: VMUpdateParams) -> "VM":
        return self._client.vms.update(self.workload.uid, request)

    def reboot(self) -> "VM":
        return self._client.vms.reboot(self.workload.uid)


class VMClient:
    def __init__(self, client: "Client") -> None:
        self.client = client

    def _hydrate(self, workload: WorkloadResponse) -> VM:
        return VM(self.client, workload)

    @staticmethod
    def _registration_payload(params: VMCreateParams) -> Dict[str, Any]:
        if not isinstance(params, VMCreateParams):
            raise ValidationError(
                "params must be VMCreateParams",
                field="params",
                value=params,
            )
        password = _validate_non_empty(params.password, "vm_config.password")
        if len(password) < 4:
            raise ValidationError(
                "vm_config.password must be at least 4 characters",
                field="vm_config.password",
                value=params.password,
            )
        payload: Dict[str, Any] = {
            "type": "VM",
            "name": _validate_non_empty(params.name, "name"),
            "image": _validate_non_empty(params.image, "image"),
            "resource_name": _validate_non_empty(
                params.resource_name,
                "resource_name",
            ),
            "vm_config": {"password": password},
        }
        if params.project_id is not None:
            payload["project_id"] = params.project_id
        if params.reservation_uid is not None:
            payload["reservation_uid"] = params.reservation_uid
        if params.ssh_keys:
            payload["ssh_keys"] = list(params.ssh_keys)
        if params.ports:
            payload["ports"] = _ports_payload(params.ports)
        return payload

    def register(self, params: VMCreateParams) -> VM:
        workload = register_typed(
            self.client,
            self._registration_payload(params),
            "VM",
        )
        return self._hydrate(workload)

    def deploy(
        self,
        uid: str,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> VM:
        return self._hydrate(
            deploy_typed(
                self.client,
                uid,
                "VM",
                wait=wait,
                timeout=timeout,
                poll_interval=poll_interval,
            )
        )

    def create(
        self,
        params: VMCreateParams,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> VM:
        uid = register_workload(
            self.client,
            self._registration_payload(params),
        )
        return self.deploy(
            uid,
            wait=params.wait_until_running,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def get(self, uid: str) -> VM:
        return self._hydrate(get_typed(self.client, uid, "VM"))

    def update(self, uid: str, request: VMUpdateParams) -> VM:
        uid = _validate_non_empty(uid, "workload_uid")
        if not isinstance(request, VMUpdateParams):
            raise ValidationError(
                "request must be VMUpdateParams",
                field="request",
                value=request,
            )
        payload: Dict[str, Any] = {}
        if request.name is not None:
            payload["name"] = _validate_non_empty(request.name, "name")
        if request.ports is not None:
            payload["ports"] = _ports_payload(request.ports)
        if not payload:
            raise ValidationError(
                "at least one VM field must be supplied",
                field="update",
            )
        data = self.client.workload._request_no_retry(
            "PATCH",
            self.client.workload._path(uid),
            _workload_uid=uid,
            json=payload,
        )
        return self._hydrate(self._typed_response(data))

    def reboot(self, uid: str) -> VM:
        uid = _validate_non_empty(uid, "workload_uid")
        self.client.workload._request_no_retry(
            "POST",
            self.client.workload._path(uid, "reboot"),
            _workload_uid=uid,
        )
        return self.get(uid)

    @staticmethod
    def _typed_response(data: Any) -> WorkloadResponse:
        workload = WorkloadResponse.from_dict(data)
        require_type(workload.uid, workload.type, "VM")
        return workload
