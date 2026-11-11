from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence

from targon.client._workload_lifecycle import (
    deploy_typed,
    get_typed,
    register_typed,
    register_workload,
    require_type,
)
from targon.client.workload import (
    PortConfig,
    VMConfig,
    VMConfigInput,
    WorkloadImage,
    WorkloadResponse,
    _optional_hostname,
    _optional_str,
    _ssh_keys_payload,
    _validate_non_empty,
    _WorkloadFields,
)
from targon.core.exceptions import ValidationError

if TYPE_CHECKING:
    from targon.client.client import Client

VALID_LOG_TYPES = frozenset({"serial", "qemu"})
VM_PORT_PROTOCOLS = frozenset({"TCP", "UDP"})

VMImage = WorkloadImage


def _ports_payload(ports: Sequence[PortConfig]) -> list:
    payload = []
    seen = set()
    for port in ports:
        if not isinstance(port, PortConfig):
            raise ValidationError(
                "ports must contain PortConfig instances",
                field="ports",
                value=port,
            )
        if (
            isinstance(port.port, bool)
            or not isinstance(port.port, int)
            or not 1 <= port.port <= 65535
        ):
            raise ValidationError(
                "VM port must be between 1 and 65535",
                field="ports",
                value=port.port,
            )
        if port.port == 22:
            raise ValidationError(
                "VM port 22 is reserved for SSH and cannot be forwarded",
                field="ports",
                value=port.port,
            )
        if port.protocol is not None and not isinstance(port.protocol, str):
            raise ValidationError(
                "VM port protocol must be a string",
                field="ports",
                value=port.protocol,
            )
        protocol = (port.protocol or "").strip().upper() or "TCP"
        if protocol not in VM_PORT_PROTOCOLS:
            raise ValidationError(
                "VM ports only support TCP or UDP",
                field="ports",
                value=port.protocol,
            )
        key = (port.port, protocol)
        if key in seen:
            raise ValidationError(
                "Duplicate VM port and protocol combination",
                field="ports",
                value=port.port,
            )
        seen.add(key)
        # VM ports are direct forwards; the backend discards routing.
        payload.append({"port": port.port, "protocol": protocol})
    return payload


def _log_params(
    *,
    since: Optional[str],
    tail: Optional[int],
    previous: bool,
    log_type: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if since:
        params["since"] = since
    if tail is not None:
        params["tail"] = tail
    if previous:
        params["previous"] = "true"
    if log_type is not None:
        normalized = _validate_non_empty(log_type, "log_type").lower()
        if normalized not in VALID_LOG_TYPES:
            raise ValidationError(
                "log_type must be one of qemu, serial",
                field="log_type",
                value=log_type,
            )
        params["type"] = normalized
    return params


@dataclass(frozen=True)
class VMCreateParams:
    name: str
    image: str
    resource_name: str
    vm_config: Optional[VMConfigInput] = None
    project_id: Optional[str] = None
    reservation_uid: Optional[str] = None
    ssh_keys: Sequence[str] = ()
    # Forwarded TCP/UDP ports. Port 22 is reserved for SSH.
    ports: Sequence[PortConfig] = ()
    wait_until_running: bool = True


@dataclass(frozen=True)
class VMUpdateParams:
    name: Optional[str] = None
    ports: Optional[Sequence[PortConfig]] = None


@dataclass(frozen=True)
class VM(_WorkloadFields):
    """A VM workload bound to the client that fetched it.

    Workload fields are exposed as read-only properties; the full response is
    available as :attr:`workload`.
    """

    _client: "Client" = field(repr=False, compare=False)
    workload: WorkloadResponse

    @property
    def vm_config(self) -> Optional[VMConfig]:
        return self.workload.vm_config

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

    def get_logs(
        self,
        *,
        since: Optional[str] = None,
        tail: Optional[int] = None,
        previous: bool = False,
        log_type: Optional[str] = None,
    ) -> str:
        return self._client.vms.get_logs(
            self.workload.uid,
            since=since,
            tail=tail,
            previous=previous,
            log_type=log_type,
        )

    def stream_logs(
        self,
        *,
        since: Optional[str] = None,
        tail: Optional[int] = None,
        previous: bool = False,
        log_type: Optional[str] = None,
    ) -> Iterator[str]:
        return self._client.vms.stream_logs(
            self.workload.uid,
            since=since,
            tail=tail,
            previous=previous,
            log_type=log_type,
        )


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
        payload: Dict[str, Any] = {
            "type": "VM",
            "name": _validate_non_empty(params.name, "name"),
            "image": _validate_non_empty(params.image, "image"),
            "resource_name": _validate_non_empty(
                params.resource_name,
                "resource_name",
            ),
        }
        if params.vm_config is not None:
            if not isinstance(params.vm_config, VMConfigInput):
                raise ValidationError(
                    "vm_config must be VMConfigInput",
                    field="vm_config",
                    value=params.vm_config,
                )
            hostname = _optional_hostname(
                params.vm_config.hostname, "vm_config.hostname"
            )
            if hostname is not None:
                payload["vm_config"] = {"hostname": hostname}
        project_id = _optional_str(params.project_id, "project_id")
        if project_id is not None:
            payload["project_id"] = project_id
        reservation_uid = _optional_str(params.reservation_uid, "reservation_uid")
        if reservation_uid is not None:
            payload["reservation_uid"] = reservation_uid
        ssh_keys = _ssh_keys_payload(params.ssh_keys)
        if ssh_keys:
            payload["ssh_keys"] = ssh_keys
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

    def get_logs(
        self,
        uid: str,
        *,
        since: Optional[str] = None,
        tail: Optional[int] = None,
        previous: bool = False,
        log_type: Optional[str] = None,
    ) -> str:
        """Fetch buffered VM logs. ``log_type`` is ``serial`` or ``qemu``."""
        uid = _validate_non_empty(uid, "workload_uid")
        params = _log_params(
            since=since, tail=tail, previous=previous, log_type=log_type
        )
        result = self.client.workload._get(
            self.client.workload._path(uid, "logs"),
            params=params or None,
            _workload_uid=uid,
        )
        return result if isinstance(result, str) else str(result)

    def stream_logs(
        self,
        uid: str,
        *,
        since: Optional[str] = None,
        tail: Optional[int] = None,
        previous: bool = False,
        log_type: Optional[str] = None,
    ) -> Iterator[str]:
        """Follow VM logs, yielding one line at a time."""
        uid = _validate_non_empty(uid, "workload_uid")
        params = _log_params(
            since=since, tail=tail, previous=previous, log_type=log_type
        )
        params["follow"] = "true"
        return self.client.workload._stream_text(
            self.client.workload._path(uid, "logs"), params, uid
        )

    def images(self) -> List[VMImage]:
        """List the images VM workloads can boot."""
        return self.client.workload._images("vm-images")

    @staticmethod
    def _typed_response(data: Any) -> WorkloadResponse:
        workload = WorkloadResponse.from_dict(data)
        require_type(workload.uid, workload.type, "VM")
        return workload
