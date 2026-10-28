from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from targon.client._workload_lifecycle import (
    deploy_typed,
    get_typed,
    register_typed,
    register_workload,
)
from targon.client.workload import (
    BMConfig,
    BMConfigInput,
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

#: Create/deploy wait timeout in seconds when the caller does not pass one.
#: Bare Metal provisioning is much slower than VM or Sandbox provisioning.
DEFAULT_BARE_METAL_WAIT_TIMEOUT = 30 * 60

BareMetalImage = WorkloadImage


@dataclass(frozen=True)
class BareMetalCreateParams:
    name: str
    image: str
    resource_name: str
    region: str
    ssh_keys: Sequence[str]
    bm_config: Optional[BMConfigInput] = None
    project_id: Optional[str] = None
    reservation_uid: Optional[str] = None
    wait_until_running: bool = True


@dataclass(frozen=True)
class BareMetalReinstallParams:
    """Reinstall request. Empty ``ssh_keys`` reuses the attached keys."""

    image: str
    ssh_keys: Sequence[str] = ()
    hostname: Optional[str] = None


@dataclass(frozen=True)
class BareMetal(_WorkloadFields):
    """A Bare Metal workload bound to the client that fetched it.

    Workload fields are exposed as read-only properties; the full response is
    available as :attr:`workload`.
    """

    _client: "Client" = field(repr=False, compare=False)
    workload: WorkloadResponse

    @property
    def bm_config(self) -> Optional[BMConfig]:
        return self.workload.bm_config

    def refresh(self) -> "BareMetal":
        return self._client.bare_metal.get(self.workload.uid)

    def deploy(
        self,
        *,
        wait: bool = True,
        timeout: float = DEFAULT_BARE_METAL_WAIT_TIMEOUT,
        poll_interval: float = 5,
    ) -> "BareMetal":
        return self._client.bare_metal.deploy(
            self.workload.uid,
            wait=wait,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def reboot(self) -> "BareMetal":
        return self._client.bare_metal.reboot(self.workload.uid)

    def power_on(self) -> "BareMetal":
        return self._client.bare_metal.power_on(self.workload.uid)

    def power_off(self) -> "BareMetal":
        return self._client.bare_metal.power_off(self.workload.uid)

    def reinstall(self, request: BareMetalReinstallParams) -> "BareMetal":
        return self._client.bare_metal.reinstall(self.workload.uid, request)


class BareMetalClient:
    def __init__(self, client: "Client") -> None:
        self.client = client

    def _hydrate(self, workload: WorkloadResponse) -> BareMetal:
        return BareMetal(self.client, workload)

    @staticmethod
    def _registration_payload(params: BareMetalCreateParams) -> Dict[str, Any]:
        if not isinstance(params, BareMetalCreateParams):
            raise ValidationError(
                "params must be BareMetalCreateParams",
                field="params",
                value=params,
            )
        payload: Dict[str, Any] = {
            "type": "BM",
            "name": _validate_non_empty(params.name, "name"),
            "image": _validate_non_empty(params.image, "image"),
            "resource_name": _validate_non_empty(
                params.resource_name,
                "resource_name",
            ),
            "region": _validate_non_empty(params.region, "region"),
        }
        ssh_keys = _ssh_keys_payload(params.ssh_keys)
        if not ssh_keys:
            raise ValidationError(
                "at least one SSH key is required",
                field="ssh_keys",
                value=params.ssh_keys,
            )
        payload["ssh_keys"] = ssh_keys
        if params.bm_config is not None:
            if not isinstance(params.bm_config, BMConfigInput):
                raise ValidationError(
                    "bm_config must be BMConfigInput",
                    field="bm_config",
                    value=params.bm_config,
                )
            hostname = _optional_hostname(
                params.bm_config.hostname, "bm_config.hostname"
            )
            if hostname is not None:
                payload["bm_config"] = {"hostname": hostname}
        project_id = _optional_str(params.project_id, "project_id")
        if project_id is not None:
            payload["project_id"] = project_id
        reservation_uid = _optional_str(params.reservation_uid, "reservation_uid")
        if reservation_uid is not None:
            payload["reservation_uid"] = reservation_uid
        return payload

    def register(self, params: BareMetalCreateParams) -> BareMetal:
        workload = register_typed(
            self.client,
            self._registration_payload(params),
            "BM",
        )
        return self._hydrate(workload)

    def deploy(
        self,
        uid: str,
        *,
        wait: bool = True,
        timeout: float = DEFAULT_BARE_METAL_WAIT_TIMEOUT,
        poll_interval: float = 5,
    ) -> BareMetal:
        return self._hydrate(
            deploy_typed(
                self.client,
                uid,
                "BM",
                wait=wait,
                timeout=timeout,
                poll_interval=poll_interval,
            )
        )

    def create(
        self,
        params: BareMetalCreateParams,
        *,
        timeout: float = DEFAULT_BARE_METAL_WAIT_TIMEOUT,
        poll_interval: float = 5,
    ) -> BareMetal:
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

    def get(self, uid: str) -> BareMetal:
        return self._hydrate(get_typed(self.client, uid, "BM"))

    def _operation(
        self,
        uid: str,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> BareMetal:
        uid = _validate_non_empty(uid, "workload_uid")
        kwargs = {"json": payload} if payload is not None else {}
        self.client.workload._request_no_retry(
            "POST",
            self.client.workload._path(uid, action),
            _workload_uid=uid,
            **kwargs,
        )
        return self.get(uid)

    def reboot(self, uid: str) -> BareMetal:
        return self._operation(uid, "reboot")

    def power_on(self, uid: str) -> BareMetal:
        return self._operation(uid, "power-on")

    def power_off(self, uid: str) -> BareMetal:
        return self._operation(uid, "power-off")

    def reinstall(
        self,
        uid: str,
        request: BareMetalReinstallParams,
    ) -> BareMetal:
        if not isinstance(request, BareMetalReinstallParams):
            raise ValidationError(
                "request must be BareMetalReinstallParams",
                field="request",
                value=request,
            )
        payload: Dict[str, Any] = {
            "image": _validate_non_empty(request.image, "image"),
        }
        ssh_keys = _ssh_keys_payload(request.ssh_keys)
        if ssh_keys:
            payload["ssh_keys"] = ssh_keys
        hostname = _optional_hostname(request.hostname, "hostname")
        if hostname is not None:
            payload["hostname"] = hostname
        return self._operation(uid, "reinstall", payload)

    def images(self) -> List[BareMetalImage]:
        """List the images Bare Metal workloads can install."""
        return self.client.workload._images("bm-images")
