from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from targon.client._workload_lifecycle import (
    deploy_typed,
    get_typed,
    register_typed,
    register_workload,
)
from targon.client.workload import WorkloadResponse, _validate_non_empty
from targon.core.exceptions import ValidationError

if TYPE_CHECKING:
    from targon.client.client import Client


@dataclass(frozen=True)
class BareMetalCreateParams:
    name: str
    image: str
    resource_name: str
    region: str
    ssh_keys: Sequence[str]
    hostname: Optional[str] = None
    project_id: Optional[str] = None
    wait_until_running: bool = True


@dataclass(frozen=True)
class BareMetalReinstallParams:
    image: str
    ssh_keys: Sequence[str]
    hostname: Optional[str] = None


@dataclass(frozen=True)
class BareMetal:
    _client: "Client" = field(repr=False, compare=False)
    workload: WorkloadResponse

    def __getattr__(self, name: str) -> Any:
        return getattr(self.workload, name)

    def refresh(self) -> "BareMetal":
        return self._client.bare_metal.get(self.workload.uid)

    def deploy(
        self,
        *,
        wait: bool = True,
        timeout: float = 300,
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
    def _require_ssh_keys(ssh_keys: Sequence[str]) -> list:
        if not ssh_keys:
            raise ValidationError(
                "at least one SSH key is required",
                field="ssh_keys",
                value=ssh_keys,
            )
        return list(ssh_keys)

    @classmethod
    def _registration_payload(
        cls,
        params: BareMetalCreateParams,
    ) -> Dict[str, Any]:
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
            "ssh_keys": cls._require_ssh_keys(params.ssh_keys),
        }
        if params.hostname:
            payload["hostname"] = params.hostname
        if params.project_id:
            payload["project_id"] = params.project_id
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
        timeout: float = 300,
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
        timeout: float = 300,
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
            "ssh_keys": self._require_ssh_keys(request.ssh_keys),
        }
        if request.hostname:
            payload["hostname"] = request.hostname
        return self._operation(uid, "reinstall", payload)
