from typing import TYPE_CHECKING, Any, Dict, Optional

from targon.client.constants import org_path
from targon.client.sandbox.capabilities import (
    SandboxFilesClient,
    SandboxHTTPClient,
    SandboxTerminalsClient,
    poll,
)
from targon.client.sandbox.models import (
    MAX_COMMAND_BYTES,
    MAX_LIST_LIMIT,
    TEMPLATE_NAME_RE,
    AccessTicket,
    DesktopInfo,
    ExecResult,
    ForkRequest,
    ListPage,
    PublishRequest,
    SandboxConfigInput,
    SandboxCreateParams,
    SandboxState,
    SandboxStatus,
    SandboxSummary,
    SandboxTemplateKind,
    SandboxTemplateStatus,
    SandboxUpdateParams,
    bounded_int,
)
from targon.client.sandbox.models import defined as _defined
from targon.client.sandbox.models import (
    invalid,
    non_empty,
    ports_payload,
    validate_name,
    validate_timeouts,
    validate_update_timeouts,
)
from targon.client.sandbox.resources import Sandbox, SandboxTemplate
from targon.core.exceptions import (
    SandboxStateError,
    SandboxTemplateError,
    ValidationError,
)

if TYPE_CHECKING:
    from targon.client.client import Client


class SandboxTemplatesClient(SandboxHTTPClient):
    def _path(self, uid: Optional[str] = None) -> str:
        path = org_path(self.client.require_org(), "sandbox-templates")
        return f"{path}/{uid}" if uid else path

    def _hydrate(self, data: Dict[str, Any]) -> SandboxTemplate:
        return SandboxTemplate._from_dict(self.client, data)

    def list(
        self,
        *,
        kind: Optional[SandboxTemplateKind] = None,
        status: Optional[SandboxTemplateStatus] = None,
        limit: int = MAX_LIST_LIMIT,
        cursor: Optional[str] = None,
    ) -> ListPage[SandboxTemplate]:
        params = _defined(
            limit=bounded_int(limit, "limit", 1, MAX_LIST_LIMIT),
            kind=SandboxTemplateKind(kind).value if kind is not None else None,
            status=SandboxTemplateStatus(status).value if status is not None else None,
            cursor=cursor,
        )
        data = self._get(self._path(), params=params)
        return ListPage(
            tuple(self._hydrate(item) for item in data["items"]),
            data.get("next_cursor"),
        )

    def get(self, uid: str) -> SandboxTemplate:
        return self._hydrate(self._get(self._path(non_empty(uid, "uid"))))

    def update(
        self,
        uid: str,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SandboxTemplate:
        if display_name is None and description is None:
            raise invalid("display_name or description is required", "display_name")
        if display_name is not None and len(display_name.strip()) > 128:
            raise invalid("display_name must be at most 128 characters", "display_name")
        payload = _defined(
            display_name=display_name.strip() if display_name is not None else None,
            description=description.strip() if description is not None else None,
        )
        return self._hydrate(
            self._request_no_retry(
                "PATCH", self._path(non_empty(uid, "uid")), json=payload
            )
        )

    def delete(self, uid: str) -> None:
        self._request_no_retry("DELETE", self._path(non_empty(uid, "uid")))


class SandboxesClient(SandboxHTTPClient):
    def __init__(self, client: "Client") -> None:
        super().__init__(client)
        self.templates = SandboxTemplatesClient(client)
        self.files = SandboxFilesClient(client)
        self.terminals = SandboxTerminalsClient(client)

    def _hydrate(self, data: Dict[str, Any]) -> Sandbox:
        return Sandbox._from_dict(self.client, data)

    @staticmethod
    def _operation_uid(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict) or not data.get("uid"):
            raise ValidationError("sandbox operation is missing uid", field="uid")
        if data.get("type") not in (None, "", "SANDBOX"):
            raise invalid(
                "workload operation is not a SANDBOX", "type", data.get("type")
            )
        return data["uid"]

    def _wait_or_get(
        self,
        workload_uid: str,
        target: SandboxStatus,
        wait: bool,
        timeout: float,
        poll_interval: float,
    ) -> Sandbox:
        if not wait:
            return self.get(workload_uid)
        return self.wait_for_status(
            workload_uid,
            target,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def create(
        self,
        request: SandboxCreateParams,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        name = validate_name(request.name)
        if not isinstance(request.template, SandboxTemplate):
            raise invalid(
                "template must be a SandboxTemplate resource",
                "template",
                request.template,
            )
        request.template._require_client()
        template_uid = non_empty(request.template.uid, "template.uid")
        if request.template.status is not SandboxTemplateStatus.READY:
            raise invalid(
                "template must be READY before creating a sandbox",
                "template.status",
                request.template.status.value,
            )
        validate_timeouts(request.ttl_sec, request.idle_timeout_sec)
        config = SandboxConfigInput(
            request.ttl_sec, request.idle_timeout_sec
        ).to_payload()
        payload: Dict[str, Any] = {
            "type": "SANDBOX",
            "name": name,
            "image": template_uid,
        }
        payload.update(
            _defined(
                project_id=request.project_id,
                ssh_keys=list(request.ssh_keys) if request.ssh_keys else None,
                ports=ports_payload(request.ports) if request.ports else None,
                sandbox_config=config or None,
            )
        )
        created_uid = self._operation_uid(
            self._request_no_retry("POST", self._workloads_path(), json=payload)
        )
        self._request_no_retry(
            "POST",
            self._workloads_path(created_uid, "deploy"),
            workload_uid=created_uid,
        )
        return self._wait_or_get(
            created_uid,
            SandboxStatus.RUNNING,
            request.wait_until_running,
            timeout,
            poll_interval,
        )

    def get(self, workload_uid: str) -> Sandbox:
        workload_uid = non_empty(workload_uid, "workload_uid")
        return self._hydrate(
            self._request_for(workload_uid, "GET", self._workloads_path(workload_uid))
        )

    def list(
        self,
        *,
        status: Optional[SandboxStatus] = None,
        project_id: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = MAX_LIST_LIMIT,
        cursor: Optional[str] = None,
    ) -> ListPage[SandboxSummary]:
        params = _defined(
            type="SANDBOX",
            limit=bounded_int(limit, "limit", 1, MAX_LIST_LIMIT),
            status=SandboxStatus(status).value if status is not None else None,
            project_id=project_id,
            name=name,
            cursor=cursor,
        )
        data = self._get(self._workloads_path(), params=params)
        return ListPage(
            items=tuple(SandboxSummary.from_dict(item) for item in data["items"]),
            next_cursor=data.get("next_cursor"),
        )

    def get_state(self, workload_uid: str) -> SandboxState:
        workload_uid = non_empty(workload_uid, "workload_uid")
        data = self._request_for(
            workload_uid,
            "GET",
            self._workloads_path(workload_uid, "state"),
        )
        if not isinstance(data, dict) or data.get("workload_type") != "SANDBOX":
            value = data.get("workload_type") if isinstance(data, dict) else None
            raise invalid("workload state is not for a SANDBOX", "workload_type", value)
        state = SandboxState.from_dict(data)
        if state is None:
            raise SandboxStateError(
                502,
                "Sandbox state response is missing status",
                workload_uid=workload_uid,
            )
        return state

    def update(self, workload_uid: str, request: SandboxUpdateParams) -> Sandbox:
        workload_uid = non_empty(workload_uid, "workload_uid")
        payload = _defined(
            name=validate_name(request.name) if request.name is not None else None,
            project_id=request.project_id,
            ssh_keys=list(request.ssh_keys) if request.ssh_keys is not None else None,
            ports=ports_payload(request.ports) if request.ports is not None else None,
        )
        if request.sandbox_config is not None:
            validate_update_timeouts(
                request.sandbox_config.ttl_sec,
                request.sandbox_config.idle_timeout_sec,
            )
            payload["sandbox_config"] = request.sandbox_config.to_payload()
        if not payload:
            raise ValidationError("update requires at least one field", field="request")
        return self._hydrate(
            self._request_no_retry(
                "PATCH",
                self._workloads_path(workload_uid),
                workload_uid=workload_uid,
                json=payload,
            )
        )

    def delete(self, workload_uid: str) -> None:
        workload_uid = non_empty(workload_uid, "workload_uid")
        self._request_no_retry(
            "DELETE",
            self._workloads_path(workload_uid),
            workload_uid=workload_uid,
        )

    def attach_ssh_key(self, workload_uid: str, key_uid: str) -> None:
        workload_uid = non_empty(workload_uid, "workload_uid")
        key_uid = non_empty(key_uid, "key_uid")
        self._request_no_retry(
            "PUT",
            self._workloads_path(workload_uid, "ssh-keys", key_uid),
            workload_uid=workload_uid,
        )

    def detach_ssh_key(self, workload_uid: str, key_uid: str) -> None:
        workload_uid = non_empty(workload_uid, "workload_uid")
        key_uid = non_empty(key_uid, "key_uid")
        self._request_no_retry(
            "DELETE",
            self._workloads_path(workload_uid, "ssh-keys", key_uid),
            workload_uid=workload_uid,
        )

    def _mutate_and_wait(
        self,
        workload_uid: str,
        action: str,
        target: SandboxStatus,
        wait: bool,
        timeout: float,
        poll_interval: float,
    ) -> Sandbox:
        workload_uid = non_empty(workload_uid, "workload_uid")

        self._request_no_retry(
            "POST",
            self._workloads_path(workload_uid, action),
            workload_uid=workload_uid,
        )
        return self._wait_or_get(workload_uid, target, wait, timeout, poll_interval)

    def freeze(
        self,
        workload_uid: str,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        return self._mutate_and_wait(
            workload_uid, "freeze", SandboxStatus.FROZEN, wait, timeout, poll_interval
        )

    def thaw(
        self,
        workload_uid: str,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        return self._mutate_and_wait(
            workload_uid, "thaw", SandboxStatus.RUNNING, wait, timeout, poll_interval
        )

    def fork(
        self,
        workload_uid: str,
        request: Optional[ForkRequest] = None,
        *,
        wait: bool = True,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        workload_uid = non_empty(workload_uid, "workload_uid")
        request = request or ForkRequest()
        payload = _defined(
            name=validate_name(request.name) if request.name is not None else None,
            project_id=request.project_id,
        )
        if request.sandbox_config is not None:
            payload["sandbox_config"] = request.sandbox_config.to_payload()
        child_uid = self._operation_uid(
            self._request_no_retry(
                "POST",
                self._workloads_path(workload_uid, "fork"),
                workload_uid=workload_uid,
                json=payload,
            )
        )
        return self._wait_or_get(
            child_uid, SandboxStatus.RUNNING, wait, timeout, poll_interval
        )

    def publish(
        self,
        workload_uid: str,
        request: PublishRequest,
        *,
        wait: bool = True,
        timeout: float = 600,
        poll_interval: float = 5,
    ) -> SandboxTemplate:
        workload_uid = non_empty(workload_uid, "workload_uid")
        name = non_empty(request.name, "name")
        if not TEMPLATE_NAME_RE.fullmatch(name):
            raise invalid(
                "name must be 1-64 letters, digits, '.', '_' or '-', starting "
                "with a letter or digit",
                "name",
                name,
            )
        if request.display_name is not None:
            if len(request.display_name.strip()) > 128:
                raise invalid(
                    "display_name must be at most 128 characters", "display_name"
                )
        payload: Dict[str, str] = {"name": name}
        if request.display_name is not None:
            payload["display_name"] = request.display_name.strip()
        if request.description is not None:
            payload["description"] = request.description.strip()

        template = self.templates._hydrate(
            self._request_no_retry(
                "POST",
                self._workloads_path(workload_uid, "publish"),
                workload_uid=workload_uid,
                json=payload,
            )
        )
        if not wait:
            return template

        def check_failure(current: SandboxTemplate) -> None:
            if current.status is SandboxTemplateStatus.FAILED:
                raise SandboxTemplateError(
                    409,
                    current.status_message or "Sandbox template publishing failed",
                    reason="WORKLOAD_SANDBOX_TEMPLATE_FAILED",
                    workload_uid=workload_uid,
                )

        return poll(
            lambda: self.templates.get(template.uid),
            is_done=lambda current: current.status is SandboxTemplateStatus.READY,
            check_failure=check_failure,
            timeout=timeout,
            poll_interval=poll_interval,
            timeout_message=lambda current: (
                f"Sandbox template {current.uid} was not ready within {timeout:.0f}s"
            ),
        )

    def wait_for_status(
        self,
        workload_uid: str,
        status: SandboxStatus,
        *,
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> Sandbox:
        workload_uid = non_empty(workload_uid, "workload_uid")
        target = SandboxStatus(status)
        terminal = {
            SandboxStatus.ERROR,
            SandboxStatus.SUSPENDED,
            SandboxStatus.DELETED,
        }

        def check_failure(state: SandboxState) -> None:
            if state.status in terminal and state.status is not target:
                raise SandboxStateError(
                    409,
                    f"Sandbox entered {state.status.value} while waiting for "
                    f"{target.value}: {state.message}",
                    reason="WORKLOAD_SANDBOX_INVALID_STATE",
                    workload_uid=workload_uid,
                )

        poll(
            lambda: self.get_state(workload_uid),
            is_done=lambda state: state.status is target,
            check_failure=check_failure,
            timeout=timeout,
            poll_interval=poll_interval,
            timeout_message=lambda state: (
                f"Sandbox {workload_uid} did not reach {target.value} within "
                f"{timeout:.0f}s (last status: {state.status.value})"
            ),
        )
        return self.get(workload_uid)

    def exec(self, workload_uid: str, cmd: str, *, timeout_sec: int = 60) -> ExecResult:
        workload_uid = non_empty(workload_uid, "workload_uid")
        non_empty(cmd, "cmd")
        command_bytes = len(cmd.encode("utf-8"))
        if command_bytes > MAX_COMMAND_BYTES:
            raise ValidationError(
                "cmd exceeds 64 KiB", field="cmd", value=command_bytes
            )
        bounded_int(timeout_sec, "timeout_sec", 1, 600)
        return ExecResult.from_dict(
            self._request_no_retry(
                "POST",
                self._workloads_path(workload_uid, "exec"),
                workload_uid=workload_uid,
                json={"cmd": cmd, "timeout_sec": timeout_sec},
            )
        )

    def mint_access_ticket(
        self, workload_uid: str, *, ttl_sec: int = 60
    ) -> AccessTicket:
        workload_uid = non_empty(workload_uid, "workload_uid")
        bounded_int(ttl_sec, "ttl_sec", 1, 300)
        data = self._request_no_retry(
            "POST",
            self._workloads_path(workload_uid, "access-tickets"),
            workload_uid=workload_uid,
            json={"ttl_sec": ttl_sec},
        )
        return AccessTicket(ticket=data["ticket"], expires_at=data["expires_at"])

    def get_desktop(self, workload_uid: str) -> DesktopInfo:
        workload_uid = non_empty(workload_uid, "workload_uid")
        data = self._request_for(
            workload_uid,
            "GET",
            self._workloads_path(workload_uid, "desktop"),
        )
        return DesktopInfo(
            available=bool(data.get("available", False)),
            port=data.get("port"),
            listening=data.get("listening"),
            ws_url=data.get("ws_url"),
        )
