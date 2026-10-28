from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict

from targon.client.workload import WorkloadResponse, _validate_non_empty
from targon.core.exceptions import HydrationError, TimeoutError, api_error

if TYPE_CHECKING:
    from targon.client.client import Client


TERMINAL_WORKLOAD_STATES = frozenset({"error", "suspended", "deleted"})


def require_type(
    uid: str,
    workload_type: str,
    expected_type: str,
) -> None:
    if workload_type.strip().upper() == expected_type.strip().upper():
        return
    raise api_error(
        502,
        f"expected {expected_type} workload, got {workload_type!r}",
        reason="WORKLOAD_TYPE_MISMATCH",
        workload_uid=uid,
    )


def _operation_uid(data: Any) -> str:
    if not isinstance(data, dict) or not data.get("uid"):
        raise HydrationError(
            "Workload operation response is missing uid",
            object_type="WorkloadResponse",
        )
    return data["uid"]


def register_workload(client: "Client", payload: Dict[str, Any]) -> str:
    data = client.workload._request_no_retry(
        "POST",
        client.workload._path(),
        json=payload,
    )
    return _operation_uid(data)


def get_typed(client: "Client", uid: str, expected_type: str) -> WorkloadResponse:
    workload = client.workload.get(uid)
    require_type(workload.uid, workload.type, expected_type)
    return workload


def register_typed(
    client: "Client",
    payload: Dict[str, Any],
    expected_type: str,
) -> WorkloadResponse:
    return get_typed(
        client,
        register_workload(client, payload),
        expected_type,
    )


def deploy_typed(
    client: "Client",
    uid: str,
    expected_type: str,
    *,
    wait: bool,
    timeout: float,
    poll_interval: float,
) -> WorkloadResponse:
    uid = _validate_non_empty(uid, "workload_uid")
    client.workload._request_no_retry(
        "POST",
        client.workload._path(uid, "deploy"),
        _workload_uid=uid,
    )
    if not wait:
        return get_typed(client, uid, expected_type)
    return wait_typed(
        client,
        uid,
        expected_type,
        target="running",
        timeout=timeout,
        poll_interval=poll_interval,
    )


def wait_typed(
    client: "Client",
    uid: str,
    expected_type: str,
    *,
    target: str,
    timeout: float,
    poll_interval: float,
) -> WorkloadResponse:
    uid = _validate_non_empty(uid, "workload_uid")
    target = _validate_non_empty(target, "status").lower()
    deadline = time.monotonic() + timeout
    while True:
        state = client.workload.get_state(uid)
        require_type(state.uid, state.workload_type, expected_type)
        status = state.status.lower()
        if status == target:
            return get_typed(client, uid, expected_type)
        if status in TERMINAL_WORKLOAD_STATES:
            message = f"workload {uid} entered terminal state {state.status!r}" + (
                f": {state.message}" if state.message else ""
            )
            raise api_error(
                409,
                message,
                reason="WORKLOAD_INVALID_STATE",
                workload_uid=uid,
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Workload {uid} did not reach {target!r} within "
                f"{timeout:.0f}s (last status: {state.status!r})",
                timeout=timeout,
            )
        time.sleep(poll_interval)
