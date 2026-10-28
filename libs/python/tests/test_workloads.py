import importlib.util

import pytest
from conftest import Response

import targon
import targon.client as client_exports
import targon.client.workload as workload_models
from targon import Client
from targon.core.exceptions import TargonAPIError
from targon.core.exceptions import TimeoutError as TargonTimeoutError


def test_deprecated_resources_are_not_exported():
    client = Client(api_key="test", org="acme")
    try:
        assert not hasattr(client, "volume")
        assert not hasattr(client, "rental")
        # Generic mutations stay removed. Logs and images moved to the typed
        # VM / Bare Metal clients because the backend rejects them for other
        # workload types.
        for method in (
            "create",
            "update",
            "deploy",
            "suspend",
            "reboot",
            "get_logs",
            "stream_logs",
            "exec_stream",
            "exec",
            "vm_images",
            "attach_volume",
            "detach_volume",
        ):
            assert not hasattr(client.workload, method)
    finally:
        client.close()


def test_common_workload_surface_is_preserved():
    client = Client(api_key="test", org="acme")
    try:
        for method in (
            "get",
            "list",
            "delete",
            "get_state",
            "get_events",
            "wait_until_ready",
            "verify",
            "attach_ssh_key",
            "detach_ssh_key",
        ):
            assert callable(getattr(client.workload, method))
        for method in ("reboot", "get_logs", "stream_logs", "images"):
            assert callable(getattr(client.vms, method))
        for method in (
            "reboot",
            "power_on",
            "power_off",
            "reinstall",
            "images",
        ):
            assert callable(getattr(client.bare_metal, method))
        for method in ("power_on", "power_off", "reinstall"):
            assert not hasattr(client.vms, method)
        for method in ("get_logs", "stream_logs"):
            assert not hasattr(client.bare_metal, method)
    finally:
        client.close()


def test_deprecated_workload_models_are_not_exported():
    removed_names = (
        "CreateWorkloadRequest",
        "UpdateWorkloadRequest",
        "EnvVar",
        "RegistryAuth",
        "VmConfig",
        "VolumeMount",
        "WorkloadVolumeMount",
        "WorkloadDeployResponse",
        "WorkloadOperationResponse",
        "VolumeMountResponse",
        "ExecResponse",
        "VmImage",
        "Volume",
        "VolumeClient",
        "VolumeCreateResponse",
        "VolumeEvent",
        "VolumeEventsResponse",
        "VolumeListResponse",
        "VolumeState",
        "VolumeStateResponse",
    )
    for name in removed_names:
        assert not hasattr(workload_models, name)
        assert not hasattr(client_exports, name)
        assert not hasattr(targon, name)


def test_volume_module_and_response_fields_are_removed():
    assert importlib.util.find_spec("targon.client.volume") is None

    workload = workload_models.WorkloadResponse.from_dict(
        {
            "uid": "wrk-1",
            "volumes": [{"uid": "vol-1"}],
        }
    )
    item = workload_models.WorkloadListItem.from_dict(
        {
            "uid": "wrk-1",
            "volumes": [{"uid": "vol-1"}],
        }
    )

    assert not hasattr(workload, "volumes")
    assert not hasattr(item, "volumes")


def test_workload_events_do_not_expose_backend_container_fields():
    event = workload_models.WorkloadEvent.from_dict(
        {
            "workload_uid": "wrk-1",
            "pod_name": "backend-pod",
            "container_name": "backend-container",
            "container_image": "backend-image",
        }
    )

    assert not hasattr(event, "pod_name")
    assert not hasattr(event, "container_name")
    assert not hasattr(event, "container_image")


@pytest.mark.parametrize("workload_type", ("VM", "BM", "SANDBOX"))
def test_common_workload_operations_keep_org_scoped_routes(client, workload_type):
    client.session.request.side_effect = [
        Response(body={"uid": "wrk-1", "type": workload_type}),
        Response(
            body={
                "items": [{"uid": "wrk-1", "type": workload_type}],
                "next_cursor": "next",
            }
        ),
        Response(body=None),
        Response(
            body={
                "uid": "wrk-1",
                "workload_type": workload_type,
                "status": "running",
            }
        ),
        Response(
            body={"items": [{"workload_uid": "wrk-1", "workload_type": workload_type}]}
        ),
    ]

    assert client.workload.get("wrk-1").type == workload_type
    page = client.workload.list(limit=10)
    assert page.next_cursor == "next"
    assert page.items[0].type == workload_type
    assert client.workload.delete("wrk-1") is None
    state = client.workload.get_state("wrk-1")
    assert state.status == "running"
    assert state.workload_type == workload_type
    event = client.workload.get_events("wrk-1").items[0]
    assert event.workload_uid == "wrk-1"
    assert event.workload_type == workload_type

    calls = client.session.request.call_args_list
    assert [call.args[0] for call in calls] == [
        "GET",
        "GET",
        "DELETE",
        "GET",
        "GET",
    ]
    assert [call.args[1] for call in calls] == [
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-1",
        "https://api.test/tha/v3/orgs/acme/workloads",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-1",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-1/state",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-1/events",
    ]
    assert calls[1].kwargs["params"] == {"limit": 10}
    assert all(call.kwargs["timeout"] == 30 for call in calls)


# GET /workloads/{uid} for BM in the backend shape: region and hostname are
# nested under bm_config, never top-level.
BM_WORKLOAD_RESPONSE = {
    "uid": "wrk-bm",
    "type": "BM",
    "created_by": {
        "token_type": "USER",
        "email": "dev@example.com",
        "username": "dev",
    },
    "project_id": "proj-1",
    "name": "metal",
    "image": "ubuntu-24.04",
    "resource_name": "bm-h100-8x",
    "reservation_uid": "res-1",
    "ssh_keys": [{"uid": "ssh-1", "name": "laptop", "public_key_raw": "ssh-ed25519"}],
    "bm_config": {"region": "us-east", "hostname": "metal-01"},
    "state": {
        "status": "running",
        "message": "",
        "public_ip": "203.0.113.10",
        "ssh_port": 22,
        "ready_replicas": 1,
        "total_replicas": 1,
    },
    "created_at": "2026-10-28T12:00:00Z",
    "updated_at": "2026-10-28T12:05:00Z",
}


def test_workload_response_matches_backend_shape():
    workload = workload_models.WorkloadResponse.from_dict(BM_WORKLOAD_RESPONSE)

    assert workload.reservation_uid == "res-1"
    assert workload.created_by == workload_models.WorkloadCreator(
        token_type="USER", email="dev@example.com", username="dev"
    )
    assert workload.bm_config == workload_models.BMConfig(
        region="us-east", hostname="metal-01"
    )
    assert workload.vm_config is None
    assert workload.state.ssh_port == 22
    assert not hasattr(workload, "region")
    assert not hasattr(workload, "hostname")

    vm = workload_models.WorkloadResponse.from_dict(
        {"uid": "wrk-vm", "type": "VM", "vm_config": {"hostname": "vm-host"}}
    )
    assert vm.vm_config == workload_models.VMConfig(hostname="vm-host")
    assert vm.bm_config is None


def test_workload_list_item_matches_backend_operation_shape():
    item = workload_models.WorkloadListItem.from_dict(
        {
            "uid": "wrk-1",
            "type": "VM",
            "name": "vm",
            "reservation_uid": "res-1",
            "created_by": {"token_type": "SERVICE", "token_name": "ci"},
        }
    )

    assert item.reservation_uid == "res-1"
    assert item.created_by.token_name == "ci"


def test_verify_posts_uid_and_digest(client):
    client.session.request.return_value = Response(body={"verified": True})

    assert client.workload.verify(" wrk-1 ", "abc123") is True
    request = client.session.request.call_args
    assert request.args == (
        "POST",
        "https://api.test/tha/v3/orgs/acme/workloads/verify",
    )
    assert request.kwargs["json"] == {"uid": "wrk-1", "digest": "abc123"}


def test_ssh_key_attach_and_detach_routes(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"workload_uid": "wrk-1", "ssh_key_uid": "ssh-9"}
    )
    client.session.request.return_value = Response(status=204)

    attached = client.workload.attach_ssh_key("wrk-1", "ssh-9")
    client.workload.detach_ssh_key("wrk-1", "ssh-9")

    assert attached == targon.client.SshKeyAttachResponse("wrk-1", "ssh-9")
    url = "https://api.test/tha/v3/orgs/acme/workloads/wrk-1/ssh-keys/ssh-9"
    assert client.sandbox_no_retry_session.request.call_args.args == ("PUT", url)
    assert client.session.request.call_args.args == ("DELETE", url)


@pytest.mark.parametrize("workload_type", ("VM", "BM", "SANDBOX"))
def test_wait_until_ready_works_for_every_type(client, workload_type):
    client.session.request.side_effect = [
        Response(
            body={"uid": "wrk-1", "workload_type": workload_type, "status": "pending"}
        ),
        Response(
            body={"uid": "wrk-1", "workload_type": workload_type, "status": "running"}
        ),
    ]

    state = client.workload.wait_until_ready("wrk-1", timeout=1, poll_interval=0)

    assert state.status == "running"
    assert client.session.request.call_count == 2


def test_wait_until_ready_terminal_and_timeout(client):
    client.session.request.return_value = Response(
        body={"uid": "wrk-1", "workload_type": "VM", "status": "error", "message": "x"}
    )
    with pytest.raises(TargonAPIError) as exc:
        client.workload.wait_until_ready("wrk-1", timeout=1, poll_interval=0)
    assert exc.value.reason == "WORKLOAD_INVALID_STATE"

    client.session.request.return_value = Response(
        body={"uid": "wrk-1", "workload_type": "VM", "status": "provisioning"}
    )
    with pytest.raises(TargonTimeoutError):
        client.workload.wait_until_ready("wrk-1", timeout=0, poll_interval=0)


@pytest.mark.parametrize("status", ("stopped", "frozen", "Frozen"))
def test_wait_until_ready_continues_through_stopped_and_frozen(client, status):
    client.session.request.side_effect = [
        Response(body={"uid": "wrk-1", "workload_type": "VM", "status": status}),
        Response(body={"uid": "wrk-1", "workload_type": "VM", "status": "running"}),
    ]

    state = client.workload.wait_until_ready("wrk-1", timeout=1, poll_interval=0)

    assert state.status == "running"
    assert client.session.request.call_count == 2


@pytest.mark.parametrize("status", ("error", "suspended", "deleted", "Deleted"))
def test_wait_until_ready_keeps_error_suspended_deleted_terminal(client, status):
    client.session.request.return_value = Response(
        body={"uid": "wrk-1", "workload_type": "VM", "status": status}
    )

    with pytest.raises(TargonAPIError) as exc:
        client.workload.wait_until_ready("wrk-1", timeout=1, poll_interval=0)

    assert exc.value.reason == "WORKLOAD_INVALID_STATE"
    assert client.session.request.call_count == 1


def test_wait_typed_target_is_checked_before_terminal_states(client):
    from targon.client._workload_lifecycle import wait_typed

    client.session.request.side_effect = [
        Response(body={"uid": "wrk-1", "workload_type": "VM", "status": "frozen"}),
        Response(body={"uid": "wrk-1", "type": "VM", "name": "vm"}),
    ]

    workload = wait_typed(
        client, "wrk-1", "VM", target="frozen", timeout=1, poll_interval=0
    )

    assert workload.uid == "wrk-1"
    assert client.session.request.call_count == 2

    client.session.request.reset_mock()
    client.session.request.side_effect = [
        Response(body={"uid": "wrk-1", "workload_type": "VM", "status": "stopped"}),
        Response(body={"uid": "wrk-1", "workload_type": "VM", "status": "error"}),
    ]
    with pytest.raises(TargonAPIError) as exc:
        wait_typed(client, "wrk-1", "VM", target="frozen", timeout=1, poll_interval=0)
    assert exc.value.reason == "WORKLOAD_INVALID_STATE"
    assert client.session.request.call_count == 2
