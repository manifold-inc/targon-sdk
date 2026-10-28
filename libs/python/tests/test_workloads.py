import importlib.util

from conftest import Response

import targon
import targon.client as client_exports
import targon.client.workload as workload_models
from targon import Client


def test_deprecated_resources_are_not_exported():
    client = Client(api_key="test", org="acme")
    try:
        assert not hasattr(client, "volume")
        for method in (
            "create",
            "update",
            "deploy",
            "suspend",
            "reboot",
            "wait_until_ready",
            "get_logs",
            "stream_logs",
            "exec_stream",
            "exec",
            "verify",
            "vm_images",
            "attach_volume",
            "detach_volume",
            "attach_ssh_key",
            "detach_ssh_key",
        ):
            assert not hasattr(client.workload, method)
    finally:
        client.close()


def test_common_workload_surface_is_preserved():
    client = Client(api_key="test", org="acme")
    try:
        for method in ("get", "list", "delete", "get_state", "get_events"):
            assert callable(getattr(client.workload, method))
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
        "SshKeyAttachResponse",
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


def test_common_workload_operations_keep_org_scoped_routes(client):
    client.session.request.side_effect = [
        Response(body={"uid": "wrk-1", "type": "VM"}),
        Response(body={"items": [{"uid": "wrk-1"}], "next_cursor": "next"}),
        Response(body=None),
        Response(body={"uid": "wrk-1", "status": "running"}),
        Response(body={"items": [{"workload_uid": "wrk-1"}]}),
    ]

    assert client.workload.get("wrk-1").type == "VM"
    assert client.workload.list(limit=10).next_cursor == "next"
    assert client.workload.delete("wrk-1") is None
    assert client.workload.get_state("wrk-1").status == "running"
    assert client.workload.get_events("wrk-1").items[0].workload_uid == "wrk-1"

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
