from unittest.mock import call

import pytest
from conftest import Response

import targon
import targon.client as client_exports
from targon.client.workload import PortConfig
from targon.core.exceptions import TargonAPIError
from targon.core.exceptions import TimeoutError as TargonTimeoutError
from targon.core.exceptions import ValidationError


def vm_body(uid="wrk-vm", name="vm-example"):
    return {"uid": uid, "type": "VM", "name": name}


def test_vm_types_are_exported_and_client_is_lazy(client):
    assert client._vms is None
    assert isinstance(client.vms, targon.VMClient)
    assert client.vms is client.vms
    for name in ("VM", "VMClient", "VMCreateParams", "VMUpdateParams"):
        assert getattr(targon, name) is getattr(client_exports, name)


def test_register_uses_exact_payload_and_hydrates(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(body=vm_body())

    vm = client.vms.register(
        targon.VMCreateParams(
            name=" vm-example ",
            image=" ubuntu-24.04 ",
            resource_name=" cpu-small ",
            password=" secret ",
            project_id="proj-1",
            reservation_uid="res-1",
            ssh_keys=("ssh-1",),
            ports=(PortConfig(22, protocol="TCP", routing="PUBLIC"),),
        )
    )

    assert vm.uid == "wrk-vm"
    assert vm.name == "vm-example"
    request = client.sandbox_no_retry_session.request.call_args
    assert request.args == (
        "POST",
        "https://api.test/tha/v3/orgs/acme/workloads",
    )
    assert request.kwargs["json"] == {
        "type": "VM",
        "name": "vm-example",
        "image": "ubuntu-24.04",
        "resource_name": "cpu-small",
        "vm_config": {"password": "secret"},
        "project_id": "proj-1",
        "reservation_uid": "res-1",
        "ssh_keys": ["ssh-1"],
        "ports": [{"port": 22, "protocol": "TCP", "routing": "PUBLIC"}],
    }
    assert client.session.request.call_args.args == (
        "GET",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-vm",
    )


@pytest.mark.parametrize("password", ("", "abc"))
def test_register_validates_password_before_request(client, password):
    with pytest.raises(ValidationError) as exc:
        client.vms.register(
            targon.VMCreateParams("vm", "ubuntu", "cpu-small", password)
        )

    assert exc.value.field == "vm_config.password"
    client.sandbox_no_retry_session.request.assert_not_called()


def test_register_requires_typed_params(client):
    with pytest.raises(ValidationError) as exc:
        client.vms.register({"name": "vm"})
    assert exc.value.field == "params"


def test_create_deploys_once_and_waits_until_running(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"uid": "wrk-vm"}),
        Response(body={"uid": "wrk-vm"}),
    ]
    client.session.request.side_effect = [
        Response(
            body={
                "uid": "wrk-vm",
                "workload_type": "VM",
                "status": "provisioning",
            }
        ),
        Response(body={"uid": "wrk-vm", "workload_type": "VM", "status": "running"}),
        Response(body=vm_body(name="ready")),
    ]

    vm = client.vms.create(
        targon.VMCreateParams("vm", "ubuntu", "cpu-small", "secret"),
        timeout=1,
        poll_interval=0,
    )

    assert vm.name == "ready"
    assert [
        item.args[0] for item in client.sandbox_no_retry_session.request.call_args_list
    ] == [
        "POST",
        "POST",
    ]
    assert (
        client.sandbox_no_retry_session.request.call_args_list[1]
        .args[1]
        .endswith("/workloads/wrk-vm/deploy")
    )
    assert [
        item.args[1].rsplit("/", 1)[-1]
        for item in client.session.request.call_args_list
    ] == [
        "state",
        "state",
        "wrk-vm",
    ]


def test_create_can_skip_wait_but_still_hydrates(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"uid": "wrk-vm"}),
        Response(body={"uid": "wrk-vm"}),
    ]
    client.session.request.return_value = Response(body=vm_body())

    vm = client.vms.create(
        targon.VMCreateParams(
            "vm", "ubuntu", "cpu-small", "secret", wait_until_running=False
        )
    )

    assert vm.uid == "wrk-vm"
    assert client.session.request.call_count == 1
    assert client.session.request.call_args.args[1].endswith("/workloads/wrk-vm")


def test_get_rejects_mismatched_workload_type(client):
    client.session.request.return_value = Response(
        body={"uid": "wrk-wrong", "type": "BM"}
    )

    with pytest.raises(TargonAPIError) as exc:
        client.vms.get("wrk-wrong")

    assert exc.value.status_code == 502
    assert exc.value.reason == "WORKLOAD_TYPE_MISMATCH"
    assert exc.value.workload_uid == "wrk-wrong"


def test_deploy_rejects_mismatched_state_type(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(
        body={"uid": "wrk-vm", "workload_type": "BM", "status": "running"}
    )

    with pytest.raises(TargonAPIError) as exc:
        client.vms.deploy("wrk-vm", timeout=1, poll_interval=0)

    assert exc.value.reason == "WORKLOAD_TYPE_MISMATCH"


def test_deploy_preserves_terminal_state_error_semantics(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(
        body={
            "uid": "wrk-vm",
            "workload_type": "VM",
            "status": "error",
            "message": "capacity lost",
        }
    )

    with pytest.raises(TargonAPIError) as exc:
        client.vms.deploy("wrk-vm", timeout=1, poll_interval=0)

    assert exc.value.status_code == 409
    assert exc.value.reason == "WORKLOAD_INVALID_STATE"
    assert "capacity lost" in str(exc.value)


def test_deploy_preserves_timeout_semantics(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(
        body={
            "uid": "wrk-vm",
            "workload_type": "VM",
            "status": "provisioning",
        }
    )

    with pytest.raises(TargonTimeoutError) as exc:
        client.vms.deploy("wrk-vm", timeout=0, poll_interval=0)

    assert exc.value.timeout == 0
    assert "provisioning" in str(exc.value)


def test_update_and_reboot_use_no_retry_transport(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body=vm_body(name="renamed")),
        Response(body={"uid": "wrk-vm"}),
    ]
    client.session.request.return_value = Response(body=vm_body(name="rebooted"))

    updated = client.vms.update(
        "wrk-vm",
        targon.VMUpdateParams(
            name=" renamed ",
            ports=(PortConfig(443, protocol="TCP"),),
        ),
    )
    rebooted = client.vms.reboot("wrk-vm")

    assert updated.name == "renamed"
    assert rebooted.name == "rebooted"
    assert client.sandbox_no_retry_session.request.call_args_list == [
        call(
            "PATCH",
            "https://api.test/tha/v3/orgs/acme/workloads/wrk-vm",
            json={
                "name": "renamed",
                "ports": [{"port": 443, "protocol": "TCP", "routing": "PROXIED"}],
            },
            timeout=30,
            verify=True,
        ),
        call(
            "POST",
            "https://api.test/tha/v3/orgs/acme/workloads/wrk-vm/reboot",
            timeout=30,
            verify=True,
        ),
    ]


def test_update_requires_a_field(client):
    with pytest.raises(ValidationError) as exc:
        client.vms.update("wrk-vm", targon.VMUpdateParams())
    assert exc.value.field == "update"


def test_update_requires_typed_request(client):
    with pytest.raises(ValidationError) as exc:
        client.vms.update("wrk-vm", {"name": "renamed"})
    assert exc.value.field == "request"


def test_bound_vm_forwards_lifecycle_calls(client):
    client.session.request.side_effect = [
        Response(body=vm_body()),
        Response(body=vm_body(name="refreshed")),
        Response(body=vm_body(name="deployed")),
        Response(body=vm_body(name="rebooted")),
    ]
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"uid": "wrk-vm"}),
        Response(body=vm_body(name="updated")),
        Response(body={"uid": "wrk-vm"}),
    ]
    vm = client.vms.get("wrk-vm")

    refreshed = vm.refresh()
    deployed = vm.deploy(wait=False)
    updated = vm.update(targon.VMUpdateParams(name="updated"))
    rebooted = vm.reboot()

    assert refreshed.name == "refreshed"
    assert deployed.name == "deployed"
    assert updated.name == "updated"
    assert rebooted.name == "rebooted"
    assert all(
        item._client is client for item in (refreshed, deployed, updated, rebooted)
    )
