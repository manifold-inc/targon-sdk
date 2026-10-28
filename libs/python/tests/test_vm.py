from dataclasses import FrozenInstanceError, fields
from unittest.mock import Mock, call

import pytest
from conftest import Response

import targon
import targon.client as client_exports
from targon.client._workload_lifecycle import require_type
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
            resource_name=f" {targon.Resources.H100_SMALL} ",
            vm_config=targon.VMConfigInput(hostname=" VM-Host.Example "),
            project_id=" proj-1 ",
            reservation_uid=" res-1 ",
            ssh_keys=("ssh-1",),
            ports=(PortConfig(8080, protocol="TCP"),),
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
        "resource_name": "h100-small",
        "vm_config": {"hostname": "vm-host.example"},
        "project_id": "proj-1",
        "reservation_uid": "res-1",
        "ssh_keys": ["ssh-1"],
        "ports": [{"port": 8080, "protocol": "TCP"}],
    }
    assert client.session.request.call_args.args == (
        "GET",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-vm",
    )


def test_register_omits_empty_optional_fields(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(body=vm_body())

    client.vms.register(
        targon.VMCreateParams(
            "vm",
            "ubuntu-24.04",
            targon.Resources.H100_SMALL,
            vm_config=targon.VMConfigInput(hostname="  "),
            project_id="",
            reservation_uid="  ",
        )
    )

    assert client.sandbox_no_retry_session.request.call_args.kwargs["json"] == {
        "type": "VM",
        "name": "vm",
        "image": "ubuntu-24.04",
        "resource_name": "h100-small",
    }


def test_vm_params_have_no_password():
    assert "password" not in {f.name for f in fields(targon.VMCreateParams)}
    assert {f.name for f in fields(targon.VMConfigInput)} == {"hostname"}


@pytest.mark.parametrize(
    ("overrides", "field"),
    (
        (
            {"vm_config": targon.VMConfigInput(hostname="bad_host")},
            "vm_config.hostname",
        ),
        ({"vm_config": targon.VMConfigInput(hostname="a" * 64)}, "vm_config.hostname"),
        ({"vm_config": {"hostname": "vm"}}, "vm_config"),
        ({"ports": (PortConfig(22),)}, "ports"),
        ({"ports": (PortConfig(70000),)}, "ports"),
        ({"ports": (PortConfig(8080, protocol="SCTP"),)}, "ports"),
        ({"ports": (PortConfig(8080, protocol=6),)}, "ports"),
        (
            {"ports": (PortConfig(8080), PortConfig(8080, protocol="tcp"))},
            "ports",
        ),
        ({"ssh_keys": "ssh-key-uid"}, "ssh_keys"),
        ({"ssh_keys": b"ssh-key-uid"}, "ssh_keys"),
        ({"ssh_keys": ("ssh-key-uid", "")}, "ssh_keys"),
        ({"ssh_keys": (42,)}, "ssh_keys"),
        ({"project_id": 42}, "project_id"),
    ),
)
def test_register_validates_before_request(client, overrides, field):
    with pytest.raises(ValidationError) as exc:
        client.vms.register(
            targon.VMCreateParams(
                "vm", "ubuntu-24.04", targon.Resources.H100_SMALL, **overrides
            )
        )

    assert exc.value.field == field
    client.sandbox_no_retry_session.request.assert_not_called()


def test_ports_allow_same_port_on_tcp_and_udp_and_normalize_protocol(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(
        body={"uid": "wrk-vm", "type": "VM", "name": "vm"}
    )

    client.vms.register(
        targon.VMCreateParams(
            "vm",
            "ubuntu-24.04",
            targon.Resources.H100_SMALL,
            ports=(
                PortConfig(8080, protocol="udp", routing="DIRECT"),
                PortConfig(8080, protocol=""),
                PortConfig(9090, protocol=" TCP "),
            ),
        )
    )

    payload = client.sandbox_no_retry_session.request.call_args.kwargs["json"]
    assert payload["ports"] == [
        {"port": 8080, "protocol": "UDP"},
        {"port": 8080, "protocol": "TCP"},
        {"port": 9090, "protocol": "TCP"},
    ]


def test_update_rejects_invalid_ports(client):
    for ports in (
        (PortConfig(8080, protocol="SCTP"),),
        (PortConfig(8080), PortConfig(8080, protocol="TCP")),
    ):
        with pytest.raises(ValidationError) as exc:
            client.vms.update("wrk-vm", targon.VMUpdateParams(ports=ports))
        assert exc.value.field == "ports"
    client.sandbox_no_retry_session.request.assert_not_called()


@pytest.mark.parametrize("status", ("stopped", "frozen"))
def test_deploy_treats_stopped_and_frozen_as_terminal(client, status):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(
        body={"uid": "wrk-vm", "workload_type": "VM", "status": status}
    )

    with pytest.raises(TargonAPIError) as exc:
        client.vms.deploy("wrk-vm", timeout=1, poll_interval=0)

    assert exc.value.reason == "WORKLOAD_INVALID_STATE"
    assert status in str(exc.value)


def test_update_rejects_ssh_port(client):
    with pytest.raises(ValidationError) as exc:
        client.vms.update("wrk-vm", targon.VMUpdateParams(ports=(PortConfig(22),)))

    assert exc.value.field == "ports"
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
        targon.VMCreateParams("vm", "ubuntu-24.04", targon.Resources.H100_SMALL),
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
            "vm",
            "ubuntu-24.04",
            targon.Resources.H100_SMALL,
            wait_until_running=False,
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


@pytest.mark.parametrize("workload_type", (None, 42))
def test_get_rejects_null_or_non_string_workload_type(client, workload_type):
    client.session.request.return_value = Response(
        body={"uid": "wrk-wrong", "type": workload_type}
    )

    with pytest.raises(TargonAPIError) as exc:
        client.vms.get("wrk-wrong")

    assert exc.value.reason == "WORKLOAD_TYPE_MISMATCH"
    assert "expected 'VM'" in str(exc.value)
    assert f"got {workload_type!r}" in str(exc.value)


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


@pytest.mark.parametrize("workload_type", (None, 42))
def test_deploy_rejects_null_or_non_string_state_type(client, workload_type):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-vm"}
    )
    client.session.request.return_value = Response(
        body={
            "uid": "wrk-vm",
            "workload_type": workload_type,
            "status": "running",
        }
    )

    with pytest.raises(TargonAPIError) as exc:
        client.vms.deploy("wrk-vm", timeout=1, poll_interval=0)

    assert exc.value.reason == "WORKLOAD_TYPE_MISMATCH"
    assert "expected 'VM'" in str(exc.value)
    assert f"got {workload_type!r}" in str(exc.value)


@pytest.mark.parametrize("expected_type", (None, 42))
def test_require_type_rejects_invalid_expected_discriminator(expected_type):
    with pytest.raises(ValidationError) as exc:
        require_type("wrk-vm", "VM", expected_type)

    assert exc.value.field == "expected_type"
    assert exc.value.value == expected_type
    assert "'VM'" in str(exc.value)
    assert repr(expected_type) in str(exc.value)


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
                "ports": [{"port": 443, "protocol": "TCP"}],
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


def test_get_logs_uses_vm_log_query(client):
    text_response = Response(content_type="text/plain")
    text_response.text = "booted\n"
    client.session.request.side_effect = [Response(body=vm_body()), text_response]

    logs = client.vms.get("wrk-vm").get_logs(tail=25, log_type="SERIAL", previous=True)

    assert logs == "booted\n"
    request = client.session.request.call_args
    assert request.args == (
        "GET",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-vm/logs",
    )
    assert request.kwargs["params"] == {
        "tail": 25,
        "previous": "true",
        "type": "serial",
    }


def test_get_logs_rejects_unknown_log_type(client):
    with pytest.raises(ValidationError) as exc:
        client.vms.get_logs("wrk-vm", log_type="kernel")

    assert exc.value.field == "log_type"
    client.session.request.assert_not_called()


def test_stream_logs_follows_without_retries(client):
    streamed = Mock(status_code=200, headers={"Content-Type": "text/plain"})
    streamed.iter_lines.return_value = iter(["line one", "", "line two\r"])
    client.sandbox_no_retry_session.request.return_value = streamed

    lines = list(client.vms.stream_logs("wrk-vm", tail=5, log_type="qemu"))

    assert lines == ["line one", "line two"]
    request = client.sandbox_no_retry_session.request.call_args
    assert request.args == (
        "GET",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-vm/logs",
    )
    assert request.kwargs["params"] == {"tail": 5, "type": "qemu", "follow": "true"}
    assert request.kwargs["stream"] is True
    assert request.kwargs["headers"] == {"Accept": "text/plain"}
    streamed.close.assert_called_once()


def test_stream_logs_raises_api_errors(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        status=501,
        body={"error": "nope", "reason": "WORKLOAD_LOGS_TYPE_NOT_SUPPORTED"},
    )
    client.sandbox_no_retry_session.request.return_value.close = Mock()

    with pytest.raises(TargonAPIError) as exc:
        list(client.vms.stream_logs("wrk-vm"))

    assert exc.value.reason == "WORKLOAD_LOGS_TYPE_NOT_SUPPORTED"


def test_images_uses_vm_images_route(client):
    client.session.request.return_value = Response(
        body=[
            {
                "name": "ubuntu-24.04",
                "display_name": "Ubuntu 24.04",
                "os_family": "linux",
                "description": "LTS",
            }
        ]
    )

    images = client.vms.images()

    assert images == [
        targon.VMImage("ubuntu-24.04", "Ubuntu 24.04", "linux", "LTS"),
    ]
    assert client.session.request.call_args.args == (
        "GET",
        "https://api.test/tha/v3/orgs/acme/workloads/vm-images",
    )


def test_vm_exposes_read_only_properties_without_getattr(client):
    client.session.request.return_value = Response(
        body={
            **vm_body(),
            "reservation_uid": "res-1",
            "created_by": {"token_type": "SERVICE", "token_name": "ci"},
            "vm_config": {"hostname": "vm-host"},
        }
    )
    vm = client.vms.get("wrk-vm")

    assert "__getattr__" not in vars(targon.VM)
    assert vm.reservation_uid == "res-1"
    assert vm.created_by.token_name == "ci"
    assert vm.vm_config.hostname == "vm-host"
    assert not hasattr(vm, "bm_config")
    assert not hasattr(vm, "password")
    with pytest.raises(AttributeError):
        vm.region
    with pytest.raises((AttributeError, FrozenInstanceError)):
        vm.name = "renamed"
    # Unpickling or copying calls getattr before fields exist; this used to
    # recurse through __getattr__.
    blank = object.__new__(targon.VM)
    with pytest.raises(AttributeError):
        blank.missing
