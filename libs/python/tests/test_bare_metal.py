from dataclasses import fields
from unittest.mock import call

import pytest
from conftest import Response

import targon
import targon.client as client_exports
from targon.core.exceptions import TargonAPIError, ValidationError


def bare_metal_body(uid="wrk-bm", name="metal-example"):
    return {
        "uid": uid,
        "type": "BM",
        "name": name,
        "region": "us-central",
        "hostname": "metal-01",
    }


def create_params(**overrides):
    values = {
        "name": "metal-example",
        "image": "ubuntu-24.04",
        "resource_name": "bm-a100",
        "region": "us-central",
        "ssh_keys": ("ssh-1", "ssh-2"),
    }
    values.update(overrides)
    return targon.BareMetalCreateParams(**values)


def test_bare_metal_types_are_exported_and_client_is_lazy(client):
    assert client._bare_metal is None
    assert isinstance(client.bare_metal, targon.BareMetalClient)
    assert client.bare_metal is client.bare_metal
    for name in (
        "BareMetal",
        "BareMetalClient",
        "BareMetalCreateParams",
        "BareMetalReinstallParams",
    ):
        assert getattr(targon, name) is getattr(client_exports, name)


def test_register_uses_exact_payload_and_hydrates(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"uid": "wrk-bm"}
    )
    client.session.request.return_value = Response(body=bare_metal_body())

    metal = client.bare_metal.register(
        create_params(
            name=" metal-example ",
            image=" ubuntu-24.04 ",
            resource_name=" bm-a100 ",
            region=" us-central ",
            hostname="metal-01",
            project_id="proj-1",
        )
    )

    assert metal.uid == "wrk-bm"
    assert metal.region == "us-central"
    assert metal.hostname == "metal-01"
    assert client.sandbox_no_retry_session.request.call_args == call(
        "POST",
        "https://api.test/tha/v3/orgs/acme/workloads",
        json={
            "type": "BM",
            "name": "metal-example",
            "image": "ubuntu-24.04",
            "resource_name": "bm-a100",
            "region": "us-central",
            "ssh_keys": ["ssh-1", "ssh-2"],
            "hostname": "metal-01",
            "project_id": "proj-1",
        },
        timeout=30,
        verify=True,
    )
    assert client.session.request.call_args.args == (
        "GET",
        "https://api.test/tha/v3/orgs/acme/workloads/wrk-bm",
    )


@pytest.mark.parametrize(
    ("overrides", "field"),
    (
        ({"name": ""}, "name"),
        ({"image": ""}, "image"),
        ({"resource_name": ""}, "resource_name"),
        ({"region": ""}, "region"),
        ({"ssh_keys": ()}, "ssh_keys"),
    ),
)
def test_register_validates_required_fields_before_request(client, overrides, field):
    with pytest.raises(ValidationError) as exc:
        client.bare_metal.register(create_params(**overrides))

    assert exc.value.field == field
    client.sandbox_no_retry_session.request.assert_not_called()


def test_register_requires_typed_params(client):
    with pytest.raises(ValidationError) as exc:
        client.bare_metal.register({"name": "metal"})

    assert exc.value.field == "params"


def test_create_params_do_not_expose_unsupported_workload_fields():
    names = {item.name for item in fields(targon.BareMetalCreateParams)}

    assert names == {
        "name",
        "image",
        "resource_name",
        "region",
        "ssh_keys",
        "hostname",
        "project_id",
        "wait_until_running",
    }


def test_create_deploys_once_and_waits_until_running(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
    ]
    client.session.request.side_effect = [
        Response(
            body={
                "uid": "wrk-bm",
                "workload_type": "BM",
                "status": "provisioning",
            }
        ),
        Response(body={"uid": "wrk-bm", "workload_type": "BM", "status": "running"}),
        Response(body=bare_metal_body(name="ready")),
    ]

    metal = client.bare_metal.create(
        create_params(),
        timeout=1,
        poll_interval=0,
    )

    assert metal.name == "ready"
    assert [
        item.args[:2] for item in client.sandbox_no_retry_session.request.call_args_list
    ] == [
        ("POST", "https://api.test/tha/v3/orgs/acme/workloads"),
        (
            "POST",
            "https://api.test/tha/v3/orgs/acme/workloads/wrk-bm/deploy",
        ),
    ]
    assert [
        item.args[1].rsplit("/", 1)[-1]
        for item in client.session.request.call_args_list
    ] == ["state", "state", "wrk-bm"]


def test_create_can_skip_wait_but_still_hydrates(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
    ]
    client.session.request.return_value = Response(body=bare_metal_body())

    metal = client.bare_metal.create(create_params(wait_until_running=False))

    assert metal.uid == "wrk-bm"
    assert client.session.request.call_count == 1


def test_get_rejects_mismatched_workload_type(client):
    client.session.request.return_value = Response(
        body={"uid": "wrk-wrong", "type": "VM"}
    )

    with pytest.raises(TargonAPIError) as exc:
        client.bare_metal.get("wrk-wrong")

    assert exc.value.status_code == 502
    assert exc.value.reason == "WORKLOAD_TYPE_MISMATCH"
    assert exc.value.workload_uid == "wrk-wrong"


def test_operations_use_exact_routes_payloads_and_no_retry_transport(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
    ]
    client.session.request.return_value = Response(body=bare_metal_body())

    rebooted = client.bare_metal.reboot("wrk-bm")
    powered_on = client.bare_metal.power_on("wrk-bm")
    powered_off = client.bare_metal.power_off("wrk-bm")
    reinstalled = client.bare_metal.reinstall(
        "wrk-bm",
        targon.BareMetalReinstallParams(
            image=" ubuntu-24.04 ",
            ssh_keys=("ssh-2",),
            hostname="metal-new",
        ),
    )

    assert all(
        item.uid == "wrk-bm"
        for item in (rebooted, powered_on, powered_off, reinstalled)
    )
    base = "https://api.test/tha/v3/orgs/acme/workloads/wrk-bm"
    assert client.sandbox_no_retry_session.request.call_args_list == [
        call("POST", f"{base}/reboot", timeout=30, verify=True),
        call("POST", f"{base}/power-on", timeout=30, verify=True),
        call("POST", f"{base}/power-off", timeout=30, verify=True),
        call(
            "POST",
            f"{base}/reinstall",
            json={
                "image": "ubuntu-24.04",
                "ssh_keys": ["ssh-2"],
                "hostname": "metal-new",
            },
            timeout=30,
            verify=True,
        ),
    ]


@pytest.mark.parametrize(
    ("request_value", "field"),
    (
        (None, "request"),
        ({"image": "ubuntu"}, "request"),
    ),
)
def test_reinstall_requires_typed_request(client, request_value, field):
    with pytest.raises(ValidationError) as exc:
        client.bare_metal.reinstall("wrk-bm", request_value)

    assert exc.value.field == field
    client.sandbox_no_retry_session.request.assert_not_called()


@pytest.mark.parametrize(
    ("request_factory", "field"),
    (
        (
            lambda: targon.BareMetalReinstallParams(image="", ssh_keys=("ssh-1",)),
            "image",
        ),
        (
            lambda: targon.BareMetalReinstallParams(image="ubuntu", ssh_keys=()),
            "ssh_keys",
        ),
    ),
)
def test_reinstall_validates_required_fields_before_request(
    client, request_factory, field
):
    with pytest.raises(ValidationError) as exc:
        client.bare_metal.reinstall("wrk-bm", request_factory())

    assert exc.value.field == field
    client.sandbox_no_retry_session.request.assert_not_called()


def test_bound_bare_metal_forwards_lifecycle_calls(client):
    client.session.request.side_effect = [
        Response(body=bare_metal_body()),
        Response(body=bare_metal_body(name="refreshed")),
        Response(body=bare_metal_body(name="deployed")),
        Response(body=bare_metal_body(name="rebooted")),
        Response(body=bare_metal_body(name="powered-on")),
        Response(body=bare_metal_body(name="powered-off")),
        Response(body=bare_metal_body(name="reinstalled")),
    ]
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
        Response(body={"uid": "wrk-bm"}),
    ]
    metal = client.bare_metal.get("wrk-bm")

    refreshed = metal.refresh()
    deployed = metal.deploy(wait=False)
    rebooted = metal.reboot()
    powered_on = metal.power_on()
    powered_off = metal.power_off()
    reinstalled = metal.reinstall(
        targon.BareMetalReinstallParams(
            image="ubuntu",
            ssh_keys=("ssh-1",),
        )
    )

    assert [
        item.name
        for item in (
            refreshed,
            deployed,
            rebooted,
            powered_on,
            powered_off,
            reinstalled,
        )
    ] == [
        "refreshed",
        "deployed",
        "rebooted",
        "powered-on",
        "powered-off",
        "reinstalled",
    ]
    assert all(
        item._client is client
        for item in (
            refreshed,
            deployed,
            rebooted,
            powered_on,
            powered_off,
            reinstalled,
        )
    )
