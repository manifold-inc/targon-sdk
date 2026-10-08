import base64
import json
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock, patch

import pytest

from targon import (
    Client,
    ForkRequest,
    PortProtocol,
    PublishRequest,
    SandboxConfigInput,
    SandboxCreateParams,
    SandboxStatus,
    SandboxSummary,
    SandboxTemplateKind,
    SandboxTemplateStatus,
    SandboxUpdateParams,
)
from targon.client.workload import PortConfig
from targon.core.exceptions import (
    PayloadTooLargeError,
    SandboxStateError,
    TerminalLimitError,
    ValidationError,
)


class Response:
    def __init__(self, status=200, body=None, content_type="application/json"):
        self.status_code = status
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.text = "" if body is None else json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def client():
    value = Client(
        api_key="pat_test",
        org="acme",
        base_url="https://api.test",
        max_retries=3,
    )
    value.session.request = Mock()
    value.sandbox_no_retry_session.request = Mock()
    yield value
    value.close()


def sandbox_body(uid="wrk-1", status="running"):
    return {
        "uid": uid,
        "type": "SANDBOX",
        "name": "devbox",
        "image": "sbt-python",
        "sandbox_config": {
            "template_uid": "sbt-python",
            "ttl_sec": 3600,
            "idle_timeout_sec": 300,
        },
        "state": {"status": status},
        "frozen_cost_per_hour": 0.01,
    }


def sandbox_summary_body(uid="wrk-1", status="running"):
    return {
        "uid": uid,
        "type": "SANDBOX",
        "name": "devbox",
        "state": {"status": status},
        "frozen_cost_per_hour": 0.01,
        "created_at": "2026-10-07T12:00:00Z",
        "updated_at": "2026-10-07T12:01:00Z",
    }


def template_body(status="READY"):
    return {
        "uid": "sbt-user",
        "name": "user-template",
        "kind": "USER",
        "status": status,
        "resource_name": "cpu-small",
    }


def hydrated_template(client, status="READY"):
    body = template_body(status=status)
    body["uid"] = "sbt-python"
    return client.sandboxes.templates._hydrate(body)


def test_create_deploys_and_waits_on_org_scoped_path(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body=sandbox_summary_body(status="registered")),
        Response(body=sandbox_summary_body(status="provisioning")),
    ]
    client.session.request.side_effect = [
        Response(
            body={
                "uid": "wrk-1",
                "workload_type": "SANDBOX",
                "status": "running",
            }
        ),
        Response(body=sandbox_body()),
    ]

    sandbox = client.sandboxes.create(
        SandboxCreateParams(
            name="devbox",
            template=hydrated_template(client),
            ports=(PortConfig(8080, protocol="TCP"),),
            ttl_sec=3600,
            idle_timeout_sec=300,
        ),
        poll_interval=0,
    )

    assert sandbox.uid == "wrk-1"
    create_call = client.sandbox_no_retry_session.request.call_args_list[0]
    assert create_call.args[:2] == (
        "POST",
        "https://api.test/tha/v3/orgs/acme/workloads",
    )
    assert create_call.kwargs["json"] == {
        "type": "SANDBOX",
        "name": "devbox",
        "image": "sbt-python",
        "ports": [{"port": 8080, "protocol": "TCP", "routing": "PROXIED"}],
        "sandbox_config": {"ttl_sec": 3600, "idle_timeout_sec": 300},
    }
    assert (
        client.sandbox_no_retry_session.request.call_args_list[1]
        .args[1]
        .endswith("/workloads/wrk-1/deploy")
    )


def test_create_can_return_without_waiting(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body=sandbox_summary_body(status="registered")),
        Response(body=sandbox_summary_body(status="provisioning")),
    ]
    client.session.request.return_value = Response(
        body=sandbox_body(status="provisioning")
    )
    result = client.sandboxes.create(
        SandboxCreateParams(
            name="devbox",
            template=hydrated_template(client),
            wait_until_running=False,
        )
    )
    assert result.state.status is SandboxStatus.PROVISIONING
    assert client.sandbox_no_retry_session.request.call_count == 2
    client.session.request.assert_called_once()


def test_create_accepts_sparse_operations_with_fetched_template(client):
    client.session.request.side_effect = [
        Response(body=template_body()),
        Response(body=sandbox_body(status="provisioning")),
    ]
    client.sandbox_no_retry_session.request.side_effect = [
        Response(
            body={
                "uid": "wrk-1",
                "type": "",
                "state": {"status": "registered"},
            }
        ),
        Response(
            body={
                "uid": "wrk-1",
                "type": "",
                "state": {"status": "provisioning"},
            }
        ),
    ]

    template = client.sandboxes.templates.get("sbt-user")
    sandbox = template.create_sandbox("devbox", wait_until_running=False)

    assert sandbox.uid == "wrk-1"
    assert sandbox.state.status is SandboxStatus.PROVISIONING
    assert client.sandbox_no_retry_session.request.call_count == 2


def test_list_always_filters_to_sandbox(client):
    client.session.request.return_value = Response(
        body={"items": [sandbox_summary_body()], "next_cursor": "next"}
    )
    page = client.sandboxes.list(status=SandboxStatus.RUNNING, limit=25)
    assert page.next_cursor == "next"
    assert isinstance(page.items[0], SandboxSummary)
    assert page.items[0].type == "SANDBOX"
    assert not hasattr(page.items[0], "image")
    assert client.session.request.call_args.kwargs["params"] == {
        "type": "SANDBOX",
        "limit": 25,
        "status": "running",
    }


def test_models_are_frozen_and_enums_are_closed(client):
    client.session.request.return_value = Response(body=sandbox_body())
    sandbox = client.sandboxes.get("wrk-1")
    with pytest.raises(FrozenInstanceError):
        sandbox.name = "changed"
    with pytest.raises(ValueError):
        SandboxStatus("new-server-state")
    assert PortProtocol("UDP") is PortProtocol.UDP
    with pytest.raises(ValueError):
        PortProtocol("SCTP")


def test_hydrated_sandbox_exposes_thin_resource_methods(client):
    client.session.request.return_value = Response(body=sandbox_body())
    sandbox = client.sandboxes.get("wrk-1")
    service = client.sandboxes
    sentinel = object()

    service.get = Mock(return_value=sentinel)
    service.get_state = Mock(return_value=sentinel)
    service.update = Mock(return_value=sentinel)
    service.freeze = Mock(return_value=sentinel)
    service.thaw = Mock(return_value=sentinel)
    service.delete = Mock()
    service.fork = Mock(return_value=sentinel)
    service.publish = Mock(return_value=sentinel)
    service.exec = Mock(return_value=sentinel)
    service.files.read = Mock(return_value=b"data")
    service.files.write = Mock()
    service.mint_access_ticket = Mock(return_value=sentinel)
    service.get_desktop = Mock(return_value=sentinel)
    service.terminals.list = Mock(return_value=())
    service.terminals.create = Mock(return_value=sentinel)
    service.terminals.delete = Mock()
    service.terminals.connect = Mock(return_value=sentinel)

    update = SandboxUpdateParams(name="renamed")
    fork = ForkRequest(name="child")
    publish = PublishRequest(name="snapshot")
    assert sandbox.refresh() is sentinel
    assert sandbox.get_state() is sentinel
    assert sandbox.update(update) is sentinel
    assert sandbox.freeze(wait=False) is sentinel
    assert sandbox.thaw(wait=False) is sentinel
    sandbox.delete()
    assert sandbox.fork(fork, wait=False) is sentinel
    assert sandbox.publish(publish, wait=False) is sentinel
    assert sandbox.exec("true") is sentinel
    assert sandbox.read_file("/tmp/x") == b"data"
    sandbox.write_file("/tmp/y", b"data")
    assert sandbox.mint_access_ticket() is sentinel
    assert sandbox.get_desktop() is sentinel
    assert sandbox.list_terminals() == ()
    assert sandbox.create_terminal() is sentinel
    sandbox.delete_terminal("term-1")
    assert sandbox.connect_terminal("term-1") is sentinel

    service.get.assert_called_once_with("wrk-1")
    service.update.assert_called_once_with("wrk-1", update)
    service.files.read.assert_called_once_with("wrk-1", "/tmp/x", as_text=False)
    service.terminals.connect.assert_called_once_with(
        "wrk-1",
        "term-1",
        ticket=None,
        ticket_ttl_sec=60,
        use_bearer=False,
        open_timeout=10,
    )


def test_retry_scope_isolated_to_non_idempotent_sandbox_calls(client):
    assert client.session.get_adapter("https://").max_retries.total == 3
    assert (
        client.sandbox_no_retry_session.get_adapter("https://").max_retries.total == 0
    )

    client.session.request.return_value = Response(body=sandbox_body())
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"stdout": "", "stderr": "", "code": 0, "timed_out": False}
    )
    client.sandboxes.get("wrk-1")
    client.sandboxes.exec("wrk-1", "true")
    client.session.request.assert_called_once()
    client.sandbox_no_retry_session.request.assert_called_once()


@pytest.mark.parametrize(
    "body",
    [
        {key: value for key, value in sandbox_body().items() if key != "type"},
        {**sandbox_body(), "type": "VM"},
    ],
)
def test_full_reads_require_sandbox_type(client, body):
    client.session.request.return_value = Response(body=body)
    with pytest.raises(ValidationError):
        client.sandboxes.get("wrk-1")


@pytest.mark.parametrize(
    "body",
    [
        {"uid": "wrk-1", "status": "running"},
        {"uid": "wrk-1", "workload_type": "VM", "status": "running"},
    ],
)
def test_state_reads_require_sandbox_workload_type(client, body):
    client.session.request.return_value = Response(body=body)
    with pytest.raises(ValidationError):
        client.sandboxes.get_state("wrk-1")


@pytest.mark.parametrize(
    "operation",
    [
        {"uid": "wrk-1"},
        {"uid": "wrk-1", "type": None},
        {"uid": "wrk-1", "type": ""},
        {"uid": "wrk-1", "type": "SANDBOX"},
    ],
)
def test_operation_uid_accepts_sparse_sandbox_types(client, operation):
    assert client.sandboxes._operation_uid(operation) == "wrk-1"


@pytest.mark.parametrize(
    "operation",
    [
        {"type": ""},
        {"uid": "", "type": ""},
        {"uid": "wrk-1", "type": "VM"},
    ],
)
def test_operation_uid_still_requires_uid_and_rejects_other_types(client, operation):
    with pytest.raises(ValidationError):
        client.sandboxes._operation_uid(operation)


def test_no_wait_operations_fetch_complete_resources(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(
            body={
                "uid": "wrk-1",
                "type": "",
                "state": {"status": "frozen"},
            }
        ),
        Response(
            body={
                "uid": "wrk-1",
                "type": "",
                "state": {"status": "provisioning"},
            }
        ),
    ]
    client.session.request.side_effect = [
        Response(body=sandbox_body(status="frozen")),
        Response(body=sandbox_body(status="provisioning")),
    ]

    frozen = client.sandboxes.freeze("wrk-1", wait=False)
    thawing = client.sandboxes.thaw("wrk-1", wait=False)
    assert frozen.image == "sbt-python"
    assert thawing.sandbox_config.template_uid == "sbt-python"
    assert client.session.request.call_count == 2


def test_all_sandbox_http_mutations_use_no_retry_transport(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body=sandbox_body()),
        Response(status=204, body=None, content_type=""),
        Response(body={"workload_uid": "wrk-1", "ssh_key_uid": "key-1"}),
        Response(status=204, body=None, content_type=""),
        Response(status=204, body=None, content_type=""),
        Response(body=template_body()),
        Response(status=204, body=None, content_type=""),
        Response(status=204, body=None, content_type=""),
    ]

    client.sandboxes.update("wrk-1", SandboxUpdateParams(name="renamed"))
    client.sandboxes.delete("wrk-1")
    client.sandboxes.attach_ssh_key("wrk-1", "key-1")
    client.sandboxes.detach_ssh_key("wrk-1", "key-1")
    client.sandboxes.files.write("wrk-1", "/tmp/x", b"x")
    client.sandboxes.templates.update("sbt-user", description="new")
    client.sandboxes.templates.delete("sbt-user")
    client.sandboxes.terminals.delete("wrk-1", "term-1")

    client.session.request.assert_not_called()
    methods = [
        call.args[0] for call in client.sandbox_no_retry_session.request.call_args_list
    ]
    assert methods == [
        "PATCH",
        "DELETE",
        "PUT",
        "DELETE",
        "PUT",
        "PATCH",
        "DELETE",
        "DELETE",
    ]


def test_exec_validates_trimmed_value_but_sends_original_command(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"stdout": "", "stderr": "", "code": 0, "timed_out": False}
    )
    command = "  printf hello  \n"
    client.sandboxes.exec("wrk-1", command)
    assert (
        client.sandbox_no_retry_session.request.call_args.kwargs["json"]["cmd"]
        == command
    )

    with pytest.raises(ValidationError):
        client.sandboxes.exec("wrk-1", " \n\t ")


def test_update_rejects_empty_and_unsupported_inputs(client):
    with pytest.raises(ValidationError):
        client.sandboxes.update("wrk-1", SandboxUpdateParams())
    with pytest.raises(ValidationError):
        client.sandboxes.create(
            SandboxCreateParams(
                name="-bad",
                template=hydrated_template(client),
                ports=(PortConfig(22),),
            )
        )


def test_sandbox_ports_reject_sctp(client):
    with pytest.raises(ValidationError, match="TCP or UDP"):
        client.sandboxes.create(
            SandboxCreateParams(
                name="valid",
                template=hydrated_template(client),
                ports=(PortConfig(8080, protocol="SCTP"),),
            )
        )
    client.sandbox_no_retry_session.request.assert_not_called()


def test_update_rejects_zero_idle_timeout_but_create_and_fork_allow_it(client):
    zero_idle = SandboxConfigInput(idle_timeout_sec=0)
    with pytest.raises(ValidationError, match="must be positive"):
        client.sandboxes.update(
            "wrk-1",
            SandboxUpdateParams(sandbox_config=zero_idle),
        )
    client.sandbox_no_retry_session.request.assert_not_called()

    client.sandbox_no_retry_session.request.side_effect = [
        Response(body=sandbox_summary_body(status="registered")),
        Response(body=sandbox_summary_body(status="provisioning")),
        Response(body=sandbox_summary_body(uid="wrk-child", status="provisioning")),
    ]
    client.session.request.side_effect = [
        Response(body=sandbox_body(status="provisioning")),
        Response(body=sandbox_body(uid="wrk-child", status="provisioning")),
    ]
    client.sandboxes.create(
        SandboxCreateParams(
            name="valid",
            template=hydrated_template(client),
            idle_timeout_sec=0,
            wait_until_running=False,
        )
    )
    client.sandboxes.fork(
        "wrk-1",
        ForkRequest(sandbox_config=zero_idle),
        wait=False,
    )
    create_payload = client.sandbox_no_retry_session.request.call_args_list[0].kwargs[
        "json"
    ]
    fork_payload = client.sandbox_no_retry_session.request.call_args_list[2].kwargs[
        "json"
    ]
    assert create_payload["sandbox_config"] == {"idle_timeout_sec": 0}
    assert fork_payload["sandbox_config"] == {"idle_timeout_sec": 0}


def test_exec_and_files_are_binary_safe(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"stdout": "ok\n", "stderr": "", "code": 0, "timed_out": False}),
        Response(status=204, body=None, content_type=""),
    ]
    client.session.request.return_value = Response(
        body={"path": "/tmp/x", "content_b64": base64.b64encode(b"\x00x").decode()}
    )
    result = client.sandboxes.exec("wrk-1", "printf ok")
    assert (result.stdout, result.code, result.timed_out) == ("ok\n", 0, False)
    assert client.sandboxes.files.read("wrk-1", "/tmp/x") == b"\x00x"
    client.sandboxes.files.write("wrk-1", "/tmp/y", b"\xff")
    assert (
        client.sandbox_no_retry_session.request.call_args.kwargs["json"]["content_b64"]
        == "/w=="
    )


def test_terminal_ticket_crud_and_desktop(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body={"ticket": "sat_once", "expires_at": "2026-10-07T12:00:00Z"}),
        Response(
            body={
                "id": "term-1",
                "pid": 12,
                "started_at": "2026-10-07T12:00:00Z",
                "exited": False,
            }
        ),
        Response(status=204, body=None, content_type=""),
    ]
    client.session.request.side_effect = [
        Response(body=[]),
        Response(
            body={
                "available": True,
                "port": 5900,
                "listening": True,
                "ws_url": "wss://x",
            }
        ),
    ]
    ticket = client.sandboxes.mint_access_ticket("wrk-1")
    terminal = client.sandboxes.terminals.create("wrk-1", cols=100, rows=30)
    assert ticket.ticket == "sat_once"
    assert terminal.id == "term-1"
    assert client.sandboxes.terminals.list("wrk-1") == ()
    client.sandboxes.terminals.delete("wrk-1", "term-1")
    assert client.sandboxes.get_desktop("wrk-1").listening is True


def test_terminal_connect_mints_fresh_ticket_then_uses_raw_binary(client):
    client.sandbox_no_retry_session.request.return_value = Response(
        body={"ticket": "sat_once", "expires_at": "2026-10-07T12:00:00Z"}
    )
    websocket = Mock()
    websocket.recv.return_value = b"hello"
    with patch("websockets.sync.client.connect", return_value=websocket) as connect:
        connection = client.sandboxes.terminals.connect("wrk-1", "term-1")
        assert client.sandbox_no_retry_session.request.call_args.args[1].endswith(
            "/workloads/wrk-1/access-tickets"
        )
        assert connect.call_args.args[0].endswith(
            "/workloads/wrk-1/terminals/term-1/ws?ticket=sat_once"
        )
        connection.send(b"input")
        assert connection.recv() == b"hello"
        websocket.send.assert_called_once_with(b"input")
        with pytest.raises(TypeError):
            connection.send("not bytes")


def test_templates_list_and_update(client):
    client.session.request.side_effect = [
        Response(body={"items": [template_body()], "next_cursor": None}),
        Response(body=template_body()),
    ]
    client.sandbox_no_retry_session.request.return_value = Response(
        body=template_body()
    )
    page = client.sandboxes.templates.list(
        kind=SandboxTemplateKind.USER,
        status=SandboxTemplateStatus.READY,
    )
    assert page.items[0].kind is SandboxTemplateKind.USER
    assert page.items[0]._client is client
    fetched = client.sandboxes.templates.get("sbt-user")
    assert fetched._client is client
    updated = client.sandboxes.templates.update("sbt-user", description="new")
    assert updated.status is SandboxTemplateStatus.READY
    assert updated._client is client
    with pytest.raises(FrozenInstanceError):
        fetched.name = "changed"


def test_template_resource_methods_and_ergonomic_create(client):
    template = hydrated_template(client)
    service = client.sandboxes.templates
    sentinel = object()
    service.get = Mock(return_value=sentinel)
    service.update = Mock(return_value=sentinel)
    service.delete = Mock()
    client.sandboxes.create = Mock(return_value=sentinel)

    assert template.refresh() is sentinel
    assert template.update(description="new") is sentinel
    template.delete()
    assert (
        template.create_sandbox("devbox", ttl_sec=60, wait_until_running=False)
        is sentinel
    )

    service.get.assert_called_once_with("sbt-python")
    service.update.assert_called_once_with(
        "sbt-python", display_name=None, description="new"
    )
    service.delete.assert_called_once_with("sbt-python")
    create_params = client.sandboxes.create.call_args.args[0]
    assert create_params.name == "devbox"
    assert create_params.template is template
    assert create_params.ttl_sec == 60
    assert create_params.wait_until_running is False


def test_template_resource_create_accepts_params(client):
    template = hydrated_template(client)
    params = SandboxCreateParams(name="devbox", template=template)
    client.sandboxes.create = Mock(return_value=object())

    template.create_sandbox(params, timeout=10, poll_interval=0)

    client.sandboxes.create.assert_called_once_with(params, timeout=10, poll_interval=0)


@pytest.mark.parametrize("invalid_template", ["unbound", "empty_uid", "pending"])
def test_create_requires_bound_ready_template_with_uid(client, invalid_template):
    template = hydrated_template(client)
    if invalid_template == "unbound":
        template = replace(template, _client=None)
    elif invalid_template == "empty_uid":
        template = replace(template, _data=replace(template._data, uid=""))
    else:
        template = hydrated_template(client, status="PENDING")

    with pytest.raises(ValidationError):
        client.sandboxes.create(SandboxCreateParams(name="devbox", template=template))
    client.sandbox_no_retry_session.request.assert_not_called()


def test_fork_and_publish_without_polling(client):
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body=sandbox_summary_body(uid="wrk-child", status="provisioning")),
        Response(body=template_body(status="PENDING")),
    ]
    client.session.request.return_value = Response(
        body=sandbox_body(uid="wrk-child", status="provisioning")
    )
    child = client.sandboxes.fork("wrk-parent", ForkRequest(name="child"), wait=False)
    template = client.sandboxes.publish(
        "wrk-parent", PublishRequest(name="snapshot"), wait=False
    )
    assert child.uid == "wrk-child"
    assert template.status is SandboxTemplateStatus.PENDING
    assert template._client is client


@pytest.mark.parametrize(
    ("status", "reason", "exception"),
    [
        (413, "WORKLOAD_SANDBOX_PAYLOAD_TOO_LARGE", PayloadTooLargeError),
        (429, "WORKLOAD_SANDBOX_SESSION_LIMIT", TerminalLimitError),
        (409, "WORKLOAD_SANDBOX_INVALID_STATE", SandboxStateError),
    ],
)
def test_typed_errors_include_workload_uid(client, status, reason, exception):
    client.session.request.return_value = Response(
        status=status, body={"error": "failed", "reason": reason}
    )
    with pytest.raises(exception) as caught:
        client.sandboxes.get("wrk-1")
    assert caught.value.reason == reason
    assert caught.value.workload_uid == "wrk-1"


@pytest.mark.parametrize("invalid_name", [True, False])
def test_create_validation_is_client_side(client, invalid_name):
    create_params = SandboxCreateParams(
        name="Bad_Name" if invalid_name else "valid",
        template=hydrated_template(client),
        ttl_sec=None if invalid_name else 60,
        idle_timeout_sec=None if invalid_name else 60,
    )
    with pytest.raises(ValidationError):
        client.sandboxes.create(create_params)
    client.session.request.assert_not_called()
    client.sandbox_no_retry_session.request.assert_not_called()


def test_list_limit_validation_is_strict(client):
    with pytest.raises(ValidationError):
        client.sandboxes.list(limit=1001)
    with pytest.raises(ValidationError):
        client.sandboxes.templates.list(limit=1001)
