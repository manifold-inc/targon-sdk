import pytest
from conftest import (  # noqa: F401
    Response,
    hydrated_template,
    operation_body,
    sandbox_body,
    sandbox_summary_body,
    template_body,
)

from targon import (  # noqa: F401
    ForkRequest,
    PublishRequest,
    SandboxConfigInput,
    SandboxCreateParams,
    SandboxStatus,
    SandboxSummary,
    SandboxTemplateStatus,
    SandboxUpdateParams,
)
from targon.client.workload import PortConfig  # noqa: F401
from targon.core.exceptions import (  # noqa: F401
    PayloadTooLargeError,
    SandboxStateError,
    TerminalLimitError,
    ValidationError,
)


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
    deploy_url = client.sandbox_no_retry_session.request.call_args_list[1].args[1]
    assert deploy_url.endswith("/workloads/wrk-1/deploy")


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
        Response(body=operation_body(status="frozen")),
        Response(body=operation_body()),
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
    calls = client.sandbox_no_retry_session.request.call_args_list
    assert calls[0].kwargs["json"]["sandbox_config"] == {"idle_timeout_sec": 0}
    assert calls[2].kwargs["json"]["sandbox_config"] == {"idle_timeout_sec": 0}


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
    assert child.uid == "wrk-child" and template.status is SandboxTemplateStatus.PENDING
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
