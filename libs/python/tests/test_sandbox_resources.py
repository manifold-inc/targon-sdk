import base64
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest
from conftest import Response, sandbox_body

from targon import (
    ForkRequest,
    PortProtocol,
    PublishRequest,
    SandboxStatus,
    SandboxUpdateParams,
)


def test_models_are_frozen_and_enums_are_closed(client):
    client.session.request.return_value = Response(body=sandbox_body())
    sandbox = client.sandboxes.get("wrk-1")
    assert sandbox.uid == "wrk-1" and not hasattr(sandbox, "_data")
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

    for method in ("get", "get_state", "update", "freeze", "thaw", "fork", "publish"):
        setattr(service, method, Mock(return_value=sentinel))
    service.delete = Mock()
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
    assert sandbox.files.read("/tmp/x") == b"data"
    sandbox.files.write("/tmp/y", b"data")
    assert sandbox.mint_access_ticket() is sentinel
    assert sandbox.get_desktop() is sentinel
    assert sandbox.terminals.list() == ()
    assert sandbox.terminals.create() is sentinel
    sandbox.terminals.delete("term-1")
    assert sandbox.terminals.connect("term-1") is sentinel
    for removed_alias in (
        "read_file",
        "write_file",
        "list_terminals",
        "create_terminal",
        "delete_terminal",
        "connect_terminal",
    ):
        assert not hasattr(sandbox, removed_alias)

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
    payload = client.sandbox_no_retry_session.request.call_args.kwargs["json"]
    assert payload["content_b64"] == "/w=="


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
    assert ticket.ticket == "sat_once" and terminal.id == "term-1"
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
        ticket_url = client.sandbox_no_retry_session.request.call_args.args[1]
        assert ticket_url.endswith("/workloads/wrk-1/access-tickets")
        assert connect.call_args.args[0].endswith(
            "/workloads/wrk-1/terminals/term-1/ws?ticket=sat_once"
        )
        connection.send(b"input")
        assert connection.recv() == b"hello"
        websocket.send.assert_called_once_with(b"input")
        with pytest.raises(TypeError):
            connection.send("not bytes")
