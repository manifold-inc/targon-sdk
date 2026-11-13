import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread

import pytest
import requests

from targon import Client
from targon.core.exceptions import APIError

# --- Loopback fixture helpers ---


class _HTTPStub:
    def __init__(self):
        self.received = Event()
        self.release_response = Event()
        self.calls = []
        self.errors = []
        self.received_at = 0.0
        self.responded_at = 0.0
        self.status = 200
        self.body = {"stdout": "done\n", "stderr": "", "code": 0, "timed_out": False}
        self.base_url = ""


@pytest.fixture
def http_stub():
    """Keep responses silent until released without leaving blocked server workers."""
    stub = _HTTPStub()
    stopping = Event()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(1)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.calls.append((self.command, self.path, body))
                stub.received_at = time.monotonic()
                stub.received.set()
                if not stub.release_response.wait(timeout=3):
                    raise AssertionError("test did not release the HTTP response")
                stub.responded_at = time.monotonic()
                payload = json.dumps(stub.body).encode("utf-8")
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Request-ID", "req-exec-test")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                # The ordinary-timeout control intentionally disconnects first.
                pass
            except Exception as exc:
                stub.errors.append(exc)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.05
    stub.base_url = "http://127.0.0.1:{}".format(server.server_address[1])

    def serve():
        while not stopping.is_set():
            server.handle_request()

    worker = Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield stub
    finally:
        stub.release_response.set()
        stopping.set()
        worker.join(timeout=5)
        server.server_close()
        assert not worker.is_alive(), "loopback server did not stop"
        assert not stub.errors, stub.errors


@pytest.fixture
def http_client(http_stub):
    with Client(
        api_key="pat_fake_loopback",
        org="fake-org",
        profile="fake-profile",
        base_url=http_stub.base_url,
        timeout=0.1,
        max_retries=3,
    ) as client:
        # Avoid proxy and netrc discovery even on machines with local credentials.
        client.session.trust_env = False
        client.sandbox_no_retry_session.trust_env = False
        yield client


# --- End loopback fixture helpers ---


def test_exec_receives_silent_result_after_ordinary_timeout(http_stub, http_client):
    finished = Event()
    results = []
    failures = []

    def execute():
        try:
            results.append(
                http_client.sandboxes.exec(
                    "wrk-1", "sleep 0.4; printf done", timeout_sec=5
                )
            )
        except Exception as exc:
            failures.append(exc)
        finally:
            finished.set()

    worker = Thread(target=execute, daemon=True)
    worker.start()
    try:
        assert http_stub.received.wait(timeout=2), "exec POST did not reach the server"
        # This deliberate silence crosses the 0.1s transport timeout on the base.
        assert not finished.wait(timeout=0.4), failures
        http_stub.release_response.set()
        assert finished.wait(timeout=2), "exec did not receive the released response"
        assert not failures, failures
        result = results[0]
        assert (result.stdout, result.stderr, result.code, result.timed_out) == (
            "done\n",
            "",
            0,
            False,
        )
        assert 0.1 < http_stub.responded_at - http_stub.received_at < 5
        assert http_stub.calls == [
            (
                "POST",
                "/tha/v3/orgs/fake-org/workloads/wrk-1/exec",
                {"cmd": "sleep 0.4; printf done", "timeout_sec": 5},
            )
        ]
    finally:
        http_stub.release_response.set()
        worker.join(timeout=3)
        assert not worker.is_alive(), "exec caller did not stop"


def test_ordinary_sandbox_request_still_uses_client_timeout(http_stub, http_client):
    with pytest.raises(requests.ReadTimeout):
        http_client.sandboxes.mint_access_ticket("wrk-1")
    assert http_stub.received.is_set()
    assert http_client.config.timeout == 0.1
    assert len(http_stub.calls) == 1
    assert http_stub.calls[0][1].endswith("/access-tickets")


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_exec_posts_once_on_server_failure(http_stub, http_client, status):
    http_stub.status = status
    http_stub.body = {"error": "exec failed", "reason": "EXEC_TEST_FAILURE"}
    http_stub.release_response.set()

    with pytest.raises(APIError) as caught:
        http_client.sandboxes.exec("wrk-1", "true", timeout_sec=1)

    assert caught.value.status_code == status
    assert caught.value.reason == "EXEC_TEST_FAILURE"
    assert caught.value.workload_uid == "wrk-1"
    assert caught.value.request_id == "req-exec-test"
    assert len(http_stub.calls) == 1
    assert http_stub.calls[0][0] == "POST"
