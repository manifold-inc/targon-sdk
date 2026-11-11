import json
from unittest.mock import Mock

import pytest

from targon import Client


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
    value = Client("pat_test", org="acme", base_url="https://api.test", max_retries=3)
    value.session.request = Mock()
    value.sandbox_no_retry_session.request = Mock()
    yield value
    value.close()


def sandbox_body(uid="wrk-1", status="running"):
    return {
        **sandbox_summary_body(uid, status),
        "image": "sbt-python",
        "sandbox_config": {
            "template_uid": "sbt-python",
            "ttl_sec": 3600,
            "idle_timeout_sec": 300,
        },
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


def operation_body(uid="wrk-1", status="provisioning"):
    return {"uid": uid, "type": "", "state": {"status": status}}


def hydrated_template(client, status="READY"):
    body = template_body(status=status)
    body["uid"] = "sbt-python"
    return client.sandboxes.templates._hydrate(body)
