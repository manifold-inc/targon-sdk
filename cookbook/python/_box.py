"""Shared create/delete setup for the sandbox cookbook recipes."""

import time
from contextlib import contextmanager

import targon


def open_client():
    return targon.Client.from_env()


def ready_template(client):
    page = client.sandboxes.templates.list(status=targon.SandboxTemplateStatus.READY)
    if not page.items:
        raise SystemExit("no READY sandbox template is available")
    return page.items[0]


def sandbox_name(action):
    # 1-32 lowercase letters, digits, or hyphens. Cannot start or end with '-'.
    return "cb-%s-%d" % (action, int(time.time()) % 1_000_000)


@contextmanager
def sandbox(client, action):
    """Create a short-lived sandbox and delete it on the way out."""
    created = ready_template(client).create_sandbox(
        sandbox_name(action),
        ttl_sec=900,
        idle_timeout_sec=300,
        timeout=600,
        poll_interval=2,
    )
    print("created %s (%s)" % (created.uid, created.name))
    try:
        yield created
    finally:
        print("deleting %s" % created.uid)
        created.delete()
