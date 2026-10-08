from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest
from conftest import (
    Response,
    hydrated_template,
    operation_body,
    sandbox_body,
    template_body,
)

from targon import (
    SandboxCreateParams,
    SandboxStatus,
    SandboxTemplateKind,
    SandboxTemplateStatus,
)
from targon.core.exceptions import ValidationError


def test_create_accepts_sparse_operations_with_fetched_template(client):
    client.session.request.side_effect = [
        Response(body=template_body()),
        Response(body=sandbox_body(status="provisioning")),
    ]
    client.sandbox_no_retry_session.request.side_effect = [
        Response(body=operation_body(status="registered")),
        Response(body=operation_body()),
    ]

    template = client.sandboxes.templates.get("sbt-user")
    sandbox = template.create_sandbox("devbox", wait_until_running=False)

    assert sandbox.uid == "wrk-1"
    assert sandbox.state.status is SandboxStatus.PROVISIONING
    assert client.sandbox_no_retry_session.request.call_count == 2


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


def test_template_resource_create_rejects_removed_params_overload(client):
    template = hydrated_template(client)
    params = SandboxCreateParams(name="devbox", template=template)

    with pytest.raises(TypeError):
        template.create_sandbox(request=params)


@pytest.mark.parametrize("invalid_template", ["unbound", "empty_uid", "pending"])
def test_create_requires_bound_ready_template_with_uid(client, invalid_template):
    template = hydrated_template(client)
    if invalid_template == "unbound":
        template = replace(template, _client=None)
    elif invalid_template == "empty_uid":
        template = replace(template, uid="")
    else:
        template = hydrated_template(client, status="PENDING")

    with pytest.raises(ValidationError):
        client.sandboxes.create(SandboxCreateParams(name="devbox", template=template))
    client.sandbox_no_retry_session.request.assert_not_called()
