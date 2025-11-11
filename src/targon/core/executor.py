from contextlib import asynccontextmanager
import dataclasses
import asyncio
import time
import inspect
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

from targon.client.publish import PublishResponse
from targon.client.app import AppResponse, AppStatusResponse
from targon.client.client import Client
from targon.core.console import Console
from targon.core.objects import BaseApp, _Object
from targon.core.exceptions import ValidationError, ServerlessError
from targon.core.resolver import Resolver
from targon.core.function import _Function
from targon.core.utils import check_object_name


@dataclasses.dataclass
class RunningApp:
    app_id: str
    app_name: str
    project_name: str
    function_ids: Dict[str, str] = dataclasses.field(default_factory=dict)
    class_ids: Dict[str, str] = dataclasses.field(default_factory=dict)
    app_page_url: Optional[str] = None


async def _init_local_app_from_name(
    client: Client,
    name: str,
    project_name: str = "default",
) -> RunningApp:

    app_resp: AppResponse = await client.async_app.get_app(
        name=name,
        project_name=project_name,
    )
    return RunningApp(
        app_id=app_resp.AppID,
        app_name=app_resp.Name,
        project_name=app_resp.ProjectName,
        app_page_url=app_resp.WebURL,
    )


async def _create_all_objects(
    client: Client,
    running_app: RunningApp,
    functions: Dict[str, _Function],
    classes: Dict[str, _Function],
    console_instance: Optional[Console] = None,
    app_file: Optional[Path] = None,
    app_module_name: str = "app",
) -> None:

    indexed_objects: Dict[str, _Object] = {**functions, **classes}
    resolver = Resolver(
        client,
        app_id=running_app.app_id,
        project_name=running_app.project_name,
        console_instance=console_instance,
        app_file=app_file,
        app_module_name=app_module_name,
    )

    if console_instance:
        console_instance.step("Creating Objects")

    async def _load(tag: str, obj: _Object) -> None:
        await resolver.load(obj)
        if _Function._is_id_type(obj.object_id):
            running_app.function_ids[tag] = obj.object_id
            if console_instance:
                console_instance.resource(tag, obj.object_id)
        else:
            raise ServerlessError(
                f"Failed to create object: unexpected object ID format",
                function_id=obj.object_id,
            )

    await asyncio.gather(*(_load(tag, obj) for tag, obj in indexed_objects.items()))

    if console_instance:
        function_count = len(running_app.function_ids)
        console_instance.success(
            f"Created {function_count} function{'s' if function_count != 1 else ''}"
        )


async def _publish_app(
    client: Client,
    running_app: RunningApp,
) -> PublishResponse:
    function_uids = list(running_app.function_ids.values())
    if not function_uids:
        raise ValidationError(
            "No functions were created; nothing to publish.",
            field="functions",
            value=[],
        )

    response = await client.async_publish.publish_serverless(
        app_id=running_app.app_id,
        app_name=running_app.app_name,
        functions=function_uids,
    )

    running_app.app_page_url = response.web_url
    return response


async def deploy_app(
    app: BaseApp,
    client: Client,
    console_instance: Optional[Console] = None,
    name: Optional[str] = None,
    app_file_path: Optional[str] = None,
    running_app: Optional[RunningApp] = None,
) -> None:

    resolved_name = name or app.name or ""
    if not resolved_name or not resolved_name.strip():
        raise ValidationError(
            "You must provide a deployment name. Example: app = targon.App(name=\"my-app\")",
            field="name",
            value=resolved_name,
        )

    check_object_name(resolved_name)

    t0 = time.time()

    if app_file_path:
        # Use the file path provided by the CLI (the actual user script)
        app_file: Optional[Path] = Path(app_file_path).resolve()
        app_module_name = app_file.stem
    else:
        try:
            app_file = Path(inspect.getfile(app.__class__)).resolve()
            app_module_name = app_file.stem
        except (TypeError, OSError):
            app_file = None
            app_module_name = "app"

    if not running_app:
        if console_instance:
            console_instance.step("Initializing app")
        running_app = await _init_local_app_from_name(
            client, resolved_name, project_name="default"
        )
        if console_instance:
            console_instance.success("Initialized")

    await _create_all_objects(
        client,
        running_app,
        app._functions,
        app._classes,
        console_instance=console_instance,
        app_file=app_file,
        app_module_name=app_module_name,
    )
    if console_instance:
        console_instance.step("Publishing app")
    publish_response = await _publish_app(client, running_app)
    if console_instance:
        console_instance.success("Published")

    total_time = time.time() - t0

    if publish_response.summary and console_instance:
        summary_msg = f"Functions: {publish_response.summary.deployed}/{publish_response.summary.total_functions} deployed"
        if publish_response.summary.failed > 0:
            summary_msg += f", {publish_response.summary.failed} failed"

        details_list = [
            f"URL: {running_app.app_page_url}" if running_app.app_page_url else None,
            summary_msg,
            "",
        ]

        # Add invocation instructions for each function
        for tag, func_obj in app._functions.items():
            func_id = running_app.function_ids.get(tag)
            if not func_id:
                continue

            # Check if function has webhook config (is a web endpoint)
            if not func_obj._webhook_config:
                # It's a webhook endpoint - suggest using targon run with local entry point
                details_list.append(
                    f"  • {tag}: use 'targon run' with local entry point"
                )
            else:
                # No webhook config - show gRPC invoke URL
                details_list.append(f"  • {tag}: {func_obj.web_url}")

        details_list.extend(
            [
                "",
                "Next steps:",
                f"  • View logs: targon app logs {resolved_name}",
            ]
        )

        console_instance.final(
            f"Deployed successfully in {total_time:.1f}s!", details_list
        )


@asynccontextmanager
async def run_app(
    app: BaseApp,
    *,
    client: Optional[Client] = None,
    console_instance: Optional[Console] = None,
    name: Optional[str] = None,
    app_file_path: Optional[str] = None,
) -> AsyncGenerator[BaseApp, None]:
    if not app_file_path:
        raise ValidationError(
            "Unable to locate the script file",
            field="app_file_path",
            value=app_file_path,
        )

    if not client:
        client = Client.from_env()

    if console_instance:
        console_instance.step("Initializing app")
    running_app = await _init_local_app_from_name(
        client, name or app.name or "", "default"
    )
    if console_instance:
        console_instance.success("Initialized")

    await deploy_app(app, client, console_instance, name, app_file_path, running_app)

    try:
        if console_instance:
            console_instance.step("Executing the function")
        yield app

    except Exception as e:
        if console_instance:
            console_instance.error(f"Failed to run app: {e}")
        raise

    finally:
        if console_instance:
            console_instance.success("Local execution completed")
