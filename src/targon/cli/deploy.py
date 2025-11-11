import click
import asyncio
from ..client.client import Client
from ..core.exceptions import TargonError, APIError
from ..core.executor import deploy_app
from .imports import parse_import_ref, import_app_from_ref
from ..core.console import console
from ..core.console import _rich_console


@click.command()
@click.argument("app", required=True)
@click.option("--name", help="Name of the deployment.")
@click.pass_context
def deploy(ctx, app, name):
    """Deploy an application to Targon."""
    client: Client = ctx.obj["client"]
    import_ref = parse_import_ref(app)
    app_obj = import_app_from_ref(import_ref)

    # Use app name for console display
    display_name = name or app_obj.name or "app"

    try:
        with console(display_name) as c:
            # Run async deployment with console
            asyncio.run(
                deploy_app(
                    app=app_obj,
                    name=name,
                    client=client,
                    console_instance=c,
                    app_file_path=import_ref.file_path,
                )
            )

    except (TargonError, APIError) as e:
        _rich_console.print(
            f"\n[bold red]✖[/bold red] [bold]Deployment failed:[/bold] {e}\n"
        )
        raise SystemExit(1)
