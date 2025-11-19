import click
import asyncio
from rich.table import Table
from rich.console import Console
from targon.client.client import Client
from targon.core.exceptions import TargonError, APIError
from datetime import datetime

console = Console(stderr=True)


@click.group()
def app():
    """Manage Targon applications."""
    pass


@click.pass_context
def list_apps(ctx):
    """List targon apps that are currently deployed/running or recently stopped."""
    client: Client = ctx.obj["client"]

    async def _list_apps_with_functions(c: Client):
        async with c:
            # First get the list of apps
            apps_response = await c.async_app.list_apps()
            
            # Then fetch detailed status for each app to get function information
            detailed_apps = []
            for app in apps_response.apps:
                try:
                    app_status = await c.async_app.get_app_status(app.app_id)
                    detailed_apps.append(app_status)
                except Exception:
                    # If we can't get status for an app, include it with basic info
                    detailed_apps.append(None)
            
            return apps_response, detailed_apps

    try:
        # Run the async function
        response, detailed_apps = asyncio.run(_list_apps_with_functions(client))

        if not response.apps:
            console.print("\n[bright_blue]ℹ[/bright_blue] No apps found.")
            return

        console.print()
        console.print(f"[bold bright_cyan]Targon Apps[/bold bright_cyan] [dim bright_black]({response.total} total)[/dim bright_black]")
        console.print()
        
        # Display apps and their functions in hierarchical view
        for i, app_item in enumerate(response.apps):
            detailed_app = detailed_apps[i]
            console.print(f"[bold bright_cyan]└─ {app_item.name}[/bold bright_cyan] [dim]({app_item.app_id})[/dim]")
            
            if detailed_app and detailed_app.functions:
                for func_id, func_data in detailed_app.functions.items():
                    status_display = get_status_display(func_data.status) if func_data.status else "[dim]●[/dim] Unknown"
                    console.print(f"     {status_display} [bold]{func_data.name}[/bold] [dim]({func_data.uid})[/dim]")
                    if func_data.url:
                        console.print(f"     [dim]URL:[/dim] [bright_blue]{func_data.url}[/bright_blue]")
            else:
                console.print(f"   [dim]No functions deployed[/dim]")
            
            console.print()
        
        console.print()

    except (TargonError, APIError) as e:
        console.print(f"\n[bold red]✖[/bold red] [bold]Error:[/bold] {e}\n")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[bold red]✖[/bold red] [bold]Unexpected error:[/bold] {e}\n")
        raise SystemExit(1)

app.command("list")(list_apps)
app.command("ls")(list_apps)


@app.command("get")
@click.argument("identifier", required=True)
@click.pass_context
def app_get(ctx, identifier):
    """Get detailed information about an app or function.

    IDENTIFIER: App ID or Function UID (e.g., app-xxxxx or fnc-xxxxx)
    """
    client: Client = ctx.obj["client"]

    if identifier.startswith("fnc-"):
        # Handle function details
        async def _get_function_details(c: Client):
            async with c:
                return await c.async_app.get_function(identifier)

        try:
            response = asyncio.run(_get_function_details(client))

            # Display function information
            console.print()
            console.print(f"[bold bright_cyan]{response.name}[/bold bright_cyan]")
            console.print(f"[dim]Function UID:[/dim] [bright_cyan]{response.uid}[/bright_cyan]")
            console.print(f"[dim]App ID:[/dim]       [bright_blue]{response.app_id}[/bright_blue]")
            if response.status:
                console.print(f"[dim]Status:[/dim]       {get_status_display(response.status)}")
            console.print(f"[dim]Module:[/dim]       {response.module or '[dim]-[/dim]'}")
            console.print(f"[dim]Qualname:[/dim]     {response.qualname or '[dim]-[/dim]'}")
            console.print(f"[dim]Image ID:[/dim]     {response.image_id or '[dim]-[/dim]'}")
            console.print(f"[dim]Resource:[/dim]     {response.resource_name or '[dim]-[/dim]'}")
            console.print(f"[dim]URL:[/dim]          {response.url or '[dim]-[/dim]'}")
            
            # Display timeout settings if available
            if response.timeout_secs is not None or response.startup_timeout is not None:
                console.print()
                console.print("[bold bright_cyan]Timeouts:[/bold bright_cyan]")
                if response.timeout_secs is not None:
                    console.print(f"  [dim]Execution:[/dim] {response.timeout_secs}s")
                if response.startup_timeout is not None:
                    console.print(f"  [dim]Startup:[/dim]   {response.startup_timeout}s")
            
            # Display webhook config if available
            if response.webhook_config:
                console.print()
                console.print("[bold bright_cyan]Webhook Configuration:[/bold bright_cyan]")
                console.print(f"  [dim]Type:[/dim]         {response.webhook_config.type}")
                console.print(f"  [dim]Method:[/dim]       {response.webhook_config.method}")
                console.print(f"  [dim]Requires Auth:[/dim] {response.webhook_config.requires_auth}")
                if response.webhook_config.port:
                    console.print(f"  [dim]Port:[/dim]         {response.webhook_config.port}")
                if response.webhook_config.label:
                    console.print(f"  [dim]Label:[/dim]        {response.webhook_config.label}")
                console.print(f"  [dim]Docs:[/dim]         {response.webhook_config.docs}")
            
            # Display autoscaler settings if available
            if response.autoscaler_settings:
                console.print()
                console.print("[bold bright_cyan]Autoscaler Settings:[/bold bright_cyan]")
                console.print(f"  [dim]Min Replicas:[/dim]              {response.autoscaler_settings.min_replicas}")
                console.print(f"  [dim]Max Replicas:[/dim]              {response.autoscaler_settings.max_replicas}")
                if response.autoscaler_settings.initial_replicas is not None:
                    console.print(f"  [dim]Initial Concurrency:[/dim]     {response.autoscaler_settings.container_concurrency}")
                if response.autoscaler_settings.container_concurrency is not None:
                    console.print(f"  [dim]Container Concurrency:[/dim]     {response.autoscaler_settings.container_concurrency}")
                if response.autoscaler_settings.target_concurrency is not None:
                    console.print(f"  [dim]Target Concurrency:[/dim]        {response.autoscaler_settings.target_concurrency}")
                if response.autoscaler_settings.scaling_metric:
                    console.print(f"  [dim]Scaling Metric:[/dim]            {response.autoscaler_settings.scaling_metric}")
                if response.autoscaler_settings.target_value is not None:
                    console.print(f"  [dim]Target Value:[/dim]              {response.autoscaler_settings.target_value}")
            
            console.print()
            console.print(f"[dim]Created:[/dim] {format_timestamp(response.created_at)}")
            console.print(f"[dim]Updated:[/dim] {format_timestamp(response.updated_at)}")
            
            console.print()
            console.print(f"[dim]Tips:[/dim]")
            console.print(f"  • View logs: [cyan]targon logs {response.uid}[/cyan]")
            console.print(f"  • View app:  [cyan]targon app get {response.app_id}[/cyan]")
            console.print()

        except APIError as e:
            console.print(f"\n[bold red]✖[/bold red] [bold]Error:[/bold] {e}\n")
            if hasattr(e, 'is_not_found') and e.is_not_found:
                console.print(f"[dim]Function '{identifier}' may not exist or may not be deployed.[/dim]")
            raise SystemExit(1)
        except TargonError as e:
            console.print(f"\n[bold red]✖[/bold red] [bold]Error:[/bold] {e}\n")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"\n[bold red]✖[/bold red] [bold]Unexpected error:[/bold] {e}\n")
            raise SystemExit(1)
    elif identifier.startswith("app-"):
        # Handle app status (existing functionality)
        async def _get_app_status(c: Client):
            async with c:
                return await c.async_app.get_app_status(identifier)

        try:
            response = asyncio.run(_get_app_status(client))

            # Display app information
            console.print()
            console.print(f"[bold bright_cyan]{response.name}[/bold bright_cyan]")
            console.print(f"[dim]App ID:[/dim]        [bright_cyan]{response.app_id}[/bright_cyan]")
            console.print(f"[dim]Project ID:[/dim]    {response.project_id or '[dim]-[/dim]'}")
            console.print(f"[dim]Project Name:[/dim]  {response.project_name or '[dim]-[/dim]'}")
            console.print(f"[dim]Function Count:[/dim] {response.function_count}")
            console.print(f"[dim]Created:[/dim]       {format_timestamp(response.created_at)}")
            console.print(f"[dim]Updated:[/dim]       {format_timestamp(response.updated_at)}")

            # Display functions if any exist
            if response.functions:
                table = Table(
                    title=f"[bold bright_cyan]Functions[/bold bright_cyan] [dim bright_black]({response.function_count} total)[/dim bright_black]",
                    caption=f"[dim]App ID: {response.app_id}[/dim]",
                    border_style="dim bright_black",
                    header_style="bold bright_cyan",
                    show_lines=False,
                )
                table.add_column("UID", style="bright_cyan", no_wrap=True)
                table.add_column("Name", style="bold")
                table.add_column("Status", style="")
                table.add_column("URL", style="bright_blue")

                for func_id, func_data in response.functions.items():
                    table.add_row(
                        func_data.uid,
                        func_data.name,
                        get_status_display(func_data.status),
                        func_data.url or "[dim]-[/dim]",
                    )

                console.print()
                console.print(table)
            else:
                console.print("\n[bright_blue]ℹ[/bright_blue] No functions deployed in this app.")

            console.print()

        except (TargonError, APIError) as e:
            console.print(f"\n[bold red]✖[/bold red] [bold]Error:[/bold] {e}\n")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"\n[bold red]✖[/bold red] [bold]Unexpected error:[/bold] {e}\n")
            raise SystemExit(1)
    else:
        console.print("[yellow]⚠[/yellow] Use [cyan]'targon app get <fnc-xxx> or <app-xxx>'[/cyan].\n")


# Keep 'status' as an alias for backwards compatibility
@app.command("status", hidden=True)
@click.argument("app_id", required=True)
@click.pass_context
def app_status(ctx, app_id):
    """[Deprecated] Use 'targon app get' instead."""
    console.print("[yellow]⚠[/yellow] [dim]'targon app status' is deprecated. Use [cyan]'targon app get'[/cyan] instead.[/dim]\n")
    ctx.invoke(app_get, identifier=app_id)


@app.command("delete")
@click.argument("app_ids", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete_app(ctx, app_ids, yes):
    """Delete one or more Targon apps and all their deployments."""
    client: Client = ctx.obj["client"]

    if not app_ids:
        console.print("[bold red]✖[/bold red] At least one app ID is required")
        raise SystemExit(1)

    if not yes:
        if len(app_ids) == 1:
            click.confirm(
                f"Are you sure you want to delete app '{app_ids[0]}' and all its deployments?",
                abort=True,
            )
        else:
            console.print(f"\n[bold]Apps to delete:[/bold]")
            for app_id in app_ids:
                console.print(f"  • [bright_cyan]{app_id}[/bright_cyan]")
            console.print()
            click.confirm(
                f"Are you sure you want to delete these {len(app_ids)} apps and all their deployments?",
                abort=True,
            )

    async def _delete_apps(c: Client):
        successful = []
        failed = []
        
        async with c:
            # Delete each app
            for app_id in app_ids:
                try:
                    with console.status(
                        f"[bold cyan]Deleting app [bright_magenta]{app_id}[/bright_magenta]...[/bold cyan]",
                        spinner="dots",
                    ):
                        result = await c.async_app.delete_app(app_id)

                    # Display success message
                    console.print(
                        f"[bold green]✔[/bold green] [bold]Successfully deleted app:[/bold] [bright_cyan]{app_id}[/bright_cyan]"
                    )

                    # Display additional info if available
                    if isinstance(result, dict):
                        if result.get("message"):
                            console.print(f"[dim italic]  {result['message']}[/dim italic]")
                        if result.get("deleted_resources"):
                            console.print(
                                f"[dim italic]  Deleted resources: {result['deleted_resources']}[/dim italic]"
                            )

                    successful.append(app_id)

                except (TargonError, APIError) as e:
                    console.print(
                        f"[bold red]✖[/bold red] [bold]Failed to delete app:[/bold] [bright_cyan]{app_id}[/bright_cyan]"
                    )
                    console.print(f"[dim italic]  Error: {e}[/dim italic]")
                    failed.append((app_id, str(e)))
                except Exception as e:
                    console.print(
                        f"[bold red]✖[/bold red] [bold]Unexpected error deleting app:[/bold] [bright_cyan]{app_id}[/bright_cyan]"
                    )
                    console.print(f"[dim italic]  Error: {e}[/dim italic]")
                    failed.append((app_id, str(e)))
        
        return successful, failed

    # Execute all deletions
    successful, failed = asyncio.run(_delete_apps(client))

    # Summary
    console.print()
    if len(app_ids) > 1:
        console.print(f"[bold]Summary:[/bold]")
        console.print(f"  [green]Successful:[/green] {len(successful)}/{len(app_ids)}")
        if failed:
            console.print(f"  [red]Failed:[/red] {len(failed)}/{len(app_ids)}")
        console.print()

    # Exit with error if any deletions failed
    if failed:
        raise SystemExit(1)


@app.command("rm")
@click.argument("app_ids", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def remove_app(ctx, app_ids, yes):
    """Delete one or more Targon apps and all their deployments (alias for delete)."""
    ctx.invoke(delete_app, app_ids=app_ids, yes=yes)


@app.command("functions")
@click.argument("app_id", required=True)
@click.pass_context
def list_functions(ctx, app_id):
    """List all functions for a given app."""
    client: Client = ctx.obj["client"]

    async def _list_functions(c: Client):
        async with c:
            return await c.async_app.list_functions(app_id)

    try:
        response = asyncio.run(_list_functions(client))

        if not response.functions:
            console.print(
                f"\n[bright_blue]ℹ[/bright_blue] No functions found for app: [bright_cyan]{app_id}[/bright_cyan]"
            )
            return

        table = Table(
            title=f"[bold bright_cyan]Functions[/bold bright_cyan] [dim bright_black]({response.total} total)[/dim bright_black]",
            caption=f"[dim]App ID: {app_id}[/dim]",
            border_style="dim bright_black",
            header_style="bold bright_cyan",
            show_lines=False,
        )
        table.add_column("UID", style="bright_cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Module", style="bright_blue")
        table.add_column("Qualname", style="bright_magenta")
        table.add_column("Image ID", style="yellow", no_wrap=True)
        table.add_column("Created", style="dim")
        table.add_column("Updated", style="dim")

        for func in response.functions:
            created_at = format_timestamp(func.created_at)
            updated_at = format_timestamp(func.updated_at)

            table.add_row(
                func.uid,
                func.name,
                func.module or "[dim]-[/dim]",
                func.qualname or "[dim]-[/dim]",
                func.image_id or "[dim]-[/dim]",
                created_at,
                updated_at,
            )

        console.print()
        console.print(table)
        console.print()

    except (TargonError, APIError) as e:
        console.print(f"\n[bold red]✖[/bold red] [bold]Error:[/bold] {e}\n")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[bold red]✖[/bold red] [bold]Unexpected error:[/bold] {e}\n")
        raise SystemExit(1)


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp string for display."""
    if not timestamp_str:
        return "-"

    try:
        # Parse ISO format timestamp
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return timestamp_str


def get_status_display(status: str) -> str:
    """Return formatted status with icon and color."""
    status_lower = status.lower()
    if status_lower in ["running", "active", "deployed"]:
        return f"[bold green]●[/bold green] {status}"
    elif status_lower in ["stopped", "inactive"]:
        return f"[bold red]●[/bold red] {status}"
    elif status_lower in ["pending", "deploying"]:
        return f"[bold yellow]●[/bold yellow] {status}"
    else:
        return f"[dim]●[/dim] {status}"
