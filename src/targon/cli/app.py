import click
import asyncio
from rich.table import Table
from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Any, Union

from targon.client.client import Client
from targon.core.exceptions import TargonError, APIError
from datetime import datetime

console = Console(stderr=True)


def display_error(err: Union[Exception, str], title: str = "Error"):
    """Display a formatted error message without heavy borders."""
    console.print()
    console.print(f"[red]{str(err)}[/red]")
    console.print()


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
        return f"[bold green]● {status}[/bold green]"
    elif status_lower in ["stopped", "inactive"]:
        return f"[bold red]● {status}[/bold red]"
    elif status_lower in ["pending", "deploying", "building"]:
        return f"[bold yellow]● {status}[/bold yellow]"
    else:
        return f"[dim]● {status}[/dim]"


@click.group()
def app():
    """Manage Targon applications."""
    pass


@click.pass_context
def list_apps(ctx):
    """List targon apps that are currently deployed/running."""
    client: Client = ctx.obj["client"]

    try:
        with console.status("[bold cyan]Fetching apps.[/bold cyan]", spinner="dots"):
            response, detailed_apps = client.run_async(
                lambda: client.async_app.list_apps_with_details()
            )

        if not response.apps:
            console.print()
            console.print("[dim]No applications found. Deploy one using [cyan]targon deploy[/cyan].[/dim]")
            console.print()
            return

        # Create a tree for hierarchical view
        tree = Tree(
            f"[bold bright_cyan]Targon Apps[/bold bright_cyan] [dim]({response.total} total)[/dim]",
            guide_style="bright_black"
        )

        for i, app_item in enumerate(response.apps):
            detailed_app = detailed_apps[i]
            app_label = Text()
            app_label.append(app_item.name, style="bold bright_cyan")
            app_label.append(f" ({app_item.app_id})", style="dim")
            
            app_node = tree.add(app_label)
            
            if detailed_app and detailed_app.functions:
                for func_id, func_data in detailed_app.functions.items():
                    status_text = get_status_display(func_data.status) if func_data.status else "[dim]● Unknown[/dim]"
                    
                    func_label = Text.from_markup(f"[bold]{func_data.name}[/bold] {status_text} [dim]({func_data.uid})[/dim]")
                    func_node = app_node.add(func_label)
                    
                    if func_data.url:
                        func_node.add(f"[dim]URL:[/dim] [bright_blue underline]{func_data.url}[/bright_blue underline]")
                    
            else:
                app_node.add("[dim italic]No functions deployed[/dim italic]")
            
        console.print()
        console.print(tree)
        console.print()

    except (TargonError, APIError) as e:
        display_error(e, "Failed to list apps")
        raise SystemExit(1)
    except Exception as e:
        display_error(e, "Unexpected error")
        raise SystemExit(1)


app.command("list")(list_apps)
app.command("ls")(list_apps)


def _display_function_details(response: Any):
    """Helper to display function details."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", justify="right")
    grid.add_column()

    # Basic Info
    grid.add_row("Function UID:", f"[bright_cyan]{response.uid}[/bright_cyan]")
    grid.add_row("App ID:", f"[bright_blue]{response.app_id}[/bright_blue]")
    if response.status:
        grid.add_row("Status:", get_status_display(response.status))
    
    grid.add_row("Module:", str(response.module or "[dim]-[/dim]"))
    grid.add_row("Qualname:", str(response.qualname or "[dim]-[/dim]"))
    grid.add_row("Image ID:", str(response.image_id or "[dim]-[/dim]"))
    grid.add_row("Resource:", str(response.resource_name or "[dim]-[/dim]"))
    grid.add_row("URL:", f"[bright_blue underline]{response.url}[/bright_blue underline]" if response.url else "[dim]-[/dim]")
    grid.add_row("Created:", format_timestamp(response.created_at))
    grid.add_row("Updated:", format_timestamp(response.updated_at))

    # Timeouts
    if response.timeout_secs is not None or response.startup_timeout is not None:
        grid.add_row("", "")  # Spacer
        grid.add_row("[bold bright_cyan]Timeouts[/bold bright_cyan]", "")
        if response.timeout_secs is not None:
            grid.add_row("Execution:", f"{response.timeout_secs}s")
        if response.startup_timeout is not None:
            grid.add_row("Startup:", f"{response.startup_timeout}s")

    # Webhook Config
    if response.webhook_config:
        grid.add_row("", "")  # Spacer
        grid.add_row("[bold bright_cyan]Webhook Configuration[/bold bright_cyan]", "")
        grid.add_row("Type:", str(response.webhook_config.type))
        grid.add_row("Method:", str(response.webhook_config.method))
        grid.add_row("Auth Required:", str(response.webhook_config.requires_auth))
        if response.webhook_config.port:
            grid.add_row("Port:", str(response.webhook_config.port))
        if response.webhook_config.label:
            grid.add_row("Label:", str(response.webhook_config.label))
        grid.add_row("Docs:", str(response.webhook_config.docs))

    # Autoscaler Settings
    if response.autoscaler_settings:
        grid.add_row("", "")  # Spacer
        grid.add_row("[bold bright_cyan]Autoscaler Settings[/bold bright_cyan]", "")
        grid.add_row("Min Replicas:", str(response.autoscaler_settings.min_replicas))
        grid.add_row("Max Replicas:", str(response.autoscaler_settings.max_replicas))
        
        if response.autoscaler_settings.container_concurrency is not None:
            grid.add_row("Container Concurrency:", str(response.autoscaler_settings.container_concurrency))
        if response.autoscaler_settings.target_concurrency is not None:
            grid.add_row("Target Concurrency:", str(response.autoscaler_settings.target_concurrency))
        if response.autoscaler_settings.scaling_metric:
            grid.add_row("Scaling Metric:", str(response.autoscaler_settings.scaling_metric))
        if response.autoscaler_settings.target_value is not None:
            grid.add_row("Target Value:", str(response.autoscaler_settings.target_value))

    console.print()
    console.print(grid)
    console.print("[bold]Tips[/bold]")
    console.print(f"  • View logs: [cyan]targon logs {response.uid}[/cyan]")
    console.print(f"  • View app:  [cyan]targon app get {response.app_id}[/cyan]")
    console.print()


def _display_app_details(response: Any):
    """Helper to display app details."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", justify="right")
    grid.add_column()

    grid.add_row("App ID:", f"[bright_cyan]{response.app_id}[/bright_cyan]")
    grid.add_row("Project ID:", str(response.project_id or "[dim]-[/dim]"))
    grid.add_row("Project Name:", str(response.project_name or "[dim]-[/dim]"))
    grid.add_row("Function Count:", str(response.function_count))
    grid.add_row("Created:", format_timestamp(response.created_at))
    grid.add_row("Updated:", format_timestamp(response.updated_at))

    console.print()
    console.print(grid)

    # Display functions if any exist
    if response.functions:
        table = Table(
            title=f"[bold bright_cyan]Functions[/bold bright_cyan] [dim]({response.function_count} total)[/dim]",
            border_style="dim bright_black",
            header_style="bold bright_cyan",
            show_lines=False,
            box=None,
            pad_edge=False,
            collapse_padding=True
        )
        table.add_column("UID", style="bright_cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Status", style="")
        table.add_column("URL", style="bright_blue underline")

        for func_id, func_data in response.functions.items():
            table.add_row(
                func_data.uid,
                func_data.name,
                get_status_display(func_data.status),
                func_data.url or "[dim]-[/dim]",
            )

        console.print()
        console.print(table)
        console.print()
    else:
        console.print("\n[dim italic]No functions deployed in this app.[/dim italic]")


@app.command("get")
@click.argument("identifier", required=True)
@click.pass_context
def app_get(ctx, identifier):
    """Get detailed information about an app or function.

    IDENTIFIER: App ID or Function UID (e.g., app-xxxxx or fnc-xxxxx)
    """
    client: Client = ctx.obj["client"]

    try:
        if identifier.startswith("fnc-"):
            # Handle function details
            with console.status("[bold cyan]Fetching function details...[/bold cyan]", spinner="dots"):
                response = client.run_async(
                    lambda: client.async_app.get_function(identifier)
                )
            
            _display_function_details(response)

        elif identifier.startswith("app-"):
            # Handle app status
            with console.status("[bold cyan]Fetching app details...[/bold cyan]", spinner="dots"):
                response = client.run_async(
                    lambda: client.async_app.get_app_status(identifier)
                )
            
            _display_app_details(response)

        else:
            console.print("[yellow]⚠[/yellow] Use [cyan]'targon app get <fnc-xxx> or <app-xxx>'[/cyan].")

    except APIError as e:
        if hasattr(e, 'is_not_found') and e.is_not_found:
             display_error(f"Resource '{identifier}' not found.", "Not Found")
        else:
            display_error(e, "API Error")
        raise SystemExit(1)
    except TargonError as e:
        display_error(e, "Targon Error")
        raise SystemExit(1)
    except Exception as e:
        display_error(e, "Unexpected Error")
        raise SystemExit(1)


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

    try:
        results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Deleting {len(app_ids)} apps...[/cyan]", total=len(app_ids))
            
            # Run parallel deletion using client.run_async
            async def _delete_apps_parallel():
                # Create tasks for all deletions
                tasks = [client.async_app.delete_app(app_id) for app_id in app_ids]
                
                # Execute in parallel, gathering results/exceptions
                # We map results back to app_ids by index since gather maintains order
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                return {app_id: responses[i] for i, app_id in enumerate(app_ids)}
            
            results = client.run_async(_delete_apps_parallel)
            progress.update(task, completed=len(app_ids))

        # Process and display results
        successful = []
        failed = []

        console.print()
        for app_id, result in results.items():
            if isinstance(result, Exception):
                failed.append((app_id, result))
                console.print(f"[bold red]✖[/bold red] Failed to delete [bright_cyan]{app_id}[/bright_cyan]: {str(result)}")
            else:
                successful.append(app_id)
                console.print(f"[bold green]✔[/bold green] Successfully deleted [bright_cyan]{app_id}[/bright_cyan]")
                # Display optional success details
                if isinstance(result, dict) and result.get("deleted_resources"):
                    console.print(f"    [dim]Deleted resources: {result['deleted_resources']}[/dim]")

        console.print()
        if failed:
            console.print(f"[green]Successful:[/green] {len(successful)}")
            console.print(f"[red]Failed:[/red] {len(failed)}")
            raise SystemExit(1)
        else:
             console.print(f"[bold green]All {len(successful)} apps deleted successfully.[/bold green]\n")

    except (TargonError, APIError) as e:
        display_error(e, "Deletion Failed")
        raise SystemExit(1)
    except Exception as e:
        display_error(e, "Unexpected Error")
        raise SystemExit(1)


@app.command("rm")
@click.argument("app_ids", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def remove_app(ctx, app_ids, yes):
    """Delete one or more Targon apps and all their deployments (alias for delete)."""
    ctx.invoke(delete_app, app_ids=app_ids, yes=yes)
