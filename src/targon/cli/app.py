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

@app.command("ls")
@click.pass_context
def list_apps(ctx):
    """List of deployed/running targon apps."""
    client: Client = ctx.obj["client"]

    try:
        with console.status("[bold cyan]Fetching apps.[/bold cyan]", spinner="dots"):
            response, detailed_apps = client.run_async(
                lambda: client.async_app.list_apps_with_details()
            )

        if not response.apps:
            console.print()
            console.print(
                "[dim]No applications found. Deploy one using [cyan]targon deploy[/cyan].[/dim]"
            )
            console.print()
            return

        # Create a tree for hierarchical view
        tree = Tree(
            f"[bold bright_cyan]Targon Apps[/bold bright_cyan] [dim]({response.total} total)[/dim]",
            guide_style="bright_black",
        )

        for i, app_item in enumerate(response.apps):
            detailed_app = detailed_apps[i]
            app_label = Text()
            app_label.append(app_item.name, style="bold bright_cyan")
            app_label.append(f" ({app_item.uid})", style="dim")

            app_node = tree.add(app_label)

            if detailed_app and detailed_app.functions:
                for fn in detailed_app.functions:
                    status_text = (
                        get_status_display(fn.state.status)
                        if fn.state
                        else "[dim]● unknown[/dim]"
                    )
                    func_label = Text.from_markup(
                        f"[bold]{fn.name}[/bold] {status_text} [dim]({fn.uid})[/dim]"
                    )
                    app_node.add(func_label)
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


def _display_function_details(response: Any):
    """Helper to display function details."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", justify="right")
    grid.add_column()

    # Basic Info
    grid.add_row("Workload UID:", f"[bright_cyan]{response.uid}[/bright_cyan]")
    grid.add_row("App ID:", f"[bright_blue]{response.app_id or '[dim]-[/dim]'}[/bright_blue]")
    grid.add_row("Name:", response.name)
    grid.add_row("Image:", response.image or "[dim]-[/dim]")
    grid.add_row("Resource:", response.resource_name or "[dim]-[/dim]")

    if response.cost_per_hour is not None:
        grid.add_row("Cost/hr:", f"${response.cost_per_hour:.4f}")

    if response.resource:
        grid.add_row("", "")
        grid.add_row("[bold bright_cyan]Resource[/bold bright_cyan]", "")
        grid.add_row("Display Name:", response.resource.display_name)
        grid.add_row("vCPU:", str(response.resource.vcpu))
        grid.add_row("Memory:", f"{response.resource.memory} MB")
        if response.resource.gpu_type:
            grid.add_row("GPU Type:", response.resource.gpu_type)
        if response.resource.gpu_count:
            grid.add_row("GPU Count:", str(response.resource.gpu_count))

    if response.state:
        grid.add_row("", "")
        grid.add_row("[bold bright_cyan]State[/bold bright_cyan]", "")
        grid.add_row("Status:", get_status_display(response.state.status))
        grid.add_row("Message:", response.state.message or "[dim]-[/dim]")
        grid.add_row("Ready Replicas:", str(response.state.ready_replicas))
        grid.add_row("Total Replicas:", str(response.state.total_replicas))

    if response.serverless_config:
        sc = response.serverless_config
        grid.add_row("", "")
        grid.add_row("[bold bright_cyan]Serverless Config[/bold bright_cyan]", "")
        if sc.module:
            grid.add_row("Module:", sc.module)
        if sc.qualname:
            grid.add_row("Qualname:", sc.qualname)
        if sc.definition_type:
            grid.add_row("Definition Type:", sc.definition_type)
        if sc.min_replicas is not None:
            grid.add_row("Min Replicas:", str(sc.min_replicas))
        if sc.max_replicas is not None:
            grid.add_row("Max Replicas:", str(sc.max_replicas))
        if sc.container_concurrency is not None:
            grid.add_row("Container Concurrency:", str(sc.container_concurrency))
        if sc.target_concurrency is not None:
            grid.add_row("Target Concurrency:", str(sc.target_concurrency))
        if sc.timeout_seconds is not None:
            grid.add_row("Timeout:", f"{sc.timeout_seconds}s")
        if sc.startup_timeout is not None:
            grid.add_row("Startup Timeout:", f"{sc.startup_timeout}s")

    grid.add_row("", "")
    grid.add_row("Created:", format_timestamp(response.created_at))
    grid.add_row("Updated:", format_timestamp(response.updated_at))

    console.print()
    console.print(grid)
    console.print("[bold]Tips[/bold]")
    console.print(f"  • View logs: [cyan]targon logs {response.uid}[/cyan]")
    if response.app_id:
        console.print(f"  • View app:  [cyan]targon app get {response.app_id}[/cyan]")
    console.print()


def _display_list_function_details(response: Any):
    """Helper to display list functions (workloads) for an app."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", justify="right")
    grid.add_column()

    grid.add_row("App ID:", f"[bright_cyan]{response.app_id}[/bright_cyan]")
    grid.add_row("App Name:", response.app_name or "[dim]-[/dim]")
    grid.add_row("Function Count:", str(response.total))

    console.print()
    console.print(grid)

    if not response.functions:
        console.print("\n[dim italic]No functions deployed in this app.[/dim italic]\n")
        return

    table = Table(
        title=f"[bold bright_cyan]Functions[/bold bright_cyan] [dim]({response.total} total)[/dim]",
        border_style="dim bright_black",
        header_style="bold bright_cyan",
        show_lines=False,
        box=None,
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("UID", style="bright_cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Image", style="dim", no_wrap=True)
    table.add_column("Status", style="")
    table.add_column("Message", style="dim")
    table.add_column("Ready", style="dim", justify="right")
    table.add_column("Created", style="dim")

    for fn in response.functions:
        status_text = (
            get_status_display(fn.state.status) if fn.state else "[dim]● unknown[/dim]"
        )
        message = fn.state.message if fn.state else "[dim]-[/dim]"
        replicas = (
            f"{fn.state.ready_replicas}/{fn.state.total_replicas}" if fn.state else "[dim]-[/dim]"
        )
        table.add_row(
            fn.uid,
            fn.name,
            fn.image or "[dim]-[/dim]",
            status_text,
            message,
            replicas,
            format_timestamp(fn.created_at),
        )

    console.print()
    console.print(table)
    console.print(
        f"\n  [dim]Run [cyan]targon app get <wrk-uid>[/cyan] for full workload details.[/dim]\n"
    )


@app.command("get")
@click.argument("identifier", required=True)
@click.pass_context
def app_get(ctx, identifier):
    """Get detailed info about an app or function."""
    client: Client = ctx.obj["client"]

    try:
        if identifier.startswith("wrk-"):
            # Handle function/workload details
            with console.status(
                "[bold cyan]Fetching workload details...[/bold cyan]", spinner="dots"
            ):
                response = client.run_async(
                    lambda: client.async_app.get_function(identifier)
                )

            _display_function_details(response)

        elif identifier.startswith("app-") or identifier.startswith("app_"):
            # Handle app details + function workloads
            with console.status(
                "[bold cyan]Fetching app + function workloads...[/bold cyan]", spinner="dots"
            ):
                response = client.run_async(
                    lambda: client.async_app.list_functions(identifier)
                )

            _display_list_function_details(response)

        else:
            console.print(
                "[yellow]⚠[/yellow] Use [cyan]'targon app get <wrk-xxx> or <app-xxx>'[/cyan]."
            )

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


@app.command("rm")
@click.argument("app_ids", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete_app(ctx, app_ids, yes):
    """Remove apps or individual workloads."""
    client: Client = ctx.obj["client"]

    if not app_ids:
        console.print("[red]✗[/red] At least one app or workload ID is required")
        raise SystemExit(1)

    invalid_ids = [
        resource_id
        for resource_id in app_ids
        if not (
            resource_id.startswith("app-")
            or resource_id.startswith("wrk-")
        )
    ]
    if invalid_ids:
        console.print(
            f"[red]✗[/red] Unsupported identifier(s): {', '.join(invalid_ids)}"
        )
        console.print(
            "[dim]Use [cyan]app-<uid>[/cyan] apps or [cyan]wrk-<uid>[/cyan] for workloads.[/dim]"
        )
        raise SystemExit(1)

    if not yes:
        if len(app_ids) == 1:
            click.confirm(
                f"Are you sure you want to delete '{app_ids[0]}'?",
                abort=True,
            )
        else:
            console.print(f"\n[bold]Resources to delete:[/bold]")
            for resource_id in app_ids:
                console.print(f"  • [bright_cyan]{resource_id}[/bright_cyan]")
            console.print()
            click.confirm(
                f"Are you sure you want to delete these {len(app_ids)} resources?",
                abort=True,
            )

    try:
        results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Deleting {len(app_ids)} resources...[/cyan]", total=len(app_ids)
            )

            async def _delete_resources_parallel():
                tasks = []
                for resource_id in app_ids:
                    if resource_id.startswith("wrk-") or resource_id.startswith("wrk_"):
                        tasks.append(client.async_functions.delete_function(resource_id))
                    else:
                        tasks.append(client.async_app.delete_app(resource_id))

                responses = await asyncio.gather(*tasks, return_exceptions=True)

                return {
                    resource_id: responses[i]
                    for i, resource_id in enumerate(app_ids)
                }

            results = client.run_async(_delete_resources_parallel)
            progress.update(task, completed=len(app_ids))

        # Process and display results
        successful = []
        failed = []

        console.print()
        for resource_id, result in results.items():
            if isinstance(result, Exception):
                failed.append((resource_id, result))
                console.print(
                    f"[red]✗[/red] Failed to delete [bright_cyan]{resource_id}[/bright_cyan]: {str(result)}"
                )
            else:
                successful.append(resource_id)
                console.print(
                    f"[green]✓[/green] Successfully deleted [bright_cyan]{resource_id}[/bright_cyan]"
                )
                # Display optional success details
                if isinstance(result, dict) and result.get("deleted_resources"):
                    console.print(
                        f"    [dim]Deleted resources: {result['deleted_resources']}[/dim]"
                    )

        console.print()
        if failed:
            console.print(f"[green]Successful:[/green] {len(successful)}")
            console.print(f"[red]Failed:[/red] {len(failed)}")
            raise SystemExit(1)
        else:
            console.print(
                f"[bold green]All {len(successful)} resources deleted successfully.[/bold green]\n"
            )

    except (TargonError, APIError) as e:
        display_error(e, "Deletion Failed")
        raise SystemExit(1)
    except Exception as e:
        display_error(e, "Unexpected Error")
        raise SystemExit(1)
