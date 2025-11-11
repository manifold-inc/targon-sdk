import click
import asyncio
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from targon.client.client import Client
from targon.core.exceptions import TargonError, APIError
from datetime import datetime

console = Console(stderr=True)


@click.group()
def app():
    """Manage Targon applications."""
    pass


@app.command("list")
@click.pass_context
def list_apps(ctx):
    """List targon apps that are currently deployed/running or recently stopped."""
    client: Client = ctx.obj["client"]

    async def _list_apps(c: Client):
        async with c:
            return await c.async_app.list_apps()

    try:
        # Run the async function
        response = asyncio.run(_list_apps(client))

        if not response.apps:
            console.print("\n[bright_blue]ℹ[/bright_blue] No apps found.")
            return

        table = Table(
            title=f"[bold bright_cyan]Targon Apps[/bold bright_cyan] [dim bright_black]({response.total} total)[/dim bright_black]",
            border_style="dim bright_black",
            header_style="bold bright_cyan",
            show_lines=False,
        )
        table.add_column("App ID", style="bright_cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Project ID", style="bright_blue")
        table.add_column("Created", style="dim")
        table.add_column("Updated", style="dim")

        for app_item in response.apps:
            # Format timestamps for better readability
            created_at = format_timestamp(app_item.created_at)
            updated_at = format_timestamp(app_item.updated_at)

            table.add_row(
                app_item.app_id,
                app_item.name,
                app_item.project_id or "[dim]-[/dim]",
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


@app.command("delete")
@click.argument("app_id", required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete_app(ctx, app_id, yes):
    """Delete the app and all the corresponding deployments."""
    client: Client = ctx.obj["client"]

    # Confirm deletion unless --yes flag is provided
    if not yes:
        click.confirm(
            f"Are you sure you want to delete app '{app_id}' and all its deployments?",
            abort=True,
        )

    async def _delete_app(c: Client):
        async with c:
            return await c.async_app.delete_app(app_id)

    try:
        with console.status(
            f"[bold cyan]Deleting app [bright_magenta]{app_id}[/bright_magenta]...[/bold cyan]",
            spinner="dots",
        ):
            result = asyncio.run(_delete_app(client))

        # Display success message
        console.print(
            f"\n[bold green]✔[/bold green] [bold]Successfully deleted app:[/bold] [bright_cyan]{app_id}[/bright_cyan]"
        )

        # Display additional info if available
        if isinstance(result, dict):
            if result.get("message"):
                console.print(f"[dim italic]  {result['message']}[/dim italic]")
            if result.get("deleted_resources"):
                console.print(
                    f"[dim italic]  Deleted resources: {result['deleted_resources']}[/dim italic]"
                )

        console.print()

    except (TargonError, APIError) as e:
        console.print(f"\n[bold red]✖[/bold red] [bold]Error:[/bold] {e}\n")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[bold red]✖[/bold red] [bold]Unexpected error:[/bold] {e}\n")
        raise SystemExit(1)


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
