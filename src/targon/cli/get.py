import click
from rich.console import Console
from rich.table import Table

from targon.client.client import Client
from targon.core.exceptions import APIError, TargonError

console = Console(stderr=True)


@click.group()
def get():
    """Get workload state and events."""
    pass


@get.command("state")
@click.argument("uid", required=True)
@click.pass_context
def state(ctx, uid):
    """Get workload state by workload UID."""
    client: Client = ctx.obj["client"]

    try:
        with console.status(
            f"[bold cyan]Fetching state for [bright_cyan]{uid}[/bright_cyan]...[/bold cyan]",
            spinner="dots",
        ):
            response = client.run_async(lambda: client.async_serverless.get_state(uid))

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold dim", justify="right")
        grid.add_column()

        grid.add_row("UID:", f"[bright_cyan]{response.uid}[/bright_cyan]")
        grid.add_row("Type:", response.workload_type or "[dim]-[/dim]")
        grid.add_row("Status:", response.status or "[dim]-[/dim]")
        grid.add_row("Message:", response.message or "[dim]-[/dim]")
        grid.add_row(
            "Replicas:", f"{response.ready_replicas}/{response.total_replicas}"
        )
        grid.add_row("Updated:", response.updated_at or "[dim]-[/dim]")

        console.print()
        console.print(grid)

        if response.urls:
            table = Table(
                title="[bold bright_cyan]URLs[/bold bright_cyan]",
                border_style="dim bright_black",
                header_style="bold bright_cyan",
                show_lines=False,
                box=None,
                pad_edge=False,
                collapse_padding=True,
            )
            table.add_column("Port", style="bright_cyan", no_wrap=True)
            table.add_column("URL", style="bright_blue")

            for item in response.urls:
                table.add_row(str(item.port), item.url or "[dim]-[/dim]")

            console.print()
            console.print(table)

        console.print()

    except APIError as e:
        console.print(f"\n[red]✗[/red] [bold]API Error:[/bold] {e.message}\n")
        raise SystemExit(1)
    except TargonError as e:
        console.print(f"\n[red]✗[/red] [bold]Error:[/bold] {e.message}\n")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[red]✗[/red] [bold]Unexpected error:[/bold] {e}\n")
        raise SystemExit(1)


@get.command("events")
@click.argument("uid", required=True)
@click.option("--limit", type=int, default=None, help="Limit number of events.")
@click.option("--cursor", type=str, default=None, help="Pagination cursor.")
@click.pass_context
def events(ctx, uid, limit, cursor):
    """Get workload events by workload UID."""
    client: Client = ctx.obj["client"]

    try:
        with console.status(
            f"[bold cyan]Fetching events for [bright_cyan]{uid}[/bright_cyan]...[/bold cyan]",
            spinner="dots",
        ):
            response = client.run_async(
                lambda: client.async_serverless.get_events(
                    uid, limit=limit, cursor=cursor
                )
            )

        if not response.items:
            console.print("\n[dim]No events found.[/dim]\n")
            return

        table = Table(
            title="[bold bright_cyan]Workload Events[/bold bright_cyan]",
            border_style="dim bright_black",
            header_style="bold bright_cyan",
            show_lines=False,
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("Created", style="dim")
        table.add_column("Event", style="bright_cyan", no_wrap=True)
        table.add_column("Status", style="yellow")
        table.add_column("Message", style="")
        table.add_column("Pod", style="dim")

        for item in response.items:
            message = (
                item.display_message
                or item.message
                or item.reason
                or "[dim]-[/dim]"
            )
            table.add_row(
                item.created_at or "[dim]-[/dim]",
                item.event_type or "[dim]-[/dim]",
                item.new_status or "[dim]-[/dim]",
                message,
                item.pod_name or "[dim]-[/dim]",
            )

        console.print()
        console.print(table)
        if response.next_cursor:
            console.print(
                f"\n[dim]Next cursor: [bright_cyan]{response.next_cursor}[/bright_cyan][/dim]"
            )
        console.print()

    except APIError as e:
        console.print(f"\n[red]✗[/red] [bold]API Error:[/bold] {e.message}\n")
        raise SystemExit(1)
    except TargonError as e:
        console.print(f"\n[red]✗[/red] [bold]Error:[/bold] {e.message}\n")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[red]✗[/red] [bold]Unexpected error:[/bold] {e}\n")
        raise SystemExit(1)
