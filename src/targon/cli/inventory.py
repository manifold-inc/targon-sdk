import click
import asyncio
from targon.core.exceptions import TargonError
from targon.core.console import _rich_console
from rich.table import Table


@click.command("capacity", help="Get compute capacity information")
@click.option("--json", is_flag=True, help="Output in JSON format")
@click.pass_context
def capacity(ctx, json):
    """Get compute capacity information (async wrapper)."""
    client = ctx.obj["client"]

    async def _async_capacity():
        capacities = await client.async_inventory.capacity()
        if not capacities:
            _rich_console.print(
                "\n[bright_blue]ℹ[/bright_blue] No capacity data available.\n"
            )
            return

        if json:
            import json as js

            click.echo(js.dumps([c.__dict__ for c in capacities], indent=2))
            return

        table = Table(
            title="[bold bright_cyan]Compute Capacity[/bold bright_cyan]",
            border_style="dim bright_black",
            header_style="bold bright_cyan",
            show_lines=False,
        )
        table.add_column("Name", style="bold", no_wrap=True)
        table.add_column("Count", justify="right", style="bright_green")

        for cap in capacities:
            table.add_row(cap.name, str(cap.count))

        _rich_console.print()
        _rich_console.print(table)
        _rich_console.print()

    asyncio.run(_async_capacity())


@click.command("inventory", help="List available inventory capacities")
@click.pass_context
def inventory(ctx):
    """List available inventory capacities (async wrapper)."""
    client = ctx.obj["client"]

    async def _async_inventory():
        capacities = await client.async_inventory.capacity()
        if not capacities:
            _rich_console.print(
                "\n[bright_blue]ℹ[/bright_blue] No inventory data available.\n"
            )
            return

        table = Table(
            title="[bold bright_cyan]Inventory[/bold bright_cyan]",
            border_style="dim bright_black",
            header_style="bold bright_cyan",
            show_lines=False,
        )
        table.add_column("Name", style="bold", no_wrap=True)
        table.add_column("Available", justify="right", style="bright_green")

        for cap in capacities:
            table.add_row(cap.name, str(cap.count))

        _rich_console.print()
        _rich_console.print(table)
        _rich_console.print()

    asyncio.run(_async_inventory())
