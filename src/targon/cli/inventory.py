import math
import click
import asyncio
from dataclasses import asdict
from targon.core.exceptions import TargonError
from targon.core.console import _rich_console
from rich.table import Table


@click.command("capacity")
@click.option("--gpu", "gpu", flag_value=True, default=None, help="Show GPU inventory only")
@click.option("--no-gpu", "gpu", flag_value=False, help="Show non-GPU inventory only")
@click.option("--json", is_flag=True, help="Output in JSON format")
@click.pass_context
def capacity(ctx, gpu, json):
    """Get inventory information."""
    client = ctx.obj["client"]

    async def _async_capacity():
        inventory = await client.async_inventory.capacity(
            gpu=gpu,
        )
        if not inventory:
            _rich_console.print(
                "\n[bright_blue]ℹ[/bright_blue] No inventory data available.\n"
            )
            return

        if json:
            import json as js

            click.echo(js.dumps([asdict(item) for item in inventory], indent=2))
            return

        _rich_console.print(
            "[dim]Note: `targon capacity` will move to `targon inventory`. "
            "Both commands are supported for now.[/dim]"
        )

        table = Table(
            title="[bold bright_cyan]Inventory[/bold bright_cyan]",
            border_style="dim bright_black",
            header_style="bold bright_cyan",
            show_lines=False,
        )
        table.add_column("Name", style="bold", no_wrap=True)
        # table.add_column("Type", style="bright_blue")
        table.add_column("CPU", style="bright_magenta")
        table.add_column("MEM", style="bright_magenta")
        table.add_column("GPU", justify="center")
        table.add_column("Cost/Hr", justify="right", style="bright_yellow")
        table.add_column("Available", justify="right", style="bright_green")
        # table.add_column("Description", style="dim")

        for item in inventory:
            table.add_row(
                item.name,
                # item.type or "-",
                str(math.ceil(item.spec.vcpu/1024)),
                str(math.ceil(item.spec.memory/1024)),
                str(item.spec.gpu_count if item.spec.gpu_count else 0),
                f"${item.cost_per_hour:.2f}",
                str(item.available),
            )

        _rich_console.print()
        _rich_console.print(table)
        _rich_console.print()

    asyncio.run(_async_capacity())
