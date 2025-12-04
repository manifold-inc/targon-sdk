import asyncio
import click
from typing import List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from targon.client.client import Client
from targon.core.exceptions import TargonError, APIError

console = Console(stderr=True)


def display_error(err: BaseException | str, title: str = "Error") -> None:
    console.print()
    console.print(f"[red]{str(err)}[/red]")
    console.print()


@click.group()
def container():
    """Manage serverless containers."""
    pass


@container.command("ls")
@click.argument("uid", required=False)
@click.option(
    "--sort",
    type=click.Choice(["created", "name", "cost"], case_sensitive=False),
    default="created",
    help="Sort containers by field (default: created)",
)
@click.option(
    "--reverse",
    is_flag=True,
    help="Reverse sort order (descending)",
)
@click.pass_context
def list_containers(ctx, uid=None, sort: str = "created", reverse: bool = False) -> None:
    """List all running containers."""
    client: Client = ctx.obj["client"]

    try:
        with console.status("[bold cyan]Fetching containers...[/bold cyan]", spinner="dots"):
            resources = client.run_async(
                lambda: client.async_serverless.list_container()
            )
        
        # Display results in a table
        if not resources:
            console.print("\n[dim]No containers found. Deploy one using [cyan]targon deploy[/cyan].[/dim]\n")
            return

        # Sort resources based on the selected field
        def sort_key(resource):
            if sort == "name":
                return (resource.name or "").lower()
            elif sort == "cost":
                return resource.cost if resource.cost is not None else 0
            else:  # created
                return resource.created_at or ""
        
        resources = sorted(resources, key=sort_key, reverse=reverse)
        
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="bright_black",
            title="[bold]Serverless Containers[/bold]",
            title_style="bold magenta",
            padding=(0, 1),
        )
        
        table.add_column("NAME", style="bright_cyan", no_wrap=True)
        table.add_column("UID", style="dim white")
        table.add_column("URL", style="blue")
        table.add_column("COST", justify="right", style="yellow")
        table.add_column("CREATED", style="green")
        
        for resource in resources:
            table.add_row(
                resource.name or "—",
                resource.uid,
                resource.url if resource.url else "[dim]—[/dim]",
                f"${resource.cost:,.1f}" if resource.cost else "[dim]—[/dim]",
                resource.created_at if resource.created_at else "[dim]—[/dim]",
            )
        
        console.print()
        console.print(table)
        console.print()

    except (TargonError, APIError) as e:
        display_error(e, "Failed to list containers")
        raise SystemExit(1)
    except Exception as e:
        display_error(e, "Unexpected error")
        raise SystemExit(1)


@container.command("rm")
@click.argument("container_ids", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def remove_containers(ctx, container_ids: List[str], yes: bool) -> None:
    """Delete one or more containers by UID."""
    client: Client = ctx.obj["client"]

    if not yes:
        if len(container_ids) == 1:
            click.confirm(
                f"Delete container '{container_ids[0]}'?", abort=True
            )
        else:
            console.print("\n[bold]Containers to delete:[/bold]")
            for cid in container_ids:
                console.print(f"  • [bright_cyan]{cid}[/bright_cyan]")
            console.print()
            click.confirm(
                f"Delete all {len(container_ids)} containers?", abort=True
            )

    try:
        async def _delete_all():
            tasks = [client.async_serverless.delete_container(cid) for cid in container_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {container_ids[i]: results[i] for i in range(len(container_ids))}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                f"[cyan]Deleting {len(container_ids)} container(s).[/cyan]",
                total=len(container_ids),
            )
            results = client.run_async(_delete_all)
            progress.update(task_id, completed=len(container_ids))

        console.print()
        failures = []
        for cid, result in results.items():
            if isinstance(result, Exception):
                failures.append(cid)
                console.print(f"[bold red]✖[/bold red] {cid}: {result}")
            else:
                console.print(f"[bold green]✔[/bold green] Deleted [bright_cyan]{cid}[/bright_cyan]")

        console.print()
        if failures:
            console.print(f"[red]Failed:[/red] {len(failures)} / {len(container_ids)}")
            raise SystemExit(1)
        console.print(f"[bold green]All {len(container_ids)} containers deleted.[/bold green]\n")

    except (TargonError, APIError) as e:
        display_error(e, "Deletion failed")
        raise SystemExit(1)
    except Exception as e:
        display_error(e, "Unexpected error")
        raise SystemExit(1)