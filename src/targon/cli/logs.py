import click
import asyncio
from rich.console import Console
from targon.client.client import Client
from targon.core.exceptions import TargonError, APIError

console = Console(stderr=True)


@click.command()
@click.argument("uid", required=True)
@click.option(
    "--follow/--no-follow",
    default=True,
    help="Follow logs in real-time (default: True)",
)
@click.pass_context
def logs(ctx, uid, follow):
    """Stream logs from a serverless deployments."""
    client: Client = ctx.obj["client"]

    async def _stream_logs(c: Client):
        async with c:
            try:
                async for log_line in c.async_logs.stream_logs(uid, follow=follow):
                    # Print directly to stdout (not stderr like console)
                    print(log_line, flush=True)
            except KeyboardInterrupt:
                console.print("\n[dim]Log streaming stopped.[/dim]")
            except Exception as e:
                raise e

    try:
        if follow:
            console.print(
                f"[dim]Streaming logs for deployment [bright_cyan]{uid}[/bright_cyan]... (Ctrl+C to stop)[/dim]\n"
            )
        else:
            console.print(
                f"[dim]Fetching logs for deployment [bright_cyan]{uid}[/bright_cyan]...[/dim]\n"
            )

        asyncio.run(_stream_logs(client))

    except KeyboardInterrupt:
        console.print("\n[dim]Log streaming stopped.[/dim]")
    except APIError as e:
        console.print(f"\n[red]✗[/red] [bold]API Error:[/bold] {e.message}\n")
        if e.is_not_found:
            console.print(
                f"[dim]Hint: deployment '{uid}' may not exist or may not be deployed yet.[/dim]"
            )
        else:
            console.print(
                f"[dim]Hint: Try checking the deployment state with [cyan]targon app get {uid}[/cyan][/dim]"
            )
        raise SystemExit(1)
    except TargonError as e:
        console.print(f"\n[red]✗[/red] [bold]Error:[/bold] {e.message}\n")
        console.print(
            f"[dim]Hint: Try checking the deployment state with [cyan]targon app get {uid}[/cyan][/dim]"
        )
        raise SystemExit(1)
    except Exception as e:
        console.print(f"\n[red]✗[/red] [bold]Unexpected error:[/bold] {e}\n")
        console.print(
            f"[dim]Hint: Try checking the deployment state with [cyan]targon app get {uid}[/cyan][/dim]"
        )
        raise SystemExit(1)
