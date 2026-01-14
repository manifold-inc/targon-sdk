import click
import sys
import asyncio
from typing import Union
from rich.console import Console
from targon import Client
from targon.cli.auth import get_api_key
from targon.core.exceptions import TargonError, APIError
from targon.cli.inventory import capacity
from targon.cli.setup import setup
from targon.cli.deploy import deploy
from targon.cli.run import run
from targon.cli.app import app
from targon.cli.logs import logs
from targon.cli.container import container
from targon.version import __version__

console = Console(stderr=True)


def display_error(err: Union[Exception, str], title: str = "Error"):
    """Display a formatted error message."""
    console.print()
    console.print(f"[red]{str(err)}[/red]")
    console.print()


def _cleanup_client(client: Client):
    """Cleanup function to close client sessions on exit."""
    def cleanup():
        client.close()
        if client._async_session is not None and not client._async_session.closed:
            try:
                asyncio.run(client._async_session.close())
            except Exception:
                pass  
    return cleanup


class SafeGroup(click.Group):
    """A Click Group that catches Targon errors and displays them nicely."""

    def __call__(self, *args, **kwargs):
        try:
            return super().__call__(*args, **kwargs)
        except (TargonError, APIError) as e:
            display_error(e, "Error")
            sys.exit(1)
        except Exception as e:
            if isinstance(e, SystemExit):
                raise
            # Let Click handle Abort (Ctrl+C)
            if isinstance(e, click.Abort):
                raise

            display_error(e, "Unexpected Error")
            sys.exit(1)


@click.group(cls=SafeGroup, invoke_without_command=True)
@click.version_option(__version__, prog_name="Targon CLI")
@click.pass_context
def cli(ctx):
    """Targon SDK CLI - Interact with Targon for secure compute."""
    ctx.ensure_object(dict)
    
    # If no subcommand is provided, print help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return
    
    # Skip client initialization for commands that don't need auth
    if ctx.invoked_subcommand in ("setup", "completion"):
        return
    
    try:
        resolved_api_key = get_api_key()
        client = Client(api_key=resolved_api_key)
        ctx.obj["client"] = client
        ctx.call_on_close(_cleanup_client(client))
    except Exception as e:
        # Catch auth errors specifically if they happen during init
        if isinstance(e, (TargonError, APIError)):
            raise
        # For other errors during setup, we might want to let them bubble
        # to the SafeGroup handler, but providing context is good.
        raise TargonError(f"Failed to initialize client: {e}") from e


@cli.command()
@click.option(
    "-s", "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell type (auto-detected if not provided).",
)
def completion(shell: str):
    """Install shell completion to virtualenv."""
    import os
    from pathlib import Path
    
    # Auto-detect shell if not provided
    if not shell:
        shell = Path(os.environ.get("SHELL", "bash")).name
        if shell not in ("bash", "zsh", "fish"):
            shell = "bash"
    
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        raise click.ClickException("Not in a virtualenv. Activate one first.")
    
    activate = Path(venv) / "bin" / "activate"
    if not activate.exists():
        raise click.ClickException(f"Activate script not found: {activate}")
    
    content = activate.read_text()
    
    if "_TARGON_COMPLETE" in content:
        click.echo("Completions already installed.")
        return
    
    line = f'\neval "$(_TARGON_COMPLETE={shell}_source targon)"\n'
    activate.write_text(content + line)
    click.echo(f"Installed completions to {activate}. Run: source {activate}")


# Register commands
cli.add_command(setup)
cli.add_command(capacity)
cli.add_command(deploy, name="deploy")
cli.add_command(run, name="run")
cli.add_command(app, name="app")
cli.add_command(logs, name="logs")
cli.add_command(container, name="container")

if __name__ == '__main__':
    cli()
