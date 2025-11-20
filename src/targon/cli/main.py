import click
import sys
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
from targon.version import __version__

console = Console(stderr=True)


def display_error(err: Union[Exception, str], title: str = "Error"):
    """Display a formatted error message."""
    console.print()
    console.print(f"[red]{str(err)}[/red]")
    console.print()


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


@click.group(cls=SafeGroup)
@click.version_option(__version__, prog_name="Targon CLI")
@click.pass_context
def cli(ctx):
    """Targon SDK CLI - Interact with Targon for secure compute."""
    ctx.ensure_object(dict)
    try:
        resolved_api_key = get_api_key()
        ctx.obj["client"] = Client(api_key=resolved_api_key)
    except Exception as e:
        # Catch auth errors specifically if they happen during init
        if isinstance(e, (TargonError, APIError)):
            raise
        # For other errors during setup, we might want to let them bubble 
        # to the SafeGroup handler, but providing context is good.
        raise TargonError(f"Failed to initialize client: {e}") from e


# Register commands
cli.add_command(setup)
cli.add_command(capacity)
cli.add_command(deploy, name="deploy")
cli.add_command(run, name="run")
cli.add_command(app, name="app")
cli.add_command(logs, name="logs")

if __name__ == '__main__':
    cli()
