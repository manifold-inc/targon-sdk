import click
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


@click.group()
@click.version_option(__version__, prog_name="Targon CLI")
@click.pass_context
def cli(ctx):
    """Targon SDK CLI - Interact with Targon for secure compute."""
    ctx.ensure_object(dict)
    resolved_api_key = get_api_key()
    ctx.obj["client"] = Client(api_key=resolved_api_key)


# Register commands
cli.add_command(setup)
cli.add_command(capacity)
cli.add_command(deploy, name="deploy")
cli.add_command(run, name="run")
cli.add_command(app, name="app")
cli.add_command(logs, name="logs")

if __name__ == '__main__':
    try:
        cli()
    except (TargonError, APIError) as e:
        click.secho(f"Error: {e}", fg='red', err=True)
        raise SystemExit(1)
