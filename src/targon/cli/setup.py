import click
from targon.cli.auth import get_stored_key, save_api_key
from ..core.console import _rich_console


@click.command("setup", help="Configure Targon CLI with your API key")
def setup():
    _rich_console.print("\n[bold bright_cyan]Targon CLI Setup[/bold bright_cyan]\n")

    # Check if already configured
    existing_key = get_stored_key()
    if existing_key:
        _rich_console.print("[bright_blue]ℹ[/bright_blue] API key already configured.")
        if not click.confirm("Do you want to update it?"):
            _rich_console.print("[dim]Setup cancelled.[/dim]\n")
            return

    # Get API key
    api_key = click.prompt(
        "Enter your Targon API key (starts with 'sn4_')", hide_input=True
    )

    # Validate format
    if not api_key.startswith("sn4_") or len(api_key) != 32:
        _rich_console.print(
            "\n[bold red]✖[/bold red] [bold]Invalid API key format.[/bold] Key should start with 'sn4_' and be 32 characters long.\n"
        )
        return

    if save_api_key(api_key):
        _rich_console.print(
            "\n[bold green]✔[/bold green] [bold]API key saved successfully![/bold]"
        )
        _rich_console.print(
            "[dim italic]  You're ready to use Targon. Try 'targon app list' to get started.[/dim italic]\n"
        )
    else:
        _rich_console.print(
            "\n[bold red]✖[/bold red] [bold]Failed to save API key.[/bold] You can set TARGON_API_KEY environment variable instead.\n"
        )
