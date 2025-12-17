from pathlib import Path
import click
import asyncio
from targon.utils.config_parser import load_config
from targon.client.client import Client
from targon.core.exceptions import TargonError, APIError, ValidationError
from targon.core.executor import deploy_app, deploy_config
from targon.cli.imports import parse_import_ref, import_app_from_ref
from targon.core.console import console
from targon.core.console import _rich_console


def is_config_file(path: str) -> bool:
    return Path(path).suffix.lower() in ['.yaml', '.yml', '.json']


def find_default_config() -> Path | None:
    """
    Find default configuration file in current directory.
    Looks for (in order)
    """
    candidates = [
        "targon.yaml",
        "targon.yml",
        "deploy.yaml",
        "deploy.yml",
        "targon.json",
        "deploy.json",
    ]

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    return None


@click.command()
@click.argument("target", required=True)
@click.option("--name", help="Override Name of the deployment.")
@click.option("--config", help="Path to configuration file (YAML/JSON)")
@click.pass_context
def deploy(ctx, target, name, config):
    """Deploy an application to Targon."""
    client: Client = ctx.obj["client"]

    config_path = None

    if config:
        config_path = Path(config)
        if not config_path.exists():
            raise ValidationError(
                f"Config file not found: {config}",
                field="config",
                value=config,
            )
    elif target:
        if is_config_file(target):
            config_path = Path(target)
    else:
        default_config = find_default_config()
        if default_config:
            config_path = default_config
            _rich_console.print(f"[dim]Using config: {config_path}[/dim]\n")
        else:
            raise ValidationError(
                "No target specified. Provide a Python file or config file.\n"
                "Examples:\n"
                "  targon deploy app.py\n"
                "  targon deploy deploy.yaml\n"
                "  targon deploy --config deploy.yaml",
                field="target",
                value=None,
            )

    try:
        if config_path:
            cfg = load_config(config_path)

            display_name = name or cfg.app_name

            with console(display_name) as c:
                asyncio.run(
                    deploy_config(
                        config=cfg,
                        client=client,
                        console_instance=c,
                    )
                )
        else:
            import_ref = parse_import_ref(target)
            app_obj = import_app_from_ref(import_ref)

            # Use app name for console display
            display_name = name or app_obj.name or "app"

            with console(display_name) as c:
                # Run async deployment with console
                asyncio.run(
                    deploy_app(
                        app=app_obj,
                        name=name,
                        client=client,
                        console_instance=c,
                        app_file_path=import_ref.file_path,
                    )
                )

    except (TargonError, APIError) as e:
        _rich_console.print(f"\n[red]✗[/red] [bold]Deployment failed:[/bold] {e}\n")
        raise SystemExit(1)
