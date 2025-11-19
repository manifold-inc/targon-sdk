from datetime import datetime
from pydoc import cli
import re
from typing import Any, Callable, Tuple, TypedDict
import typing
from attr import dataclass
import click
import json
import asyncio
import inspect
from pathlib import Path
from rich.panel import Panel
from rich.syntax import Syntax

from ..core.executor import run_app
from .imports import parse_import_ref, import_app_from_ref
from ..core.console import console
from ..core.console import _rich_console
from ..client.client import Client


@dataclass
class ParameterMetadata:
    name: str
    default: Any
    annotation: Any
    type_hint: Any  # same as annotation but evaluated by typing.get_type_hints
    kind: Any


@dataclass
class CliRunnableSignature:
    parameters: dict[str, ParameterMetadata]
    has_variadic_args: bool


def _get_cli_runnable_signature(
    fn_signature: inspect.Signature, type_map: dict[str, type]
) -> Tuple[dict[str, ParameterMetadata], bool]:
    """Extract CLI-compatible signature from function signature."""
    variadic = False
    params_meta: dict[str, ParameterMetadata] = {}

    for arg in fn_signature.parameters.values():
        if arg.kind == inspect.Parameter.VAR_POSITIONAL:
            variadic = True
        else:
            params_meta[arg.name] = ParameterMetadata(
                name=arg.name,
                default=arg.default,
                annotation=arg.annotation,
                type_hint=type_map.get(arg.name, Any),
                kind=arg.kind,
            )

    if variadic and params_meta:
        raise click.ClickException(
            "A function using *args cannot define any additional parameters."
        )

    return params_meta, variadic


def _get_param_type_as_str(annot: Any) -> str:
    """Return annotation as a string, handling various optional type spellings."""

    # Handle missing annotation
    if annot is inspect.Signature.empty:
        return "Any"

    annot_str = str(annot)
    annot_patterns = [
        r"typing\.Optional\[([\w.]+)\]",
        r"typing\.Union\[([\w.]+), NoneType\]",
        r"([\w.]+) \| None",
        r"<class '([\w\.]+)'>",
    ]
    for pat in annot_patterns:
        m = re.match(pat, annot_str)
        if m is not None:
            return m.group(1)
    return annot_str


class AnyParamType(click.ParamType):
    name = "any"

    def convert(self, value, param, ctx):
        return value


option_parsers = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "datetime.datetime": click.DateTime(),
    "Any": AnyParamType(),
}


def _add_click_options(func, parameters: dict[str, ParameterMetadata]):
    """Add @click.option decorators based on function signature."""
    # reversed so that help will print in order
    for param in reversed(parameters.values()):
        # parse the annotation if empty use Any
        param_type_str = (
            "Any"
            if param.annotation is inspect.Signature.empty
            else _get_param_type_as_str(param.type_hint)
        )
        param_name = param.name.replace("_", "-")
        cli_name = "--" + param_name

        if param_type_str == "bool":
            cli_name += "/--no-" + param_name

        parser = option_parsers.get(param_type_str)
        if parser is None:
            msg = (
                f"Parameter `{param_name}` has unparseable annotation: "
                f"{param.annotation!r}"
            )
            raise click.ClickException(msg)

        kwargs: Any = {"type": parser}
        if param.default is not inspect.Signature.empty:
            kwargs["default"] = param.default
        else:
            kwargs["required"] = True
        click.option(cli_name, **kwargs)(func)
    return func

    # XXX:
    # This can be used to define the args
    # ex no need for --message "hello targon" simply pass argument as "hello targon"
    # for param in reversed(positional_parameters.values()params):
    #     param_type_str = "Any" if param.annotation is inspect.Signature.empty else _get_param_type_as_str(param.type_hint)
    #     param_name = param.name.replace("_", "-")

    #     parser = option_parsers.get(param_type_str)
    #     if parser is None:
    #         msg = (
    #             f"Parameter `{param_name}` has unparseable annotation: "
    #             f"{param.annotation!r}"
    #         )
    #         raise click.ClickException(msg)

    #     click.argument(param_name, type=parser)(func)
    # return func


def safe_get_type_hints(f):
    try:
        return typing.get_type_hints(f)
    except Exception as exc:
        raise click.ClickException(f"Unable to parse type hints:\n{exc}") from exc


def _get_click_command_for_local_entrypoint(app, entrypoint):
    fnc = entrypoint.raw_f
    is_async = asyncio.iscoroutinefunction(fnc)
    parameters, has_variadic_args = _get_cli_runnable_signature(
        inspect.signature(fnc),
        safe_get_type_hints(fnc),
    )

    @click.pass_context
    def f(ctx, *args, **kwargs):
        if has_variadic_args:
            assert len(args) == 0 and len(kwargs) == 0
            args = ctx.args

        # Use asyncio.run for the entire execution context
        async def _run_with_app():
            # @TODO:get the app name from the object and replace with display name
            with console(app.name) as c:
                async with run_app(
                    app=app,
                    client=ctx.obj["client"],
                    console_instance=c,
                    app_file_path=ctx.obj["app_file_path"],
                ):
                    if is_async:
                        return await fnc(*args, **kwargs)
                    else:
                        result = fnc(*args, **kwargs)
                        # If sync function returned a coroutine, await it
                        if asyncio.iscoroutine(result):
                            return await result
                        return result

        res = asyncio.run(_run_with_app())

        if res is not None:
            _rich_console.print()

            # Format result based on type
            if isinstance(res, (dict, list)):
                result_json = json.dumps(res, indent=2)
                syntax = Syntax(
                    result_json,
                    "json",
                    theme="ansi_dark",
                    line_numbers=False,
                    padding=(1, 2),
                )
                panel = Panel(
                    syntax,
                    title="[bold bright_cyan]Result[/bold bright_cyan]",
                    border_style="dim bright_black",
                    padding=(0, 0),
                )
                _rich_console.print(panel)
            else:
                result_str = str(res)
                panel = Panel(
                    f"[bright_cyan]{result_str}[/bright_cyan]",
                    title="[dim bright_black]┤[/dim bright_black][bold bright_cyan] Result[/bold bright_cyan][dim bright_black]├[/dim bright_black]",
                    border_style="dim bright_black",
                    padding=(1, 2),
                )
                _rich_console.print(panel)

    click_options = _add_click_options(f, parameters)
    if has_variadic_args:
        return click.command(
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
        )(click_options)
    else:
        return click.command(click_options)


class RunGroup(click.Group):
    def get_command(self, ctx: click.Context, func_ref: str) -> click.Command | None:
        import_ref = parse_import_ref(func_ref)
        app_obj = import_app_from_ref(import_ref)

        ctx.obj["app_file_path"] = import_ref.file_path

        # Determining the what entrypoint to use
        # @TODO: need to add the functionality to call a function or class directly
        # even if the localentrypoint is not defined
        if not app_obj._local_entrypoints:
            _rich_console.print(
                "[bold red]Error:[/bold red] No local entrypoint found. Please define an entrypoint using [bold]@app.local_entrypoint[/bold]."
            )
            ctx.exit(1)

        entrypoint_name = list(app_obj._local_entrypoints.keys())[0]
        entrypoint = app_obj._local_entrypoints[entrypoint_name]
        if len(app_obj._local_entrypoints) > 1:
            _rich_console.print(
                f"[bright_blue]ℹ[/bright_blue] Running first entrypoint: [bold]{entrypoint_name}[/bold]"
            )

        return _get_click_command_for_local_entrypoint(app_obj, entrypoint)


@click.group(
    cls=RunGroup,
    subcommand_metavar="FUNC_REF",
)
@click.pass_context
def run(ctx):
    """Deplpoy and execute local entrypoints defined in Targon app."""
    pass
