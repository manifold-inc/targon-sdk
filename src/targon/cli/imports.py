from ..core.objects import BaseApp
from ..core.app import App
from ..core.exceptions import TargonError
from ..core.console import console

import importlib.util
import sys
import dataclasses
import inspect
from typing import Optional
from pathlib import Path
import click
from rich.markdown import Markdown


@dataclasses.dataclass
class ImportRef:
    file_path: str
    object_path: Optional[str] = None


# This is the object that will
# be searched if nothing is mentioned
DEFAULT_APP_NAME = "app"


def parse_import_ref(file_ref: str) -> ImportRef:
    """
    Parse a file reference like 'examples/getting_started.py' or 'examples/getting_started.py::main'
    """
    if '::' in file_ref:
        file_path, object_path = file_ref.split('::', 1)
    else:
        file_path, object_path = file_ref, ""
    return ImportRef(file_path=file_path, object_path=object_path)


# Hey Python, I have got this random file somewhere on my computer.
# Please load it as a module so I can use its functions/classes.
def import_file_or_module(import_ref: ImportRef, base_cmd: str = ""):
    if "" not in sys.path:
        sys.path.insert(0, "")

    full_path = Path(import_ref.file_path).resolve()
    if "." in full_path.name.removesuffix(".py"):
        raise ValueError(
            f"Invalid source filename: {full_path.name!r}."
            "\n\nSource filename cannot contain additional period characters."
        )
    sys.path.insert(0, str(full_path.parent))

    module_name = inspect.getmodulename(import_ref.file_path)
    assert module_name is not None
    spec = importlib.util.spec_from_file_location(module_name, import_ref.file_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        assert spec.loader
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TargonError(f"Failed to execute module {full_path}: {exc}") from exc

    return module


def import_app_from_ref(
    import_ref: ImportRef, base_cmd: str = "targon deploy"
) -> BaseApp:
    import_path = import_ref.file_path
    object_path = import_ref.object_path or DEFAULT_APP_NAME

    module = import_file_or_module(import_ref, base_cmd)
    if "." in object_path:
        raise click.UsageError(
            f"Nested object paths are not supported. Use simple object names like '{object_path.split('.')[0]}'"
        )
    # from module import object_path
    app = getattr(module, object_path)

    if app is None:
        console.print(
            f"[bold red]Could not find Targon app '{object_path}' in {import_path}.[/bold red]"
        )

        if not object_path:
            guidance_msg = Markdown(
                f"Expected to find an app variable named **`{DEFAULT_APP_NAME}`** (the default app name). "
                "If your `App` is assigned to a different variable name, "
                "you must specify it in the app ref argument. "
                f"For example an App variable `app_2 = App()` in `{import_path}` would "
                f"be specified as `{import_path}::app_2`."
            )
            console.print(guidance_msg)

        sys.exit(1)

    if not isinstance(app, App):
        raise click.UsageError(f"{app} is not a Targon App")

    return app
