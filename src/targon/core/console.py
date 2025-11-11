from typing import Any, Generator, List, Optional, Tuple, Union
import time

from rich.console import Console as RichConsole, Group
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.align import Align
from rich.style import Style
from rich import box
from contextlib import contextmanager

from targon.core.exceptions import ValidationError

_rich_console = RichConsole(stderr=True)


class Console:
    console: RichConsole
    app_name: Optional[str]
    _live: Optional[Live]
    _start_time: float
    _step_start_time: float
    _lines: List[Union[str, Spinner, Tuple[Spinner, str]]]
    _current_step: str
    _active_spinner: Optional[Spinner]
    _active_step_message: str
    _step_count: int
    _success_count: int

    def __init__(self, app_name: Optional[str] = None) -> None:

        self.console = _rich_console
        self.app_name = app_name.strip() if app_name and app_name.strip() else None
        self._live = None
        self._start_time = 0.0
        self._step_start_time = 0.0
        self._lines = []
        self._current_step = ""
        self._active_spinner = None
        self._active_step_message = ""
        self._step_count = 0
        self._success_count = 0

    def __enter__(self) -> "Console":
        self._start_time = time.time()
        self._lines = []
        self._step_count = 0
        self._success_count = 0

        self._live = Live(
            self._render(), console=self.console, refresh_per_second=10, transient=False
        )
        self._live.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._live:
            self._live.stop()
        self._live = None

    def step(self, message: str, detail: str = "") -> None:
        if self._active_step_message == message:
            return

        self._step_start_time = time.time()
        self._current_step = message
        self._active_step_message = message
        self._step_count += 1

        self._active_spinner = Spinner(
            "dots", text=f"[bold cyan]{message}[/bold cyan]", style="bright_cyan"
        )

        if detail:
            self._lines.append((self._active_spinner, detail))
        else:
            self._lines.append(self._active_spinner)

        self._update()

    def substep(self, message: str, detail: str = "") -> None:
        line = f"  [bright_blue]▸[/bright_blue] {message}"
        if detail:
            line += f" [dim italic]{detail}[/dim italic]"

        if (
            self._lines
            and isinstance(self._lines[-1], str)
            and self._lines[-1].startswith("  [bright_blue]▸[/bright_blue]")
        ):
            self._lines[-1] = line
        else:
            self._lines.append(line)
        self._update()

    def resource(self, name: str, resource_id: str) -> None:
        line = f"  [dim]╰─[/dim] [bold yellow]{name}[/bold yellow] [dim]→[/dim] [bold green]{resource_id}[/bold green]"
        self._lines.append(line)
        self._update()

    def success(
        self, message: str, detail: str = "", duration: Optional[float] = None
    ) -> None:
        if duration is not None and (
            not isinstance(duration, (int, float)) or duration < 0
        ):
            raise ValidationError(
                "duration must be a non-negative number or None",
                field="duration",
                value=duration,
            )

        if duration is None and self._step_start_time:
            duration = time.time() - self._step_start_time

        if self._lines and (
            isinstance(self._lines[-1], Spinner)
            or (
                isinstance(self._lines[-1], tuple)
                and isinstance(self._lines[-1][0], Spinner)
            )
        ):
            self._lines.pop()

        self._active_spinner = None
        self._success_count += 1

        line = f"[bold green]✔[/bold green] [bold]{message}[/bold]"
        if detail:
            line += f" [dim italic]{detail}[/dim italic]"
        if duration is not None and duration > 0:
            line += f" [dim bright_black]· {duration:.2f}s[/dim bright_black]"

        self._lines.append(line)
        self._update()

    def error(self, message: str, detail: str = "") -> None:
        if self._lines and (
            isinstance(self._lines[-1], Spinner)
            or (
                isinstance(self._lines[-1], tuple)
                and isinstance(self._lines[-1][0], Spinner)
            )
        ):
            self._lines.pop()

        self._active_spinner = None

        line = f"[bold red]✖[/bold red] [bold]{message}[/bold]"
        if detail:
            line += f" [dim italic]{detail}[/dim italic]"

        self._lines.append(line)
        self._update()

    def info(self, message: str, detail: str = "") -> None:
        line = f"[bright_blue]ℹ[/bright_blue] {message}"
        if detail:
            line += f" [dim italic]{detail}[/dim italic]"

        self._lines.append(line)
        self._update()

    def separator(self) -> None:
        self._lines.append("")
        self._update()

    def final(self, message: str, details: Optional[List[str]] = None) -> None:
        if details is not None:
            for i, detail in enumerate(details):
                if not isinstance(detail, str):
                    raise ValidationError(
                        f"details[{i}] must be a string",
                        field=f"details[{i}]",
                        value=type(detail).__name__,
                    )

        total_duration = time.time() - self._start_time if self._start_time else 0

        self.separator()
        self._lines.append(f"[bold bright_green]✨ {message}[/bold bright_green]")

        if total_duration > 0:
            self._lines.append(
                f"[dim bright_black]   Completed in {total_duration:.2f}s[/dim bright_black]"
            )

        if details:
            self._lines.append("")
            for detail in details:
                self._lines.append(f"[dim]   {detail}[/dim]")

        self._update()

    def _update(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _create_header(self) -> Panel:
        if not self.app_name:
            # Create a simple targon header
            header_content = Text()
            header_content.append("TARGON", style="bold blue")
            return Panel(
                Align.center(header_content),
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(0, 0),
            )

        # Create app-specific header with stats
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left")
        table.add_column(justify="right")

        app_text = Text()
        app_text.append(self.app_name, style="bold blue")

        stats_text = Text()
        if self._step_count > 0:
            stats_text.append(
                f"{self._success_count}/{self._step_count} steps",
                style="dim bright_black",
            )

        table.add_row(app_text, stats_text)

        return Panel(
            table,
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def _render(self) -> Union[Panel, Group, Text]:
        rendered_lines: List[Any] = []

        # Add header if we have an app name or want to show branding
        if self.app_name or self._step_count > 0:
            rendered_lines.append(self._create_header())
            rendered_lines.append(Text(""))

        # Render content lines
        for item in self._lines:
            if isinstance(item, Spinner):
                rendered_lines.append(item)
            elif (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], Spinner)
            ):
                spinner, detail = item
                rendered_lines.append(spinner)
                rendered_lines.append(
                    Text.from_markup(f"  [dim italic]{detail}[/dim italic]")
                )
            elif isinstance(item, str):
                if item:
                    rendered_lines.append(Text.from_markup(item))
                else:
                    rendered_lines.append(Text(""))

        if not rendered_lines:
            return Text("")

        content = Group(*rendered_lines)

        # Wrap everything in a subtle panel for a cleaner look
        return Panel(
            content,
            border_style="dim bright_black",
            padding=(1, 2),
            box=box.ROUNDED,
        )


@contextmanager
def console(app_name: Optional[str] = None) -> Generator[Console, None, None]:
    c = Console(app_name)
    with c:
        yield c
