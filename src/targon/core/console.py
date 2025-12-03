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
    _current_step: str
    _active_spinner: Optional[Spinner]
    _active_step_message: str
    _active_substep_message: str
    _active_substep_detail: str
    _active_detail: str
    _step_count: int
    _success_count: int
    _header_printed: bool

    def __init__(self, app_name: Optional[str] = None) -> None:
        self.console = _rich_console
        self.app_name = app_name.strip() if app_name and app_name.strip() else None
        self._live = None
        self._start_time = 0.0
        self._step_start_time = 0.0
        self._current_step = ""
        self._active_spinner = None
        self._active_step_message = ""
        self._active_substep_message = ""
        self._active_substep_detail = ""
        self._active_detail = ""
        self._step_count = 0
        self._success_count = 0
        self._header_printed = False

    def __enter__(self) -> "Console":
        self._start_time = time.time()
        self._step_count = 0
        self._success_count = 0
        self._header_printed = False

        self._print_header()

        self._live = Live(
            "", console=self.console, refresh_per_second=10, transient=True
        )
        self._live.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._live:
            self._live.stop()
        self._live = None

    def _print_header(self) -> None:
        if self._header_printed:
            return

        # Simple clean divider
        self.console.print()

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", no_wrap=True)
        table.add_column(justify="right")

        left_text = Text()
        if self.app_name:
            left_text.append("▸ ", style="bright_blue")
            left_text.append(self.app_name, style="bold")
        else:
            left_text.append("▸ ", style="bright_blue")
            left_text.append("TARGON", style="bold")

        right_text = Text()
        right_text.append(time.strftime("%H:%M:%S"), style="dim")

        table.add_row(left_text, right_text)

        self.console.print(table)
        self.console.print("─" * (self.console.width - 2), style="dim")
        self.console.print()

        self._header_printed = True

    def _print_line(self, line: str) -> None:
        """Print a completed line to the console"""
        text = Text.from_markup(line)
        # Ensure truncation for long lines
        if len(text) > self.console.width - 2:
            text.truncate(self.console.width - 2, overflow="ellipsis")
        self.console.print(text, overflow="ellipsis", no_wrap=True)

    def step(self, message: str, detail: str = "") -> None:
        if self._active_step_message == message:
            return

        # Clear any previous spinner
        if self._active_spinner and self._live:
            self._live.update("")

        self._step_start_time = time.time()
        self._current_step = message
        self._active_step_message = message
        self._active_substep_message = ""
        self._active_substep_detail = ""
        self._active_detail = detail
        self._step_count += 1

        self._active_spinner = Spinner("dots", text=f"{message}", style="cyan")

        self._update_spinner()

    def substep(self, message: str, detail: str = "") -> None:
        # Update live substep
        self._active_substep_message = message
        self._active_substep_detail = detail
        self._update_spinner()

    def resource(self, name: str, resource_id: str) -> None:
        line = f"  [dim]╰[/dim] [cyan]{name}[/cyan] [dim]→[/dim] [green]{resource_id}[/green]"
        self._print_line(line)

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

        # Clear spinner
        if self._live:
            self._live.update("")

        self._active_spinner = None
        self._active_substep_message = ""
        self._active_substep_detail = ""
        self._success_count += 1

        line = f"[green]✓[/green] {message}"
        if detail:
            line += f" [dim]· {detail}[/dim]"
        if duration is not None and duration > 0:
            line += f" [dim]{duration:.2f}s[/dim]"

        self._print_line(line)

    def error(self, message: str, detail: str = "") -> None:
        # Clear spinner
        if self._live:
            self._live.update("")

        self._active_spinner = None
        self._active_substep_message = ""
        self._active_substep_detail = ""

        line = f"[red]✗[/red] {message}"
        if detail:
            line += f" [dim]· {detail}[/dim]"

        self._print_line(line)

    def info(self, message: str, detail: str = "") -> None:
        line = f"[cyan]→[/cyan] {message}"
        if detail:
            line += f" [dim]· {detail}[/dim]"

        self._print_line(line)

    def separator(self) -> None:
        self.console.print()

    def final(self, message: str, details: Optional[List[str]] = None) -> None:
        if details is not None:
            for i, detail in enumerate(details):
                if not isinstance(detail, str):
                    raise ValidationError(
                        f"details[{i}] must be a string",
                        field=f"details[{i}]",
                        value=type(detail).__name__,
                    )

        # Clear spinner
        if self._live:
            self._live.update("")

        total_duration = time.time() - self._start_time if self._start_time else 0

        self.separator()
        self.console.print("─" * (self.console.width - 2), style="dim")

        # Success message
        msg_line = f"[bold green]✓[/bold green] {message}"
        if total_duration > 0:
            msg_line += f" [dim]({total_duration:.2f}s)[/dim]"

        self._print_line(msg_line)

        if self._step_count > 0:
            stats_line = (
                f"  [dim]{self._success_count}/{self._step_count} steps completed[/dim]"
            )
            self._print_line(stats_line)

        if details:
            self.console.print()
            for detail in details:
                self._print_line(f"  [dim]{detail}[/dim]")

        self.console.print()

    def _update_spinner(self) -> None:
        """Update the live spinner display"""
        if not self._live or not self._active_spinner:
            return

        rendered_lines: List[Any] = [self._active_spinner]

        if self._active_detail:
            detail_text = Text.from_markup(f"  [dim]{self._active_detail}[/dim]")
            if len(detail_text) > self.console.width - 4:
                detail_text.truncate(self.console.width - 4, overflow="ellipsis")
            rendered_lines.append(detail_text)

        if self._active_substep_message:
            line = f"  [dim]│[/dim] {self._active_substep_message}"
            if self._active_substep_detail:
                line += f" [dim]· {self._active_substep_detail}[/dim]"

            substep_text = Text.from_markup(line)
            if len(substep_text) > self.console.width - 2:
                substep_text.truncate(self.console.width - 2, overflow="ellipsis")
            rendered_lines.append(substep_text)

        content = Group(*rendered_lines)
        self._live.update(content)


@contextmanager
def console(app_name: Optional[str] = None) -> Generator[Console, None, None]:
    c = Console(app_name)
    with c:
        yield c
