import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from aiohttp import ClientResponse, ClientTimeout, ClientError

from targon.client.constants import HEIM_BASE_URL, HEIM_BUILD_ENDPOINT
from targon.core.console import Console
from targon.core.exceptions import APIError, NetworkError, TimeoutError, TargonError
from targon.core.objects import AsyncBaseHTTPClient

if TYPE_CHECKING:
    from targon.client.client import Client


class AsyncHeimClient(AsyncBaseHTTPClient):
    """Async client for Heim build service with NDJSON streaming support."""

    def __init__(self, client: "Client") -> None:
        super().__init__(client)
        self.base_url = HEIM_BASE_URL

    async def build_image(
        self,
        app_id: str,
        dockerfile_commands: List[str],
        context_files: Dict[str, str],
        console_instance: Console,
        verbose: bool = False,
    ) -> str:
        if not app_id:
            raise TargonError("app_id cannot be empty")
        if not dockerfile_commands:
            raise TargonError("dockerfile_commands cannot be empty")

        payload = {
            "app_id": app_id,
            "dockerfile": dockerfile_commands,
            "files": [
                {"name": name, "content": content}
                for name, content in context_files.items()
            ],
        }

        build_url = f"{self.base_url}{HEIM_BUILD_ENDPOINT}"

        try:
            async with self.session.post(
                build_url,
                json=payload,
                headers={
                    "Accept": "application/x-ndjson",
                    "Content-Type": "application/json",
                },
                timeout=ClientTimeout(
                    total=30 * 60,
                    # Allow long pauses between streamed lines (e.g., exporting/pushing layers)
                    # while still enforcing an overall cap for the request.
                    sock_read=None,
                ),
            ) as response:
                if response.status != 200:
                    try:
                        error_text = await response.text()
                    except Exception:
                        error_text = f"HTTP {response.status}"
                    raise APIError(response.status, f"Build failed: {error_text}")

                return await self._process_build_stream(
                    response, console_instance, verbose
                )

        except APIError:
            raise
        except TimeoutError as e:
            raise TimeoutError(
                f"Build request timed out: {str(e)}", timeout=30 * 60
            ) from e
        except ClientError as e:
            raise NetworkError(f"Network error during build: {str(e)}", cause=e) from e
        except TargonError:
            raise
        except Exception as e:
            raise TargonError(f"Build request failed: {str(e)}") from e

    async def _process_build_stream(
        self, response: ClientResponse, console_instance: Console, verbose: bool
    ) -> str:
        registry_ref: Optional[str] = None
        current_step: Optional[str] = None

        try:
            async for line in response.content:
                if not line:
                    continue

                try:
                    if isinstance(line, bytes):
                        line_str = line.decode("utf-8", errors="replace").strip()
                    else:
                        line_str = str(line).strip()

                    if not line_str:
                        continue

                    data = json.loads(line_str)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if verbose:
                        console_instance.info(
                            "[dim yellow]⚠ Skipping malformed line[/dim yellow]"
                        )
                    continue

                msg_type = data.get("type")

                if msg_type == "log":
                    message = self._format_log_message(data, current_step, verbose)
                    if message:
                        step = data.get("step")
                        if step and step != current_step:
                            current_step = step
                            self._display_step(console_instance, step, verbose)

                        if verbose:
                            console_instance.info(message)
                        else:
                            console_instance.substep(message)

                elif msg_type == "progress":
                    self._handle_progress(data, console_instance, verbose)

                elif msg_type == "complete":
                    registry_ref = data.get("image_id")
                    if registry_ref:
                        console_instance.success(
                            "Build complete", detail=f"Image: {registry_ref}"
                        )
                    break

                elif msg_type == "error":
                    error_message = data.get("error", "Unknown error")
                    console_instance.error("Build failed", detail=error_message)
                    raise APIError(400, f"Build failed: {error_message}")

                elif msg_type == "status":
                    self._handle_status(data, console_instance, verbose)

        except APIError:
            raise
        except Exception as e:
            raise TargonError(f"Error processing build stream: {str(e)}") from e

        if not registry_ref:
            raise TargonError("Build completed without image identifier")

        return registry_ref

    def _handle_progress(
        self, data: Dict[str, Any], console_instance: Console, verbose: bool
    ) -> None:
        percent = data.get("percent", 0)
        description = data.get("description", "Building")

        if verbose:
            message = f"  [cyan]⏳ {description}: {percent}%[/cyan]"
            console_instance.info(message)
        else:
            message = f"[cyan]{description}: {percent}%[/cyan]"
            console_instance.substep(message)

    def _handle_status(
        self, data: Dict[str, Any], console_instance: Console, verbose: bool
    ) -> None:
        status_msg = data.get("message", "").strip()
        if not status_msg:
            return

        if verbose:
            message = f"  [yellow]ℹ[/yellow] {status_msg}"
            console_instance.info(message)
        else:
            console_instance.substep(status_msg)

    def _display_step(
        self, console_instance: Console, step: str, verbose: bool
    ) -> None:
        step_message = f"Step {step}"
        if verbose:
            console_instance.info(f"[bold blue]{step_message}[/bold blue]")
        else:
            console_instance.substep(step_message)

    def _format_log_message(
        self, data: Dict[str, Any], current_step: Optional[str], verbose: bool
    ) -> Optional[str]:
        message = data.get("message", "").strip()
        if not message:
            return None

        stream = data.get("stream", "stdout")
        is_stderr = stream == "stderr"

        if verbose:
            prefix = "[red]▸[/red]" if is_stderr else "[dim]▸[/dim]"
            return f"  {prefix} {message}"

        return f"[red]{message}[/red]" if is_stderr else message
