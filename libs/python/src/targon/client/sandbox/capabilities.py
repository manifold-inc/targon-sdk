import base64
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
from urllib.parse import urlencode

from targon.client.constants import org_path
from targon.client.sandbox.models import (
    MAX_FILE_BYTES,
    MAX_TERMINAL_DIMENSION,
    TerminalSession,
    bounded_int,
    invalid,
    non_empty,
    validate_path,
    validate_terminal_id,
)
from targon.core.exceptions import AccessTicketError, TimeoutError, ValidationError
from targon.core.objects import BaseHTTPClient

if TYPE_CHECKING:
    from targon.client.client import Client

T = TypeVar("T")


def poll(
    fetch: Callable[[], T],
    *,
    is_done: Callable[[T], bool],
    check_failure: Callable[[T], None],
    timeout: float,
    poll_interval: float,
    timeout_message: Callable[[T], str],
) -> T:
    deadline = time.monotonic() + timeout
    while True:
        value = fetch()
        if is_done(value):
            return value
        check_failure(value)
        if time.monotonic() >= deadline:
            raise TimeoutError(timeout_message(value), timeout=timeout)
        time.sleep(poll_interval)


class SandboxHTTPClient(BaseHTTPClient):
    def _workloads_path(self, *parts: str) -> str:
        path = org_path(self.client.require_org(), "workloads")
        return f"{path}/{'/'.join(parts)}" if parts else path

    def _request_for(self, workload_uid: str, method: str, path: str, **kwargs: Any):
        return self._request(method, path, _workload_uid=workload_uid, **kwargs)

    def _request_no_retry(
        self,
        method: str,
        path: str,
        *,
        workload_uid: Optional[str] = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("timeout", self.client.config.timeout)
        kwargs.setdefault("verify", self.client.config.verify_ssl)
        response = self.client.sandbox_no_retry_session.request(
            method, f"{self.base_url}{path}", **kwargs
        )
        return self._handle_response(response, workload_uid=workload_uid)


class _BoundCapability:
    def __init__(self, client: "Client", workload_uid: str) -> None:
        self._client = client
        self._workload_uid = workload_uid


class SandboxFiles(_BoundCapability):
    def read(self, path: str, *, as_text: bool = False):
        return self._client.sandboxes.files.read(
            self._workload_uid, path, as_text=as_text
        )

    def write(self, path: str, data: Union[bytes, str]) -> None:
        self._client.sandboxes.files.write(self._workload_uid, path, data)


class SandboxFilesClient(SandboxHTTPClient):
    def read(self, workload_uid: str, path: str, *, as_text: bool = False):
        workload_uid = non_empty(workload_uid, "workload_uid")
        path = validate_path(path)
        data = self._request_for(
            workload_uid,
            "GET",
            self._workloads_path(workload_uid, "files"),
            params={"path": path},
        )
        raw = base64.b64decode(data["content_b64"], validate=True)
        return raw.decode("utf-8") if as_text else raw

    def write(self, workload_uid: str, path: str, data: Union[bytes, str]) -> None:
        workload_uid = non_empty(workload_uid, "workload_uid")
        path = validate_path(path)
        if isinstance(data, str):
            raw = data.encode()
        elif isinstance(data, bytes):
            raw = data
        else:
            raise ValidationError("data must be bytes or str", field="data", value=data)
        if len(raw) > MAX_FILE_BYTES:
            raise invalid("file content exceeds 256 MiB", "data", len(raw))
        self._request_no_retry(
            "PUT",
            self._workloads_path(workload_uid, "files"),
            workload_uid=workload_uid,
            json={"path": path, "content_b64": base64.b64encode(raw).decode("ascii")},
        )


class TerminalConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def send(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("terminal data must be bytes")
        self._connection.send(data)

    def recv(self, timeout: Optional[float] = None) -> bytes:
        data = self._connection.recv(timeout=timeout)
        if not isinstance(data, bytes):
            raise RuntimeError("terminal server sent a non-binary WebSocket frame")
        return data

    def __iter__(self) -> Iterator[bytes]:
        for data in self._connection:
            if not isinstance(data, bytes):
                raise RuntimeError("terminal server sent a non-binary WebSocket frame")
            yield data

    def close(self, code: int = 1000, reason: str = "") -> None:
        self._connection.close(code=code, reason=reason)

    def __enter__(self) -> "TerminalConnection":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class SandboxTerminals(_BoundCapability):
    def list(self) -> Sequence[TerminalSession]:
        return self._client.sandboxes.terminals.list(self._workload_uid)

    def create(self, *, cols: int = 80, rows: int = 24) -> TerminalSession:
        return self._client.sandboxes.terminals.create(
            self._workload_uid, cols=cols, rows=rows
        )

    def delete(self, terminal_id: str) -> None:
        self._client.sandboxes.terminals.delete(self._workload_uid, terminal_id)

    def connect(
        self,
        terminal_id: str,
        *,
        ticket: Optional[str] = None,
        ticket_ttl_sec: int = 60,
        use_bearer: bool = False,
        open_timeout: float = 10,
    ) -> TerminalConnection:
        return self._client.sandboxes.terminals.connect(
            self._workload_uid,
            terminal_id,
            ticket=ticket,
            ticket_ttl_sec=ticket_ttl_sec,
            use_bearer=use_bearer,
            open_timeout=open_timeout,
        )


class SandboxTerminalsClient(SandboxHTTPClient):
    def _path(self, workload_uid: str, *parts: str) -> str:
        return self._workloads_path(workload_uid, "terminals", *parts)

    def list(self, workload_uid: str) -> Sequence[TerminalSession]:
        workload_uid = non_empty(workload_uid, "workload_uid")
        data = self._request_for(workload_uid, "GET", self._path(workload_uid))
        return tuple(TerminalSession.from_dict(item) for item in data)

    def create(
        self, workload_uid: str, *, cols: int = 80, rows: int = 24
    ) -> TerminalSession:
        workload_uid = non_empty(workload_uid, "workload_uid")
        bounded_int(cols, "cols", 1, MAX_TERMINAL_DIMENSION)
        bounded_int(rows, "rows", 1, MAX_TERMINAL_DIMENSION)
        return TerminalSession.from_dict(
            self._request_no_retry(
                "POST",
                self._path(workload_uid),
                workload_uid=workload_uid,
                json={"cols": cols, "rows": rows},
            )
        )

    def delete(self, workload_uid: str, terminal_id: str) -> None:
        workload_uid = non_empty(workload_uid, "workload_uid")
        terminal_id = validate_terminal_id(terminal_id)
        self._request_no_retry(
            "DELETE",
            self._path(workload_uid, terminal_id),
            workload_uid=workload_uid,
        )

    def connect(
        self,
        workload_uid: str,
        terminal_id: str,
        *,
        ticket: Optional[str] = None,
        ticket_ttl_sec: int = 60,
        use_bearer: bool = False,
        open_timeout: float = 10,
    ) -> TerminalConnection:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "Terminal WebSocket support requires the optional dependency; "
                "install with `pip install 'targon-sdk[sandbox]'`."
            ) from exc

        workload_uid = non_empty(workload_uid, "workload_uid")
        terminal_id = non_empty(terminal_id, "terminal_id")
        headers = None
        query = ""
        if use_bearer:
            if ticket is not None:
                raise invalid("ticket and use_bearer cannot be used together", "ticket")
            headers = {"Authorization": f"Bearer {self.client.config.api_key}"}
        else:
            if ticket is None:
                ticket = self.client.sandboxes.mint_access_ticket(
                    workload_uid, ttl_sec=ticket_ttl_sec
                ).ticket
            query = "?" + urlencode({"ticket": non_empty(ticket, "ticket")})

        if self.base_url.startswith("https://"):
            base_url = "wss://" + self.base_url[8:]
        elif self.base_url.startswith("http://"):
            base_url = "ws://" + self.base_url[7:]
        else:
            raise invalid(
                "base_url must use http or https",
                "base_url",
                self.base_url,
            )
        try:
            connection = connect(
                f"{base_url}{self._path(workload_uid, terminal_id, 'ws')}{query}",
                additional_headers=headers,
                open_timeout=open_timeout,
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None) or getattr(
                exc, "status_code", None
            )
            if status_code == 401:
                raise AccessTicketError(
                    401,
                    "Terminal access ticket is invalid, expired, or already used",
                    reason="WORKLOAD_ACCESS_TICKET_INVALID",
                    workload_uid=workload_uid,
                    cause=exc,
                ) from exc
            raise
        return TerminalConnection(connection)
