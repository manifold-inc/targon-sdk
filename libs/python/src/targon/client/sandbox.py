from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

from targon.client.workload import (
    CreateWorkloadRequest,
    EnvVar,
    ExecResponse,
    PortConfig,
    WorkloadStateResponse,
)
from targon.core.resources import Resources

if TYPE_CHECKING:
    from targon.client.client import Client

DEFAULT_SANDBOX_IMAGE = "debian:stable-slim"
KEEP_ALIVE_COMMAND = ["sleep", "infinity"]


class Sandbox:
    """An interactive, isolated container you can run commands in.

    A Sandbox is backed by a RENTAL workload. Use :meth:`create` to provision a
    new one or :meth:`from_workload` to attach to an existing workload, then run
    commands with :meth:`exec`::

        import targon

        s = targon.Sandbox.create(image="debian:stable-slim")
        response = s.exec('echo "Hello World from exec!"', timeout=10)
        if response.exit_code != 0:
            print(f"Error: {response.exit_code} {response.result}")
        else:
            print(response.result)

        targon.Sandbox.terminate(s)
    """

    def __init__(
        self,
        workload_uid: str,
        client: "Client",
        *,
        _owned: bool = False,
    ) -> None:
        self._workload_uid = workload_uid
        self._client = client
        # Whether this Sandbox provisioned its own workload (and may auto-clean
        # it up on context-manager exit).
        self._owned = _owned

    # -- Construction --------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        image: str = DEFAULT_SANDBOX_IMAGE,
        resource: str = Resources.CPU_SMALL,
        name: Optional[str] = None,
        ports: Optional[List[PortConfig]] = None,
        envs: Optional[Union[dict, List[EnvVar]]] = None,
        project_id: Optional[str] = None,
        command: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
        keep_alive: bool = True,
        ready_timeout: float = 300.0,
        client: Optional["Client"] = None,
    ) -> "Sandbox":
        """Provision and deploy a new sandbox, waiting until it is running.

        Args:
            image: Container image to run.
            resource: Resource tier (see :class:`targon.Resources`).
            name: Optional workload name; auto-generated when omitted.
            ports: Ports to expose.
            envs: Environment variables (dict or list of ``EnvVar``).
            project_id: Optional project to create the workload under.
            command: Override the container entrypoint command.
            args: Arguments for the container command.
            keep_alive: When no ``command`` is given, run ``sleep infinity`` so
                the container stays up to exec into.
            ready_timeout: Seconds to wait for the workload to become ready.
            client: Optional client override; defaults to ``Client.from_env()``.
        """
        from targon.client.client import Client

        client = client or Client.from_env()

        if command is None and keep_alive:
            command = list(KEEP_ALIVE_COMMAND)

        if not name:
            name = f"sandbox-{uuid.uuid4().hex[:8]}"

        request = CreateWorkloadRequest(
            name=name,
            image=image,
            resource_name=resource,
            type="RENTAL",
            project_id=project_id,
            ports=ports,
            envs=envs,
            commands=command,
            args=args,
        )

        workload = client.workload.create(request)
        client.workload.deploy(workload.uid)
        client.workload.wait_until_ready(workload.uid, timeout=ready_timeout)
        return cls(workload.uid, client, _owned=True)

    @classmethod
    def from_workload(
        cls,
        workload_uid: str,
        *,
        client: Optional["Client"] = None,
    ) -> "Sandbox":
        """Attach to an existing running workload by its UID."""
        from targon.client.client import Client

        client = client or Client.from_env()
        return cls(workload_uid, client, _owned=False)

    # Alias matching the workload-centric naming.
    from_id = from_workload

    # -- Properties ----------------------------------------------------------

    @property
    def workload_uid(self) -> str:
        """UID of the underlying workload."""
        return self._workload_uid

    @property
    def id(self) -> str:
        """Alias for :attr:`workload_uid`."""
        return self._workload_uid

    # -- Command execution ---------------------------------------------------

    def exec(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
    ) -> ExecResponse:
        """Run a shell command and return its exit code and output."""
        return self._client.workload.exec(self._workload_uid, command, timeout=timeout)

    def exec_stream(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
    ) -> Iterator[str]:
        """Run a shell command, streaming merged stdout/stderr lines."""
        return self._client.workload.exec_stream(
            self._workload_uid, command, timeout=timeout
        )

    # -- State / lifecycle ---------------------------------------------------

    def get_state(self) -> WorkloadStateResponse:
        """Return the current state of the underlying workload."""
        return self._client.workload.get_state(self._workload_uid)

    def wait_until_ready(
        self,
        *,
        timeout: float = 300.0,
        poll_interval: float = 5.0,
    ) -> WorkloadStateResponse:
        """Block until the underlying workload is running."""
        return self._client.workload.wait_until_ready(
            self._workload_uid, timeout=timeout, poll_interval=poll_interval
        )

    def terminate(self) -> None:
        """Delete the underlying workload.

        Works both as an instance method (``sandbox.terminate()``) and as
        ``targon.Sandbox.terminate(sandbox)``.
        """
        self._client.workload.delete(self._workload_uid)

    # Alias.
    kill = terminate

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Only auto-terminate sandboxes we created; leave attached ones alone.
        if self._owned:
            self.terminate()
