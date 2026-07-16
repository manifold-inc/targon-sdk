from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from targon.client.inventory import InventoryClient
from targon.client.projects import ProjectClient
from targon.client.ssh_key import SshKeyClient
from targon.client.user import UserClient
from targon.client.volume import VolumeClient
from targon.client.workload import WorkloadClient
from targon.core.auth import get_api_key
from targon.core.config import Config


class Client:
    """Targon SDK Client.

    Handles authentication and configuration, and exposes the various service
    clients lazily.

    Attributes:
        config (Config): Configuration including API key, timeout, retries, etc.
    """

    def __init__(self, api_key: str, timeout: int = 30):
        self.config = Config(
            api_key=api_key, timeout=timeout, max_retries=3, verify_ssl=True
        )
        self.session = self._init_session()

        self._inventory: Optional[InventoryClient] = None
        self._workload: Optional[WorkloadClient] = None
        self._volume: Optional[VolumeClient] = None
        self._ssh_key: Optional[SshKeyClient] = None
        self._user: Optional[UserClient] = None
        self._project: Optional[ProjectClient] = None

    def _init_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.config.headers)

        retries = Retry(
            total=self.config.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )

        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.verify = self.config.verify_ssl

        return session

    @property
    def inventory(self) -> InventoryClient:
        if self._inventory is None:
            self._inventory = InventoryClient(self)
        return self._inventory

    @property
    def workload(self) -> WorkloadClient:
        if self._workload is None:
            self._workload = WorkloadClient(self)
        return self._workload

    @property
    def volume(self) -> VolumeClient:
        if self._volume is None:
            self._volume = VolumeClient(self)
        return self._volume

    @property
    def ssh_key(self) -> SshKeyClient:
        if self._ssh_key is None:
            self._ssh_key = SshKeyClient(self)
        return self._ssh_key

    @property
    def user(self) -> UserClient:
        if self._user is None:
            self._user = UserClient(self)
        return self._user

    @property
    def project(self) -> ProjectClient:
        if self._project is None:
            self._project = ProjectClient(self)
        return self._project

    @classmethod
    def from_env(cls) -> "Client":
        api_key = get_api_key()
        if not api_key:
            raise ValueError("TARGON_API_KEY environment variable not set")

        return cls(api_key=api_key)

    def close(self):
        self.session.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
