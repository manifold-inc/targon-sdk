from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from targon.client.constants import DEFAULT_BASE_URL
from targon.client.inventory import InventoryClient
from targon.client.projects import ProjectClient
from targon.client.sandbox import SandboxesClient
from targon.client.ssh_key import SshKeyClient
from targon.client.volume import VolumeClient
from targon.client.workload import WorkloadClient
from targon.core.auth import get_api_key
from targon.core.config import Config
from targon.core.exceptions import ConfigurationError


class Client:
    """Targon SDK Client.

    Handles authentication and configuration, and exposes the various service
    clients lazily.

    Attributes:
        config (Config): Configuration including API key, timeout, retries, etc.
    """

    def __init__(
        self,
        api_key: str,
        org: Optional[str] = None,
        profile: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True,
        user_agent: Optional[str] = None,
    ) -> None:
        self.config = Config(
            api_key=api_key,
            org=org,
            profile=profile,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            user_agent=user_agent,
        )
        self.session = self._init_session()
        self.sandbox_no_retry_session = self._init_sandbox_no_retry_session()
        self._owns_session = True
        self._init_lazy_clients()

    def _init_lazy_clients(self) -> None:
        self._inventory: Optional[InventoryClient] = None
        self._workload: Optional[WorkloadClient] = None
        self._volume: Optional[VolumeClient] = None
        self._ssh_key: Optional[SshKeyClient] = None
        self._project: Optional[ProjectClient] = None
        self._sandboxes: Optional[SandboxesClient] = None
        self._orgs: Optional[Any] = None
        self._members: Optional[Any] = None
        self._api_tokens: Optional[Any] = None
        self._service_tokens: Optional[Any] = None
        self._wallet: Optional[Any] = None
        self._credits: Optional[Any] = None

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

    def _init_sandbox_no_retry_session(self) -> requests.Session:
        """Build the transport used only for non-idempotent sandbox calls."""
        session = requests.Session()
        session.headers.update(self.config.headers)
        adapter = HTTPAdapter(max_retries=0, pool_connections=10, pool_maxsize=20)
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
    def org(self) -> Optional[str]:
        return self.config.org

    @property
    def project(self) -> ProjectClient:
        if self._project is None:
            self._project = ProjectClient(self)
        return self._project

    @property
    def sandboxes(self) -> SandboxesClient:
        if self._sandboxes is None:
            self._sandboxes = SandboxesClient(self)
        return self._sandboxes

    @property
    def orgs(self) -> Any:
        if self._orgs is None:
            from targon.client.org import OrgClient

            self._orgs = OrgClient(self)
        return self._orgs

    @property
    def members(self) -> Any:
        if self._members is None:
            from targon.client.member import MemberClient

            self._members = MemberClient(self)
        return self._members

    @property
    def api_tokens(self) -> Any:
        if self._api_tokens is None:
            from targon.client.token import ApiTokenClient

            self._api_tokens = ApiTokenClient(self)
        return self._api_tokens

    @property
    def service_tokens(self) -> Any:
        if self._service_tokens is None:
            from targon.client.token import ServiceTokenClient

            self._service_tokens = ServiceTokenClient(self)
        return self._service_tokens

    @property
    def wallet(self) -> Any:
        if self._wallet is None:
            from targon.client.org import WalletClient

            self._wallet = WalletClient(self)
        return self._wallet

    @property
    def credits(self) -> Any:
        if self._credits is None:
            from targon.client.org import CreditsClient

            self._credits = CreditsClient(self)
        return self._credits

    @classmethod
    def from_env(
        cls,
        org: Optional[str] = None,
        profile: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True,
        user_agent: Optional[str] = None,
    ) -> "Client":
        api_key = get_api_key(profile)
        if not api_key:
            raise ConfigurationError(
                "API key is required. Set TARGON_API_KEY or authenticate the "
                "selected profile with `targon login`.",
                config_key="api_key",
            )

        return cls(
            api_key=api_key,
            org=org,
            profile=profile,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            user_agent=user_agent,
        )

    def require_org(self) -> str:
        return self.config.require_org()

    def for_org(self, slug: str) -> "Client":
        if not isinstance(slug, str) or not slug.strip():
            raise ConfigurationError(
                "Organization slug must be a non-empty string.",
                config_key="org",
            )

        scoped = object.__new__(type(self))
        scoped.config = Config(
            api_key=self.config.api_key,
            org=slug,
            profile=self.config.profile,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            verify_ssl=self.config.verify_ssl,
            user_agent=self.config.user_agent,
        )
        scoped.session = self.session
        scoped.sandbox_no_retry_session = self.sandbox_no_retry_session
        scoped._owns_session = False
        scoped._init_lazy_clients()
        return scoped

    def close(self) -> None:
        if self._owns_session:
            self.session.close()
            self.sandbox_no_retry_session.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
