import requests
import aiohttp
import asyncio
from typing import Callable, Any, TypeVar
from requests.adapters import HTTPAdapter, Retry
from targon.cli.auth import get_api_key
from targon.client.inventory import AsyncInventoryClient
from targon.client.serverless import AsyncServerlessClient
from targon.core.config import Config
from targon.client.heim import AsyncHeimClient
from targon.client.functions import AsyncFunctionsClient
from targon.client.app import AsyncAppClient
from targon.client.publish import AsyncPublishClient
from targon.client.logs import AsyncLogsClient

T = TypeVar('T')


class Client:
    """
    Targon SDK Client
    Handles authentication, configuration, and exposes various service clients lazily.
    Attributes:
        config (Config): Configuration object including API key, timeout, retries, etc.
    """

    def __init__(self, api_key: str, timeout: int = 30):
        self.config = Config(
            api_key=api_key, timeout=timeout, max_retries=3, verify_ssl=True
        )
        self.session = self._init_session()
        self._async_session = None

        # Lazy-loaded async clients
        self._async_inventory = None
        self._async_heim = None
        self._async_functions = None
        self._async_app = None
        self._async_publish = None
        self._async_logs = None
        self._async_serverless = None
        
    def _init_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self.config.headers)

        retries = Retry(
            total=self.config.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )

        # Add connection pooling for production performance
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        session.verify = self.config.verify_ssl

        return session

    def _init_async_session(self) -> aiohttp.ClientSession:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)

        session = aiohttp.ClientSession(
            headers=self.config.headers, timeout=timeout, connector=connector
        )
        return session

    @property
    def async_session(self) -> aiohttp.ClientSession:
        if self._async_session is None:
            self._async_session = self._init_async_session()
        return self._async_session

    @property
    def async_inventory(self) -> AsyncInventoryClient:
        """Lazy initialization of async inventory client."""
        if self._async_inventory is None:
            self._async_inventory = AsyncInventoryClient(self)
        return self._async_inventory

    @property
    def async_heim(self) -> AsyncHeimClient:
        if self._async_heim is None:
            self._async_heim = AsyncHeimClient(self)
        return self._async_heim

    @property
    def async_functions(self) -> AsyncFunctionsClient:
        if self._async_functions is None:
            self._async_functions = AsyncFunctionsClient(self)
        return self._async_functions

    @property
    def async_app(self) -> AsyncAppClient:
        if self._async_app is None:
            self._async_app = AsyncAppClient(self)
        return self._async_app

    @property
    def async_publish(self) -> AsyncPublishClient:
        if self._async_publish is None:
            self._async_publish = AsyncPublishClient(self)
        return self._async_publish

    @property
    def async_logs(self) -> AsyncLogsClient:
        if self._async_logs is None:
            self._async_logs = AsyncLogsClient(self)
        return self._async_logs
    
    @property
    def async_serverless(self) -> AsyncServerlessClient:
        if self._async_serverless is None:
            self._async_serverless = AsyncServerlessClient(self)
        return self._async_serverless

    @classmethod
    def from_env(cls):
        import os

        api_key = get_api_key()
        if not api_key:
            raise ValueError("TARGON_API_KEY environment variable not set")

        return cls(api_key=api_key)

    def close(self):
        # Clean up the session when done.
        self.session.close()

    async def aclose(self):
        """Clean up both async and sync sessions."""
        if self._async_session is not None:
            await self._async_session.close()
        # Also close sync session for complete cleanup
        if self.session:
            self.session.close()

    def __enter__(self):
        # Enable context manager usage.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Auto-close the session when exiting context.
        self.close()

    async def __aenter__(self):
        # Enable async context manager usage.
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # Auto-close the async session when exiting context.
        await self.aclose()

    def run_async(self, coro_func: Callable[[], Any]) -> Any:
        async def _run():
            async with self:
                return await coro_func()

        return asyncio.run(_run())
