from typing import Dict, Optional

from targon.client.constants import DEFAULT_BASE_URL
from targon.core.auth import get_api_key, get_org, get_profile
from targon.core.exceptions import ConfigurationError
from targon.version import __version__


class Config:
    api_key: str
    org: Optional[str]
    profile: str
    base_url: str
    timeout: int
    max_retries: int
    verify_ssl: bool
    version: str
    user_agent: str
    headers: Dict[str, str]

    def __init__(
        self,
        api_key: Optional[str] = None,
        org: Optional[str] = None,
        profile: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True,
        user_agent: Optional[str] = None,
    ) -> None:

        resolved_profile = get_profile(profile)
        resolved_key = api_key or get_api_key(resolved_profile)
        if (
            not resolved_key
            or not isinstance(resolved_key, str)
            or not resolved_key.strip()
        ):
            raise ConfigurationError(
                "API key is required. Provide it via the 'api_key' parameter or "
                "set the TARGON_API_KEY environment variable.",
                config_key="api_key",
            )

        if not base_url or not isinstance(base_url, str) or not base_url.strip():
            raise ConfigurationError(
                "base_url must be a non-empty string", config_key="base_url"
            )

        if timeout <= 0:
            raise ConfigurationError(
                "timeout must be greater than 0", config_key="timeout"
            )

        if max_retries < 0:
            raise ConfigurationError(
                "max_retries must be non-negative", config_key="max_retries"
            )

        self.api_key = resolved_key.strip()
        self.profile = resolved_profile
        self.org = get_org(org, resolved_profile)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.version = __version__
        self.user_agent = (
            user_agent.strip() if user_agent else f"targon-sdk-python/{__version__}"
        )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

    def require_org(self) -> str:
        if self.org:
            return self.org
        raise ConfigurationError(
            "organization is required. Pass org='your-org' to Client(), set "
            "TARGON_ORG, or select one with `targon org use <slug>`.",
            config_key="org",
        )
