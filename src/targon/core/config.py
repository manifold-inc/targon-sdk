import os
from typing import Dict, Optional
from targon.client.constants import DEFAULT_BASE_URL
from targon.core.exceptions import ConfigurationError
from targon.version import __version__
import logging

DEFAULT_LOG_FORMAT = "\033[32m[targon-sdk]\033[0m %(asctime)s %(levelname)s %(message)s"
DEFAULT_LOG_LEVEL = logging.CRITICAL

LOG_LEVEL_LOOKUP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("targon-sdk")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)

    log_level_str = os.getenv("TARGON_LOG_LEVEL")
    if log_level_str:
        logger.setLevel(LOG_LEVEL_LOOKUP.get(log_level_str.upper(), logging.INFO))
    else:
        logger.setLevel(DEFAULT_LOG_LEVEL)

    logger.propagate = False
    return logger

logger = _configure_logger()


class Config:
    api_key: str
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
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = True,
        user_agent: Optional[str] = None,
    ) -> None:

        resolved_key = api_key or os.getenv("TARGON_API_KEY")
        if (
            not resolved_key
            or not isinstance(resolved_key, str)
            or not resolved_key.strip()
        ):
            raise ConfigurationError(
                "API key is required. Provide it via the 'api_key' parameter or do targon setup or do "
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
        }
