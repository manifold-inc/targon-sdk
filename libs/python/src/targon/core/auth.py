import os
import re
from pathlib import Path
from typing import Optional

API_KEY_ENV = "TARGON_API_KEY"
DEFAULT_PROFILE = "default"

CONFIG_DIR = Path.home() / ".targon"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def _credentials_file(profile: str) -> Path:
    return CONFIG_DIR / f"credentials-{profile}"


def _current_profile() -> str:
    try:
        contents = CONFIG_FILE.read_text()
    except OSError:
        return DEFAULT_PROFILE

    match = re.search(r'^\s*current\s*=\s*"([^"]*)"', contents, re.MULTILINE)
    if match and match.group(1):
        return match.group(1)
    return DEFAULT_PROFILE


def _get_from_file() -> Optional[str]:
    path = _credentials_file(_current_profile())
    if path.exists():
        try:
            key = path.read_text().strip()
        except OSError:
            return None
        return key or None
    return None


def get_api_key() -> Optional[str]:
    env_key = os.environ.get(API_KEY_ENV)
    if env_key and env_key.strip():
        return env_key.strip()

    return _get_from_file()
