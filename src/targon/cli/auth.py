import os
import stat
import click
from pathlib import Path

APP_NAME = "com.targon.cli"
CONFIG_DIR = Path.home() / ".config" / "targon"
CREDENTIALS_FILE = CONFIG_DIR / "credentials"

# Try to use keyring, but gracefully handle when it's not available
_keyring_available = False
try:
    import keyring
    # Test if keyring backend is actually usable
    keyring.get_password(APP_NAME, "__test__")
    _keyring_available = True
except Exception:
    _keyring_available = False


def _get_from_file() -> str | None:
    """Read API key from file-based storage."""
    if CREDENTIALS_FILE.exists():
        try:
            return CREDENTIALS_FILE.read_text().strip()
        except Exception:
            return None
    return None


def _save_to_file(api_key: str) -> None:
    """Save API key to file with secure permissions."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CREDENTIALS_FILE.write_text(api_key)
        # Set file permissions to user-only (600)
        CREDENTIALS_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass  # Silently fail, env var can still be used


def _get_from_keyring() -> str | None:
    """Try to get API key from system keyring."""
    if not _keyring_available:
        return None
    try:
        return keyring.get_password(APP_NAME, "default")
    except Exception:
        return None


def _save_to_keyring(api_key: str) -> bool:
    """Try to save API key to system keyring. Returns True if successful."""
    if not _keyring_available:
        return False
    try:
        stored = keyring.get_password(APP_NAME, "default")
        if stored != api_key:
            keyring.set_password(APP_NAME, "default", api_key)
        return True
    except Exception:
        return False


def get_stored_key() -> str | None:
    """Get stored API key from keyring or file."""
    # Try keyring first
    key = _get_from_keyring()
    if key:
        return key
    # Fall back to file
    return _get_from_file()


def save_api_key(api_key: str) -> bool:
    """Save API key to keyring (preferred) or file. Returns True if successful."""
    # Try keyring first
    if _save_to_keyring(api_key):
        return True
    # Fall back to file
    try:
        _save_to_file(api_key)
        return True
    except Exception:
        return False


def get_api_key():
    """Get API key from env, keyring, file, or prompt user."""
    env_key = os.environ.get("TARGON_API_KEY")
    if env_key:
        save_api_key(env_key)
        return env_key

    stored_key = get_stored_key()
    if stored_key:
        return stored_key

    api_key = click.prompt("Enter your Targon API key", hide_input=True)
    save_api_key(api_key)
    return api_key


def is_keyring_available() -> bool:
    """Check if system keyring is available."""
    return _keyring_available
