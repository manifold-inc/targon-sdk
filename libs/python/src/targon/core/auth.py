import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

API_KEY_ENV = "TARGON_API_KEY"
ORG_ENV = "TARGON_ORG"
DEFAULT_PROFILE = "default"

CONFIG_DIR = Path.home() / ".targon"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def _credentials_file(profile: str) -> Path:
    return CONFIG_DIR / f"credentials-{profile}"


def _read_config() -> Optional[str]:
    try:
        return CONFIG_FILE.read_text()
    except OSError:
        return None


def _parse_config(contents: str) -> Dict[str, Any]:
    try:
        import importlib

        tomllib = importlib.import_module("tomllib")
        parsed = tomllib.loads(contents)
        return parsed if isinstance(parsed, dict) else {}
    except (ImportError, ValueError):
        return {}


def _toml_string(value: str) -> Optional[str]:
    value = value.strip()
    if len(value) < 2:
        return None
    if value.startswith('"') and value.endswith('"'):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, str) else None
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return None


def _fallback_value(
    contents: str,
    key: str,
    profile: Optional[str] = None,
) -> Optional[str]:
    if profile is None:
        root = contents.split("[", 1)[0]
        match = re.search(
            rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*(?:#.*)?$",
            root,
            re.MULTILINE,
        )
        return _toml_string(match.group(1)) if match else None

    section_pattern = re.compile(
        r"^\s*\[\s*profiles\s*\.\s*"
        r'(?:"((?:\\.|[^"])*)"|\'([^\']*)\'|([A-Za-z0-9_-]+))'
        r"\s*\]\s*(?:#.*)?$",
        re.MULTILINE,
    )
    sections = list(section_pattern.finditer(contents))
    for index, section in enumerate(sections):
        double_quoted, literal, bare = section.groups()
        if double_quoted is not None:
            name = _toml_string(f'"{double_quoted}"') or ""
        else:
            name = literal if literal is not None else bare
        if name != profile:
            continue
        end = (
            sections[index + 1].start() if index + 1 < len(sections) else len(contents)
        )
        body = contents[section.end() : end]
        match = re.search(
            rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*(?:#.*)?$",
            body,
            re.MULTILINE,
        )
        return _toml_string(match.group(1)) if match else None
    return None


def _config_value(key: str, profile: Optional[str] = None) -> Optional[str]:
    contents = _read_config()
    if contents is None:
        return None

    parsed = _parse_config(contents)
    if parsed:
        value: Any
        if profile is None:
            value = parsed.get(key)
        else:
            profiles = parsed.get("profiles", {})
            profile_config = (
                profiles.get(profile, {}) if isinstance(profiles, dict) else {}
            )
            value = (
                profile_config.get(key) if isinstance(profile_config, dict) else None
            )
        return value.strip() if isinstance(value, str) and value.strip() else None

    return _fallback_value(contents, key, profile)


def get_profile(profile: Optional[str] = None) -> str:
    if isinstance(profile, str) and profile.strip():
        return profile.strip()

    return _config_value("current") or DEFAULT_PROFILE


def _get_from_file(profile: Optional[str] = None) -> Optional[str]:
    path = _credentials_file(get_profile(profile))
    if path.exists():
        try:
            key = path.read_text().strip()
        except OSError:
            return None
        return key or None
    return None


def get_api_key(profile: Optional[str] = None) -> Optional[str]:
    env_key = os.environ.get(API_KEY_ENV)
    if env_key and env_key.strip():
        return env_key.strip()

    return _get_from_file(profile)


def get_profile_org(profile: Optional[str] = None) -> Optional[str]:
    return _config_value("org", get_profile(profile))


def get_org(
    org: Optional[str] = None,
    profile: Optional[str] = None,
) -> Optional[str]:
    if isinstance(org, str) and org.strip():
        return org.strip()

    env_org = os.environ.get(ORG_ENV)
    if env_org and env_org.strip():
        return env_org.strip()

    return get_profile_org(profile)
