import os
import click
import keyring

APP_NAME = "com.targon.cli"


def get_api_key():
    env_key = os.environ.get("TARGON_API_KEY")
    if env_key:
        _save_to_keyring(env_key)
        return env_key

    keyring_key = keyring.get_password(APP_NAME, "default")
    if keyring_key:
        return keyring_key

    api_key = click.prompt("Enter your Targon API key", hide_input=True)
    _save_to_keyring(api_key)
    return api_key


def _save_to_keyring(api_key: str):
    stored = keyring.get_password(APP_NAME, "default")
    if stored != api_key:
        keyring.set_password(APP_NAME, "default", api_key)
