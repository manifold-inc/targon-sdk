import json
from typing import TYPE_CHECKING, Any

import requests

from targon.core.exceptions import APIError

if TYPE_CHECKING:
    from targon.client.client import Client


class BaseHTTPClient:
    """Synchronous base client wrapping a shared ``requests.Session``."""

    def __init__(self, client: "Client") -> None:
        self.client = client
        self.session = client.session
        self.base_url = client.config.base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "BaseHTTPClient":
        from targon.client.client import Client

        client = Client.from_env()
        return cls(client)

    def _get(self, path: str, **kwargs: Any):
        res = self.session.get(f"{self.base_url}{path}", **kwargs)
        return self._handle_response(res)

    def _post(self, path: str, **kwargs: Any):
        res = self.session.post(f"{self.base_url}{path}", **kwargs)
        return self._handle_response(res)

    def _put(self, path: str, **kwargs: Any):
        res = self.session.put(f"{self.base_url}{path}", **kwargs)
        return self._handle_response(res)

    def _patch(self, path: str, **kwargs: Any):
        res = self.session.patch(f"{self.base_url}{path}", **kwargs)
        return self._handle_response(res)

    def _delete(self, path: str, **kwargs: Any):
        res = self.session.delete(f"{self.base_url}{path}", **kwargs)
        return self._handle_response(res)

    def _handle_response(self, res: requests.Response):
        if res.status_code >= 400:
            text = res.text
            message = text
            reason = None
            try:
                body = json.loads(text)
                if isinstance(body, dict):
                    message = body.get("error", text)
                    reason = body.get("reason")
            except (json.JSONDecodeError, ValueError):
                pass
            raise APIError(
                res.status_code,
                message,
                response={"reason": reason} if reason else None,
            )

        content_type = res.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                return res.json()
            except (ValueError, json.JSONDecodeError):
                return res.text
        return res.text
