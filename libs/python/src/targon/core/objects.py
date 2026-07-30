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

    def _request(self, method: str, path: str, **kwargs: Any):
        kwargs.setdefault("timeout", self.client.config.timeout)
        kwargs.setdefault("verify", self.client.config.verify_ssl)
        res = self.session.request(
            method,
            f"{self.base_url}{path}",
            **kwargs,
        )
        return self._handle_response(res)

    def _get(self, path: str, **kwargs: Any):
        return self._request("GET", path, **kwargs)

    def _post(self, path: str, **kwargs: Any):
        return self._request("POST", path, **kwargs)

    def _put(self, path: str, **kwargs: Any):
        return self._request("PUT", path, **kwargs)

    def _patch(self, path: str, **kwargs: Any):
        return self._request("PATCH", path, **kwargs)

    def _delete(self, path: str, **kwargs: Any):
        return self._request("DELETE", path, **kwargs)

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
