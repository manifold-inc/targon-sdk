from dataclasses import dataclass
from typing import Any, Dict, Optional

from targon.client.constants import DEFAULT_BASE_URL, MOUNT_FILE_ENDPOINT
from targon.core.exceptions import ValidationError
from targon.core.function import HydrationError
from targon.core.objects import AsyncBaseHTTPClient


@dataclass(slots=True)
class MountFileRequest:
    sha256_hex: str
    data: Optional[str] = None
    data_blob_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"sha256_hex": self.sha256_hex}
        if self.data is not None:
            payload["data"] = self.data
        if self.data_blob_id is not None:
            payload["data_blob_id"] = self.data_blob_id
        return payload


@dataclass(slots=True)
class MountFileResponse:
    exists: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MountFileResponse":
        if not isinstance(data, dict):
            raise HydrationError(
                f"Invalid response format: expected dict, got {type(data).__name__}",
                object_type="MountFileResponse",
            )
        exists = data.get("exists")
        if not isinstance(exists, bool):
            raise HydrationError(
                "Missing or invalid 'exists' field in mount response",
                object_type="MountFileResponse",
            )
        return cls(exists=exists)


class AsyncMountClient(AsyncBaseHTTPClient):
    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    async def mount_put_file(
        self, sha256_hex: str, data: Optional[str] = None, data_blob_id: Optional[str] = None
    ) -> bool:

        if sha256_hex and not isinstance(sha256_hex, str):
            raise ValidationError("sha256_hex must be a hex-encoded string", field="sha256_hex", value=type(sha256_hex).__name__)

        if data is not None and not isinstance(data, str):
            raise ValidationError("data must be a base64-encoded string", field="data", value=type(data).__name__)
        
        if data_blob_id is not None and not isinstance(data_blob_id, str):
            raise ValidationError("data_blob_id must be a base64-encoded string", field="data_blob_id", value=type(data_blob_id).__name__)

        request = MountFileRequest(sha256_hex=sha256_hex, data=data, data_blob_id=data_blob_id)
        
        result = await self._async_post(MOUNT_FILE_ENDPOINT, json=request.to_payload())

        response = MountFileResponse.from_dict(result)
        return response.exists