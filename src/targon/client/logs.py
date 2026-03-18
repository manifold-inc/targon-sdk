import aiohttp
from typing import AsyncGenerator
from targon.core.exceptions import ValidationError, APIError
from targon.client.constants import WORKLOAD_LOGS_ENDPOINT, DEFAULT_BASE_URL
from targon.core.objects import AsyncBaseHTTPClient


class AsyncLogsClient(AsyncBaseHTTPClient):
    """Async client for workload logs."""

    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    async def stream_logs(
        self, workload_uid: str, follow: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream logs from a workload."""

        if not workload_uid or not workload_uid.strip():
            raise ValidationError(
                "workload_uid is required",
                field="workload_uid",
                value=workload_uid,
            )

        url = f"{self.base_url}{WORKLOAD_LOGS_ENDPOINT.format(workload_uid=workload_uid.strip())}"
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_read=None,
        )
        params = {"follow": str(follow).lower()}
        headers = {"Accept": "text/plain"}

        try:
            async with self.session.get(
                url, params=params, headers=headers, timeout=timeout
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise APIError(response.status, error_text)

                async for line in response.content:
                    if line:
                        decoded_line = line.decode("utf-8", errors="replace").rstrip("\n\r")
                        if not decoded_line:
                            continue
                        yield decoded_line

        except aiohttp.ClientError as e:
            raise APIError(
                500, f"Failed to connect to logs endpoint: {str(e)}", cause=e
            )
