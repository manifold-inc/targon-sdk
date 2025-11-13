import aiohttp
from typing import AsyncGenerator
from targon.core.exceptions import ValidationError, APIError
from targon.client.constants import LOGS_ENDPOINT, DEFAULT_BASE_URL
from targon.core.objects import AsyncBaseHTTPClient


class AsyncLogsClient(AsyncBaseHTTPClient):
    """Async client for streaming function logs."""

    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL

    async def stream_logs(
        self, serverless_uid: str, follow: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream logs from a serverless function."""

        if not serverless_uid or not serverless_uid.strip():
            raise ValidationError(
                "serverless_uid is required",
                field="serverless_uid",
                value=serverless_uid
            )
        
        payload = {
            "serverless_uid": serverless_uid.strip(),
            "follow": follow
        }
        
        url = f"{self.base_url}{LOGS_ENDPOINT}"
        timeout = aiohttp.ClientTimeout(
                total=None,  # No total timeout for streaming
                connect=10,  # 10 seconds to establish connection
                sock_read=None  # No timeout on reading (for long-running streams)
            )

        try:
            async with self.session.post(url, json=payload,timeout=timeout) as response:
                # Check for HTTP errors before streaming
                if response.status >= 400:
                    error_text = await response.text()
                    raise APIError(response.status, error_text)
                
                # Stream log lines
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8').rstrip('\n\r')
                        if decoded_line:  
                            yield decoded_line
                            
        except aiohttp.ClientError as e:
            raise APIError(
                500,
                f"Failed to connect to logs endpoint: {str(e)}",
                cause=e
            )
