import aiohttp
from typing import AsyncGenerator, Dict
from targon.core.exceptions import ValidationError, APIError
from targon.client.constants import LOGS_ENDPOINT, DEFAULT_BASE_URL
from targon.core.objects import AsyncBaseHTTPClient
import re


class AsyncLogsClient(AsyncBaseHTTPClient):
    """Async client for streaming function logs."""

    COLORS = [
        '\033[36m',  # Cyan
        '\033[35m',  # Magenta
        '\033[33m',  # Yellow
        '\033[32m',  # Green
        '\033[34m',  # Blue
        '\033[96m',  # Bright Cyan
        '\033[95m',  # Bright Magenta
        '\033[93m',  # Bright Yellow
    ]
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    def __init__(self, client):
        super().__init__(client)
        self.base_url = DEFAULT_BASE_URL
        self._pod_colors: Dict[str, str] = {}
        self._color_index = 0

    def _get_pod_color(self, pod_name: str) -> str:
        """Get or assign a color for a pod."""
        if pod_name not in self._pod_colors:
            self._pod_colors[pod_name] = self.COLORS[self._color_index % len(self.COLORS)]
            self._color_index += 1
        return self._pod_colors[pod_name]

    def _extract_pod_name(self, event_line: str) -> str:
        """Extract pod name from event line"""
        match = re.search(r'event:\s*log:\s*(.+)', event_line)
        if match:
            return match.group(1).strip()
        return "unknown"

    def _format_log_line(self, pod_name: str, log_data: str) -> str:
        """Format a log line with pod name and color coding."""
        color = self._get_pod_color(pod_name)
        
        # Shorten pod name for display (show last 12 chars which is typically the unique part)
        short_pod_name = pod_name[-12:] if len(pod_name) > 12 else pod_name
        
        # Format: [POD-ID] log message (only pod name is colored)
        formatted = f"{color}{self.BOLD}[{short_pod_name}]{self.RESET} {log_data}"
        return formatted

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

        current_pod = None
        
        try:
            async with self.session.post(url, json=payload, timeout=timeout) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise APIError(response.status, error_text)
                
                # Stream and parse SSE format
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8').rstrip('\n\r')
                        if not decoded_line:
                            continue
                        
                        if decoded_line.startswith('event:'):
                            current_pod = self._extract_pod_name(decoded_line)
                            
                        elif decoded_line.startswith('data:'):
                            log_data = decoded_line[5:].strip()
                            
                            if log_data and current_pod:
                                # Yield formatted log line
                                yield self._format_log_line(current_pod, log_data)
                        
                        elif not decoded_line.startswith('event:') and not decoded_line.startswith('data:'):
                            if current_pod:
                                yield self._format_log_line(current_pod, decoded_line)
                            else:
                                yield decoded_line
                            
        except aiohttp.ClientError as e:
            raise APIError(
                500,
                f"Failed to connect to logs endpoint: {str(e)}",
                cause=e
            )
