from dataclasses import dataclass
from typing import Any, Dict, List, cast
from targon.core.objects import AsyncBaseHTTPClient
from targon.client.constants import INVENTORY_ENDPOINT


@dataclass
class Capacity:
    name: str
    count: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(name=data.get("name", ""), count=data.get("count", 0))

    def __str__(self):
        return f"{self.name:<15} {self.count:>5}"

    def __repr__(self):
        return f"{self.name} ({self.count})"


class AsyncInventoryClient(AsyncBaseHTTPClient):
    """Async inventory client for capacity and resource queries."""

    async def capacity(self) -> List[Capacity]:
        """Get available compute capacity."""
        res = await self._async_get(INVENTORY_ENDPOINT)
        data_list = cast(List[Dict[str, Any]], res)
        return [Capacity.from_dict(data) for data in data_list]
