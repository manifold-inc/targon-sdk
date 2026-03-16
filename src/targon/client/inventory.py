import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from targon.core.objects import AsyncBaseHTTPClient
from targon.client.constants import INVENTORY_ENDPOINT


@dataclass
class InventorySpec:
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    vcpu: Optional[int] = None
    memory: Optional[int] = None
    storage: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            data = {}
        return cls(
            gpu_type=data.get("gpu_type"),
            gpu_count=data.get("gpu_count"),
            vcpu=data.get("vcpu"),
            memory=data.get("memory"),
            storage=data.get("storage"),
        )


@dataclass
class Inventory:
    name: str
    display_name: str
    description: str
    type: str
    gpu: bool
    spec: InventorySpec
    cost_per_hour: float
    available: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError(f"Expected inventory item dict, got {type(data).__name__}")

        raw_spec = data.get("spec", {})
        if not isinstance(raw_spec, dict):
            raw_spec = {}

        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description", ""),
            type=data.get("type", ""),
            gpu=bool(data.get("gpu", False)),
            spec=InventorySpec.from_dict(cast(Dict[str, Any], raw_spec)),
            cost_per_hour=float(data.get("cost_per_hour", 0)),
            available=int(data.get("available", 0)),
        )

    def __repr__(self):
        return f"{self.name} ({self.available} available)"


class AsyncInventoryClient(AsyncBaseHTTPClient):
    """Async inventory client for resource queries."""

    async def capacity(
        self,
        inventory_type: Optional[str] = "serverless",
        gpu: Optional[bool] = None,
    ) -> List[Inventory]:
        """Get inventory entries, optionally filtered by type and GPU support."""
        params: Dict[str, Any] = {}
        if inventory_type is not None:
            params["type"] = inventory_type
        if gpu is not None:
            params["gpu"] = str(gpu).lower()

        res = await self._async_get(INVENTORY_ENDPOINT, params=params)
        if isinstance(res, str):
            res = json.loads(res)

        if not isinstance(res, list):
            raise TypeError(f"Expected inventory list response, got {type(res).__name__}")

        data_list = cast(List[Dict[str, Any]], res)
        return [Inventory.from_dict(data) for data in data_list]
