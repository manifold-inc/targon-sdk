import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from targon.client.constants import INVENTORY_ENDPOINT
from targon.core.exceptions import ValidationError
from targon.core.objects import BaseHTTPClient

VALID_INVENTORY_TYPES = frozenset({"rental", "storage", "vm"})
DEPRECATED_INVENTORY_TYPES = frozenset({"serverless"})


@dataclass
class InventorySpec:
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    vcpu: int = 0
    memory: int = 0
    storage: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            data = {}
        return cls(
            gpu_type=data.get("gpu_type"),
            gpu_count=int(data.get("gpu_count", 0)),
            vcpu=int(data.get("vcpu", 0)),
            memory=int(data.get("memory", 0)),
            storage=int(data.get("storage", 0)),
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


class InventoryClient(BaseHTTPClient):
    """Inventory client for resource queries."""

    def list(
        self,
        inventory_type: Optional[str] = None,
        gpu: Optional[bool] = None,
    ) -> List[Inventory]:
        """Get inventory entries, optionally filtered by type and GPU support."""
        params: Dict[str, Any] = {}
        if inventory_type is not None:
            if not isinstance(inventory_type, str) or not inventory_type.strip():
                raise ValidationError(
                    "inventory_type must be a non-empty string",
                    field="inventory_type",
                    value=inventory_type,
                )
            normalized_type = inventory_type.strip().lower()
            if normalized_type in DEPRECATED_INVENTORY_TYPES:
                raise ValidationError(
                    f"inventory type {normalized_type} has been deprecated",
                    field="inventory_type",
                    value=inventory_type,
                )
            if normalized_type not in VALID_INVENTORY_TYPES:
                raise ValidationError(
                    "inventory_type must be one of rental, storage, vm",
                    field="inventory_type",
                    value=inventory_type,
                )
            params["type"] = normalized_type
        if gpu is not None:
            params["gpu"] = str(gpu).lower()

        res = self._get(INVENTORY_ENDPOINT, params=params)
        if isinstance(res, str):
            res = json.loads(res)

        if not isinstance(res, list):
            raise TypeError(
                f"Expected inventory list response, got {type(res).__name__}"
            )

        data_list = cast(List[Dict[str, Any]], res)
        return [Inventory.from_dict(data) for data in data_list]

    def capacity(
        self,
        inventory_type: Optional[str] = None,
        gpu: Optional[bool] = None,
    ) -> List[Inventory]:
        """Compatibility alias for :meth:`list`."""
        return self.list(inventory_type=inventory_type, gpu=gpu)
