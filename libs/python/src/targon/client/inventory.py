import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from targon.client.constants import INVENTORY_ENDPOINT
from targon.core.exceptions import ValidationError
from targon.core.objects import BaseHTTPClient

VALID_INVENTORY_TYPES = frozenset({"vm", "bm"})
DEPRECATED_INVENTORY_TYPES = frozenset({"rental", "storage", "serverless"})

_REGION_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class InventorySpec:
    gpu_vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    cpu_millicores: int = 0
    cpu_vendor: Optional[str] = None
    cpu_model: Optional[str] = None
    cpu_sockets: Optional[int] = None
    memory_mib: int = 0
    disk_size_mib: int = 0
    disk_label: Optional[str] = None
    network_mode: Optional[str] = None
    cc_enabled: Optional[bool] = None
    vcpu: int = 0
    memory: int = 0
    storage: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            data = {}
        return cls(
            gpu_vendor=data.get("gpu_vendor"),
            gpu_model=data.get("gpu_model"),
            gpu_type=data.get("gpu_type"),
            gpu_count=int(data.get("gpu_count", 0)),
            cpu_millicores=int(data.get("cpu_millicores", 0)),
            cpu_vendor=data.get("cpu_vendor"),
            cpu_model=data.get("cpu_model"),
            cpu_sockets=data.get("cpu_sockets"),
            memory_mib=int(data.get("memory_mib", 0)),
            disk_size_mib=int(data.get("disk_size_mib", 0)),
            disk_label=data.get("disk_label"),
            network_mode=data.get("network_mode"),
            cc_enabled=data.get("cc_enabled"),
            vcpu=int(data.get("vcpu", 0)),
            memory=int(data.get("memory", 0)),
            storage=int(data.get("storage", 0)),
        )


@dataclass
class InventoryRegion:
    uid: str
    display_name: str = ""
    country: str = ""
    continent: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> Optional["InventoryRegion"]:
        if not isinstance(data, dict) or not data.get("uid"):
            return None
        return cls(
            uid=data["uid"],
            display_name=data.get("display_name", ""),
            country=data.get("country", ""),
            continent=data.get("continent", ""),
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
    region: Optional[InventoryRegion] = None

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
            region=InventoryRegion.from_dict(data.get("region")),
        )

    def __repr__(self):
        return f"{self.name} ({self.available} available)"


def _inventory_params(
    inventory_type: Optional[str],
    gpu: Optional[bool],
    region: Optional[str],
) -> Dict[str, Any]:
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
                "inventory_type must be one of vm, bm",
                field="inventory_type",
                value=inventory_type,
            )
        params["type"] = normalized_type
    if gpu is not None:
        params["gpu"] = str(gpu).lower()
    if region is not None:
        normalized_region = region.strip().lower() if isinstance(region, str) else ""
        if len(normalized_region) > 32 or not _REGION_RE.match(normalized_region):
            raise ValidationError(
                "region must be a lowercase slug no longer than 32 characters",
                field="region",
                value=region,
            )
        params["region"] = normalized_region
    return params


class InventoryClient(BaseHTTPClient):
    """Inventory client for resource queries."""

    def list(
        self,
        inventory_type: Optional[str] = None,
        gpu: Optional[bool] = None,
        region: Optional[str] = None,
    ) -> List[Inventory]:
        """Get inventory entries filtered by type (``vm``/``bm``), GPU, region."""
        params = _inventory_params(inventory_type, gpu, region)
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
        region: Optional[str] = None,
    ) -> List[Inventory]:
        """Compatibility alias for :meth:`list`."""
        return self.list(inventory_type=inventory_type, gpu=gpu, region=region)
