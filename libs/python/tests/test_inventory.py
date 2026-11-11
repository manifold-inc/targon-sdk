import pytest
from conftest import Response

from targon.client import InventoryRegion
from targon.core.exceptions import ValidationError

# /tha/v3/inventory rows in the shape the backend returns for vm and bm.
INVENTORY_RESPONSE = [
    {
        "name": "h100-small",
        "display_name": "H100 Small",
        "description": "1x H100",
        "type": "vm",
        "gpu": True,
        "spec": {
            "gpu_vendor": "nvidia",
            "gpu_type": "H100",
            "gpu_count": 1,
            "cpu_millicores": 16000,
            "memory_mib": 131072,
            "cc_enabled": True,
            "vcpu": 16,
            "memory": 128,
        },
        "cost_per_hour": 2.5,
        "available": 3,
    },
    {
        "name": "bm-h100-8x",
        "display_name": "H100 8x Bare Metal",
        "description": "8x H100",
        "type": "bm",
        "region": {
            "uid": "us-east",
            "display_name": "US East",
            "country": "US",
            "continent": "NA",
        },
        "gpu": True,
        "spec": {"gpu_count": 8, "cpu_sockets": 2, "memory_mib": 2097152},
        "cost_per_hour": 20,
        "available": 1,
    },
]


def test_list_sends_current_filters_and_decodes_backend_shape(client):
    client.session.request.return_value = Response(body=INVENTORY_RESPONSE)

    items = client.inventory.list(inventory_type=" BM ", gpu=True, region=" US-East ")

    request = client.session.request.call_args
    assert request.args == ("GET", "https://api.test/tha/v3/inventory")
    assert request.kwargs["params"] == {
        "type": "bm",
        "gpu": "true",
        "region": "us-east",
    }
    vm, bm = items
    assert vm.spec.memory_mib == 131072
    assert vm.spec.cc_enabled is True
    assert vm.region is None
    assert bm.region == InventoryRegion("us-east", "US East", "US", "NA")
    assert bm.spec.cpu_sockets == 2


def test_capacity_accepts_vm_type(client):
    client.session.request.return_value = Response(body=[])

    assert client.inventory.capacity("vm") == []
    assert client.session.request.call_args.kwargs["params"] == {"type": "vm"}


@pytest.mark.parametrize(
    ("kwargs", "field", "text"),
    (
        ({"inventory_type": "rental"}, "inventory_type", "deprecated"),
        ({"inventory_type": "storage"}, "inventory_type", "deprecated"),
        ({"inventory_type": "serverless"}, "inventory_type", "deprecated"),
        ({"inventory_type": "sandbox"}, "inventory_type", "one of vm, bm"),
        ({"inventory_type": " "}, "inventory_type", "non-empty"),
        ({"region": "us_east"}, "region", "slug"),
        ({"region": "a" * 33}, "region", "slug"),
        ({"region": ""}, "region", "slug"),
    ),
)
def test_list_rejects_deprecated_and_invalid_filters(client, kwargs, field, text):
    with pytest.raises(ValidationError) as exc:
        client.inventory.list(**kwargs)

    assert exc.value.field == field
    assert text in str(exc.value)
    client.session.request.assert_not_called()
