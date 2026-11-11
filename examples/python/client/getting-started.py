"""Create a VM and a Bare Metal workload with the typed Create helpers.

Required: TARGON_API_KEY, TARGON_ORG, TARGON_SSH_KEY_UID.
Optional: TARGON_VM_IMAGE, TARGON_BM_IMAGE, TARGON_REGION, TARGON_BM_RESOURCE.
Images default to the first catalog entry. The Bare Metal region and resource
default to the first available ``bm`` inventory entry.
"""

import os

import targon


def pick_bare_metal_capacity(client: targon.Client):
    region = os.environ.get("TARGON_REGION") or None
    resource = os.environ.get("TARGON_BM_RESOURCE")
    if resource:
        if not region:
            raise SystemExit("set TARGON_REGION when TARGON_BM_RESOURCE is set")
        return resource, region
    for item in client.inventory.list(inventory_type="bm", region=region):
        if item.available > 0 and item.region is not None:
            return item.name, item.region.uid
    raise SystemExit("no Bare Metal inventory available")


def main() -> None:
    ssh_key = os.environ.get("TARGON_SSH_KEY_UID", "").strip()
    if not ssh_key:
        raise SystemExit("set TARGON_SSH_KEY_UID to an SSH key UID")
    client = targon.Client.from_env(org=os.environ["TARGON_ORG"])

    with client:
        vm_image = os.environ.get("TARGON_VM_IMAGE") or client.vms.images()[0].name
        # SSH is reachable on the VM's public IP and SSH port; port 22 cannot
        # be forwarded.
        vm = client.vms.create(
            targon.VMCreateParams(
                name="example-vm",
                image=vm_image,
                resource_name=targon.Resources.H100_SMALL,
                vm_config=targon.VMConfigInput(hostname="example-vm"),
                ssh_keys=(ssh_key,),
                ports=(targon.PortConfig(port=8080, protocol="TCP"),),
            ),
            timeout=600,
        )
        try:
            print(f"VM {vm.uid} is running")
        finally:
            client.workload.delete(vm.uid)

        resource, region = pick_bare_metal_capacity(client)
        bm_image = (
            os.environ.get("TARGON_BM_IMAGE") or client.bare_metal.images()[0].name
        )
        # Create waits up to DEFAULT_BARE_METAL_WAIT_TIMEOUT (30 minutes).
        metal = client.bare_metal.create(
            targon.BareMetalCreateParams(
                name="example-bare-metal",
                image=bm_image,
                resource_name=resource,
                region=region,
                ssh_keys=(ssh_key,),
                bm_config=targon.BMConfigInput(hostname="example-bare-metal"),
            )
        )
        try:
            print(f"Bare Metal {metal.uid} is running in {region}")
        finally:
            client.workload.delete(metal.uid)


if __name__ == "__main__":
    main()
