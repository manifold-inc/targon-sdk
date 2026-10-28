import os

import targon
from targon.client import PortConfig

client = targon.Client.from_env(org=os.environ["TARGON_ORG"])

with client:
    vm = client.vms.register(
        targon.VMCreateParams(
            name="example-vm",
            image=os.environ["TARGON_VM_IMAGE"],
            resource_name=targon.Resources.CPU_SMALL,
            password=os.environ["TARGON_VM_PASSWORD"],
            ports=(PortConfig(port=22),),
        )
    )
    try:
        print(f"Registered VM {vm.uid}")
        vm = vm.deploy(timeout=300)
        print(f"VM {vm.uid} is running")
    finally:
        client.workload.delete(vm.uid)

    metal = client.bare_metal.register(
        targon.BareMetalCreateParams(
            name="example-bare-metal",
            image=os.environ["TARGON_BM_IMAGE"],
            resource_name=os.environ["TARGON_BM_RESOURCE"],
            region=os.environ["TARGON_REGION"],
            ssh_keys=(os.environ["TARGON_SSH_KEY_UID"],),
        )
    )
    try:
        print(f"Registered Bare Metal {metal.uid}")
        metal = metal.deploy(timeout=1800)
        print(f"Bare Metal {metal.uid} is running")
    finally:
        client.workload.delete(metal.uid)
