import targon.client
from targon import Resources
from targon.client import CreateWorkloadRequest, PortConfig
from targon.core.exceptions import TargonError, TimeoutError

client = targon.client.Client.from_env()

with client:
    # Define the workload.
    req = CreateWorkloadRequest(
        name="my-workload",
        image="nginx:latest",
        resource_name=Resources.CPU_SMALL,
        ports=[PortConfig(port=80)],
        envs={"ENV": "production"},
    )

    wl = client.workload.create(req)
    print(f"Created workload {wl.uid} ({wl.name})")

    deploy = client.workload.deploy(wl.uid)
    print(f"Deploying workload {deploy.uid}...")

    try:
        state = client.workload.wait_until_ready(wl.uid, timeout=300)
    except (TargonError, TimeoutError) as e:
        print(f"Workload did not become ready: {e}")
    else:
        print(
            f"Status: {state.status} "
            f"({state.ready_replicas}/{state.total_replicas} ready)"
        )
        for url in state.urls:
            print(f"  port {url.port} -> {url.url}")

        # Stream logs (Ctrl-C to stop).
        print("--- logs ---")
        print(client.workload.get_logs(wl.uid))