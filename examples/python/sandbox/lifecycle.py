import time

import targon

with targon.Client.from_env() as client:
    templates = client.sandboxes.templates.list(
        status=targon.SandboxTemplateStatus.READY,
    )
    if not templates.items:
        raise RuntimeError("no READY sandbox template is available")

    print("Registering sandbox")
    sandbox = client.sandboxes.register(
        targon.SandboxCreateParams(
            name=f"python-example-{int(time.time()) % 1_000_000}",
            template=templates.items[0],
            ttl_sec=900,
            idle_timeout_sec=300,
        )
    )
    print(f"Sandbox registered ({sandbox.uid})")

    try:
        sandbox = sandbox.deploy(timeout=600, poll_interval=2)
        print(f"Sandbox running ({sandbox.uid})")
        response = sandbox.exec("python --version", timeout_sec=60)
        if response.code != 0:
            print(f"Error: {response.code} {response.stderr}")
        else:
            print(response.stdout)
    finally:
        print("Removing sandbox")
        sandbox.delete()

print("Sandbox removed")
