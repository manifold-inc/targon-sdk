import time

import targon

client = targon.Client.from_env()
templates = client.sandboxes.templates.list(
    status=targon.SandboxTemplateStatus.READY,
)
if not templates.items:
    raise RuntimeError("no READY sandbox template is available")

print("Creating sandbox")
sandbox = templates.items[0].create_sandbox(
    name=f"python-example-{int(time.time()) % 1_000_000}",
    ttl_sec=900,
    idle_timeout_sec=300,
    timeout=600,
    poll_interval=2,
)
print(f"Sandbox created ({sandbox.uid})")

try:
    response = sandbox.exec("python --version", timeout_sec=60)
    if response.code != 0:
        print(f"Error: {response.code} {response.stderr}")
    else:
        print(response.stdout)
finally:
    print("Removing sandbox")
    sandbox.delete()
    client.close()

print("Sandbox removed")
