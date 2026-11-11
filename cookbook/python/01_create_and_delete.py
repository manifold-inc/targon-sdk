"""Create a sandbox, confirm it is running, and always delete it."""

import targon

from _box import open_client, ready_template, sandbox_name

client = open_client()
try:
    template = ready_template(client)
    print("using template %s (%s)" % (template.uid, template.name))

    sandbox = template.create_sandbox(
        sandbox_name("create"),
        ttl_sec=900,
        idle_timeout_sec=300,
        timeout=600,
        poll_interval=2,
    )
    try:
        state = sandbox.get_state()
        print("created %s status=%s" % (sandbox.uid, state.status.value))
    finally:
        print("deleting %s" % sandbox.uid)
        sandbox.delete()
finally:
    client.close()

print("sandbox removed")
