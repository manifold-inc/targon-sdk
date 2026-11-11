"""Store a sandbox uid, then attach to that same sandbox and keep working."""

from _box import open_client, sandbox

client = open_client()
try:
    with sandbox(client, "reconnect") as box:
        uid = box.uid
        print("persist this uid:", uid)

        # A later process constructs its own client and calls get(uid).
        # list() only returns summaries; get() returns a sandbox you can exec.
        again = client.sandboxes.get(uid)
        result = again.exec("hostname", timeout_sec=30)
        if result.code != 0:
            raise SystemExit(result.stderr)
        print(result.stdout)
finally:
    client.close()
