"""Copy a sandbox. The child is a separate workload and must be deleted too."""

import targon

from _box import open_client, sandbox, sandbox_name

client = open_client()
try:
    with sandbox(client, "fork") as box:
        child = box.fork(
            targon.ForkRequest(name=sandbox_name("child")),
            timeout=600,
            poll_interval=2,
        )
        try:
            print("child", child.uid)
            result = child.exec("echo forked", timeout_sec=60)
            if result.code != 0:
                raise SystemExit(result.stderr)
            print(result.stdout, end="")
        finally:
            print("deleting child", child.uid)
            child.delete()
finally:
    client.close()
