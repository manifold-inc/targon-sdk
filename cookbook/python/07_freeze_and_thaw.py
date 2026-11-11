"""Freeze a running sandbox, then thaw it before running another command."""

from _box import open_client, sandbox

client = open_client()
try:
    with sandbox(client, "freeze") as box:
        frozen = box.freeze(timeout=300, poll_interval=2)
        print("frozen status:", frozen.get_state().status.value)

        running = frozen.thaw(timeout=300, poll_interval=2)
        result = running.exec("echo thawed", timeout_sec=60)
        if result.code != 0:
            raise SystemExit(result.stderr)
        print(result.stdout, end="")
        print("thawed status:", running.get_state().status.value)
finally:
    client.close()
