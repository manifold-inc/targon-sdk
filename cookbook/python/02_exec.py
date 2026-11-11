"""Run one command and read its exit code, stdout, and stderr."""

from _box import open_client, sandbox

client = open_client()
try:
    with sandbox(client, "exec") as box:
        result = box.exec("python --version", timeout_sec=60)
        print("code=%s timed_out=%s" % (result.code, result.timed_out))
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        if result.code != 0 or result.timed_out:
            raise SystemExit("command failed")
finally:
    client.close()
