"""Extend the TTL and idle timeout on a sandbox that is already running."""

import targon

from _box import open_client, sandbox

client = open_client()
try:
    with sandbox(client, "ttl") as box:
        updated = box.update(
            targon.SandboxUpdateParams(
                sandbox_config=targon.SandboxConfigInput(
                    ttl_sec=1800,
                    idle_timeout_sec=600,
                )
            )
        )
        config = updated.sandbox_config
        print(
            "ttl_sec=%s idle_timeout_sec=%s"
            % (config.ttl_sec, config.idle_timeout_sec)
        )
finally:
    client.close()
