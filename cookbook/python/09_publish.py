"""Publish a sandbox as a template, then delete that template and the sandbox."""

import targon

from _box import open_client, sandbox, sandbox_name

client = open_client()
try:
    with sandbox(client, "pub") as box:
        template = box.publish(
            targon.PublishRequest(
                name=sandbox_name("tpl"),
                display_name="Cookbook snapshot",
                description="Created by cookbook/python/09_publish.py",
            ),
            timeout=600,
            poll_interval=2,
        )
        try:
            print("template %s status=%s" % (template.uid, template.status.value))
        finally:
            print("deleting template", template.uid)
            template.delete()
finally:
    client.close()
