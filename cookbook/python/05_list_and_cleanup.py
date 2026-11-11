"""Delete leftover cookbook sandboxes whose names start with cb-."""

from _box import open_client

PREFIX = "cb-"

client = open_client()
try:
    cursor = None
    deleted = 0
    while True:
        page = client.sandboxes.list(limit=100, cursor=cursor)
        for item in page.items:
            if not item.name.startswith(PREFIX):
                continue
            print("deleting %s (%s)" % (item.uid, item.name))
            client.sandboxes.delete(item.uid)
            deleted += 1
        if not page.next_cursor:
            break
        cursor = page.next_cursor
    print("deleted %d sandbox(es)" % deleted)
finally:
    client.close()
