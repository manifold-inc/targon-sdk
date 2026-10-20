"""Open a PTY, send one command, read one frame, and close the session."""

from _box import open_client, sandbox

client = open_client()
try:
    with sandbox(client, "term") as box:
        session = box.terminals.create(cols=80, rows=24)
        print("terminal", session.id)
        try:
            with box.terminals.connect(session.id, open_timeout=10) as connection:
                connection.send(b"echo cookbook-terminal\n")
                frame = connection.recv(timeout=5)
                print(frame.decode("utf-8", errors="replace"))
        finally:
            box.terminals.delete(session.id)
finally:
    client.close()
