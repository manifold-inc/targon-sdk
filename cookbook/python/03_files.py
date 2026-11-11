"""Write a file in the guest, run it, and read the output file back."""

from _box import open_client, sandbox

client = open_client()
try:
    with sandbox(client, "files") as box:
        box.files.write("/tmp/hello.txt", "hello from the cookbook\n")
        result = box.exec(
            "cp /tmp/hello.txt /tmp/hello.out && wc -c /tmp/hello.out",
            timeout_sec=60,
        )
        if result.code != 0:
            raise SystemExit(result.stderr)
        text = box.files.read("/tmp/hello.out", as_text=True)
        raw = box.files.read("/tmp/hello.out")
        print(result.stdout)
        print("text:", text)
        print("bytes:", len(raw))
finally:
    client.close()
