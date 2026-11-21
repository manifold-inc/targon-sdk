# Targon sandboxes — agent instructions

You are operating Targon sandboxes through the official SDK (`targon-sdk` in Python, `github.com/manifold-inc/targon-sdk/libs/go/v4` in Go, `@targon/sdk` in TypeScript). A sandbox is a short-lived guest you create, use, and delete in the same task. A running sandbox keeps billing until you delete it, its TTL expires, or it idles out.

Prefer the smallest surface that finishes the job: one `exec` for a command, files plus `exec` for a code run, a terminal only when you need a live PTY.

## Credentials

Set these before any call:

```bash
export TARGON_API_KEY="..."
export TARGON_ORG="your-org-slug"
```

Python and Go also read the active CLI profile under `~/.targon` when those variables are unset. TypeScript always needs `TARGON_ORG` (or an `organization` argument) and reads `TARGON_API_KEY` when `apiKey` is omitted. One client is bound to one organization. Do not pass the org on each call.

## Default lifecycle

Do this unless the user asked you to keep the sandbox:

1. List templates with status `READY`.
2. Create from that template **object**. A raw template id is rejected.
3. `create` already deploys and waits until status is `running`. Leave that wait on.
4. Do the work (`exec`, files, and only then terminals).
5. Delete the sandbox in `finally` / `defer`, including after your own errors.

Set both timeouts on create so a crash cannot leave a sandbox up:

- `ttl_sec` is the hard lifetime. `0` disables it.
- `idle_timeout_sec` deletes the sandbox after it goes idle. `0` disables it on create.
- When both are positive, idle must be shorter than the TTL.

Use 15 minutes TTL and 5 minutes idle (`ttl_sec=900`, `idle_timeout_sec=300`) unless the task needs longer. Extend a live sandbox by patching `sandbox_config`. On update, `idle_timeout_sec` must be positive. `0` means "disabled" only at create time.

If you opt out of the wait (`wait_until_running=False` in Python, `Wait.NoWait` in Go, `wait_until_running: false` in TypeScript), the sandbox is not ready. Poll with `wait_for_status` / `WaitForStatus` until `running` before `exec` or files. Waiting fails on `error`, `suspended`, and `deleted`.

### Python

```python
import targon

with targon.Client.from_env() as client:
    templates = client.sandboxes.templates.list(
        status=targon.SandboxTemplateStatus.READY,
    )
    if not templates.items:
        raise RuntimeError("no READY sandbox template is available")

    sandbox = client.sandboxes.create(
        targon.SandboxCreateParams(
            name="agent-job-1",
            template=templates.items[0],
            ttl_sec=900,
            idle_timeout_sec=300,
        )
    )
    try:
        result = sandbox.exec("python --version", timeout_sec=60)
        if result.code != 0 or result.timed_out:
            raise RuntimeError(result.stderr or "command failed")
    finally:
        sandbox.delete()
```

`template.create_sandbox(name, ttl_sec=..., idle_timeout_sec=...)` is the same create, starting from a READY template resource.

### TypeScript

```typescript
import { TargonClient } from "@targon/sdk";

const client = new TargonClient({ organization: process.env.TARGON_ORG! });
const templates = await client.sandboxTemplates.list({ status: "READY" });
const template = templates.items[0];
if (!template) throw new Error("no READY sandbox template is available");

const sandbox = await client.sandboxes.create({
  name: "agent-job-1",
  template,
  ttl_sec: 900,
  idle_timeout_sec: 300,
});
try {
  const result = await sandbox.exec("python --version", 60);
  if (result.code !== 0 || result.timed_out) {
    throw new Error(result.stderr || "command failed");
  }
} finally {
  await sandbox.delete();
}
```

### Go

```go
client, err := targon.NewFromEnv()
// list READY templates, then:
ttl, idle := 900, 300
sandbox, err := client.Sandboxes.Create(ctx, targon.SandboxCreateParams{
    Name:     "agent-job-1",
    Template: template,
    SandboxConfig: &targon.SandboxConfigInput{
        TTLSec:         &ttl,
        IdleTimeoutSec: &idle,
    },
})
defer sandbox.Delete(ctx)
```

## Which call to use

| You need | Call | Skip |
| --- | --- | --- |
| One command, install, or test | `exec` | a terminal |
| Put a program or input in the guest and read a result file | `files.write`, then `exec`, then `files.read` | a terminal, and base64 (the SDK encodes it) |
| An interactive shell or a process that stays open | `terminals.create`, then `connect` | `exec` for that session |
| Pause for minutes or hours | `freeze`, later `thaw` until `running` | leaving it `running` |
| A second copy to try a change | `fork`, then delete parent and child separately | mutating the only sandbox you still need |
| Reuse this guest image later | `publish`, wait until the template is `READY` | keeping the source sandbox as the template |
| Resume work in a later process | persist `uid`, then `get` | `list` as if it returned a full sandbox |

`exec` returns stdout, stderr, `code`, and `timed_out`. A non-zero `code` is the guest failing, not an SDK exception. Check `code` and `timed_out` yourself. The command string is at most 64 KiB. `timeout_sec` is 1 to 600. `timed_out` means the guest command was killed at the deadline.

File paths are absolute (`/tmp/input.txt`). Python accepts `str` or `bytes` and `read(..., as_text=True)` returns `str`. Go and TypeScript use bytes. A single file is at most 256 MiB. Write the input, `exec` the command that reads it, then read the output. That is the usual code-runner pattern.

```python
sandbox.files.write("/tmp/hello.txt", "hello\n")
result = sandbox.exec("cp /tmp/hello.txt /tmp/hello.out", timeout_sec=60)
text = sandbox.files.read("/tmp/hello.out", as_text=True)
```

The same file calls exist on the service as `client.sandboxes.files.read/write(uid, ...)` when you only have a uid.

## Names and lookups

Sandbox names are 1–32 characters: lowercase letters, digits, and hyphens, and they cannot start or end with a hyphen. Template names are 1–64 characters of letters, digits, `.`, `_`, or `-`, and must start with a letter or digit.

`list` returns a page of summaries (`uid`, `name`, `state`, cost). It does not return a sandbox you can `exec`. Page with `limit` and `cursor` until `next_cursor` is empty (`limit` max 1000), then `get` the uid you want. Reconnect does not extend TTL or idle timeout.

## Freeze, fork, publish

`freeze` parks the sandbox and waits until `frozen`. `thaw` waits until `running`. `exec` and files require `running`, so thaw first.

`fork` copies the sandbox into a new workload and waits until the child is `running`. The child has its own uid and its own bill. Pass a new name and, when you need it, a new `sandbox_config`. Delete the child and the parent separately.

`publish` snapshots the sandbox into a template and waits until that template is `READY`. The template remains after the source sandbox is deleted. Delete a template you created only for the task.

## Terminals and desktop

Use a terminal only for a PTY. Create it with a fixed `cols` and `rows` (each at most 1000). There is no resize. Python `connect` is a context manager. Go `ConnectTerminal` opens the socket the same way. TypeScript does not bundle a WebSocket: pass the runtime `WebSocket` (Node 22+, Bun, Deno, or a browser). Each `connect` mints a new single-use access ticket immediately before the socket opens. Frames are raw bytes. Delete the terminal when the session ends, then delete the sandbox.

`get_desktop` / `GetDesktop` reports whether a desktop port is up and includes a WebSocket URL when it is. `mint_access_ticket` returns a ticket lasting 1–300 seconds. Treat the ticket as a secret. Do not print it, log it, or put it in a commit.

## Rules that keep this cheap and correct

- Delete every sandbox you create before you finish, unless the user told you to leave a uid for later.
- Delete forks and task-local published templates the same way.
- Do not create a sandbox to answer a question you can answer locally.
- Do not pass a container image string. Sandboxes start from a READY template resource from this client.
- Do not call `exec` or files while status is `frozen`, `provisioning`, or `registered`.
- Do not use `list` when you already have a uid.
- Do not open a terminal to run `python script.py`, an install, or a test.
- Do not expose port 22. Other ports are 1–65535, protocol `TCP` or `UDP`.
- SSH keys, when you attach them, are a sequence of key uid strings, not key material and not one bare string.
- If create fails because no template is `READY`, stop and say so. Do not invent a template id.
