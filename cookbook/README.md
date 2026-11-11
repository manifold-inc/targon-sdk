# Targon sandbox cookbook

Runnable recipes for creating a sandbox, doing work inside it, and deleting it.
Most applications should treat a sandbox as a short-lived process: create it,
use it, and tear it down in the same program.

These examples call the live API. A sandbox that is left running continues to
incur cost until it is deleted, it expires, or it idles out.

Languages with a sandbox SDK:

| Language | Package | Recipes |
| --- | --- | --- |
| Python | `targon-sdk` | [`cookbook/python`](python) |
| Go | `github.com/manifold-inc/targon-sdk/libs/go/v4` | [`cookbook/go`](go) |
| TypeScript | `@targon/sdk` | [`cookbook/typescript`](typescript) |

## Credentials

```bash
export TARGON_API_KEY="..."
export TARGON_ORG="your-org-slug"
```

Python and Go also read the active CLI profile under `~/.targon` when those
variables are unset. The TypeScript client always needs `TARGON_ORG` (or an
`organization` argument). It reads `TARGON_API_KEY` when `apiKey` is omitted.

You need one sandbox template in `READY` status. Every recipe loads that
template from the API and passes the template **object** into create. A raw
template id is rejected.

## Run a recipe

Python (from the repo root, with the SDK installed in the active environment):

```bash
python cookbook/python/01_create_and_delete.py
```

Go:

```bash
cd cookbook/go
go run ./create
```

TypeScript (build the local package once, then run a recipe):

```bash
npm run build --prefix libs/typescript
cd cookbook/typescript
npm install
npx tsx src/01-create-and-delete.ts
```

Recipe names match across languages. Later recipes share a small helper
(`_box.py`, `internal/box`, `src/box.ts`) that does the same create-and-delete
setup as recipe 01. Read recipe 01 before copying a later one.

## Lifecycle

```text
list READY templates
        |
        v
create SANDBOX workload from the template object
        |
        v
deploy, then poll until status is running
        |
        v
exec / files / terminals / freeze / fork / publish
        |
        v
delete the sandbox (and any fork or published template you created)
```

`create` waits until the sandbox is `running` unless you opt out
(`wait_until_running=False` in Python, `Wait.NoWait` in Go,
`wait_until_running: false` in TypeScript). Opting out returns a sandbox that
is not ready for exec yet. Poll with `wait_for_status` / `WaitForStatus`
before running commands.

Terminal failure statuses while waiting are `error`, `suspended`, and
`deleted`. The call fails instead of waiting out the timeout.

## Actions

### 01 — Create and delete

Create from a ready template, print the workload uid, and delete in a
`finally` / `defer` so a later error still tears the sandbox down.

Set a lifetime when you create it:

- `ttl_sec` is the hard lifetime. `0` disables it.
- `idle_timeout_sec` deletes the sandbox after it goes idle. `0` disables it
  on create.
- When both are positive, idle must be shorter than the TTL.

The recipes use 15 minutes TTL and 5 minutes idle so a crashed example does
not leave a sandbox up all day.

### 02 — Exec

`exec` runs one shell command and returns stdout, stderr, an exit code, and
whether the command timed out. It is the right call for `python script.py`,
package installs, and tests.

The command string is limited to 64 KiB. `timeout_sec` must be from 1 to 600.
A non-zero `code` is an application failure, not an SDK exception. `timed_out`
means the guest command was killed at the deadline.

### 03 — Files

Read and write a single absolute guest path, for example `/tmp/input.txt`.
The wire format is base64; the SDKs encode and decode that for you.

- Python accepts `str` or `bytes`, and `read(..., as_text=True)` returns `str`.
- Go and TypeScript use bytes (`[]byte`, `Uint8Array`).

Write the input file, `exec` a command that reads it, then read the output
file back. This is the usual pattern for a code runner.

### 04 — Reconnect by uid

Creation and later work do not have to be the same process. Persist
`sandbox.uid`, then in the next process call `get` / `Get` with that uid.
`list` is the wrong call for this: it returns a summary page without the
action methods. Hydrate a chosen summary with `get`.

Delete when the job is finished. Reconnect does not keep the sandbox alive by
itself; TTL and idle timeout still apply.

### 05 — List and clean up

Page through sandboxes with `limit` and `cursor` until `next_cursor` is empty.
This recipe deletes every sandbox whose name starts with `cb-`, which is the
prefix used by the other recipes. Do not point it at an org where real
workloads use that prefix.

### 06 — Change timeouts

Patch `sandbox_config` on a running sandbox to extend or shorten TTL and idle
timeout. On update, `idle_timeout_sec` must be positive when you send it.
`0` is only valid at create time, where it means "disabled".

### 07 — Freeze and thaw

`freeze` parks a sandbox and waits until status is `frozen`. `thaw` brings it
back and waits until `running`. Use this when a workflow pauses for minutes
or hours and you do not want a running sandbox the whole time. Exec and file
calls need a running sandbox, so thaw before them.

### 08 — Fork

`fork` copies a sandbox into a new workload and waits until the child is
`running`. The child has its own uid and its own bill. Delete the child and
the parent separately. Pass a new name and, when you need it, a new
`sandbox_config`.

### 09 — Publish a template

`publish` snapshots the sandbox into a template and waits until that template
is `READY` (or fails if publishing fails). The template survives after the
source sandbox is deleted. This recipe deletes the template it just created.
Template names are 1–64 characters of letters, digits, `.`, `_`, or `-`, and
must start with a letter or digit. Sandbox names are stricter: 1–32 lowercase
letters, digits, or hyphens, and they cannot start or end with a hyphen.

### 10 — Terminals

`exec` is for one command. A terminal is a PTY session: create it with a
fixed size, connect a WebSocket, and exchange raw binary frames. There is no
resize call; the size is fixed at creation.

Python's `connect` is a context manager and mints a one-time access ticket
unless you pass one. Go's `ConnectTerminal` does the same. TypeScript does
not bundle a WebSocket implementation: pass the runtime's `WebSocket`
(Node 22+, Bun, Deno, or a browser). Every `connect` mints a new single-use
ticket immediately before opening the socket. Delete the terminal when the
session is over, then delete the sandbox.

### 11 — Desktop and access tickets

`get_desktop` / `GetDesktop` reports whether a desktop port is available and
includes a WebSocket URL when it is. `mint_access_ticket` returns a
short-lived ticket (1–300 seconds) for that kind of direct connection. Treat
the ticket as a secret. This recipe prints only its expiry.

## What create does not do

- It does not accept a container image string. Sandboxes start from a
  template resource.
- It does not share organization context per call. Set the org on the client
  (`Client.from_env`, `NewFromEnv`, or `new TargonClient({ organization })`).
- Listing sandboxes does not return a full `Sandbox`. Call `get` with the uid.
