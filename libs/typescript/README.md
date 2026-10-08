# Targon TypeScript Sandbox SDK

Zero-runtime-dependency, Fetch API based sandbox client for Node.js 18+, Bun,
Deno, and browsers.

## Installation

```bash
npm install @targon/sdk@4.0.0-rc.1
```

## Quickstart

```ts
import { TargonClient } from "@targon/sdk";

const client = new TargonClient({
  organization: "my-org",
  apiKey: "pat_...", // Node may instead use TARGON_API_KEY
});

const templates = await client.sandboxTemplates.list({ status: "READY" });
const template = templates.items[0];
if (!template) throw new Error("No ready sandbox template is available");

const sandbox = await client.sandboxes.create({
  name: "agent-workspace",
  template,
});

const result = await client.sandboxes.exec(sandbox.uid, "python --version");
// The returned Sandbox is hydrated; service methods remain available too.
const sameResult = await sandbox.exec("python --version");
await client.sandboxes.writeFile(
  sandbox.uid,
  "/tmp/input.bin",
  new Uint8Array([0, 1, 2]),
);
await sandbox.files.write("/tmp/other.bin", new Uint8Array([3, 4, 5]));
```

The client also supports listing/updating/deleting sandboxes, SSH keys,
freeze/thaw, fork, publish and template polling, binary files, access tickets,
terminal CRUD, desktop discovery, and typed API errors.

Sandbox templates are service-bound resources. Template list/get/update and
sandbox publish operations return `SandboxTemplate` instances with `refresh()`,
`update()`, `delete()`, and `createSandbox()` methods:

```ts
const template = await client.sandboxTemplates.get("sbt-python");
const sandbox = await template.createSandbox({ name: "agent-workspace" });
const refreshed = await template.refresh();
```

Sandbox creation accepts a `SandboxTemplate` returned by the same client. The
template must have a nonempty UID and `READY` status; raw template UIDs are not
accepted.

`create`, `get`, `fork`, and lifecycle waits return a hydrated `Sandbox`.
`client.sandboxes.list()` intentionally returns `Page<SandboxSummary>` because
the backend list endpoint returns sparse operation records. Fetch a selected
summary with `client.sandboxes.get(summary.uid)` to hydrate it; listing never
performs hidden N+1 requests.

Sandbox ports support `TCP` and `UDP` only. On update,
`sandbox_config.idle_timeout_sec` must be positive when supplied; create and
fork continue to accept `0` as the disabled value.

## Terminals

The package does not depend on a WebSocket implementation. Supply the browser
global, Bun's implementation, or a compatible Node implementation. Every
`connect` call mints a new single-use access ticket immediately before opening
the socket. PTY messages are raw binary bytes.

```ts
const terminal = await client.sandboxes.terminals.create(sandbox.uid, {
  cols: 100,
  rows: 30,
});
const connection = await client.sandboxes.terminals.connect(
  sandbox.uid,
  terminal.id,
  {
    WebSocket,
    onData: (bytes) => console.log(new TextDecoder().decode(bytes)),
  },
);
await connection.ready;
connection.write("echo ready\n");
```

Terminal dimensions are fixed at creation. The backend has no live resize
protocol.

## Development

```bash
npm install
npm run build
npm test
```

## License

Apache 2.0 — see [LICENSE](../../LICENSE) for details.
