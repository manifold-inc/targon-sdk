import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  GatewayError,
  PayloadTooLargeError,
  Sandbox,
  SandboxTemplate,
  SandboxTemplateError,
  TargonClient,
  type SandboxData,
  type SandboxSummary,
  type SandboxTemplateData,
  type WebSocketLike,
} from "../src/index.js";

const sandbox: SandboxData = {
  uid: "wrk-1",
  type: "SANDBOX",
  name: "devbox",
  image: "sbt-base",
  sandbox_config: { template_uid: "sbt-base" },
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const summary: SandboxSummary = {
  uid: sandbox.uid,
  type: "SANDBOX",
  name: sandbox.name,
  image: sandbox.image,
  state: {
    status: "registered",
    message: "registered",
    ready_replicas: 0,
    total_replicas: 0,
  },
  created_at: sandbox.created_at,
  updated_at: sandbox.updated_at,
};

const templateData: SandboxTemplateData = {
  uid: "sbt-base",
  name: "base",
  kind: "FRESH",
  status: "READY",
  resource_name: "cpu",
  created_at: sandbox.created_at,
  updated_at: sandbox.updated_at,
};

function json(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

type FetchCall = Parameters<typeof globalThis.fetch>;
type MockFetch = ((...args: FetchCall) => Promise<Response>) & {
  calls: FetchCall[];
};

function mockFetch(...responses: Response[]): MockFetch {
  let responseIndex = 0;
  const calls: FetchCall[] = [];
  const fetch = (async (...args: FetchCall) => {
    calls.push(args);
    const response = responses[responseIndex++];
    assert.ok(response, `unexpected fetch call ${responseIndex}`);
    return response;
  }) as MockFetch;
  fetch.calls = calls;
  return fetch;
}

function assertSubset(
  actual: unknown,
  expected: Record<string, unknown>,
): void {
  assert.ok(actual !== null && typeof actual === "object");
  for (const [key, value] of Object.entries(expected)) {
    assert.deepEqual((actual as Record<string, unknown>)[key], value);
  }
}

describe("TargonClient", () => {
  it("preserves plain-text upstream error messages", async () => {
    const fetch = mockFetch(
      new Response("sandbox gateway unavailable", {
        status: 502,
        headers: { "Content-Type": "text/plain" },
      }),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "pat_secret",
      baseUrl: "https://example.test",
      fetch,
    });

    const error = await client.sandboxes.get("wrk-1").catch((caught) => caught);
    assert.ok(error instanceof GatewayError);
    assertSubset(error, {
      status: 502,
      message: "sandbox gateway unavailable",
      workload_uid: "wrk-1",
    });
  });

  it("creates, deploys, and waits for a sandbox with org auth", async () => {
    const fetch = mockFetch(
      json(templateData),
      json({ uid: sandbox.uid, type: "" }),
      json({ uid: sandbox.uid }),
        json({
          uid: "wrk-1",
          workload_type: "SANDBOX",
          status: "running",
          message: "",
          ready_replicas: 1,
          total_replicas: 1,
          updated_at: sandbox.updated_at,
        }),
      json(sandbox),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "pat_secret",
      baseUrl: "https://example.test",
      fetch,
    });
    const template = await client.sandboxTemplates.get("sbt-base");

    const result = await client.sandboxes.create({
      name: "devbox",
      template,
    });

    assert.equal(result.uid, "wrk-1");
    assert.ok(result instanceof Sandbox);
    assert.equal(typeof result.exec, "function");
    assert.equal(typeof result.files.read, "function");
    assert.equal(typeof result.terminals.connect, "function");
    assert.equal(fetch.calls.length, 5);
    const [url, init] = fetch.calls[1]!;
    assert.equal(url, "https://example.test/tha/v3/orgs/acme/workloads");
    assertSubset(init?.headers, { Authorization: "Bearer pat_secret" });
    assertSubset(JSON.parse(String(init?.body)), {
      type: "SANDBOX",
      image: "sbt-base",
    });
  });

  it("GETs the full workload after sparse no-wait create and fork operations", async () => {
    const childData: SandboxData = { ...sandbox, uid: "wrk-child", name: "child" };
    const fetch = mockFetch(
      json(templateData),
      json({ uid: sandbox.uid, type: "" }),
      json({ uid: sandbox.uid }),
      json(sandbox),
      json({ uid: childData.uid, type: "" }),
      json(childData),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      baseUrl: "https://example.test",
      fetch,
    });
    const template = await client.sandboxTemplates.get("sbt-base");

    const created = await client.sandboxes.create({
      name: "devbox",
      template,
      idle_timeout_sec: 0,
      wait_until_running: false,
    });
    const forked = await client.sandboxes.fork("wrk-1", {
      sandbox_config: { idle_timeout_sec: 0 },
      wait_until_running: false,
    });

    assert.ok(created instanceof Sandbox);
    assert.equal(created.sandbox_config?.template_uid, "sbt-base");
    assert.ok(forked instanceof Sandbox);
    assert.equal(forked.uid, "wrk-child");
    assert.equal(fetch.calls[3]![0],
      "https://example.test/tha/v3/orgs/acme/workloads/wrk-1",
    );
    assert.equal(fetch.calls[5]![0],
      "https://example.test/tha/v3/orgs/acme/workloads/wrk-child",
    );
    assertSubset(JSON.parse(String(fetch.calls[1]![1]?.body)), {
      sandbox_config: { idle_timeout_sec: 0 },
    });
    assertSubset(JSON.parse(String(fetch.calls[4]![1]?.body)), {
      sandbox_config: { idle_timeout_sec: 0 },
    });
  });

  it("accepts sparse empty-type freeze and thaw operation responses", async () => {
    const fetch = mockFetch(
      json({ uid: sandbox.uid, type: "" }),
        json({
          uid: sandbox.uid,
          workload_type: "SANDBOX",
          status: "frozen",
          message: "",
          ready_replicas: 0,
          total_replicas: 1,
          updated_at: sandbox.updated_at,
        }),
      json({ ...sandbox, state: { ...summary.state, status: "frozen" } }),
      json({ uid: sandbox.uid }),
        json({
          uid: sandbox.uid,
          workload_type: "SANDBOX",
          status: "running",
          message: "",
          ready_replicas: 1,
          total_replicas: 1,
          updated_at: sandbox.updated_at,
        }),
      json({ ...sandbox, state: { ...summary.state, status: "running" } }),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    assert.ok(await client.sandboxes.freeze(sandbox.uid) instanceof Sandbox);
    assert.ok(await client.sandboxes.thaw(sandbox.uid) instanceof Sandbox);
    assert.equal(fetch.calls.length, 6);
  });

  it("requires operation UIDs and rejects other non-empty operation types", async () => {
    const fetch = mockFetch(
      json({ type: "" }),
      json({ uid: sandbox.uid, type: "VM" }),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    await assert.rejects(client.sandboxes.freeze(sandbox.uid), /non-empty uid/);
    await assert.rejects(client.sandboxes.thaw(sandbox.uid), /not a sandbox/);
    assert.equal(fetch.calls.length, 2);
  });

  it("supports service and hydrated resource execution methods", async () => {
    const execResult = { stdout: "ok\n", stderr: "", code: 0, timed_out: false };
    const fetch = mockFetch(json(sandbox), json(execResult), json(execResult));
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      baseUrl: "https://example.test",
      fetch,
    });
    const resource = await client.sandboxes.get("wrk-1");

    assert.deepEqual(await client.sandboxes.exec("wrk-1", "echo service"), execResult);
    assert.deepEqual(await resource.exec("echo resource"), execResult);
    assert.equal(JSON.stringify(resource), JSON.stringify(sandbox));
    assert.equal("files" in JSON.parse(JSON.stringify(resource)), false);
    assert.deepEqual(fetch.calls.slice(1).map(([, init]) => JSON.parse(String(init?.body))), [
      { cmd: "echo service", timeout_sec: 60 },
      { cmd: "echo resource", timeout_sec: 60 },
    ]);
  });

  it("returns explicit sparse summaries from list without N+1 requests", async () => {
    const fetch = mockFetch(json({ items: [summary], next_cursor: "next" }));
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    const page = await client.sandboxes.list();

    assert.deepEqual(page.items[0], summary);
    assert.ok(!(page.items[0] instanceof Sandbox));
    assert.equal("sandbox_config" in page.items[0]!, false);
    assert.equal(page.next_cursor, "next");
    assert.equal(fetch.calls.length, 1);
  });

  it("hydrates templates and supports their bound resource methods", async () => {
    const updatedTemplate = { ...templateData, display_name: "Base template" };
    const fetch = mockFetch(
      json({ items: [templateData], next_cursor: null }),
      json(templateData),
      json(updatedTemplate),
      json(undefined, 204),
      json(summary),
      json({ ...summary, state: { ...summary.state, status: "provisioning" } }),
      json(sandbox),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      baseUrl: "https://example.test",
      fetch,
    });

    const page = await client.sandboxTemplates.list();
    const template = page.items[0]!;

    assert.ok(template instanceof SandboxTemplate);
    assert.ok(await template.refresh() instanceof SandboxTemplate);
    assertSubset(await template.update({ display_name: "Base template" }), {
      display_name: "Base template",
    });
    await template.delete();
    const created = await template.createSandbox({
      name: "devbox",
      wait_until_running: false,
    });

    assert.ok(created instanceof Sandbox);
    assertSubset(JSON.parse(String(fetch.calls[4]![1]?.body)), {
      type: "SANDBOX",
      image: "sbt-base",
    });
    assert.deepEqual(JSON.parse(JSON.stringify(template)), templateData);
  });

  it("requires a bound, identified, READY template when creating", async () => {
    const firstFetch = mockFetch(
      json(templateData),
      json({ ...templateData, uid: " " }),
      json({ ...templateData, status: "PENDING" }),
    );
    const first = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch: firstFetch,
    });
    const second = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch: mockFetch(json(templateData)),
    });
    const ready = await first.sandboxTemplates.get("sbt-base");
    const emptyUID = await first.sandboxTemplates.get("empty");
    const pending = await first.sandboxTemplates.get("pending");
    const otherReady = await second.sandboxTemplates.get("sbt-base");

    await assert.rejects(
      first.sandboxes.create({
        name: "devbox",
        template: otherReady,
      }),
      /from this TargonClient/,
    );
    await assert.rejects(
      first.sandboxes.create({
        name: "devbox",
        template: emptyUID,
      }),
      /uid must be non-empty/,
    );
    await assert.rejects(
      first.sandboxes.create({
        name: "devbox",
        template: pending,
      }),
      SandboxTemplateError,
    );
    await assert.rejects(
      first.sandboxes.create({
        name: "devbox",
        template: { ...templateData } as SandboxTemplate,
      }),
      /from this TargonClient/,
    );
    assert.throws(
      () => Reflect.construct(SandboxTemplate, [first.sandboxTemplates, templateData]),
      /must be obtained from a service/,
    );
    assert.equal(ready.status, "READY");
  });

  it("keeps list, full read, and state discriminators strict", async () => {
    const fetch = mockFetch(
      json({ items: [{ ...summary, type: "" }], next_cursor: null }),
      json({ ...sandbox, type: "" }),
        json({
          uid: "wrk-1",
          workload_type: "",
          status: "running",
          message: "",
          ready_replicas: 1,
          total_replicas: 1,
          updated_at: sandbox.updated_at,
        }),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    await assert.rejects(client.sandboxes.list(), /not a sandbox/);
    await assert.rejects(client.sandboxes.get("wrk-1"), /not a sandbox/);
    await assert.rejects(client.sandboxes.getState("wrk-1"), /not a sandbox/);
  });

  it("treats backend error and deleted states as terminal polling failures", async () => {
    const fetch = mockFetch(
        json({
          uid: "wrk-1",
          workload_type: "SANDBOX",
          status: "error",
          message: "provision failed",
          ready_replicas: 0,
          total_replicas: 1,
          updated_at: sandbox.updated_at,
        }),
        json({
          uid: "wrk-2",
          workload_type: "SANDBOX",
          status: "deleted",
          message: "deleted",
          ready_replicas: 0,
          total_replicas: 0,
          updated_at: sandbox.updated_at,
        }),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    await assert.rejects(
      client.sandboxes.waitForStatus("wrk-1", "running"),
      /terminal status error/,
    );
    await assert.rejects(
      client.sandboxes.waitForStatus("wrk-2", "running"),
      /terminal status deleted/,
    );
    assert.equal(fetch.calls.length, 2);
  });

  it("validates names, ports, timeouts, limits, and guest paths", async () => {
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch: mockFetch(),
    });
    await assert.rejects(
      client.sandboxes.create({
        name: "Bad Name",
        template: { ...templateData } as SandboxTemplate,
      }),
      /name must/,
    );
    assert.throws(() => client.sandboxes.list({ limit: 1001 }), /limit/);
    assert.throws(() => client.sandboxes.exec("wrk-1", "x", 601), /timeoutSec/);
    assert.throws(
      () => client.sandboxes.writeFile("wrk-1", "relative", new Uint8Array()),
      /absolute/,
    );
    await assert.rejects(client.sandboxes.update("wrk-1", {}), /at least one/);
    await assert.rejects(
      client.sandboxes.update("wrk-1", {
        sandbox_config: { idle_timeout_sec: 0 },
      }),
      /positive integer/,
    );
    await assert.rejects(
      client.sandboxes.update("wrk-1", {
        sandbox_config: { ttl_sec: 10, idle_timeout_sec: 10 },
      }),
      /less than ttl_sec/,
    );
    await assert.rejects(
      client.sandboxes.create({
        name: "devbox",
        template: { ...templateData } as SandboxTemplate,
        ports: [{ port: 3000, protocol: "SCTP" as "TCP" }],
      }),
      /TCP or UDP/,
    );
  });

  it("round-trips binary files without Buffer-specific API", async () => {
    const fetch = mockFetch(
      json({ path: "/tmp/data", content_b64: "AP+A" }),
      json(undefined, 204),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    assert.deepEqual(await client.sandboxes.readFile("wrk-1", "/tmp/data"),
      new Uint8Array([0, 255, 128]),
    );
    await client.sandboxes.writeFile(
      "wrk-1",
      "/tmp/data",
      new Uint8Array([0, 255, 128]),
    );
    assert.deepEqual(JSON.parse(String(fetch.calls[1]![1]?.body)), {
      path: "/tmp/data",
      content_b64: "AP+A",
    });
  });

  it("maps API failures to typed errors", async () => {
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch: mockFetch(
        json(
          {
            error: "too large",
            reason: "WORKLOAD_SANDBOX_PAYLOAD_TOO_LARGE",
          },
          413,
        ),
      ),
    });

    await assert.rejects(
      client.sandboxes.readFile("wrk-1", "/big"),
      PayloadTooLargeError,
    );
  });

  it("raises when publish polling reaches FAILED", async () => {
    const fetch = mockFetch(
        json({
          uid: "sbt-new",
          name: "new",
          kind: "USER",
          status: "PENDING",
          resource_name: "cpu",
          created_at: sandbox.created_at,
          updated_at: sandbox.updated_at,
        }),
        json({
          uid: "sbt-new",
          name: "new",
          kind: "USER",
          status: "FAILED",
          status_message: "snapshot failed",
          resource_name: "cpu",
          created_at: sandbox.created_at,
          updated_at: sandbox.updated_at,
        }),
    );
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    await assert.rejects(
      client.sandboxes.publish("wrk-1", { name: "new" }),
      SandboxTemplateError,
    );
  });

  it("returns a hydrated template from no-wait publish", async () => {
    const pending = { ...templateData, uid: "sbt-new", status: "PENDING" as const };
    const fetch = mockFetch(json(pending));
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      fetch,
    });

    const template = await client.sandboxes.publish("wrk-1", {
      name: "new",
      wait_until_ready: false,
    });

    assert.ok(template instanceof SandboxTemplate);
    assert.equal(typeof template.refresh, "function");
  });

  it("mints a fresh ticket immediately before opening a binary PTY", async () => {
    const fetch = mockFetch(
      json({ ticket: "one time&ticket", expires_at: "2026-01-01T00:01:00Z" }),
    );
    let openedURL = "";
    class FakeWebSocket implements WebSocketLike {
      binaryType = "";
      readyState = 0;
      constructor(url: string | URL) {
        openedURL = String(url);
      }
      send(): void {}
      close(): void {}
      addEventListener(): void {}
    }
    const client = new TargonClient({
      organization: "acme",
      apiKey: "key",
      baseUrl: "https://example.test",
      fetch,
    });

    await client.sandboxes.terminals.connect("wrk-1", "term.1", {
      WebSocket: FakeWebSocket,
    });

    assert.equal(fetch.calls.length, 1);
    assert.equal(openedURL,
      "wss://example.test/tha/v3/orgs/acme/workloads/wrk-1/terminals/term.1/ws?ticket=one+time%26ticket",
    );
  });
});
