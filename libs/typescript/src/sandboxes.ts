import { decodeBase64, encodeBase64 } from "./base64.js";
import type { TargonClient } from "./client.js";
import { Sandbox } from "./sandbox.js";
import {
  hydrateSandboxTemplate,
  type SandboxTemplate,
  type SandboxTemplatesService,
  templateUIDForService,
} from "./templates.js";
import type {
  AccessTicket,
  DesktopInfo,
  ExecResult,
  ForkRequest,
  Page,
  PublishRequest,
  SandboxData,
  SandboxCreateParams,
  SandboxListParams,
  SandboxStatus,
  SandboxSummary,
  SandboxTemplateData,
  SandboxUpdateParams,
  TerminalConnectOptions,
  TerminalConnection,
  TerminalCreateOptions,
  TerminalSession,
  WaitOptions,
  WebSocketLike,
  WorkloadStateResponse,
} from "./types.js";
import {
  assertLimit,
  assertName,
  assertPorts,
  assertSandboxConfig,
  assertSandboxConfigUpdate,
  assertTemplateName,
  assertTerminalDimension,
  MAX_COMMAND_BYTES,
  MAX_FILE_BYTES,
  sleep,
} from "./validation.js";

interface FileResponse {
  path: string;
  content_b64: string;
}

interface SandboxOperationResponse {
  uid: string;
  type?: "" | "SANDBOX";
}

export class SandboxesService {
  readonly terminals: SandboxTerminalsService;
  readonly files: SandboxFilesService;

  constructor(
    private readonly client: TargonClient,
    readonly templates: SandboxTemplatesService,
  ) {
    this.terminals = new SandboxTerminalsService(client, this);
    this.files = new SandboxFilesService(this);
  }

  async create(params: SandboxCreateParams): Promise<Sandbox> {
    assertName(params.name);
    assertPorts(params.ports);
    const templateUID = templateUIDForService(params.template, this.templates);
    const sandboxConfig = {
      ttl_sec: params.ttl_sec,
      idle_timeout_sec: params.idle_timeout_sec,
    };
    assertSandboxConfig(sandboxConfig);
    const created = this.assertOperationResponse(
      await this.client.request<unknown>(
        "POST",
        this.client.orgPath("/workloads"),
        {
          body: {
            type: "SANDBOX",
            name: params.name,
            image: templateUID,
            project_id: params.project_id,
            ssh_keys: params.ssh_keys,
            ports: params.ports,
            sandbox_config: sandboxConfig,
          },
        },
      ),
    );
    this.assertOperationResponse(
      await this.client.request<unknown>(
        "POST",
        this.workloadPath(created.uid, "/deploy"),
        { workloadUID: created.uid },
      ),
    );
    return params.wait_until_running === false
      ? this.get(created.uid)
      : this.waitForStatus(created.uid, "running", params.wait);
  }

  async get(uid: string): Promise<Sandbox> {
    const sandbox = await this.client.request<SandboxData>(
      "GET",
      this.workloadPath(uid),
      { workloadUID: uid },
    );
    if (sandbox.type !== "SANDBOX") {
      throw new TypeError(`Workload ${uid} is not a sandbox`);
    }
    return this.hydrate(sandbox);
  }

  list(params: SandboxListParams = {}): Promise<Page<SandboxSummary>> {
    assertLimit(params.limit);
    return this.client
      .request<Page<SandboxSummary>>("GET", this.client.orgPath("/workloads"), {
        query: { type: "SANDBOX", ...params },
      })
      .then((page) => ({
        ...page,
        items: page.items.map((item) => this.assertSummary(item)),
      }));
  }

  async getState(uid: string): Promise<WorkloadStateResponse> {
    const state = await this.client.request<WorkloadStateResponse>(
      "GET",
      this.workloadPath(uid, "/state"),
      {
        workloadUID: uid,
      },
    );
    if (state.workload_type !== "SANDBOX") {
      throw new TypeError(`Workload ${uid} is not a sandbox`);
    }
    assertSandboxStatus(state.status);
    return state;
  }

  async update(uid: string, update: SandboxUpdateParams): Promise<Sandbox> {
    const raw = update as SandboxUpdateParams & {
      image?: unknown;
      resource_name?: unknown;
    };
    if (raw.image !== undefined || raw.resource_name !== undefined) {
      throw new TypeError("image and resource_name cannot be updated on a sandbox");
    }
    if (
      update.name === undefined &&
      update.project_id === undefined &&
      update.ssh_keys === undefined &&
      update.ports === undefined &&
      update.sandbox_config === undefined
    ) {
      throw new TypeError("at least one sandbox update field is required");
    }
    if (update.name !== undefined) assertName(update.name);
    assertPorts(update.ports);
    assertSandboxConfigUpdate(update.sandbox_config);
    const data = await this.client.request<SandboxData>(
      "PATCH",
      this.workloadPath(uid),
      {
        body: update,
        workloadUID: uid,
      },
    );
    return this.hydrate(data);
  }

  async attachSshKey(uid: string, keyUID: string): Promise<void> {
    await this.client.request(
      "PUT",
      this.workloadPath(uid, `/ssh-keys/${encodeURIComponent(keyUID)}`),
      { workloadUID: uid },
    );
  }

  detachSshKey(uid: string, keyUID: string): Promise<void> {
    return this.client.request(
      "DELETE",
      this.workloadPath(uid, `/ssh-keys/${encodeURIComponent(keyUID)}`),
      { workloadUID: uid },
    );
  }

  async freeze(uid: string, wait: WaitOptions = {}): Promise<Sandbox> {
    this.assertOperationResponse(
      await this.client.request<unknown>(
        "POST",
        this.workloadPath(uid, "/freeze"),
        { workloadUID: uid },
      ),
    );
    return this.waitForStatus(uid, "frozen", wait);
  }

  async thaw(uid: string, wait: WaitOptions = {}): Promise<Sandbox> {
    this.assertOperationResponse(
      await this.client.request<unknown>(
        "POST",
        this.workloadPath(uid, "/thaw"),
        { workloadUID: uid },
      ),
    );
    return this.waitForStatus(uid, "running", wait);
  }

  delete(uid: string): Promise<void> {
    return this.client.request("DELETE", this.workloadPath(uid), {
      workloadUID: uid,
    });
  }

  async fork(uid: string, request: ForkRequest = {}): Promise<Sandbox> {
    if (request.name !== undefined && request.name !== "") assertName(request.name);
    assertSandboxConfig(request.sandbox_config);
    const child = this.assertOperationResponse(
      await this.client.request<unknown>(
        "POST",
        this.workloadPath(uid, "/fork"),
        {
          body: {
            name: request.name,
            project_id: request.project_id,
            sandbox_config: request.sandbox_config,
          },
          workloadUID: uid,
        },
      ),
    );
    return request.wait_until_running === false
      ? this.get(child.uid)
      : this.waitForStatus(child.uid, "running", request.wait);
  }

  async publish(uid: string, request: PublishRequest): Promise<SandboxTemplate> {
    assertTemplateName(request.name);
    if (request.display_name !== undefined && request.display_name.trim().length > 128) {
      throw new RangeError("display_name must be at most 128 characters");
    }
    const template = await this.client.request<SandboxTemplateData>(
      "POST",
      this.workloadPath(uid, "/publish"),
      {
        body: {
          name: request.name,
          display_name: request.display_name,
          description: request.description,
        },
        workloadUID: uid,
      },
    );
    return request.wait_until_ready === false
      ? hydrateSandboxTemplate(template, this.templates)
      : this.templates.waitForStatus(template.uid, ["READY", "FAILED"], request.wait);
  }

  async waitForStatus(
    uid: string,
    statuses: SandboxStatus | readonly SandboxStatus[],
    options: WaitOptions = {},
  ): Promise<Sandbox> {
    const accepted = new Set(Array.isArray(statuses) ? statuses : [statuses]);
    const timeout = options.timeout_ms ?? 10 * 60_000;
    const interval = options.interval_ms ?? 1_000;
    const deadline = Date.now() + timeout;
    for (;;) {
      if (options.signal?.aborted) throw options.signal.reason;
      const state = await this.getState(uid);
      if (accepted.has(state.status)) return this.get(uid);
      if (TERMINAL_FAILURE_STATUSES.has(state.status)) {
        throw new Error(`Sandbox ${uid} reached terminal status ${state.status}: ${state.message}`);
      }
      if (Date.now() >= deadline) {
        throw new Error(`Timed out waiting for sandbox ${uid}`);
      }
      await sleep(interval, options.signal);
    }
  }

  exec(uid: string, cmd: string, timeoutSec = 60): Promise<ExecResult> {
    const commandBytes = new TextEncoder().encode(cmd).byteLength;
    if (!cmd.trim() || commandBytes > MAX_COMMAND_BYTES) {
      throw new RangeError("cmd must be non-empty and at most 64 KiB");
    }
    if (!Number.isInteger(timeoutSec) || timeoutSec < 1 || timeoutSec > 600) {
      throw new RangeError("timeoutSec must be an integer from 1 through 600");
    }
    return this.client.request("POST", this.workloadPath(uid, "/exec"), {
      body: { cmd, timeout_sec: timeoutSec },
      workloadUID: uid,
    });
  }

  async readFile(uid: string, path: string): Promise<Uint8Array> {
    assertGuestPath(path);
    const file = await this.client.request<FileResponse>(
      "GET",
      this.workloadPath(uid, "/files"),
      { query: { path }, workloadUID: uid },
    );
    return decodeBase64(file.content_b64);
  }

  writeFile(uid: string, path: string, data: Uint8Array): Promise<void> {
    assertGuestPath(path);
    if (!(data instanceof Uint8Array)) throw new TypeError("data must be a Uint8Array");
    if (data.byteLength > MAX_FILE_BYTES) {
      throw new RangeError("file content must not exceed 256 MiB");
    }
    return this.client.request("PUT", this.workloadPath(uid, "/files"), {
      body: { path, content_b64: encodeBase64(data) },
      workloadUID: uid,
    });
  }

  mintAccessTicket(uid: string, ttlSec = 60): Promise<AccessTicket> {
    if (!Number.isInteger(ttlSec) || ttlSec < 1 || ttlSec > 300) {
      throw new RangeError("ttlSec must be an integer from 1 through 300");
    }
    return this.client.request("POST", this.workloadPath(uid, "/access-tickets"), {
      body: { ttl_sec: ttlSec },
      workloadUID: uid,
    });
  }

  getDesktop(uid: string): Promise<DesktopInfo> {
    return this.client.request("GET", this.workloadPath(uid, "/desktop"), {
      workloadUID: uid,
    });
  }

  private workloadPath(uid: string, suffix = ""): string {
    return this.client.orgPath(`/workloads/${encodeURIComponent(uid)}${suffix}`);
  }

  private hydrate(data: SandboxData): Sandbox {
    if (data.type !== "SANDBOX") {
      throw new TypeError(`Workload ${data.uid} is not a sandbox`);
    }
    if (data.state) assertSandboxStatus(data.state.status);
    return new Sandbox(this, data);
  }

  private assertOperationResponse(response: unknown): SandboxOperationResponse {
    if (typeof response !== "object" || response === null) {
      throw new TypeError("Sandbox operation response must include a non-empty uid");
    }
    const { uid, type } = response as { uid?: unknown; type?: unknown };
    if (typeof uid !== "string" || uid.trim() === "") {
      throw new TypeError("Sandbox operation response must include a non-empty uid");
    }
    if (type !== undefined && type !== "" && type !== "SANDBOX") {
      throw new TypeError(`Workload ${uid} is not a sandbox`);
    }
    return response as SandboxOperationResponse;
  }

  private assertSummary(summary: SandboxSummary): SandboxSummary {
    if (summary.type !== "SANDBOX") {
      throw new TypeError(`Workload ${summary.uid} is not a sandbox`);
    }
    if (summary.state) assertSandboxStatus(summary.state.status);
    return summary;
  }
}

export class SandboxFilesService {
  constructor(private readonly sandboxes: SandboxesService) {}

  read(uid: string, path: string): Promise<Uint8Array> {
    return this.sandboxes.readFile(uid, path);
  }

  write(uid: string, path: string, data: Uint8Array): Promise<void> {
    return this.sandboxes.writeFile(uid, path, data);
  }
}

export class SandboxTerminalsService {
  constructor(
    private readonly client: TargonClient,
    private readonly sandboxes: SandboxesService,
  ) {}

  list(uid: string): Promise<TerminalSession[]> {
    return this.client.request("GET", this.path(uid), { workloadUID: uid });
  }

  create(uid: string, options: TerminalCreateOptions = {}): Promise<TerminalSession> {
    const cols = options.cols ?? 80;
    const rows = options.rows ?? 24;
    assertTerminalDimension(cols, "cols");
    assertTerminalDimension(rows, "rows");
    return this.client.request("POST", this.path(uid), {
      body: { cols, rows },
      workloadUID: uid,
    });
  }

  delete(uid: string, terminalID: string): Promise<void> {
    assertTerminalID(terminalID);
    return this.client.request(
      "DELETE",
      `${this.path(uid)}/${encodeURIComponent(terminalID)}`,
      { workloadUID: uid },
    );
  }

  async connect(
    uid: string,
    terminalID: string,
    options: TerminalConnectOptions,
  ): Promise<TerminalConnection> {
    assertTerminalID(terminalID);
    if (!options.WebSocket) throw new TypeError("A WebSocket constructor is required");

    // Tickets are single-use, so mint one immediately before every connection.
    const ticket = await this.sandboxes.mintAccessTicket(
      uid,
      options.ticket_ttl_sec ?? 60,
    );
    const url = new URL(
      this.client.websocketUrl(
        `${this.path(uid)}/${encodeURIComponent(terminalID)}/ws`,
      ),
    );
    url.searchParams.set("ticket", ticket.ticket);
    const socket = new options.WebSocket(url.toString());
    socket.binaryType = "arraybuffer";

    let opened = false;
    const ready = new Promise<void>((resolve, reject) => {
      socket.addEventListener("open", () => {
        opened = true;
        resolve();
      });
      socket.addEventListener("error", (event) => {
        options.onError?.(event);
        if (!opened) reject(new Error("Terminal WebSocket connection failed"));
      });
      socket.addEventListener("close", (event) => {
        options.onExit?.({
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
        });
        if (!opened) reject(new Error("Terminal WebSocket closed before opening"));
      });
    });

    socket.addEventListener("message", (event) => {
      void toBytes(event.data).then(options.onData, options.onError);
    });

    return {
      socket,
      ready,
      write(data: Uint8Array | string): void {
        const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
        socket.send(bytes);
      },
      close(code?: number, reason?: string): void {
        socket.close(code, reason);
      },
    };
  }

  private path(uid: string): string {
    return this.client.orgPath(`/workloads/${encodeURIComponent(uid)}/terminals`);
  }
}

function assertGuestPath(path: string): void {
  if (!path.startsWith("/") || path.includes("\0")) {
    throw new TypeError("path must be absolute and must not contain NUL");
  }
}

function assertTerminalID(terminalID: string): void {
  if (!/^[A-Za-z0-9._-]{1,64}$/.test(terminalID)) {
    throw new TypeError("terminalID is invalid");
  }
}

async function toBytes(data: unknown): Promise<Uint8Array> {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (typeof Blob !== "undefined" && data instanceof Blob) {
    return new Uint8Array(await data.arrayBuffer());
  }
  throw new TypeError("Terminal WebSocket delivered a non-binary frame");
}

const SANDBOX_STATUSES: ReadonlySet<string> = new Set([
  "registered",
  "provisioning",
  "running",
  "error",
  "suspended",
  "deleted",
  "pending",
  "powering_on",
  "powering_off",
  "rebooting",
  "stopped",
  "frozen",
]);

const TERMINAL_FAILURE_STATUSES: ReadonlySet<SandboxStatus> = new Set([
  "error",
  "deleted",
]);

function assertSandboxStatus(status: string): asserts status is SandboxStatus {
  if (!SANDBOX_STATUSES.has(status)) {
    throw new TypeError(`Unknown sandbox status: ${status}`);
  }
}
