import type {
  SandboxFilesService,
  SandboxesService,
} from "./sandboxes.js";
import type { SandboxTemplate } from "./templates.js";
import type {
  AccessTicket,
  DesktopInfo,
  ExecResult,
  ForkRequest,
  PublishRequest,
  SandboxConfig,
  SandboxData,
  SandboxUpdateParams,
  TerminalConnectOptions,
  TerminalConnection,
  TerminalCreateOptions,
  TerminalSession,
  WaitOptions,
  WorkloadPort,
  WorkloadResource,
  WorkloadSSHKey,
  WorkloadState,
  WorkloadStateResponse,
} from "./types.js";

/** A hydrated sandbox with the wire data and thin service-backed conveniences. */
export class Sandbox implements SandboxData {
  readonly #data: SandboxData;
  readonly #service: SandboxesService;
  readonly #files: SandboxFiles;
  readonly #terminals: SandboxTerminals;

  constructor(service: SandboxesService, data: SandboxData) {
    this.#service = service;
    this.#data = { ...data };
    this.#files = new SandboxFiles(service.files, data.uid);
    this.#terminals = new SandboxTerminals(service, this.uid);
  }

  get uid(): string { return this.#data.uid; }
  get type(): "SANDBOX" { return this.#data.type; }
  get project_id(): string | undefined { return this.#data.project_id; }
  get name(): string { return this.#data.name; }
  get image(): string | undefined { return this.#data.image; }
  get resource_name(): string | undefined { return this.#data.resource_name; }
  get reservation_uid(): string | undefined { return this.#data.reservation_uid; }
  get cost_per_hour(): number | undefined { return this.#data.cost_per_hour; }
  get frozen_cost_per_hour(): number | undefined {
    return this.#data.frozen_cost_per_hour;
  }
  get resource(): WorkloadResource | undefined { return this.#data.resource; }
  get ports(): WorkloadPort[] | undefined { return this.#data.ports; }
  get ssh_keys(): WorkloadSSHKey[] | undefined { return this.#data.ssh_keys; }
  get sandbox_config(): SandboxConfig | undefined {
    return this.#data.sandbox_config;
  }
  get state(): WorkloadState | undefined { return this.#data.state; }
  get created_at(): string { return this.#data.created_at; }
  get updated_at(): string { return this.#data.updated_at; }

  get files(): SandboxFiles {
    return this.#files;
  }

  get terminals(): SandboxTerminals {
    return this.#terminals;
  }

  refresh(): Promise<Sandbox> {
    return this.#service.get(this.uid);
  }

  getState(): Promise<WorkloadStateResponse> {
    return this.#service.getState(this.uid);
  }

  update(update: SandboxUpdateParams): Promise<Sandbox> {
    return this.#service.update(this.uid, update);
  }

  freeze(wait?: WaitOptions): Promise<Sandbox> {
    return this.#service.freeze(this.uid, wait);
  }

  thaw(wait?: WaitOptions): Promise<Sandbox> {
    return this.#service.thaw(this.uid, wait);
  }

  delete(): Promise<void> {
    return this.#service.delete(this.uid);
  }

  fork(request: ForkRequest = {}): Promise<Sandbox> {
    return this.#service.fork(this.uid, request);
  }

  publish(request: PublishRequest): Promise<SandboxTemplate> {
    return this.#service.publish(this.uid, request);
  }

  exec(cmd: string, timeoutSec = 60): Promise<ExecResult> {
    return this.#service.exec(this.uid, cmd, timeoutSec);
  }

  mintAccessTicket(ttlSec = 60): Promise<AccessTicket> {
    return this.#service.mintAccessTicket(this.uid, ttlSec);
  }

  getDesktop(): Promise<DesktopInfo> {
    return this.#service.getDesktop(this.uid);
  }

  attachSshKey(keyUID: string): Promise<void> {
    return this.#service.attachSshKey(this.uid, keyUID);
  }

  detachSshKey(keyUID: string): Promise<void> {
    return this.#service.detachSshKey(this.uid, keyUID);
  }

  toJSON(): SandboxData {
    return { ...this.#data };
  }
}

export class SandboxFiles {
  constructor(
    private readonly service: SandboxFilesService,
    private readonly uid: string,
  ) {}

  read(path: string): Promise<Uint8Array> {
    return this.service.read(this.uid, path);
  }

  write(path: string, data: Uint8Array): Promise<void> {
    return this.service.write(this.uid, path, data);
  }
}

export class SandboxTerminals {
  constructor(
    private readonly service: SandboxesService,
    private readonly uid: string,
  ) {}

  list(): Promise<TerminalSession[]> {
    return this.service.terminals.list(this.uid);
  }

  create(options: TerminalCreateOptions = {}): Promise<TerminalSession> {
    return this.service.terminals.create(this.uid, options);
  }

  delete(terminalID: string): Promise<void> {
    return this.service.terminals.delete(this.uid, terminalID);
  }

  connect(
    terminalID: string,
    options: TerminalConnectOptions,
  ): Promise<TerminalConnection> {
    return this.service.terminals.connect(this.uid, terminalID, options);
  }
}
