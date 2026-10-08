import type { SandboxesService } from "./sandboxes.js";
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
  readonly uid: string;
  readonly type: "SANDBOX";
  project_id?: string;
  name: string;
  image?: string;
  resource_name?: string;
  reservation_uid?: string;
  cost_per_hour?: number;
  frozen_cost_per_hour?: number;
  resource?: WorkloadResource;
  ports?: WorkloadPort[];
  ssh_keys?: WorkloadSSHKey[];
  sandbox_config?: SandboxConfig;
  state?: WorkloadState;
  readonly created_at: string;
  updated_at: string;

  readonly #service: SandboxesService;
  readonly #files: SandboxFiles;
  readonly #terminals: SandboxTerminals;

  constructor(service: SandboxesService, data: SandboxData) {
    this.#service = service;
    this.uid = data.uid;
    this.type = data.type;
    this.project_id = data.project_id;
    this.name = data.name;
    this.image = data.image;
    this.resource_name = data.resource_name;
    this.reservation_uid = data.reservation_uid;
    this.cost_per_hour = data.cost_per_hour;
    this.frozen_cost_per_hour = data.frozen_cost_per_hour;
    this.resource = data.resource;
    this.ports = data.ports;
    this.ssh_keys = data.ssh_keys;
    this.sandbox_config = data.sandbox_config;
    this.state = data.state;
    this.created_at = data.created_at;
    this.updated_at = data.updated_at;
    this.#files = new SandboxFiles(service, this.uid);
    this.#terminals = new SandboxTerminals(service, this.uid);
  }

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

  readFile(path: string): Promise<Uint8Array> {
    return this.#service.readFile(this.uid, path);
  }

  writeFile(path: string, data: Uint8Array): Promise<void> {
    return this.#service.writeFile(this.uid, path, data);
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
    return {
      uid: this.uid,
      type: this.type,
      project_id: this.project_id,
      name: this.name,
      image: this.image,
      resource_name: this.resource_name,
      reservation_uid: this.reservation_uid,
      cost_per_hour: this.cost_per_hour,
      frozen_cost_per_hour: this.frozen_cost_per_hour,
      resource: this.resource,
      ports: this.ports,
      ssh_keys: this.ssh_keys,
      sandbox_config: this.sandbox_config,
      state: this.state,
      created_at: this.created_at,
      updated_at: this.updated_at,
    };
  }
}

export class SandboxFiles {
  constructor(
    private readonly service: SandboxesService,
    private readonly uid: string,
  ) {}

  read(path: string): Promise<Uint8Array> {
    return this.service.readFile(this.uid, path);
  }

  write(path: string, data: Uint8Array): Promise<void> {
    return this.service.writeFile(this.uid, path, data);
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
