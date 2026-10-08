import type { SandboxTemplate } from "./templates.js";
import type { SANDBOX_STATUSES } from "./validation.js";

export type SandboxTemplateKind = "FRESH" | "USER";
export type SandboxTemplateStatus = "PENDING" | "READY" | "FAILED";
export type PortProtocol = "TCP" | "UDP";
export type PortRouting = "PROXIED" | "DIRECT";
export type SandboxStatus = (typeof SANDBOX_STATUSES)[number];

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

export interface WorkloadPort {
  port: number;
  protocol: PortProtocol;
  routing?: PortRouting;
}

export interface WorkloadURL {
  port: number;
  url: string;
}

export interface WorkloadSSHKey {
  uid: string;
  name: string;
  public_key_raw: string;
}

export interface WorkloadResource {
  name: string;
  display_name: string;
  gpu_vendor?: string;
  gpu_model?: string;
  gpu_type?: string;
  gpu_count?: number;
  cpu_millicores: number;
  cpu_vendor?: string;
  cpu_model?: string;
  cpu_sockets?: number;
  memory_mib: number;
  disk_size_mib?: number;
  disk_label?: string;
  network_mode?: string;
  cc_enabled?: boolean;
  vcpu: number;
  memory: number;
}

export interface WorkloadState {
  status: SandboxStatus;
  message: string;
  urls?: WorkloadURL[];
  public_ip?: string;
  ssh_port?: number;
  ready_replicas: number;
  total_replicas: number;
}

export interface SandboxConfigInput {
  ttl_sec?: number;
  idle_timeout_sec?: number;
}

export interface SandboxConfig extends SandboxConfigInput {
  template_uid: string;
  parent_workload_uid?: string;
  expires_at?: string;
  last_activity_at?: string;
}

/** Raw JSON shape returned by the sandbox workload API. */
export interface SandboxData {
  uid: string;
  type: "SANDBOX";
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
  created_at: string;
  updated_at: string;
}

/** Sparse workload shape returned by operations and sandbox list endpoints. */
export interface SandboxSummary {
  uid: string;
  type: "SANDBOX";
  name: string;
  image?: string;
  reservation_uid?: string;
  cost_per_hour?: number;
  frozen_cost_per_hour?: number;
  resource?: WorkloadResource;
  state?: WorkloadState;
  created_at: string;
  updated_at: string;
}

export interface WorkloadStateResponse extends WorkloadState {
  uid: string;
  workload_type: "SANDBOX";
  updated_at: string;
}

/** Raw JSON shape returned by the sandbox template API. */
export interface SandboxTemplateData {
  uid: string;
  name: string;
  display_name?: string;
  description?: string;
  kind: SandboxTemplateKind;
  status: SandboxTemplateStatus;
  status_message?: string;
  resource_name: string;
  resource?: WorkloadResource;
  cost_per_hour?: number;
  frozen_cost_per_hour?: number;
  desktop_port?: number;
  source_workload_uid?: string;
  created_at: string;
  updated_at: string;
}

export interface SandboxCreateParams {
  name: string;
  template: SandboxTemplate;
  project_id?: string;
  ssh_keys?: string[];
  ports?: WorkloadPort[];
  ttl_sec?: number;
  idle_timeout_sec?: number;
  wait_until_running?: boolean;
  wait?: WaitOptions;
}

export interface SandboxUpdateParams {
  name?: string;
  project_id?: string;
  ssh_keys?: string[];
  ports?: WorkloadPort[];
  sandbox_config?: SandboxConfigInput;
}

export interface SandboxListParams {
  status?: SandboxStatus;
  project_id?: string;
  name?: string;
  limit?: number;
  cursor?: string;
}

export interface TemplateListParams {
  kind?: SandboxTemplateKind;
  status?: SandboxTemplateStatus;
  limit?: number;
  cursor?: string;
}

export interface SandboxTemplateUpdate {
  display_name?: string;
  description?: string;
}

export interface ForkRequest {
  name?: string;
  project_id?: string;
  sandbox_config?: SandboxConfigInput;
  wait_until_running?: boolean;
  wait?: WaitOptions;
}

export interface PublishRequest {
  name: string;
  display_name?: string;
  description?: string;
  wait_until_ready?: boolean;
  wait?: WaitOptions;
}

export interface WaitOptions {
  timeout_ms?: number;
  interval_ms?: number;
  signal?: AbortSignal;
}

export interface ExecResult {
  stdout: string;
  stderr: string;
  code: number;
  timed_out: boolean;
}

export interface AccessTicket {
  ticket: string;
  expires_at: string;
}

export interface TerminalSession {
  id: string;
  pid: number;
  started_at: string;
  exited: boolean;
  exit_code?: number;
}

export interface DesktopInfo {
  available: boolean;
  port?: number;
  listening?: boolean;
  ws_url?: string;
}

export interface TerminalCreateOptions {
  cols?: number;
  rows?: number;
}

export interface WebSocketLike {
  binaryType: string;
  readonly readyState: number;
  send(data: ArrayBuffer | ArrayBufferView): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: string, listener: (event: any) => void): void;
  removeEventListener?(type: string, listener: (event: any) => void): void;
}

export interface WebSocketConstructor {
  new (url: string | URL, protocols?: string | string[]): WebSocketLike;
}

export interface TerminalConnectOptions {
  WebSocket: WebSocketConstructor;
  ticket_ttl_sec?: number;
  onData?: (data: Uint8Array) => void;
  onExit?: (event: { code?: number; reason?: string; wasClean?: boolean }) => void;
  onError?: (event: unknown) => void;
}

export interface TerminalConnection {
  readonly socket: WebSocketLike;
  readonly ready: Promise<void>;
  write(data: Uint8Array | string): void;
  close(code?: number, reason?: string): void;
}
