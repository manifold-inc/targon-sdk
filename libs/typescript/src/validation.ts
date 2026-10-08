import type {
  SandboxConfigInput,
  SandboxStatus,
  WaitOptions,
  WorkloadPort,
} from "./types.js";

export const MAX_FILE_BYTES = 256 * 1024 * 1024;
export const MAX_COMMAND_BYTES = 64 * 1024;
const NAME = /^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$/;
const TEMPLATE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const TERMINAL_ID = /^[A-Za-z0-9._-]{1,64}$/;

export const SANDBOX_STATUSES = [
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
 ] as const;

const SANDBOX_STATUS_SET: ReadonlySet<string> = new Set(SANDBOX_STATUSES);
export const TERMINAL_FAILURE_STATUSES: ReadonlySet<string> = new Set([
  "error",
  "deleted",
]);

export function assertName(name: string): void {
  if (!NAME.test(name)) {
    throw new TypeError(
      "name must be 1-32 lowercase alphanumeric characters or hyphens, without leading/trailing hyphens",
    );
  }
}

export function assertTemplateName(name: string): void {
  if (!TEMPLATE_NAME.test(name)) {
    throw new TypeError(
      "template name must be 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit",
    );
  }
}

export function assertDisplayName(displayName?: string): void {
  if (displayName !== undefined && displayName.trim().length > 128) {
    throw new RangeError("display_name must be at most 128 characters");
  }
}

export function assertSandboxConfig(config?: SandboxConfigInput): void {
  if (!config) return;
  const { ttl_sec: ttl, idle_timeout_sec: idle } = config;
  if (ttl !== undefined && (!Number.isInteger(ttl) || ttl < 0)) {
    throw new RangeError("ttl_sec must be a non-negative integer");
  }
  if (idle !== undefined && (!Number.isInteger(idle) || idle < 0)) {
    throw new RangeError("idle_timeout_sec must be a non-negative integer");
  }
  if (ttl && idle && idle >= ttl) {
    throw new RangeError("idle_timeout_sec must be less than ttl_sec");
  }
}

export function assertSandboxConfigUpdate(config?: SandboxConfigInput): void {
  assertSandboxConfig(config);
  if (config?.idle_timeout_sec !== undefined && config.idle_timeout_sec === 0) {
    throw new RangeError(
      "idle_timeout_sec must be a positive integer when updating a sandbox",
    );
  }
}

export function assertPorts(ports?: WorkloadPort[]): void {
  for (const item of ports ?? []) {
    if (!Number.isInteger(item.port) || item.port < 1 || item.port > 65_535) {
      throw new RangeError("port must be an integer from 1 through 65535");
    }
    if (item.port === 22) throw new RangeError("port 22 cannot be exposed");
    if (item.protocol !== "TCP" && item.protocol !== "UDP") {
      throw new TypeError("sandbox port protocol must be TCP or UDP");
    }
  }
}

export function assertLimit(limit?: number): void {
  if (
    limit !== undefined &&
    (!Number.isInteger(limit) || limit < 1 || limit > 1_000)
  ) {
    throw new RangeError("limit must be an integer from 1 through 1000");
  }
}

export function assertTerminalDimension(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 1 || value > 1_000) {
    throw new RangeError(`${name} must be an integer from 1 through 1000`);
  }
}

export function assertSandboxStatus(status: string): asserts status is SandboxStatus {
  if (!SANDBOX_STATUS_SET.has(status)) {
    throw new TypeError(`Unknown sandbox status: ${status}`);
  }
}

export function assertGuestPath(path: string): void {
  if (!path.startsWith("/") || path.includes("\0")) {
    throw new TypeError("path must be absolute and must not contain NUL");
  }
}

export function assertTerminalID(terminalID: string): void {
  if (!TERMINAL_ID.test(terminalID)) {
    throw new TypeError("terminalID is invalid");
  }
}

export function assertCommand(cmd: string, timeoutSec: number): void {
  if (!cmd.trim() || new TextEncoder().encode(cmd).byteLength > MAX_COMMAND_BYTES) {
    throw new RangeError("cmd must be non-empty and at most 64 KiB");
  }
  if (!Number.isInteger(timeoutSec) || timeoutSec < 1 || timeoutSec > 600) {
    throw new RangeError("timeoutSec must be an integer from 1 through 600");
  }
}

export function assertFileData(data: Uint8Array): void {
  if (!(data instanceof Uint8Array)) throw new TypeError("data must be a Uint8Array");
  if (data.byteLength > MAX_FILE_BYTES) {
    throw new RangeError("file content must not exceed 256 MiB");
  }
}

export function assertTicketTTL(ttlSec: number): void {
  if (!Number.isInteger(ttlSec) || ttlSec < 1 || ttlSec > 300) {
    throw new RangeError("ttlSec must be an integer from 1 through 300");
  }
}

export async function pollUntil<T, R>(
  poll: () => Promise<T>,
  inspect: (value: T) => R | undefined | Promise<R | undefined>,
  timeoutMessage: string,
  options: WaitOptions = {},
): Promise<R> {
  const deadline = Date.now() + (options.timeout_ms ?? 10 * 60_000);
  const interval = options.interval_ms ?? 1_000;
  for (;;) {
    if (options.signal?.aborted) throw options.signal.reason;
    const result = await inspect(await poll());
    if (result !== undefined) return result;
    if (Date.now() >= deadline) throw new Error(timeoutMessage);
    await sleep(interval, options.signal);
  }
}

export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}
