import { resolveApiKey } from "./auth.js";
import { apiError, type ApiErrorBody } from "./errors.js";
import { SandboxesService } from "./sandboxes.js";
import { SandboxTemplatesService } from "./templates.js";

export const DEFAULT_BASE_URL = "https://api.targon.com";
export const API_VERSION = "/tha/v3";

export interface TargonClientConfig {
  organization: string;
  apiKey?: string;
  baseUrl?: string;
  fetch?: typeof fetch;
  requestTimeoutMs?: number;
}

interface RequestOptions {
  body?: unknown;
  query?: Record<string, string | number | undefined>;
  signal?: AbortSignal;
  workloadUID?: string;
}

export class TargonClient {
  readonly organization: string;
  readonly sandboxes: SandboxesService;
  readonly sandboxTemplates: SandboxTemplatesService;
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private readonly requestTimeoutMs: number;

  constructor(config: TargonClientConfig) {
    this.organization = config.organization.trim();
    if (!this.organization) throw new Error("organization is required");
    this.apiKey = resolveApiKey(config.apiKey);
    this.baseUrl = (config.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.fetchImpl = config.fetch ?? globalThis.fetch;
    if (typeof this.fetchImpl !== "function") {
      throw new Error("A Fetch API implementation is required");
    }
    this.requestTimeoutMs = config.requestTimeoutMs ?? 30_000;
    this.sandboxTemplates = new SandboxTemplatesService(this);
    this.sandboxes = new SandboxesService(this, this.sandboxTemplates);
  }

  orgPath(path: string): string {
    return `${API_VERSION}/orgs/${encodeURIComponent(this.organization)}${path}`;
  }

  absoluteUrl(
    path: string,
    query?: Record<string, string | number | undefined>,
  ): string {
    const url = new URL(`${this.baseUrl}${path}`);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
    return url.toString();
  }

  websocketUrl(path: string): string {
    const url = new URL(`${this.baseUrl}${path}`);
    url.protocol = url.protocol === "http:" ? "ws:" : "wss:";
    return url.toString();
  }

  async request<T>(
    method: string,
    path: string,
    options: RequestOptions = {},
  ): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.requestTimeoutMs);
    const abort = () => controller.abort(options.signal?.reason);
    options.signal?.addEventListener("abort", abort, { once: true });

    try {
      const response = await this.fetchImpl(this.absoluteUrl(path, options.query), {
        method,
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          Accept: "application/json",
          ...(options.body === undefined ? {} : { "Content-Type": "application/json" }),
        },
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: controller.signal,
      });

      if (!response.ok) {
        let body: ApiErrorBody = {};
        const rawBody = await response.text();
        try {
          body = JSON.parse(rawBody) as ApiErrorBody;
        } catch {
          if (rawBody.trim()) {
            body = { error: rawBody.trim() };
          }
        }
        throw apiError(
          response.status,
          body,
          `Targon API request failed with status ${response.status}`,
          options.workloadUID,
        );
      }
      if (response.status === 204) return undefined as T;
      return (await response.json()) as T;
    } finally {
      clearTimeout(timeout);
      options.signal?.removeEventListener("abort", abort);
    }
  }
}

export function createClient(config: TargonClientConfig): TargonClient {
  return new TargonClient(config);
}
