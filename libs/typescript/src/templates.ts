import type { TargonClient } from "./client.js";
import { SandboxTemplateError } from "./errors.js";
import type { Sandbox } from "./sandbox.js";
import type {
  Page,
  SandboxCreateParams,
  SandboxTemplateData,
  SandboxTemplateKind,
  SandboxTemplateStatus,
  SandboxTemplateUpdate,
  TemplateListParams,
  WaitOptions,
  WorkloadResource,
} from "./types.js";
import { assertLimit, sleep } from "./validation.js";

type SandboxCreateFromTemplateParams = Omit<SandboxCreateParams, "template">;

const SANDBOX_TEMPLATE_RESOURCE = Symbol("SandboxTemplateResource");
const templateBindings = new WeakMap<SandboxTemplate, SandboxTemplatesService>();

/** A hydrated sandbox template with service-backed convenience methods. */
export class SandboxTemplate implements SandboxTemplateData {
  readonly uid: string;
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
  readonly created_at: string;
  updated_at: string;

  constructor(
    service: SandboxTemplatesService,
    data: SandboxTemplateData,
    resourceToken: typeof SANDBOX_TEMPLATE_RESOURCE,
  ) {
    if (resourceToken !== SANDBOX_TEMPLATE_RESOURCE) {
      throw new TypeError("SandboxTemplate resources must be obtained from a service");
    }
    templateBindings.set(this, service);
    this.uid = data.uid;
    this.name = data.name;
    this.display_name = data.display_name;
    this.description = data.description;
    this.kind = data.kind;
    this.status = data.status;
    this.status_message = data.status_message;
    this.resource_name = data.resource_name;
    this.resource = data.resource;
    this.cost_per_hour = data.cost_per_hour;
    this.frozen_cost_per_hour = data.frozen_cost_per_hour;
    this.desktop_port = data.desktop_port;
    this.source_workload_uid = data.source_workload_uid;
    this.created_at = data.created_at;
    this.updated_at = data.updated_at;
  }

  refresh(): Promise<SandboxTemplate> {
    return templateService(this).get(this.uid);
  }

  update(update: SandboxTemplateUpdate): Promise<SandboxTemplate> {
    return templateService(this).update(this.uid, update);
  }

  delete(): Promise<void> {
    return templateService(this).delete(this.uid);
  }

  createSandbox(params: SandboxCreateFromTemplateParams): Promise<Sandbox> {
    return templateService(this).createSandbox(this, params);
  }

  toJSON(): SandboxTemplateData {
    return {
      uid: this.uid,
      name: this.name,
      display_name: this.display_name,
      description: this.description,
      kind: this.kind,
      status: this.status,
      status_message: this.status_message,
      resource_name: this.resource_name,
      resource: this.resource,
      cost_per_hour: this.cost_per_hour,
      frozen_cost_per_hour: this.frozen_cost_per_hour,
      desktop_port: this.desktop_port,
      source_workload_uid: this.source_workload_uid,
      created_at: this.created_at,
      updated_at: this.updated_at,
    };
  }
}

export class SandboxTemplatesService {
  constructor(private readonly client: TargonClient) {}

  async list(params: TemplateListParams = {}): Promise<Page<SandboxTemplate>> {
    assertLimit(params.limit);
    const page = await this.client.request<Page<SandboxTemplateData>>(
      "GET",
      this.client.orgPath("/sandbox-templates"),
      {
        query: {
          kind: params.kind,
          status: params.status,
          limit: params.limit,
          cursor: params.cursor,
        },
      },
    );
    return {
      ...page,
      items: page.items.map((template) => hydrateSandboxTemplate(template, this)),
    };
  }

  async get(uid: string): Promise<SandboxTemplate> {
    const template = await this.client.request<SandboxTemplateData>(
      "GET",
      this.client.orgPath(`/sandbox-templates/${encodeURIComponent(uid)}`),
    );
    return hydrateSandboxTemplate(template, this);
  }

  async update(
    uid: string,
    update: SandboxTemplateUpdate,
  ): Promise<SandboxTemplate> {
    if (update.display_name === undefined && update.description === undefined) {
      throw new TypeError("display_name or description is required");
    }
    if (update.display_name !== undefined && update.display_name.trim().length > 128) {
      throw new RangeError("display_name must be at most 128 characters");
    }
    const template = await this.client.request<SandboxTemplateData>(
      "PATCH",
      this.client.orgPath(`/sandbox-templates/${encodeURIComponent(uid)}`),
      { body: update },
    );
    return hydrateSandboxTemplate(template, this);
  }

  delete(uid: string): Promise<void> {
    return this.client.request(
      "DELETE",
      this.client.orgPath(`/sandbox-templates/${encodeURIComponent(uid)}`),
    );
  }

  async waitForStatus(
    uid: string,
    statuses: SandboxTemplateStatus | readonly SandboxTemplateStatus[],
    options: WaitOptions = {},
  ): Promise<SandboxTemplate> {
    const accepted = new Set(Array.isArray(statuses) ? statuses : [statuses]);
    const timeout = options.timeout_ms ?? 10 * 60_000;
    const interval = options.interval_ms ?? 1_000;
    const deadline = Date.now() + timeout;
    for (;;) {
      if (options.signal?.aborted) throw options.signal.reason;
      const template = await this.get(uid);
      if (template.status === "FAILED") {
        throw new SandboxTemplateError(
          409,
          template.status_message || `Template ${uid} failed`,
          "SANDBOX_TEMPLATE_FAILED",
        );
      }
      if (accepted.has(template.status)) return template;
      if (Date.now() >= deadline) {
        throw new Error(`Timed out waiting for template ${uid}`);
      }
      await sleep(interval, options.signal);
    }
  }

  createSandbox(
    template: SandboxTemplate,
    params: SandboxCreateFromTemplateParams,
  ): Promise<Sandbox> {
    return this.client.sandboxes.create({ ...params, template });
  }
}

export function hydrateSandboxTemplate(
  data: SandboxTemplateData,
  service: SandboxTemplatesService,
): SandboxTemplate {
  return new SandboxTemplate(service, data, SANDBOX_TEMPLATE_RESOURCE);
}

export function templateUIDForService(
  template: SandboxTemplate,
  service: SandboxTemplatesService,
): string {
  if (!(template instanceof SandboxTemplate) || templateBindings.get(template) !== service) {
    throw new TypeError("template must be a SandboxTemplate from this TargonClient");
  }
  if (!template.uid.trim()) {
    throw new TypeError("template uid must be non-empty");
  }
  if (template.status !== "READY") {
    throw new SandboxTemplateError(
      409,
      `Template ${template.uid} is not READY`,
      "SANDBOX_TEMPLATE_NOT_READY",
    );
  }
  return template.uid;
}

function templateService(template: SandboxTemplate): SandboxTemplatesService {
  const service = templateBindings.get(template);
  if (!service) throw new TypeError("SandboxTemplate is not bound to a service");
  return service;
}
