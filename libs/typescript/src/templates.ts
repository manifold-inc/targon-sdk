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
import {
  assertDisplayName,
  assertLimit,
  pollUntil,
} from "./validation.js";

type SandboxCreateFromTemplateParams = Omit<SandboxCreateParams, "template">;

/** A hydrated sandbox template with service-backed convenience methods. */
export class SandboxTemplate implements SandboxTemplateData {
  readonly #data: SandboxTemplateData;
  readonly #service: SandboxTemplatesService;

  constructor(
    service: SandboxTemplatesService,
    data: SandboxTemplateData,
  ) {
    this.#service = service;
    this.#data = { ...data };
  }

  get uid(): string { return this.#data.uid; }
  get name(): string { return this.#data.name; }
  get display_name(): string | undefined { return this.#data.display_name; }
  get description(): string | undefined { return this.#data.description; }
  get kind(): SandboxTemplateKind { return this.#data.kind; }
  get status(): SandboxTemplateStatus { return this.#data.status; }
  get status_message(): string | undefined { return this.#data.status_message; }
  get resource_name(): string { return this.#data.resource_name; }
  get resource(): WorkloadResource | undefined { return this.#data.resource; }
  get cost_per_hour(): number | undefined { return this.#data.cost_per_hour; }
  get frozen_cost_per_hour(): number | undefined {
    return this.#data.frozen_cost_per_hour;
  }
  get desktop_port(): number | undefined { return this.#data.desktop_port; }
  get source_workload_uid(): string | undefined {
    return this.#data.source_workload_uid;
  }
  get created_at(): string { return this.#data.created_at; }
  get updated_at(): string { return this.#data.updated_at; }

  refresh(): Promise<SandboxTemplate> {
    return this.#service.get(this.uid);
  }

  update(update: SandboxTemplateUpdate): Promise<SandboxTemplate> {
    return this.#service.update(this.uid, update);
  }

  delete(): Promise<void> {
    return this.#service.delete(this.uid);
  }

  createSandbox(params: SandboxCreateFromTemplateParams): Promise<Sandbox> {
    return this.#service.createSandbox(this, params);
  }

  toJSON(): SandboxTemplateData {
    return { ...this.#data };
  }

  /** @internal Validates package-owned use without exposing the bound service. */
  _uidFor(service: SandboxTemplatesService): string {
    if (this.#service !== service) {
      throw new TypeError("template must be a SandboxTemplate from this TargonClient");
    }
    if (!this.uid.trim()) throw new TypeError("template uid must be non-empty");
    if (this.status !== "READY") {
      throw new SandboxTemplateError(
        409,
        `Template ${this.uid} is not READY`,
        "SANDBOX_TEMPLATE_NOT_READY",
      );
    }
    return this.uid;
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
      items: page.items.map((template) => new SandboxTemplate(this, template)),
    };
  }

  async get(uid: string): Promise<SandboxTemplate> {
    const template = await this.client.request<SandboxTemplateData>(
      "GET",
      this.client.orgPath(`/sandbox-templates/${encodeURIComponent(uid)}`),
    );
    return new SandboxTemplate(this, template);
  }

  async update(
    uid: string,
    update: SandboxTemplateUpdate,
  ): Promise<SandboxTemplate> {
    if (update.display_name === undefined && update.description === undefined) {
      throw new TypeError("display_name or description is required");
    }
    assertDisplayName(update.display_name);
    const template = await this.client.request<SandboxTemplateData>(
      "PATCH",
      this.client.orgPath(`/sandbox-templates/${encodeURIComponent(uid)}`),
      { body: update },
    );
    return new SandboxTemplate(this, template);
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
    return pollUntil(
      () => this.get(uid),
      (template) => {
        if (template.status === "FAILED") {
          throw new SandboxTemplateError(
            409,
            template.status_message || `Template ${uid} failed`,
            "SANDBOX_TEMPLATE_FAILED",
          );
        }
        return accepted.has(template.status) ? template : undefined;
      },
      `Timed out waiting for template ${uid}`,
      options,
    );
  }

  createSandbox(
    template: SandboxTemplate,
    params: SandboxCreateFromTemplateParams,
  ): Promise<Sandbox> {
    return this.client.sandboxes.create({ ...params, template });
  }
}
