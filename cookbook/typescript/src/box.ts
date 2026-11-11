import { TargonClient, type Sandbox, type SandboxTemplate } from "@targon/sdk";

export function client(): TargonClient {
  const organization = process.env.TARGON_ORG?.trim();
  if (!organization) {
    throw new Error("TARGON_ORG is required");
  }
  return new TargonClient({ organization });
}

export function sandboxName(action: string): string {
  // 1-32 lowercase letters, digits, or hyphens. Cannot start or end with "-".
  return `cb-${action}-${Math.floor(Date.now() / 1000) % 1_000_000}`;
}

export async function readyTemplate(api: TargonClient): Promise<SandboxTemplate> {
  const page = await api.sandboxTemplates.list({ status: "READY" });
  const template = page.items[0];
  if (!template) {
    throw new Error("no READY sandbox template is available");
  }
  return template;
}

export async function createSandbox(api: TargonClient, action: string): Promise<Sandbox> {
  return api.sandboxes.create({
    name: sandboxName(action),
    template: await readyTemplate(api),
    ttl_sec: 900,
    idle_timeout_sec: 300,
    wait: { timeout_ms: 600_000, interval_ms: 2_000 },
  });
}
