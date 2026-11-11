/**
 * Register, deploy, exec, and delete a sandbox.
 *
 * Required: TARGON_API_KEY, TARGON_ORG.
 *
 * TypeScript create registers the sandbox and deploys it. Go and Python expose
 * those as separate Register and Deploy calls.
 */
import { TargonClient } from "@targon/sdk";

const organization = process.env.TARGON_ORG?.trim();
if (!organization) {
  throw new Error("TARGON_ORG is required");
}

const client = new TargonClient({ organization });
const templates = await client.sandboxTemplates.list({ status: "READY" });
const template = templates.items[0];
if (!template) {
  throw new Error("no READY sandbox template is available");
}

const name = `ts-example-${Math.floor(Date.now() / 1000) % 1_000_000}`;
console.log("Creating sandbox");
const sandbox = await client.sandboxes.create({
  name,
  template,
  ttl_sec: 900,
  idle_timeout_sec: 300,
  wait: { timeout_ms: 600_000, interval_ms: 2_000 },
});
console.log(`Sandbox running (${sandbox.uid})`);

try {
  const response = await sandbox.exec("python --version", 60);
  if (response.code !== 0) {
    console.log(`Error: ${response.code} ${response.stderr}`);
  } else {
    process.stdout.write(response.stdout);
  }
} finally {
  console.log("Removing sandbox");
  await sandbox.delete();
}

console.log("Sandbox removed");
