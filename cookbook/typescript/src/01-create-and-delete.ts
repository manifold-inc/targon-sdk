import { client, readyTemplate, sandboxName } from "./box.js";

const api = client();
const template = await readyTemplate(api);
console.log(`using template ${template.uid} (${template.name})`);

const sandbox = await api.sandboxes.create({
  name: sandboxName("create"),
  template,
  ttl_sec: 900,
  idle_timeout_sec: 300,
  wait: { timeout_ms: 600_000, interval_ms: 2_000 },
});

try {
  const state = await sandbox.getState();
  console.log(`created ${sandbox.uid} status=${state.status}`);
} finally {
  console.log(`deleting ${sandbox.uid}`);
  await sandbox.delete();
}

console.log("sandbox removed");
