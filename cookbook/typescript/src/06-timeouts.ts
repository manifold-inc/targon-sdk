import { client, createSandbox } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "ttl");
try {
  const updated = await sandbox.update({
    sandbox_config: { ttl_sec: 1800, idle_timeout_sec: 600 },
  });
  const config = updated.sandbox_config;
  console.log(
    `ttl_sec=${config?.ttl_sec} idle_timeout_sec=${config?.idle_timeout_sec}`,
  );
} finally {
  await sandbox.delete();
}
