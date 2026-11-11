import { client, createSandbox } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "freeze");
const wait = { timeout_ms: 300_000, interval_ms: 2_000 };
try {
  const frozen = await sandbox.freeze(wait);
  console.log("frozen status:", (await frozen.getState()).status);

  const running = await frozen.thaw(wait);
  const result = await running.exec("echo thawed", 60);
  if (result.code !== 0) {
    throw new Error(result.stderr);
  }
  process.stdout.write(result.stdout);
  console.log("thawed status:", (await running.getState()).status);
} finally {
  await sandbox.delete();
}
