import { client, createSandbox, sandboxName } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "fork");
try {
  const child = await sandbox.fork({
    name: sandboxName("child"),
    wait: { timeout_ms: 600_000, interval_ms: 2_000 },
  });
  try {
    console.log("child", child.uid);
    const result = await child.exec("echo forked", 60);
    if (result.code !== 0) {
      throw new Error(result.stderr);
    }
    process.stdout.write(result.stdout);
  } finally {
    console.log("deleting child", child.uid);
    await child.delete();
  }
} finally {
  await sandbox.delete();
}
