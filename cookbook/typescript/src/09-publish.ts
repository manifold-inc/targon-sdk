import { client, createSandbox, sandboxName } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "pub");
try {
  const template = await sandbox.publish({
    name: sandboxName("tpl"),
    display_name: "Cookbook snapshot",
    description: "Created by cookbook/typescript/src/09-publish.ts",
    wait: { timeout_ms: 600_000, interval_ms: 2_000 },
  });
  try {
    console.log(`template ${template.uid} status=${template.status}`);
  } finally {
    console.log("deleting template", template.uid);
    await template.delete();
  }
} finally {
  await sandbox.delete();
}
