import { client, createSandbox } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "reconn");
try {
  console.log("persist this uid:", sandbox.uid);

  // A later process constructs its own client and calls get(uid).
  // list() only returns summaries; get() returns a sandbox you can exec.
  const again = await api.sandboxes.get(sandbox.uid);
  const result = await again.exec("hostname", 30);
  if (result.code !== 0) {
    throw new Error(result.stderr);
  }
  console.log(result.stdout);
} finally {
  await sandbox.delete();
}
