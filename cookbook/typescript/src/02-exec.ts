import { client, createSandbox } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "exec");
try {
  const result = await sandbox.exec("python --version", 60);
  console.log(`code=${result.code} timed_out=${result.timed_out}`);
  console.log("stdout:", result.stdout);
  console.log("stderr:", result.stderr);
  if (result.code !== 0 || result.timed_out) {
    throw new Error("command failed");
  }
} finally {
  await sandbox.delete();
}
