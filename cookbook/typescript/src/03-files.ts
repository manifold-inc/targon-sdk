import { client, createSandbox } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "files");
try {
  const input = new TextEncoder().encode("hello from the cookbook\n");
  await sandbox.files.write("/tmp/hello.txt", input);
  const result = await sandbox.exec(
    "cp /tmp/hello.txt /tmp/hello.out && wc -c /tmp/hello.out",
    60,
  );
  if (result.code !== 0) {
    throw new Error(result.stderr);
  }
  const raw = await sandbox.files.read("/tmp/hello.out");
  console.log(result.stdout);
  console.log("text:", new TextDecoder().decode(raw));
  console.log("bytes:", raw.byteLength);
} finally {
  await sandbox.delete();
}
