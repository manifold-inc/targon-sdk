import { client, createSandbox } from "./box.js";

const WebSocketImpl = globalThis.WebSocket;
if (typeof WebSocketImpl !== "function") {
  throw new Error(
    "This runtime has no global WebSocket. Use Node 22+, Bun, Deno, or a browser.",
  );
}

const api = client();
const sandbox = await createSandbox(api, "term");
try {
  const session = await sandbox.terminals.create({ cols: 80, rows: 24 });
  console.log("terminal", session.id);
  try {
    const connection = await sandbox.terminals.connect(session.id, {
      WebSocket: WebSocketImpl,
      onData: (bytes) => {
        process.stdout.write(new TextDecoder().decode(bytes));
      },
    });
    await connection.ready;
    connection.write("echo cookbook-terminal\n");
    await new Promise((resolve) => setTimeout(resolve, 1000));
    connection.close();
  } finally {
    await sandbox.terminals.delete(session.id);
  }
} finally {
  await sandbox.delete();
}
