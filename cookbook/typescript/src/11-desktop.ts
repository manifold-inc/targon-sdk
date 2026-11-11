import { client, createSandbox } from "./box.js";

const api = client();
const sandbox = await createSandbox(api, "desk");
try {
  const desktop = await sandbox.getDesktop();
  console.log(
    `desktop available=${desktop.available} port=${desktop.port} listening=${desktop.listening}`,
  );
  if (desktop.ws_url) {
    console.log("desktop ws url:", desktop.ws_url);
  }

  const ticket = await sandbox.mintAccessTicket(60);
  // ticket.ticket is a secret. Print only when it expires.
  console.log("access ticket expires at", ticket.expires_at);
} finally {
  await sandbox.delete();
}
