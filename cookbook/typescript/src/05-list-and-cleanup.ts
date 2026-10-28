import { client } from "./box.js";

const api = client();
const prefix = "cb-";
let cursor: string | null | undefined;
let deleted = 0;

for (;;) {
  const page = await api.sandboxes.list({ limit: 100, cursor });
  for (const item of page.items) {
    if (!item.name.startsWith(prefix)) continue;
    console.log(`deleting ${item.uid} (${item.name})`);
    await api.sandboxes.delete(item.uid);
    deleted += 1;
  }
  if (!page.next_cursor) break;
  cursor = page.next_cursor;
}

console.log(`deleted ${deleted} sandbox(es)`);
