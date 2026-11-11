export function resolveApiKey(explicit?: string): string {
  const processKey =
    typeof globalThis === "object" &&
    "process" in globalThis &&
    typeof (globalThis as { process?: { env?: Record<string, string | undefined> } }).process
      ?.env?.TARGON_API_KEY === "string"
      ? (globalThis as { process: { env: Record<string, string | undefined> } }).process.env
          .TARGON_API_KEY
      : undefined;
  const apiKey = explicit?.trim() || processKey?.trim();
  if (!apiKey) {
    throw new Error("Targon API key is required (pass apiKey or set TARGON_API_KEY)");
  }
  return apiKey;
}
