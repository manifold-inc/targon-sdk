export interface ApiErrorBody {
  error?: string;
  reason?: string;
}

export class TargonApiError extends Error {
  readonly status: number;
  readonly reason?: string;
  readonly workload_uid?: string;

  constructor(status: number, message: string, reason?: string, workload_uid?: string) {
    super(message);
    this.name = "TargonApiError";
    this.status = status;
    this.reason = reason;
    this.workload_uid = workload_uid;
  }
}

export class SandboxUnavailableError extends TargonApiError {}
export class SandboxStateError extends TargonApiError {}
export class SandboxTemplateError extends TargonApiError {}
export class TerminalLimitError extends TargonApiError {}
export class AccessTicketError extends TargonApiError {}
export class PayloadTooLargeError extends TargonApiError {}
export class GatewayError extends TargonApiError {}

export function apiError(
  status: number,
  body: ApiErrorBody,
  fallback: string,
  workloadUID?: string,
): TargonApiError {
  const message = body.error || fallback;
  const reason = body.reason;
  const args: [number, string, string | undefined, string | undefined] = [
    status,
    message,
    reason,
    workloadUID,
  ];

  if (status === 413 || reason === "WORKLOAD_SANDBOX_PAYLOAD_TOO_LARGE") {
    return new PayloadTooLargeError(...args);
  }
  if (status === 429) return new TerminalLimitError(...args);
  if (status === 502) return new GatewayError(...args);
  if (
    status === 503 ||
    reason?.endsWith("_UNAVAILABLE") ||
    reason?.endsWith("_NO_CAPACITY")
  ) {
    return new SandboxUnavailableError(...args);
  }
  if (
    reason?.includes("TEMPLATE") ||
    reason === "SANDBOX_TEMPLATE_NAME_INVALID" ||
    reason === "SANDBOX_TEMPLATE_DISPLAY_NAME_INVALID"
  ) {
    return new SandboxTemplateError(...args);
  }
  if (
    reason?.includes("INVALID_STATE") ||
    reason?.includes("NOT_DEPLOYED") ||
    (status === 409 && reason?.startsWith("WORKLOAD_SANDBOX"))
  ) {
    return new SandboxStateError(...args);
  }
  if (status === 401 && reason?.includes("TICKET")) {
    return new AccessTicketError(...args);
  }
  return new TargonApiError(...args);
}
