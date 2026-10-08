type BufferLike = {
  from(data: Uint8Array | string, encoding?: string): {
    toString(encoding?: string): string;
  } & Uint8Array;
};

function buffer(): BufferLike | undefined {
  return (globalThis as { Buffer?: BufferLike }).Buffer;
}

export function encodeBase64(data: Uint8Array): string {
  if (typeof globalThis.btoa === "function") {
    const chunks: string[] = [];
    const size = 0x8000;
    for (let offset = 0; offset < data.length; offset += size) {
      chunks.push(
        String.fromCharCode(...data.subarray(offset, Math.min(offset + size, data.length))),
      );
    }
    return globalThis.btoa(chunks.join(""));
  }
  const BufferImpl = buffer();
  if (BufferImpl) return BufferImpl.from(data).toString("base64");
  throw new Error("No browser base64 API or Node Buffer implementation is available");
}

export function decodeBase64(encoded: string): Uint8Array {
  if (typeof globalThis.atob === "function") {
    const binary = globalThis.atob(encoded);
    const result = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) {
      result[index] = binary.charCodeAt(index);
    }
    return result;
  }
  const BufferImpl = buffer();
  if (BufferImpl) return new Uint8Array(BufferImpl.from(encoded, "base64"));
  throw new Error("No browser base64 API or Node Buffer implementation is available");
}
