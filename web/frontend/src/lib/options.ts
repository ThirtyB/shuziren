/** 通用工具：错误消息 / 单元格渲染。 */

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function renderCell(value: unknown): string {
  if (Array.isArray(value)) return value.join("、");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value ?? "");
}
