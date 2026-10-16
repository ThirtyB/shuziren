/** 数据集中的三个模拟平台常量与显示工具。 */

export const SIMULATED_PLATFORMS = [
  { key: "facebook", value: "facebook_simulated", label: "模拟 Facebook" },
  { key: "instagram", value: "instagram_simulated", label: "模拟 Instagram" },
  { key: "twitter", value: "twitter_simulated", label: "模拟 Twitter" },
] as const;

export type SimulatedPlatform = (typeof SIMULATED_PLATFORMS)[number];

export const DEFAULT_ALLOWED_PLATFORMS = SIMULATED_PLATFORMS.map((item) => item.value).join(",");

export function parsePlatformList(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function togglePlatform(value: string, platform: string): string {
  const current = parsePlatformList(value);
  const next = current.includes(platform)
    ? current.filter((item) => item !== platform)
    : [...current, platform];
  return next.join(",");
}

export function platformLabel(value: unknown): string {
  const text = String(value ?? "").trim();
  if (!text) return "-";
  const matched = SIMULATED_PLATFORMS.find(
    (item) => item.value === text || item.key === text,
  );
  return matched ? matched.label : text;
}

export function platformModeLabel(value: unknown): string {
  const mode = String(value ?? "").trim();
  if (!mode) return "-";
  if (mode.endsWith("_only")) {
    return `仅 ${platformLabel(mode.slice(0, -"_only".length))}`;
  }
  if (mode === "cross_platform") {
    return "跨平台（多平台混合）";
  }
  if (mode === "unspecified") {
    return "未指定平台";
  }
  return mode;
}

export function platformKeyFromValue(value: string): string {
  const matched = SIMULATED_PLATFORMS.find((item) => item.value === value);
  return matched ? matched.key : value;
}
