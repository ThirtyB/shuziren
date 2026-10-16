/** 事件对象与表单之间的双向转换。 */

import type { EventForm } from "../types";

export const FALLBACK_EVENT = {
  domain: "general",
  event_title: "请输入事件标题",
  event_summary: "请输入事件概要",
  target: "引导公众理解事件影响并进行理性讨论。",
  is_synthetic: false,
  opinion_variants: ["请输入可用于发帖内容生成的观点变体。"],
};

export function eventToForm(event: Record<string, unknown>): EventForm {
  // 兼容旧后端响应里可能仍带 ``event_id``：表单不展示它，但 ``formToEvent``
  // 不会回填，所以它对最终请求体没有影响。
  return {
    domain: String(event.domain ?? "general"),
    event_title: String(event.event_title ?? ""),
    event_summary: String(event.event_summary ?? ""),
    target: String(event.target ?? ""),
    opinion_variants: Array.isArray(event.opinion_variants)
      ? event.opinion_variants.map(String).join("\n")
      : String(event.opinion_variants ?? ""),
  };
}

export function formToEvent(form: EventForm) {
  // 故意不传 ``event_id``：后端 ``_normalize_hot_event`` 会在缺失时
  // 自动生成 ``web_event_<timestamp>``，不需要用户起名。
  return {
    domain: form.domain,
    event_title: form.event_title,
    event_summary: form.event_summary,
    target: form.target,
    is_synthetic: false,
    opinion_variants: form.opinion_variants
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean),
  };
}

export function parseRawEvent(raw: string): Record<string, unknown> {
  const payload = JSON.parse(raw);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("事件 JSON 必须是对象。");
  }
  return payload as Record<string, unknown>;
}
