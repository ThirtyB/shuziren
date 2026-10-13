import { useState } from "react";
import type { EventForm } from "../types";
import { formToEvent, parseRawEvent } from "../lib/eventForm";

export type UserEventFormProps = {
  form: EventForm;
  updateForm: <K extends keyof EventForm>(key: K, value: EventForm[keyof EventForm]) => void;
  rawJson: string;
  setRawJson: (value: string) => void;
  inputMode: "form" | "json";
  onInputModeChange: (next: "form" | "json") => void;
};

/**
 * 用户事件输入：表单（领域 / 标题 / 概要 / 传播目标 / 观点变体）
 * 或 JSON 模式互斥。该组件仅在 ``eventSource === "user"`` 时显示。
 *
 * ``event_id`` **不在表单中暴露**：后端 ``_normalize_hot_event`` 在缺失时
 * 会自动生成 ``web_event_<timestamp>``，因此用户不需要（也不应该）手动起名。
 *
 * ``inputMode`` 由父组件管理：父组件需要它来生成 ``getUserEventPayload`` 的
 * 归一化结果，从而判断是否可以发起请求。
 */
export function UserEventForm(props: UserEventFormProps) {
  const { form, updateForm, rawJson, setRawJson, inputMode, onInputModeChange } = props;
  const [error, setError] = useState("");

  function switchMode(next: "form" | "json") {
    setError("");
    if (next === inputMode) return;
    if (next === "json") {
      setRawJson(JSON.stringify(formToEvent(form), null, 2));
    } else {
      try {
        const parsed = parseRawEvent(rawJson);
        Object.entries(parsed).forEach(([k, v]) => {
          if (k in form) {
            const value =
              k === "opinion_variants"
                ? Array.isArray(v)
                  ? v.map(String).join("\n")
                  : String(v ?? "")
                : (v as EventForm[keyof EventForm]);
            updateForm(k as keyof EventForm, value as never);
          }
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }
    onInputModeChange(next);
  }

  return (
    <div className="event-input">
      <div className="segmented">
        <button
          type="button"
          className={inputMode === "form" ? "active" : ""}
          onClick={() => switchMode("form")}
        >
          表单
        </button>
        <button
          type="button"
          className={inputMode === "json" ? "active" : ""}
          onClick={() => switchMode("json")}
        >
          JSON
        </button>
      </div>
      {error && <div className="message error">{error}</div>}
      {inputMode === "form" ? (
        <div className="form-grid">
          <label>
            领域
            <input
              value={form.domain}
              onChange={(event) => updateForm("domain", event.target.value)}
            />
          </label>
          <label className="wide">
            标题
            <input
              value={form.event_title}
              onChange={(event) => updateForm("event_title", event.target.value)}
            />
          </label>
          <label className="wide">
            概要
            <textarea
              value={form.event_summary}
              rows={4}
              onChange={(event) => updateForm("event_summary", event.target.value)}
            />
          </label>
          <label className="wide">
            传播目标
            <textarea
              value={form.target}
              rows={3}
              onChange={(event) => updateForm("target", event.target.value)}
            />
          </label>
          <label className="wide">
            观点变体
            <textarea
              value={form.opinion_variants}
              rows={6}
              onChange={(event) =>
                updateForm("opinion_variants", event.target.value)
              }
            />
          </label>
          <p className="form-hint">
            事件 ID 会在提交时由后端自动生成（``web_event_&lt;时间戳&gt;``），
            无需手动填写。
          </p>
        </div>
      ) : (
        <label className="json-editor">
          事件 JSON
          <textarea
            value={rawJson}
            rows={18}
            onChange={(event) => setRawJson(event.target.value)}
            spellCheck={false}
          />
        </label>
      )}
    </div>
  );
}

export function getUserEventPayload(
  form: EventForm,
  rawJson: string,
  inputMode: "form" | "json",
): Record<string, unknown> {
  return inputMode === "json" ? parseRawEvent(rawJson) : formToEvent(form);
}