import { useEffect, useState } from "react";
import { Loader2, Zap } from "lucide-react";
import type { LlmLatencyResponse } from "../types";
import { getLlmLatency, runLlmLatency } from "../lib/api";
import { errorMessage } from "../lib/options";

const STATE_LABEL: Record<string, string> = {
  queued: "排队中",
  running: "测试中",
  completed: "已完成",
  failed: "失败",
};

/**
 * LLM 往返延迟测试面板。
 *
 * 与"一键压测"并列，但**只测一件事**：把库里的一条话题交给大模型生成文案，
 * 记录"发出请求 → 收到响应"的耗时。不跑选人 / 打分 / 策略生成，所以这个数字
 * 可以直接用来判断大模型本身的响应速度。
 *
 * 行为与压测一致：POST 立即返回 queued，之后轮询详情；按钮不因为"有任务在跑"
 * 而禁用，可以连续提交多次（后端线程池串行排队）。
 */
export function LlmLatencyPanel() {
  const [iterations, setIterations] = useState(3);
  const [label, setLabel] = useState("LLM 延迟");
  const [submitting, setSubmitting] = useState(false);
  const [detail, setDetail] = useState<LlmLatencyResponse | null>(null);
  const [errorText, setErrorText] = useState("");

  const latencyId = detail?.latency_id;
  const active = detail?.state === "queued" || detail?.state === "running";

  // 提交后持续轮询，直到不再是排队 / 测试中。
  useEffect(() => {
    if (!latencyId || !active) return;
    const timer = window.setInterval(() => {
      getLlmLatency(latencyId)
        .then((payload) => setDetail(payload))
        .catch(() => {
          /* 轮询失败静默，下一拍重试 */
        });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [latencyId, active]);

  async function handleSubmit() {
    setSubmitting(true);
    setErrorText("");
    try {
      const created = await runLlmLatency({
        iterations,
        label: label.trim() || null,
      });
      setDetail(created);
    } catch (err) {
      setErrorText(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  }

  const summary = detail?.summary ?? null;
  const hasSummary = !!summary && typeof summary.avg_seconds === "number";

  return (
    <div className="stack llm-latency-panel">
      <div className="scenario-hint">
        从库里随机抽一条话题，让大模型生成一条文案，记录
        <strong>发出请求 → 收到响应</strong> 的耗时。不经过选人 / 打分 / 策略生成等
        本地计算，因此可以直接反映大模型本身的响应速度。
      </div>

      <div className="form-grid">
        <label>
          调用次数
          <input
            type="number"
            min={1}
            max={20}
            value={iterations}
            onChange={(event) => setIterations(Number(event.target.value))}
          />
        </label>
        <label>
          测试标签
          <input
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="例如 LLM 延迟"
          />
        </label>
      </div>

      <button
        className="primary-button"
        type="button"
        onClick={handleSubmit}
        disabled={submitting}
      >
        {submitting ? <Loader2 className="spin" size={18} /> : <Zap size={18} />}
        开始测试（{iterations} 次调用）
      </button>

      {errorText && <div className="message">{errorText}</div>}

      {detail && (
        <div className="llm-latency-result">
          <div className="llm-latency-head">
            <span className={`benchmark-state-chip state-${detail.state}`}>
              {active && <Loader2 className="spin" size={12} />}
              {STATE_LABEL[detail.state] ?? detail.state}
            </span>
            <span className="benchmark-progress-text">
              {detail.completed}/{detail.iterations} 次完成
            </span>
            {(detail.provider || detail.model) && (
              <span className="benchmark-label">
                {[detail.provider, detail.model].filter(Boolean).join(" / ")}
              </span>
            )}
          </div>

          {detail.error && <div className="message">{detail.error}</div>}

          {hasSummary && summary && (
            <div className="llm-latency-stats">
              <div>
                <span>平均</span>
                <strong>{summary.avg_seconds.toFixed(2)}s</strong>
              </div>
              <div>
                <span>最快</span>
                <strong>{summary.min_seconds.toFixed(2)}s</strong>
              </div>
              <div>
                <span>最慢</span>
                <strong>{summary.max_seconds.toFixed(2)}s</strong>
              </div>
              <div>
                <span>合计</span>
                <strong>{summary.total_seconds.toFixed(2)}s</strong>
              </div>
            </div>
          )}

          {detail.calls.length > 0 && (
            <table className="llm-latency-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>耗时</th>
                  <th>话题</th>
                  <th>文案预览</th>
                </tr>
              </thead>
              <tbody>
                {detail.calls.map((call) => (
                  <tr key={call.index} className={call.ok ? "" : "failed"}>
                    <td>{call.index}</td>
                    <td className="mono">{call.elapsed_seconds.toFixed(2)}s</td>
                    <td title={call.event_title ?? ""}>{call.event_title ?? "—"}</td>
                    <td className="preview">{call.error || call.preview || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
