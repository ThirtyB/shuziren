import { Activity, Loader2 } from "lucide-react";
import type {
  SingleTargetMode,
  SubTaskKind,
  SubTaskSpec,
  TestRunPerEvent,
  TestRunResponse,
  TestRunState,
  TestRunSubTaskResult,
} from "../types";
import { BenchmarkSummaryView } from "./BenchmarkSummaryView";
import { SectionTitle } from "./SectionTitle";

const STATE_LABEL: Record<TestRunState, string> = {
  queued: "排队中",
  running: "测试中",
  completed: "完成",
  failed: "失败",
};
const KIND_LABELS: Record<SubTaskKind, string> = {
  single_target: "单人定向",
  multi_user: "多人协同",
  cross_platform: "多平台并行",
};
const MODE_LABELS: Record<SingleTargetMode, string> = {
  self_operate: "自运营闭环",
  attention: "数字人注意",
};

/**
 * 单条总体测试的详情面板：进度 / 总体汇总计时 / 每个子任务的配置与计时明细。
 * 因为测试是异步任务，"测试中"时 ``summary`` 可能为 null，此时只展示进度。
 */
export function TestRunDetailPanel({
  run,
  loading,
  error,
}: {
  run: TestRunResponse | null;
  loading: boolean;
  error: string;
}) {
  if (!run) {
    return (
      <section className="panel detail-panel">
        <SectionTitle title="测试详情" />
        {error && <div className="message error">{error}</div>}
        <div className="empty-state">
          {loading ? "加载测试详情中…" : "从左侧测试历史里点一条记录查看详情。"}
        </div>
      </section>
    );
  }

  const finished = run.completed + run.failed;
  const progress = run.total_subtasks > 0 ? Math.round((finished / run.total_subtasks) * 100) : 0;
  const running = run.state === "queued" || run.state === "running";

  return (
    <section className="panel detail-panel">
      <SectionTitle
        title="测试详情"
        extra={
          <span className={`benchmark-state-chip state-${run.state}`}>
            {running && <Loader2 className="spin" size={12} />}
            {STATE_LABEL[run.state]}
          </span>
        }
      />

      {error && <div className="message error">{error}</div>}

      <div className="status-row">
        <span className="run-id">{run.test_run_id}</span>
        {run.label && <span className="benchmark-label">{run.label}</span>}
      </div>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="benchmark-progress-text small">
        {finished}/{run.total_subtasks} 个子任务已完成
        {run.failed > 0 ? `（失败 ${run.failed}）` : ""}
      </div>

      {run.error && <div className="message error">{run.error}</div>}

      {run.summary && (
        <div className="summary-band">
          <div>
            <span>总用时</span>
            <strong>{formatSeconds(run.summary.total_seconds)}</strong>
          </div>
          <div>
            <span>单事件平均</span>
            <strong>{formatSeconds(run.summary.average_seconds)}</strong>
          </div>
          <div>
            <span>单事件最快</span>
            <strong>{formatSeconds(run.summary.min_seconds)}</strong>
          </div>
          <div>
            <span>单事件最慢</span>
            <strong>{formatSeconds(run.summary.max_seconds)}</strong>
          </div>
          <div>
            <span>成功 / 总事件</span>
            <strong>
              {run.summary.successful_iterations} / {run.summary.iterations}
            </strong>
          </div>
        </div>
      )}

      {run.summary && <BenchmarkSummaryView summary={run.summary} />}

      <div className="benchmark-runs">
        <h3>
          <Activity size={14} />
          子任务（每个独立运行，事件随机）
        </h3>
        {run.subtasks.map((st) => (
          <SubTaskBlock key={st.index} subtask={st} />
        ))}
      </div>
    </section>
  );
}

function SubTaskBlock({ subtask }: { subtask: TestRunSubTaskResult }) {
  return (
    <div className="subtask-block">
      <div className="subtask-block-head">
        <span className="subtask-index">#{subtask.index + 1}</span>
        <span className="status-chip">{KIND_LABELS[subtask.kind]}</span>
        {subtask.kind === "single_target" && subtask.single_target_mode && (
          <span className="status-chip">{MODE_LABELS[subtask.single_target_mode]}</span>
        )}
        {subtask.with_image && <span className="status-chip llm-yes">含图</span>}
        <span className="mono">{subtask.event_count} 条随机事件</span>
        {subtask.ok ? (
          <span className="status-chip ok">成功</span>
        ) : (
          <span className="status-chip fail">失败</span>
        )}
        {subtask.error && <span className="subtask-error">{subtask.error}</span>}
      </div>
      <div className="subtask-meta small mono">
        {subtask.kind === "single_target" && subtask.target_user_id && (
          <>目标: {subtask.target_user_id} · </>
        )}
        平台: {subtask.platform_specs.map((p) => `${p.platform_key}×${p.requested_count}`).join(", ") || "-"}
      </div>
      {subtask.subtask_summary && <BenchmarkSummaryView summary={subtask.subtask_summary} />}
      {subtask.per_event.length > 0 && (
        <div className="per-event-list">
          {subtask.per_event.map((ev, i) => (
            <PerEventCard key={i} ev={ev} />
          ))}
        </div>
      )}
    </div>
  );
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`;
  if (value < 60) return `${value.toFixed(2)} s`;
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}m ${seconds.toFixed(1)}s`;
}

/**
 * 单次事件的具体结果卡片：状态 / 真实墙钟耗时 / 选取数字人 / 支持节点 / 各平台选中情况 /
 * 内容生成诊断，并可展开查看完整五维调度策略产物（``strategy_output``）。
 */
function PerEventCard({ ev }: { ev: TestRunPerEvent }) {
  const ok = !ev.error && ev.summary != null;
  const secs =
    ev.elapsed_seconds ??
    Object.values(ev.stage_times ?? {}).reduce<number>((acc, b) => acc + (b || 0), 0);
  const summary = ev.summary;
  const contentLLM = summary?.content_llm as Record<string, unknown> | undefined;
  const support = summary?.support_nodes;
  const target = summary?.target_dispatch as Record<string, unknown> | undefined;
  const perPlatform = summary?.per_platform as Array<Record<string, unknown>> | undefined;
  const selectedIds: string[] = summary?.selected_digital_human_ids ?? [];
  const warnings: string[] = Array.isArray(target?.warnings)
    ? (target?.warnings as string[])
    : [];
  return (
    <div className="per-event-card">
      <div className="per-event-head">
        <strong>{ev.event_title ?? ev.event_id}</strong>
        <span className="mono small">{ev.event_id}</span>
        {ok ? (
          <span className="status-chip ok">成功</span>
        ) : (
          <span className="status-chip fail">失败</span>
        )}
        <span className="mono">{formatSeconds(secs)}</span>
      </div>
      {ev.error && <div className="message error">{ev.error}</div>}
      {summary && (
        <div className="per-event-body">
          <div className="per-event-row">
            <span className="kv-label">选取数字人</span>
            <span>
              {selectedIds.length} 个
              {selectedIds.length ? "：" + selectedIds.join("、") : ""}
            </span>
          </div>
          {warnings.length > 0 && (
            <div className="per-event-row warn">
              <span className="kv-label">定向提示</span>
              <span>{warnings.join("；")}</span>
            </div>
          )}
          {support && (
            <div className="per-event-row">
              <span className="kv-label">支持节点</span>
              <span>{support.enabled ? `已启用 ${support.count ?? 0} 人` : "未启用"}</span>
            </div>
          )}
          {perPlatform && perPlatform.length > 0 && (
            <div className="per-event-row">
              <span className="kv-label">各平台</span>
              <span>
                {perPlatform
                  .map(
                    (p) =>
                      `${p.platform_key}: ${p.selected_count ?? 0}/${p.requested_count ?? 0}`,
                  )
                  .join("，")}
              </span>
            </div>
          )}
          {contentLLM && (
            <div className="per-event-row">
              <span className="kv-label">内容生成</span>
              <span>{renderContentLLM(contentLLM)}</span>
            </div>
          )}
          {summary.strategy_output && (
            <details className="per-event-details">
              <summary>生成策略全文（点击展开）</summary>
              <pre className="json-preview">
                {truncate(JSON.stringify(summary.strategy_output, null, 2), 3000)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function renderContentLLM(meta: Record<string, unknown>): string {
  if (meta.success === true) return "大模型生成成功";
  const reason =
    typeof meta.fallback_reason === "string"
      ? meta.fallback_reason
      : typeof meta.fallbackReason === "string"
        ? meta.fallbackReason
        : null;
  if (reason) return `降级：${reason}`;
  return "无内容生成诊断";
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}\n…（已截断）` : value;
}
