import { Activity, Loader2 } from "lucide-react";
import type { BenchmarkEventEntry, BenchmarkResponse, BenchmarkState } from "../types";
import { BenchmarkSummaryView } from "./BenchmarkSummaryView";
import { SectionTitle } from "./SectionTitle";

const STATE_LABEL: Record<BenchmarkState, string> = {
  queued: "排队中",
  running: "测试中",
  completed: "完成",
  failed: "失败",
};

/** 每个测试点（一轮）在详情里的展示形状 —— 把 ``runs`` 与 ``events`` 合并。 */
type TestPoint = {
  iteration: number;
  event: BenchmarkEventEntry | null;
  status: "waiting" | "ok" | "failed";
  seconds: number | null;
};

/**
 * 单条压测的详情面板：进度 / 每个测试点随机到的事件 / 阶段柱状图 / 每轮明细。
 *
 * 因为压测是异步任务，"测试中"时 ``summary`` 可能还是 ``null``（还没成功跑完
 * 任何一轮），此时只展示进度与逐个测试点的事件，柱状图等有汇总数据再出现。
 */
export function BenchmarkDetailPanel({
  benchmark,
  loading,
  error,
}: {
  benchmark: BenchmarkResponse | null;
  loading: boolean;
  error: string;
}) {
  if (!benchmark) {
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

  const points = buildTestPoints(benchmark);
  const finished = benchmark.completed + benchmark.failed;
  const progress =
    benchmark.iterations > 0 ? Math.round((finished / benchmark.iterations) * 100) : 0;
  const running = benchmark.state === "queued" || benchmark.state === "running";

  return (
    <section className="panel detail-panel">
      <SectionTitle
        title="测试详情"
        extra={
          <span className={`benchmark-state-chip state-${benchmark.state}`}>
            {running && <Loader2 className="spin" size={12} />}
            {STATE_LABEL[benchmark.state]}
          </span>
        }
      />

      {error && <div className="message error">{error}</div>}

      <div className="status-row">
        <span className="run-id">{benchmark.benchmark_id}</span>
        {benchmark.label && <span className="benchmark-label">{benchmark.label}</span>}
      </div>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="benchmark-progress-text small">
        {finished}/{benchmark.iterations} 个测试点已完成
        {benchmark.failed > 0 ? `（失败 ${benchmark.failed}）` : ""}
        {" · "}
        LLM {benchmark.options.use_llm ? "开启" : "关闭"}
      </div>

      {benchmark.error && <div className="message error">{benchmark.error}</div>}

      {benchmark.summary && (
        <div className="summary-band">
          <div>
            <span>总用时</span>
            <strong>{formatSeconds(benchmark.summary.total_seconds)}</strong>
          </div>
          <div>
            <span>单轮平均</span>
            <strong>{formatSeconds(benchmark.summary.average_seconds)}</strong>
          </div>
          <div>
            <span>单轮最快</span>
            <strong>{formatSeconds(benchmark.summary.min_seconds)}</strong>
          </div>
          <div>
            <span>单轮最慢</span>
            <strong>{formatSeconds(benchmark.summary.max_seconds)}</strong>
          </div>
          <div>
            <span>成功 / 总轮</span>
            <strong>
              {benchmark.summary.successful_iterations} / {benchmark.iterations}
            </strong>
          </div>
        </div>
      )}

      {benchmark.summary && <BenchmarkSummaryView summary={benchmark.summary} />}

      <div className="benchmark-runs">
        <h3>
          <Activity size={14} />
          测试点（每轮随机抽取的事件）
        </h3>
        <table className="benchmark-runs-table">
          <thead>
            <tr>
              <th>轮次</th>
              <th>事件</th>
              <th>状态</th>
              <th>本轮用时</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point) => (
              <tr key={point.iteration}>
                <td className="mono">#{point.iteration}</td>
                <td>
                  {point.event ? (
                    <div className="benchmark-run-event">
                      <strong>{point.event.event_title ?? point.event.event_id}</strong>
                      <small className="mono">
                        {point.event.event_id}
                        {point.event.domain ? ` · ${point.event.domain}` : ""}
                      </small>
                    </div>
                  ) : (
                    <em className="muted">待抽取</em>
                  )}
                </td>
                <td>
                  {point.status === "ok" && <span className="status-chip ok">成功</span>}
                  {point.status === "failed" && (
                    <span className="status-chip fail">失败</span>
                  )}
                  {point.status === "waiting" && (
                    <span className="status-chip pending">测试中</span>
                  )}
                </td>
                <td className="mono">{formatSeconds(point.seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * 把 ``runs``（已跑完的轮次）与 ``events``（已抽到的事件）对齐成逐轮视图：
 * 已抽到但还没跑完的轮次显示为"测试中"。
 */
function buildTestPoints(benchmark: BenchmarkResponse): TestPoint[] {
  const byIteration = new Map<number, TestPoint>();

  benchmark.events.forEach((event, index) => {
    const iteration = event.iteration ?? index + 1;
    byIteration.set(iteration, { iteration, event, status: "waiting", seconds: null });
  });

  benchmark.runs.forEach((run) => {
    byIteration.set(run.iteration, {
      iteration: run.iteration,
      event: run.event ?? byIteration.get(run.iteration)?.event ?? null,
      status: run.ok ? "ok" : "failed",
      seconds: run.elapsed_seconds ?? null,
    });
  });

  const total = Math.max(benchmark.iterations, byIteration.size);
  return Array.from({ length: total }, (_, index) => {
    const iteration = index + 1;
    return (
      byIteration.get(iteration) ?? {
        iteration,
        event: null,
        status: "waiting" as const,
        seconds: null,
      }
    );
  });
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`;
  if (value < 60) return `${value.toFixed(2)} s`;
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}m ${seconds.toFixed(1)}s`;
}
