import type { BenchmarkSummary } from "../types";

/**
 * 5 次压测汇总视图：
 *
 *   - 顶部"每轮总用时"柱状图（per-iteration wall time）；
 *   - 中部"各阶段平均用时"柱状图，含最小 / 最大范围条；
 *   - 底部阶段数据明细表格，方便复制与对账。
 *
 * 柱状图用纯 SVG 实现（不引入 chart 依赖），柱高度按最大值等比缩放；条数随
 * 阶段数自适应。
 */
export function BenchmarkSummaryView({ summary }: { summary: BenchmarkSummary }) {
  const stageEntries = Object.entries(summary.stage_averages ?? {})
    .filter(([, seconds]) => Number.isFinite(seconds) && seconds >= 0)
    .sort(([, left], [, right]) => right - left);

  return (
    <div className="benchmark-summary-view">
      <div className="benchmark-chart-block">
        <h4>每轮总用时</h4>
        <BarChart
          bars={summary.per_iteration.map((item) => ({
            label: `#${item.iteration}`,
            value: item.seconds,
            tooltip: `第 ${item.iteration} 轮：${formatSeconds(item.seconds)}`,
          }))}
          unit="s"
        />
      </div>

      <div className="benchmark-chart-block">
        <h4>各阶段平均用时（带最小 / 最大区间）</h4>
        {stageEntries.length === 0 ? (
          <div className="empty-state">本组压测没有记录到阶段耗时。</div>
        ) : (
          <BarChart
            bars={stageEntries.map(([stage, avg]) => {
              const range = summary.stage_min_max?.[stage];
              const sampleCount = range?.samples?.length ?? 0;
              return {
                label: STAGE_LABELS[stage] ?? stage,
                value: avg,
                range,
                tooltip: range
                  ? `${stage} 平均 ${formatSeconds(avg)}（min ${formatSeconds(range.min)} / max ${formatSeconds(range.max)}，${sampleCount} 次）`
                  : `${stage} 平均 ${formatSeconds(avg)}`,
              };
            })}
            unit="s"
          />
        )}
      </div>

      <details className="benchmark-stage-detail">
        <summary>查看各阶段数据明细</summary>
        <table className="benchmark-runs-table">
          <thead>
            <tr>
              <th>阶段</th>
              <th>平均用时</th>
              <th>最小</th>
              <th>最大</th>
              <th>样本数</th>
              <th>采样明细</th>
            </tr>
          </thead>
          <tbody>
            {stageEntries.map(([stage, avg]) => {
              const range = summary.stage_min_max?.[stage];
              const samples: number[] = range?.samples ?? [];
              return (
                <tr key={stage}>
                  <td className="mono">{STAGE_LABELS[stage] ?? stage}</td>
                  <td className="mono">{formatSeconds(avg)}</td>
                  <td className="mono">{formatSeconds(range?.min)}</td>
                  <td className="mono">{formatSeconds(range?.max)}</td>
                  <td className="mono">{samples.length}</td>
                  <td className="mono benchmark-samples">
                    {samples.map((value: number, idx: number) => (
                      <span key={idx}>{formatSeconds(value)}</span>
                    ))}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </div>
  );
}

/**
 * 简易 SVG 柱状图。
 * - ``bars[i].value`` 是柱高度；按 max(value, range.max) 归一化；
 * - ``bars[i].range`` 时会额外叠一根灰色范围条；
 * - ``width`` 跟随 bar 数量自适应，最小 12px/柱；
 */
function BarChart({
  bars,
  unit,
}: {
  bars: Array<{
    label: string;
    value: number;
    range?: { min: number; max: number; samples: number[] };
    tooltip?: string;
  }>;
  unit: string;
}) {
  if (bars.length === 0) {
    return <div className="empty-state">暂无数据。</div>;
  }
  const maxValue = bars.reduce((current, item) => {
    const candidate = Math.max(item.value, item.range?.max ?? 0);
    return candidate > current ? candidate : current;
  }, 0);
  // 留出顶部空间；最小 0.1s 避免除零。
  const scaleMax = maxValue > 0 ? maxValue * 1.15 : 0.1;
  const barWidth = Math.max(38, Math.min(72, 720 / bars.length));
  const chartWidth = bars.length * barWidth + 24;
  const chartHeight = 200;
  const chartInnerHeight = chartHeight - 28;

  return (
    <div className="benchmark-bar-chart">
      <svg
        viewBox={`0 0 ${chartWidth} ${chartHeight}`}
        width="100%"
        height={chartHeight}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* 顶部网格线 */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const y = 8 + chartInnerHeight * (1 - ratio);
          return (
            <g key={ratio}>
              <line
                x1={20}
                x2={chartWidth - 4}
                y1={y}
                y2={y}
                stroke="#e2e8f0"
                strokeDasharray="2 4"
              />
              <text x={0} y={y + 3} fontSize="10" fill="#617080">
                {(scaleMax * ratio).toFixed(scaleMax >= 5 ? 0 : 1) + unit}
              </text>
            </g>
          );
        })}
        {bars.map((bar, idx) => {
          const x = 24 + idx * barWidth;
          const height = (bar.value / scaleMax) * chartInnerHeight;
          const y = 8 + chartInnerHeight - height;
          const rangeMax = bar.range?.max ?? 0;
          const rangeMin = bar.range?.min ?? 0;
          const rangeMaxHeight = (rangeMax / scaleMax) * chartInnerHeight;
          const rangeMinHeight = (rangeMin / scaleMax) * chartInnerHeight;
          const rangeY = 8 + chartInnerHeight - rangeMaxHeight;
          const tickY = 8 + chartInnerHeight - rangeMinHeight;
          return (
            <g key={`${bar.label}-${idx}`}>
              {bar.range && (
                <rect
                  x={x + 4}
                  width={barWidth - 16}
                  y={rangeY}
                  height={Math.max(rangeMaxHeight, 1)}
                  fill="#cbd5e1"
                  rx={3}
                />
              )}
              <rect
                x={x + 4}
                width={barWidth - 16}
                y={y}
                height={Math.max(height, 1)}
                fill="#0f766e"
                rx={3}
              >
                <title>{bar.tooltip ?? `${bar.label}${formatSeconds(bar.value)}`}</title>
              </rect>
              {bar.range && (
                <line
                  x1={x + 2}
                  x2={x + barWidth - 4}
                  y1={tickY}
                  y2={tickY}
                  stroke="#0f172a"
                />
              )}
              <text
                x={x + barWidth / 2}
                y={chartHeight - 12}
                fontSize="11"
                fill="#465767"
                textAnchor="middle"
              >
                {bar.label}
              </text>
              <text
                x={x + barWidth / 2}
                y={Math.max(y - 4, 12)}
                fontSize="11"
                fill="#10212f"
                textAnchor="middle"
              >
                {formatSeconds(bar.value)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  llm_client_init: "LLM 客户端初始化",
  hot_event_to_payload: "事件标准化",
  event_parser: "事件解析",
  feature_builder: "特征构建",
  scorer: "节点打分",
  selector: "数字人选人",
  strategy_generator: "策略生成",
  platform_dispatch: "跨平台分发",
  content_generation: "内容生成 (LLM)",
  write_output: "写产物 JSON",
  write_trace: "写 trace",
  pipeline_total: "Pipeline 合计",
  event_total: "单事件合计",
};

function formatSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 1) return `${(value * 1000).toFixed(0)}ms`;
  if (value < 60) return `${value.toFixed(2)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}m${seconds.toFixed(1)}s`;
}