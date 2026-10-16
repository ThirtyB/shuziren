import { useState } from "react";
import { FlaskConical, Loader2, Zap } from "lucide-react";
import type { BenchmarkResponse, RunOptions } from "../types";
import { runBenchmark } from "../lib/api";
import { errorMessage } from "../lib/options";
import { SIMULATED_PLATFORMS } from "../lib/platforms";
import { buildOptions, type SharedScenarioProps } from "./SingleTargetScenario";

/** 压测场景的 props：在共享 props 基础上追加"提交成功"回调。 */
export type BenchmarkScenarioProps = SharedScenarioProps & {
  /**
   * 提交成功后回调（后端异步返回 queued 状态）。
   * 父组件用它把新测试立刻插入"测试历史"并选中查看。
   */
  onStarted?: (benchmark: BenchmarkResponse) => void;
};

/**
 * 一键压测场景（默认 5 轮）：每次从 200 条 hot_event 里**不放回**地随机抽
 * N 条事件，串行跑 N 轮，汇总各阶段平均用时与总用时。
 *
 * 关键行为：
 *
 *  * **异步不锁按钮** —— POST 立即返回 queued；按钮只在本机提交的那一瞬间
 *    显示 loading，随后立刻恢复。上一次测试还在跑时也可以继续点，新任务在
 *    后端线程池里排队，并在"测试历史"里以"排队中"出现。
 *  * **每个测试点事件都不同** —— 随机不放回抽样，避免同一条事件的画像 / LLM
 *    缓存被预热后拉低后续轮次耗时。
 *  * **LLM 开关只跑当前这一档** —— 想对比开 / 关差异，切换顶栏开关后再点一次。
 */
export function BenchmarkScenario(props: BenchmarkScenarioProps) {
  const { onError, onStarted, useLlm } = props;
  const [iterations, setIterations] = useState(5);
  const [label, setLabel] = useState("压测");
  const [dispatchPlatform, setDispatchPlatform] = useState<string>(
    SIMULATED_PLATFORMS[0].value,
  );
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit() {
    setSubmitting(true);
    try {
      // 压测固定走 multi_user（选人池 / 平台约束最宽松），事件交给后端随机抽。
      const overrides: Parameters<typeof buildOptions>[1] = {
        dispatch_mode: "multi_user",
        event_limit: 1,
        event_id: null,
      };
      overrides.dispatch_platform = dispatchPlatform;
      const options: RunOptions = buildOptions(props, overrides);
      const benchmark = await runBenchmark({
        options,
        iterations,
        event_id: null,
        label: label.trim() || null,
      });
      onStarted?.(benchmark);
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="stack">
      <div className="scenario-hint">
        每次从系统内置的 200 条 hot_event 里随机抽 {iterations} 条**互不相同**的事件，
        串行跑 {iterations} 轮，统计"事件解析 / 特征构建 / 打分 / 选人 / 策略生成 /
        内容生成"各环节平均用时与总用时。
      </div>

      <div className="form-grid">
        <label>
          压测轮数
          <input
            type="number"
            min={1}
            max={10}
            value={iterations}
            onChange={(event) => setIterations(Number(event.target.value))}
          />
        </label>
        <label>
          测试标签
          <input
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="例如 压测"
          />
        </label>
        <label>
          多人分发平台
          <select
            value={dispatchPlatform}
            onChange={(event) => setDispatchPlatform(event.target.value)}
          >
            {SIMULATED_PLATFORMS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="scenario-hint small">
        当前顶栏 LLM 开关 = <strong>{useLlm ? "开启" : "关闭"}</strong>
        ，本次测试只跑这一档；想对比 LLM 开关差异，切换开关后再点一次。
        <br />
        提交后立即返回，测试在后台执行 —— 可以连续发起多次，新的测试会自动排队。
      </div>

      <button className="primary-button" type="button" onClick={handleSubmit} disabled={submitting}>
        {submitting ? (
          <Loader2 className="spin" size={18} />
        ) : (
          <Zap size={18} />
        )}
        开始测试（{iterations} 轮）
      </button>

      <div className="scenario-hint small benchmark-footnote">
        <FlaskConical size={13} />
        结果会写入 <span className="mono">outputs/web_runs/benchmark_runs/&lt;id&gt;/</span>
        ，与主控制台产物隔离。
      </div>
    </div>
  );
}
