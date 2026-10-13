import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import type { RunStatus } from "../types";
import { BaseOptionsForm } from "../components/BaseOptionsForm";
import { postJson } from "../lib/api";
import { errorMessage } from "../lib/options";
import { SIMULATED_PLATFORMS } from "../lib/platforms";
import { buildOptions, type SharedScenarioProps } from "./SingleTargetScenario";

/**
 * 批量评测场景：在内置事件库（默认 200 条 hot_event_opinion_variants）上
 * 重复运行所选分发场景。
 *
 * 与单事件场景的差别：事件数据来自后端 default_events_path，触发 /api/eval-runs，
 * 共享同一份 RunOptions.dispatch_mode，方便一次性回测三种场景的差异。
 */
export function BatchEvalScenario(props: SharedScenarioProps) {
  const { isRunning, onStart, onError } = props;
  const [eventLimit, setEventLimit] = useState(10);
  const [eventId, setEventId] = useState("");
  // 批量评测默认走 multi_user，可在下拉中切换成任意一种分发场景。
  const [dispatchMode, setDispatchMode] = useState<
    "multi_user" | "single_target" | "cross_platform"
  >("multi_user");
  const [dispatchPlatform, setDispatchPlatform] = useState<string>(SIMULATED_PLATFORMS[0].value);
  const [targetUserId, setTargetUserId] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSubmit() {
    setBusy(true);
    try {
      const overrides: Parameters<typeof buildOptions>[1] = {
        dispatch_mode: dispatchMode,
        event_limit: eventLimit,
        event_id: eventId.trim() || null,
      };
      if (dispatchMode === "multi_user") overrides.dispatch_platform = dispatchPlatform;
      if (dispatchMode === "single_target") overrides.target_user_id = targetUserId.trim() || null;
      if (dispatchMode === "cross_platform") {
        overrides.platform_specs = [
          { platform_key: SIMULATED_PLATFORMS[0].key, requested_count: 3 },
          { platform_key: SIMULATED_PLATFORMS[2].key, requested_count: 3 },
        ];
      }
      const options = buildOptions(props, overrides);
      const run = await postJson<RunStatus>("/api/eval-runs", { options });
      onStart(run);
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <div className="scenario-hint">
        一次性跑多条 hot_event，固定输出到 eval/output/&lt;mode&gt;/run_NNN/。
        切换下方"分发场景"可批量验证单平台 / 跨平台的差异。
      </div>

      <div className="form-grid">
        <label>
          事件数量
          <input
            type="number"
            min={1}
            max={200}
            value={eventLimit}
            onChange={(event) => setEventLimit(Number(event.target.value))}
          />
        </label>
        <label>
          指定事件 ID
          <input
            value={eventId}
            onChange={(event) => setEventId(event.target.value)}
            placeholder="留空跑前 N 条"
          />
        </label>
        <label>
          分发场景
          <select
            value={dispatchMode}
            onChange={(event) => setDispatchMode(event.target.value as typeof dispatchMode)}
          >
            <option value="multi_user">多人协同分发（默认）</option>
            <option value="single_target">单人定向分发</option>
            <option value="cross_platform">跨平台并行分发</option>
          </select>
        </label>
        {dispatchMode === "multi_user" && (
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
        )}
        {dispatchMode === "single_target" && (
          <label>
            目标 user_id
            <input
              value={targetUserId}
              onChange={(event) => setTargetUserId(event.target.value)}
              placeholder="单人分发必填"
            />
          </label>
        )}
        {dispatchMode === "cross_platform" && (
          <label className="wide">
            跨平台配置（按 facebook=3 / twitter=3 的默认示例）
          </label>
        )}
      </div>

      <div className="sub-section">
        <h3>基础运行参数</h3>
        <BaseOptionsForm
          maxSelectedNodes={props.maxSelectedNodes}
          setMaxSelectedNodes={props.setMaxSelectedNodes}
          riskLevel={props.riskLevel}
          setRiskLevel={props.setRiskLevel}
          campaignWindowHours={props.campaignWindowHours}
          setCampaignWindowHours={props.setCampaignWindowHours}
          maxFrequencyPerDay={props.maxFrequencyPerDay}
          setMaxFrequencyPerDay={props.setMaxFrequencyPerDay}
          profileLimit={props.profileLimit}
          setProfileLimit={props.setProfileLimit}
          allowedPlatforms={props.allowedPlatforms}
          setAllowedPlatforms={props.setAllowedPlatforms}
        />
      </div>

      <button
        className="primary-button"
        type="button"
        onClick={handleSubmit}
        disabled={busy}
      >
        {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
        一键运行 Eval
      </button>
    </div>
  );
}
