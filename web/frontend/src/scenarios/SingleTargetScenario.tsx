import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import type { RunOptions, RunStatus } from "../types";
import { BaseOptionsForm } from "../components/BaseOptionsForm";
import { postJson } from "../lib/api";
import { errorMessage } from "../lib/options";
import { SIMULATED_PLATFORMS } from "../lib/platforms";

/**
 * 三个分发场景共享的 props：基础运行参数 + 事件来源结果。
 *
 * 事件来源（``eventPayload``）由 ``ConsoleRoute`` 归一化后传入：
 *  - ``eventSource="user"`` 时 ``eventPayload`` 来自 ``UserEventForm`` 的表单/JSON；
 *  - ``eventSource="system"`` 时 ``eventPayload`` 来自 ``/api/events/{id}`` 的全字段。
 * 两种来源在 Scenario 内部不再做区分。
 */
export type SharedScenarioProps = {
  eventPayload: Record<string, unknown> | null;
  profileLimit: string;
  setProfileLimit: (value: string) => void;
  maxSelectedNodes: number;
  setMaxSelectedNodes: (value: number) => void;
  riskLevel: string;
  setRiskLevel: (value: string) => void;
  campaignWindowHours: number;
  setCampaignWindowHours: (value: number) => void;
  maxFrequencyPerDay: number;
  setMaxFrequencyPerDay: (value: number) => void;
  allowedPlatforms: string;
  setAllowedPlatforms: (value: string) => void;
  useLlm: boolean;
  isRunning: boolean;
  onStart: (run: RunStatus) => void;
  onError: (message: string) => void;
};

export type SingleTargetScenarioProps = SharedScenarioProps;

/**
 * 场景 1：单人定向分发。
 *
 * 关键参数：
 *  - target_user_id：唯一数字人；其余候选池参数即使配置了也不参与选人。
 *  - dispatch_platform / platform_specs：在提交前被后端 _validate_dispatch_options 拒绝。
 */
export function SingleTargetScenario(props: SingleTargetScenarioProps) {
  const { eventPayload, isRunning, onStart, onError } = props;
  const [targetUserId, setTargetUserId] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!eventPayload) {
      onError("请先在上方选择或填写事件。");
      return;
    }
    if (!targetUserId.trim()) {
      onError("单人定向分发必须填写目标数字人 user_id。");
      return;
    }
    setBusy(true);
    try {
      const options = buildOptions(props, {
        dispatch_mode: "single_target",
        target_user_id: targetUserId.trim(),
      });
      const run = await postJson<RunStatus>("/api/runs/text", {
        event: eventPayload,
        options,
      });
      onStart(run);
    } catch (err) {
      onError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="stack" onSubmit={handleSubmit}>
      <div className="scenario-hint">
        指定一个数字人作为本次事件的唯一首发者；其余数字人（互动/扩散/支持）由
        评估流水线以该数字人为锚点生成。
      </div>

      <label className="wide">
        目标数字人 user_id
        <input
          value={targetUserId}
          onChange={(event) => setTargetUserId(event.target.value)}
          placeholder="例如 110133（必填，平台由系统从该数字人画像自动推断）"
        />
      </label>

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
          // 单人定向：1 人固定、无选人池（目标数字人本身的平台由后端推断）。
          showMaxSelectedNodes={false}
          showAllowedPlatforms={false}
        />
      </div>

      <button className="primary-button" type="submit" disabled={busy}>
        {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
        运行：单人定向分发
      </button>
    </form>
  );
}

export function buildOptions(
  props: SharedScenarioProps,
  overrides: Partial<RunOptions> & { dispatch_mode: RunOptions["dispatch_mode"] },
): RunOptions {
  const platforms = props.allowedPlatforms
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return {
    profile_limit: props.profileLimit.trim() ? Number(props.profileLimit) : null,
    max_selected_nodes: props.maxSelectedNodes,
    risk_level: (props.riskLevel || null) as RunOptions["risk_level"],
    campaign_window_hours: props.campaignWindowHours,
    max_frequency_per_day: props.maxFrequencyPerDay,
    allowed_platforms: platforms.length
      ? platforms
      : SIMULATED_PLATFORMS.map((item) => item.value),
    use_llm: props.useLlm,
    event_limit: null,
    event_id: null,
    dispatch_mode: overrides.dispatch_mode,
    target_user_id: overrides.target_user_id ?? null,
    dispatch_platform: overrides.dispatch_platform ?? null,
    platform_specs: overrides.platform_specs ?? [],
  };
}