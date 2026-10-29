import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import type { RunOptions, RunStatus } from "../types";
import { BaseOptionsForm } from "../components/BaseOptionsForm";
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
  // 特殊支持节点参与开关与数量（所有分发模式共用）。
  waterArmyEnabled: boolean;
  setWaterArmyEnabled: (value: boolean) => void;
  waterArmyCount: number;
  setWaterArmyCount: (value: number) => void;
  useLlm: boolean;
  isRunning: boolean;
  /**
   * 提交运行：由父组件 ``ConsoleRoute`` 提供，分流到对应的后端端点。
   *  * ``eventSource="user"|"system"`` 时，``event`` 是表单/系统事件 payload；
   *  * 用户输入 + 多人/跨平台时，``submit`` 内部改用文字（可附图片）提交。
   * Scenario 不感知事件来源的具体实现。
   */
  submit: (event: Record<string, unknown> | null, options: RunOptions) => Promise<RunStatus>;
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
  const [singleTargetMode, setSingleTargetMode] = useState<"self_operate" | "attention">("self_operate");
  const [oldPostId, setOldPostId] = useState("POST-0001");

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    // ``eventPayload`` 由父组件统一分流校验（文字/图片/系统事件），这里只校验本场景专属参数。
    // 不再做"必须有 event"的短路校验，统一由 ``ConsoleRoute`` 的 runViaSource 处理。
    if (!targetUserId.trim()) {
      onError("单人定向分发必须填写目标数字人 user_id。");
      return;
    }
    setBusy(true);
    try {
      const options = buildOptions(props, {
        dispatch_mode: "single_target",
        target_user_id: targetUserId.trim(),
        single_target_mode: singleTargetMode,
        target_old_post_id: singleTargetMode === "attention" ? oldPostId.trim() || null : null,
      });
      const run = await props.submit(eventPayload, options);
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
        {singleTargetMode === "attention"
          ? "多数字人协同，全程 @ 指定路人（被动、不反应），使其注意到事件。只需提供路人、时间窗口与事件。"
          : "指定一个数字人作为本次事件的主发布节点。系统以该数字人为核心，自主完成「节点激活时期发布 → 节点激活时期自赞 → 信息扩散时期自转发 → 共识形成阶段自赞并再发承接帖」的自运营闭环。只需提供数字人、时间窗口与事件。"}
      </div>

      <label className="wide">
        目标数字人 user_id
        <input
          value={targetUserId}
          onChange={(event) => setTargetUserId(event.target.value)}
          placeholder="例如 110133（必填，平台由系统从该数字人画像自动推断）"
        />
      </label>

      <div className="segmented">
        <button
          type="button"
          className={singleTargetMode === "self_operate" ? "active" : ""}
          onClick={() => setSingleTargetMode("self_operate")}
        >
          单人分发
        </button>
        <button
          type="button"
          className={singleTargetMode === "attention" ? "active" : ""}
          onClick={() => setSingleTargetMode("attention")}
        >
          数字人注意
        </button>
      </div>

      {singleTargetMode === "attention" && (
        <label className="wide">
          被点名路人旧帖编号（仅用于 @ 点名）
          <input
            value={oldPostId}
            onChange={(event) => setOldPostId(event.target.value)}
            placeholder="例如 POST-0001"
          />
        </label>
      )}

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
          waterArmyEnabled={props.waterArmyEnabled}
          setWaterArmyEnabled={props.setWaterArmyEnabled}
          waterArmyCount={props.waterArmyCount}
          setWaterArmyCount={props.setWaterArmyCount}
          // 单人分发：仅保留“时间窗口”，其余进阶参数按默认值即可。
          showMaxSelectedNodes={false}
          showAllowedPlatforms={false}
          showAdvancedOptions={false}
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
    single_target_mode: overrides.single_target_mode ?? "self_operate",
    target_old_post_id: overrides.target_old_post_id ?? null,
    dispatch_platform: overrides.dispatch_platform ?? null,
    platform_specs: overrides.platform_specs ?? [],
    water_army_enabled: props.waterArmyEnabled,
    water_army_count: props.waterArmyEnabled ? props.waterArmyCount : 0,
  };
}