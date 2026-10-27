import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import type { RunStatus } from "../types";
import { BaseOptionsForm } from "../components/BaseOptionsForm";
import { errorMessage } from "../lib/options";
import { SIMULATED_PLATFORMS } from "../lib/platforms";
import { buildOptions, type SharedScenarioProps } from "./SingleTargetScenario";

/**
 * 场景 2：多人协同分发（单平台多角色）。
 *
 * 与单人定向分发的关键区别：必须选定一个 dispatch_platform，选人池只取
 * 该平台已有的数字人；主选 / 备选 / 互动 / 扩散 / 支持角色由
 * Selector 自动配齐。
 */
export function MultiUserScenario(props: SharedScenarioProps) {
  const { eventPayload, isRunning, onStart, onError } = props;
  const [dispatchPlatform, setDispatchPlatform] = useState<string>(
    SIMULATED_PLATFORMS[0].value,
  );
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    // 注意：``eventPayload === null`` 是合法的——``eventSource === "image"`` 时
    // 由 ``props.submit`` 直接走 ``/api/runs/image`` + FormData，不需要事件 payload。
    if (!dispatchPlatform) {
      onError("多人协同分发必须指定一个分发平台。");
      return;
    }
    setBusy(true);
    try {
      const options = buildOptions(props, {
        dispatch_mode: "multi_user",
        dispatch_platform: dispatchPlatform,
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
        在一个模拟平台上选出多角色数字人小队（首发 / 互动 / 扩散 / 支持），
        共同完成本次事件的全链路模拟分发。
      </div>

      <label className="wide">
        分发平台（多人在此平台内协同）
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

      <div className="sub-section">
        <h3>小队人数与传播窗口</h3>
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
          // 多人协同：用户只需指定"平台 + 小队人数"。风险等级 / 传播窗口 /
          // 频率上限 / Profile 限制保持默认值，不在本场景暴露。
          showMaxSelectedNodes={true}
          showAllowedPlatforms={false}
          showAdvancedOptions={false}
        />
      </div>

      <button className="primary-button" type="submit" disabled={busy}>
        {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
        运行：多人协同分发
      </button>
    </form>
  );
}