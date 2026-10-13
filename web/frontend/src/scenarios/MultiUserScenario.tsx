import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import type { RunStatus } from "../types";
import { BaseOptionsForm } from "../components/BaseOptionsForm";
import { postJson } from "../lib/api";
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
    if (!eventPayload) {
      onError("请先在上方选择或填写事件。");
      return;
    }
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
          // 多人协同：需要主选数量（小队人数），平台由 dispatchPlatform 决定。
          showMaxSelectedNodes={true}
          showAllowedPlatforms={false}
        />
      </div>

      <button className="primary-button" type="submit" disabled={busy}>
        {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
        运行：多人协同分发
      </button>
    </form>
  );
}