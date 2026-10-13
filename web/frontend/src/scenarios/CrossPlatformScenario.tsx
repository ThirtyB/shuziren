import { useState } from "react";
import { Loader2, Play, Plus, Trash2 } from "lucide-react";
import type { PlatformSpecPayload, RunStatus } from "../types";
import { BaseOptionsForm } from "../components/BaseOptionsForm";
import { postJson } from "../lib/api";
import { errorMessage } from "../lib/options";
import { SIMULATED_PLATFORMS } from "../lib/platforms";
import { buildOptions, type SharedScenarioProps } from "./SingleTargetScenario";

/**
 * 场景 3：跨平台并行分发。
 *
 * 与多人协同分发的关键区别：每个平台独立指定人数，PlatformDispatcher 按
 * 平台自身 requested_count 在该平台已有的数字人中独立选人，正文与动作
 * 由各平台 content_plan 模板独立生成，输出 JSON 不含旧版五维调度策略。
 */
export function CrossPlatformScenario(props: SharedScenarioProps) {
  const { eventPayload, isRunning, onStart, onError } = props;
  const [specs, setSpecs] = useState<PlatformSpecPayload[]>([
    { platform_key: SIMULATED_PLATFORMS[0].key, requested_count: 3 },
    { platform_key: SIMULATED_PLATFORMS[2].key, requested_count: 3 },
  ]);
  const [busy, setBusy] = useState(false);

  function updateSpec(index: number, patch: Partial<PlatformSpecPayload>) {
    setSpecs((prev) => prev.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  }
  function addSpec() {
    setSpecs((prev) => [
      ...prev,
      { platform_key: SIMULATED_PLATFORMS[1].key, requested_count: 2 },
    ]);
  }
  function removeSpec(index: number) {
    setSpecs((prev) => (prev.length > 1 ? prev.filter((_, i) => i !== index) : prev));
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!eventPayload) {
      onError("请先在上方选择或填写事件。");
      return;
    }
    const cleaned: PlatformSpecPayload[] = specs
      .map((item) => ({ ...item, requested_count: Number(item.requested_count) || 0 }))
      .filter((item) => item.requested_count > 0);
    if (!cleaned.length) {
      onError("跨平台并行分发至少需要一个有效平台配置（requested_count >= 1）。");
      return;
    }
    const seen = new Set<string>();
    for (const item of cleaned) {
      if (seen.has(item.platform_key)) {
        onError(`平台 ${item.platform_key} 重复：每个平台只能出现一次。`);
        return;
      }
      seen.add(item.platform_key);
    }
    setBusy(true);
    try {
      const options = buildOptions(props, {
        dispatch_mode: "cross_platform",
        platform_specs: cleaned,
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
        每个平台按自己的 requested_count 独立选人；不同平台可以重复出现不同的
        数字人（前提是该数字人已写入两个平台的画像），但每个平台只能配置一次。
      </div>

      <div className="sub-section">
        <div className="sub-section-head">
          <h3>平台人数配置</h3>
          <button type="button" className="icon-button" onClick={addSpec}>
            <Plus size={16} />
            添加平台
          </button>
        </div>
        <div className="platform-spec-table">
          {specs.map((item, index) => (
            <div key={index} className="platform-spec-row">
              <select
                value={item.platform_key}
                onChange={(event) => updateSpec(index, { platform_key: event.target.value })}
              >
                {SIMULATED_PLATFORMS.map((opt) => (
                  <option key={opt.key} value={opt.key}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <input
                type="number"
                min={1}
                value={item.requested_count}
                onChange={(event) =>
                  updateSpec(index, { requested_count: Number(event.target.value) })
                }
              />
              <span className="unit">人</span>
              <button
                type="button"
                className="icon-button ghost"
                onClick={() => removeSpec(index)}
                disabled={specs.length <= 1}
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
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
          // 跨平台：每个平台独立 requested_count（已在专属参数里配置）。
          showMaxSelectedNodes={false}
          showAllowedPlatforms={false}
        />
      </div>

      <button className="primary-button" type="submit" disabled={busy}>
        {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
        运行：跨平台并行分发
      </button>
    </form>
  );
}