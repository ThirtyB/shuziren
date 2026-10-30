import { useState } from "react";
import { Loader2, Play, Plus, Trash2 } from "lucide-react";
import type { PlatformSpecPayload, RunStatus } from "../types";
import { BaseOptionsForm } from "../components/BaseOptionsForm";
import { errorMessage } from "../lib/options";
import { SIMULATED_PLATFORMS } from "../lib/platforms";
import { buildOptions, type SharedScenarioProps } from "./SingleTargetScenario";

/**
 * 场景 3：跨平台并行分发。
 *
 * 用户输入「时间窗口 + 系统事件 + 各平台人数」，每个平台被拆成一条独立的多人协同
 * 任务，**完全复用**多人协同管线（Selector + StrategyGenerator），并行执行；各平台
 * 产出与单平台完全同构的五维调度策略，最后只做整合输出。进阶参数（风险等级 / 频率
 * 上限 / Profile 限制）不在本场景暴露，保持默认值即可。
 */
export function CrossPlatformScenario(props: SharedScenarioProps) {
  const { eventPayload, isRunning, onStart, onError } = props;
  const [specs, setSpecs] = useState<PlatformSpecPayload[]>([
    { platform_key: SIMULATED_PLATFORMS[0].key, requested_count: 5 },
    { platform_key: SIMULATED_PLATFORMS[1].key, requested_count: 5 },
    { platform_key: SIMULATED_PLATFORMS[2].key, requested_count: 5 },
  ]);
  const [busy, setBusy] = useState(false);

  function updateSpec(index: number, patch: Partial<PlatformSpecPayload>) {
    setSpecs((prev) => prev.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  }
  function addSpec() {
    setSpecs((prev) => {
      const used = new Set(prev.map((item) => item.platform_key));
      const next =
        SIMULATED_PLATFORMS.find((item) => !used.has(item.key)) ?? SIMULATED_PLATFORMS[0];
      return [...prev, { platform_key: next.key, requested_count: 5 }];
    });
  }
  function removeSpec(index: number) {
    setSpecs((prev) => (prev.length > 1 ? prev.filter((_, i) => i !== index) : prev));
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    // ``eventPayload`` 由父组件统一分流校验（文字/图片/系统事件），这里只校验本场景参数。
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
          waterArmyEnabled={props.waterArmyEnabled}
          setWaterArmyEnabled={props.setWaterArmyEnabled}
          waterArmyCount={props.waterArmyCount}
          setWaterArmyCount={props.setWaterArmyCount}
          // 跨平台：每个平台独立 requested_count（已在专属参数里配置），
          // 进阶参数（风险等级 / 频率上限 / Profile 限制）保持默认值，不在此暴露。
          showMaxSelectedNodes={false}
          showAllowedPlatforms={false}
          showAdvancedOptions={false}
        />
      </div>

      <button className="primary-button" type="submit" disabled={busy}>
        {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
        运行：跨平台并行分发
      </button>
    </form>
  );
}