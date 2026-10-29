import {
  parsePlatformList,
  SIMULATED_PLATFORMS,
  togglePlatform,
} from "../lib/platforms";

export type BaseOptionsFormProps = {
  maxSelectedNodes: number;
  setMaxSelectedNodes: (value: number) => void;
  riskLevel: string;
  setRiskLevel: (value: string) => void;
  campaignWindowHours: number;
  setCampaignWindowHours: (value: number) => void;
  maxFrequencyPerDay: number;
  setMaxFrequencyPerDay: (value: number) => void;
  profileLimit: string;
  setProfileLimit: (value: string) => void;
  allowedPlatforms: string;
  setAllowedPlatforms: (value: string) => void;
  /** 特殊支持节点参与开关与数量（所有分发模式共用）。 */
  waterArmyEnabled: boolean;
  setWaterArmyEnabled: (value: boolean) => void;
  waterArmyCount: number;
  setWaterArmyCount: (value: number) => void;
  /** 是否渲染"特殊支持节点参与"开关；默认 true。 */
  showWaterArmy?: boolean;
  /** 是否渲染"主选数量"输入；单/跨平台场景下隐藏。默认 true。 */
  showMaxSelectedNodes?: boolean;
  /** 是否渲染"候选人平台"复选框；单/多/跨平台场景下都隐藏（各自从目标
    * 数字人 / dispatchPlatform / platform_specs 推断），仅 BatchEval
    * 因为内嵌多模式共享而保留可见。默认 true。 */
  showAllowedPlatforms?: boolean;
  /** 是否渲染"风险等级 / 频率上限 / Profile 限制"这些进阶参数。
    *
    * 多人协同分发下用户只需要指定**平台 + 小队人数 + 时间窗口**，其余项保持
    * 默认值即可，全部铺开反而让人无从下手。默认 true——单人 / 跨平台 /
    * BatchEval 场景仍保持原样。 */
  showAdvancedOptions?: boolean;
  /** 是否渲染"传播窗口"（整场活动的时间跨度，小时）。
    *
    * 时间窗口是用户要直接掌控的参数，所以独立于 ``showAdvancedOptions``：
    * 多人场景只收起进阶项，窗口照常保留。默认 true。 */
  showCampaignWindow?: boolean;
};

/**
 * 三种分发场景共享的基础参数：风险等级 + 传播窗口 + 频率上限 + profile 限制。
 *
 * "主选数量"（maxSelectedNodes）和"候选人平台"（allowed_platforms）在
 * 单人 / 多人 / 跨平台场景下都由具体 Scenario 的专属参数控制：
 *   * single_target: 1 人固定，无选人池；
 *   * multi_user:    maxSelectedNodes 控制小队人数，平台由 dispatchPlatform 决定；
 *   * cross_platform:每个平台独立 requested_count，平台清单隐含在 platform_specs。
 * 因此这两个字段按需隐藏。
 */
export function BaseOptionsForm(props: BaseOptionsFormProps) {
  const {
    maxSelectedNodes,
    setMaxSelectedNodes,
    riskLevel,
    setRiskLevel,
    campaignWindowHours,
    setCampaignWindowHours,
    maxFrequencyPerDay,
    setMaxFrequencyPerDay,
    profileLimit,
    setProfileLimit,
    allowedPlatforms,
    setAllowedPlatforms,
    waterArmyEnabled,
    setWaterArmyEnabled,
    waterArmyCount,
    setWaterArmyCount,
    showWaterArmy = true,
    showMaxSelectedNodes = true,
    showAllowedPlatforms = true,
    showAdvancedOptions = true,
    showCampaignWindow = true,
  } = props;

  return (
    <div className="form-grid">
      {showMaxSelectedNodes && (
        <label>
          主选数量
          <input
            type="number"
            min={1}
            max={20}
            value={maxSelectedNodes}
            onChange={(event) => setMaxSelectedNodes(Number(event.target.value))}
          />
        </label>
      )}
      {showCampaignWindow && (
        <label>
          传播窗口（小时）
          <input
            type="number"
            min={1}
            max={168}
            value={campaignWindowHours}
            onChange={(event) => setCampaignWindowHours(Number(event.target.value))}
          />
        </label>
      )}
      {showAdvancedOptions && (
        <>
          <label>
            风险等级
            <select
              value={riskLevel}
              onChange={(event) => setRiskLevel(event.target.value)}
            >
              <option value="">自动</option>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
          <label>
            频率上限
            <input
              type="number"
              min={1}
              max={24}
              value={maxFrequencyPerDay}
              onChange={(event) => setMaxFrequencyPerDay(Number(event.target.value))}
            />
          </label>
          <label>
            Profile 限制
            <input
              value={profileLimit}
              onChange={(event) => setProfileLimit(event.target.value)}
              placeholder="留空加载全部"
            />
          </label>
        </>
      )}
      {showAllowedPlatforms && (
        <label className="wide">
          候选人平台（评分/选人池）
          <span className="platform-options">
            {SIMULATED_PLATFORMS.map((item) => (
              <span key={item.value} className="platform-option">
                <input
                  type="checkbox"
                  checked={parsePlatformList(allowedPlatforms).includes(item.value)}
                  onChange={() =>
                    setAllowedPlatforms(togglePlatform(allowedPlatforms, item.value))
                  }
                />
                {item.label}
              </span>
            ))}
          </span>
        </label>
      )}
      {showWaterArmy && (
        <label className="wide">
          <span className="checkbox-row">
            <input
              type="checkbox"
              checked={waterArmyEnabled}
              onChange={(event) => setWaterArmyEnabled(event.target.checked)}
            />
            特殊支持节点参与（仅点赞 / 转发，不计入选人上限）
          </span>
          {waterArmyEnabled && (
            <span className="water-army-count">
              支持节点数量
              <input
                type="number"
                min={0}
                max={300}
                value={waterArmyCount}
                onChange={(event) => setWaterArmyCount(Number(event.target.value))}
              />
            </span>
          )}
        </label>
      )}
    </div>
  );
}