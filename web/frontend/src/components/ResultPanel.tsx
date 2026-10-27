import { useState } from "react";
import { Download } from "lucide-react";
import type { RunResult, RunStatus } from "../types";
import { platformLabel, platformModeLabel } from "../lib/platforms";
import { DataTable } from "./DataTable";
import { SectionTitle } from "./SectionTitle";

/** 数字人原始 id → 统一编号（D01…）的映射，用于把所有出现处的原始 id 收敛成编号。 */
function buildIdMap(division: Array<Record<string, unknown>>): Record<string, string> {
  const map: Record<string, string> = {};
  for (const h of division) {
    const id = String(h["数字人id"] ?? "");
    const no = String(h["数字人编号"] ?? "");
    if (id && no) map[id] = no;
  }
  return map;
}

/** 隐藏维度表里的原始数字人 id，统一只展示编号（前端表现层面统一，不动后端结构）。 */
function withNumbering(
  items: Array<Record<string, unknown>>,
  idMap: Record<string, string>,
): Array<Record<string, unknown>> {
  return items.map((it) => {
    const copy = { ...it };
    const id = String(it["数字人id"] ?? "");
    if (it["数字人编号"] && idMap[id]) {
      delete copy["数字人id"];
    }
    return copy;
  });
}

/** 各模块耗时的中文标签；未知模块名原样展示。 */
const MODULE_LABELS: Record<string, string> = {
  llm_client_init: "LLM 客户端初始化",
  hot_event_to_payload: "事件解析",
  event_parser: "事件解析",
  feature_builder: "特征构建",
  recall: "候选召回",
  selector: "数字人筛选",
  scorer: "评分排序",
  strategy_generator: "策略生成",
  platform_dispatch: "平台分发",
  content_generation: "内容生成",
  write_output: "写出结果",
  write_trace: "写出 Trace",
  event_total: "总耗时",
};

/** 模块耗时的展示顺序；不在表中的排后面，总耗时永远最后。 */
const MODULE_ORDER = [
  "llm_client_init",
  "hot_event_to_payload",
  "event_parser",
  "feature_builder",
  "recall",
  "selector",
  "scorer",
  "strategy_generator",
  "platform_dispatch",
  "content_generation",
  "write_output",
  "write_trace",
  "event_total",
];

/**
 * 结果面板：按 ``result.dispatch_mode`` 分支渲染。
 *
 * * single_target  —— 高亮目标数字人、其所属平台、目标角色与 warnings。
 * * multi_user     —— 旧版五维调度策略（目标 / 时间 / 频率 / 平台 / 内容）。
 * * cross_platform —— 每个平台一张表，叠加总览。
 * 所有模式尾部都展示 trace 摘要 + 原始 JSON。
 */
export function ResultPanel({
  runId,
  run,
  result,
  payload,
  trace,
  loading,
}: {
  runId: string | null;
  run: RunStatus | null;
  result: RunResult | null;
  payload: Record<string, unknown> | null;
  trace: Record<string, unknown> | null;
  loading: boolean;
}) {
  const [showModules, setShowModules] = useState(false);
  return (
    <section className="panel result-panel">
      <div className="result-head">
        <SectionTitle title="策略结果" />
        {runId && result && (
          <a
            className="icon-button link-button"
            href={`/api/runs/${runId}/files/${result.event_id}`}
            target="_blank"
            rel="noreferrer"
          >
            <Download size={18} />
            JSON
          </a>
        )}
      </div>
      {loading && <div className="empty-state">加载中</div>}
      {!loading && !payload && <div className="empty-state">选择一个结果查看详情</div>}
      {!loading && payload && result && (
        <div className="detail-stack">
          <div className="summary-band">
            <div>
              <span>事件</span>
              <strong>{String(payload["事件名称"] ?? "-")}</strong>
            </div>
            <div>
              <span>格式</span>
              <strong>{String(payload["输出格式版本"] ?? "-")}</strong>
            </div>
            {run?.elapsed_seconds != null && (
              <div>
                <span>运行耗时</span>
                <strong>{run.elapsed_seconds.toFixed(2)} 秒</strong>
                <button
                  type="button"
                  className="info-toggle"
                  title="查看各模块耗时"
                  aria-label="查看各模块耗时"
                  onClick={() => setShowModules((v) => !v)}
                >
                  !
                </button>
              </div>
            )}
            {showModules && result?.stage_times && (
              <div style={{ flexBasis: "100%" }}>
                <ModuleTimes times={result.stage_times} />
              </div>
            )}
          </div>

          {result.dispatch_mode === "single_target" && (
            <SingleTargetDetail payload={payload} result={result} />
          )}

          {result.dispatch_mode === "cross_platform" && (
            <CrossPlatformOverview payload={payload} />
          )}

          {/* 多人分发与跨平台分发共用同一套五维调度策略结构，
              跨平台只是多出「平台」维度下的各平台方案。 */}
          {(result.dispatch_mode === "multi_user" ||
            result.dispatch_mode === "cross_platform") && (
            <FiveDimension payload={payload} />
          )}

          <TraceSection trace={trace} />
          <section className="result-section">
            <h3>原始 JSON</h3>
            <pre>{JSON.stringify(payload, null, 2)}</pre>
          </section>
        </div>
      )}
    </section>
  );
}

function SingleTargetDetail({
  payload,
  result,
}: {
  payload: Record<string, unknown>;
  result: RunResult;
}) {
  const diag = (payload["定向分发诊断"] as Record<string, unknown> | undefined) ?? {};
  const targetDispatch = result.target_dispatch;
  return (
    <>
      <section className="result-section highlight">
        <h3>单人定向诊断</h3>
        <div className="summary-band">
          <div>
            <span>目标数字人</span>
            <strong>{String(targetDispatch?.target_user_id ?? diag.target_user_id ?? "-")}</strong>
          </div>
          <div>
            <span>所属平台</span>
            <strong>
              {platformLabel(
                targetDispatch?.target_platform ?? diag.target_platform,
              )}
            </strong>
          </div>
          <div>
            <span>目标角色</span>
            <strong>{String(targetDispatch?.target_role ?? diag.target_role ?? "-")}</strong>
          </div>
        </div>
        {Array.isArray(targetDispatch?.warnings) && targetDispatch!.warnings!.length > 0 && (
          <ul className="warning-list">
            {targetDispatch!.warnings!.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        )}
      </section>
      <FiveDimension payload={payload} />
    </>
  );
}

/**
 * 五维调度策略渲染：多人分发与跨平台分发**共用**同一套维度结构。
 *
 * 跨平台模式额外在「平台」维度下带 ``各平台方案``，检测到就逐平台追加渲染，
 * 因此两种模式只有这一块差异，不存在第二条读取路径。
 */
function FiveDimension({ payload }: { payload: Record<string, unknown> }) {
  const strategy = payload["五维调度策略"] as Record<string, unknown> | undefined;
  const target = strategy?.["目标对象"] as Record<string, unknown> | undefined;
  const time = strategy?.["时间"] as Record<string, unknown> | undefined;
  const frequency = strategy?.["频率"] as Record<string, unknown> | undefined;
  const platform = strategy?.["平台"] as Record<string, unknown> | undefined;
  const content = strategy?.["内容"] as Record<string, unknown> | undefined;
  const selectedIds = (target?.["选取数字人id组"] as string[] | undefined) ?? [];
  const division =
    (target?.["数字人分工"] as Array<Record<string, unknown>> | undefined) ?? [];
  const idMap = buildIdMap(division);
  const platformSchemes =
    (platform?.["各平台方案"] as Array<Record<string, unknown>> | undefined) ?? [];
  const stageStats =
    (time?.["阶段统计"] as Array<Record<string, unknown>> | undefined) ?? [];
  const timePlanRows = withNumbering(
    (time?.["数字人时间安排"] as Array<Record<string, unknown>> | undefined) ?? [],
    idMap,
  );
  const frequencyItems = withNumbering(
    (frequency?.["数字人频率"] as Array<Record<string, unknown>> | undefined) ?? [],
    idMap,
  );
  const platformItems = withNumbering(
    (platform?.["数字人平台分发"] as Array<Record<string, unknown>> | undefined) ?? [],
    idMap,
  );
  const contentRows = withNumbering(
    (content?.["数字人发帖内容"] as Array<Record<string, unknown>> | undefined) ?? [],
    idMap,
  );
  const actions =
    (content?.["动作清单"] as Array<Record<string, unknown>> | undefined) ?? [];

  return (
    <>
      {/* 数字人职责：每个人在传播链里的角色 / 职责 / 人设 / 平台，统一用 D01… 编号。 */}
      <section className="result-section">
        <h3>数字人职责</h3>
        <p>{String(target?.["传播目标"] ?? "-")}</p>
        <div className="id-row">
          {selectedIds.map((id) => (
            <span key={id}>{idMap[id] ?? id}</span>
          ))}
        </div>
        <RoleView humans={division} />
      </section>

      {/* 时间维度：阶段统计（含时期—窗口）与每人数安排。时期—窗口不再单列，避免重复。 */}
      <section className="result-section">
        <h3>时间（阶段统计）</h3>
        <StageStatView stats={stageStats} />
        <DataTable items={timePlanRows} />
      </section>

      {/* 频率维度：单独占一行，精简为「编号 · 条/小时」。 */}
      <section className="result-section">
        <h3>频率（条/小时）</h3>
        <FrequencyView items={frequencyItems} />
      </section>

      <section className="result-section">
        <h3>平台</h3>
        <p>{platformModeLabel(platform?.["平台模式"])}</p>
        <DataTable items={platformItems} />
        {platformSchemes.map((scheme, index) => (
          <PlatformSchemeCard
            key={`${String(scheme["平台"] ?? index)}-${index}`}
            scheme={scheme}
            index={index}
            idMap={idMap}
          />
        ))}
      </section>

      <section className="result-section">
        <h3>内容</h3>
        <DataTable items={contentRows} />
      </section>

      <section className="result-section">
        <h3>动作清单（按阶段分组）</h3>
        <ActionListView actions={actions} />
      </section>
    </>
  );
}

/**
 * 各模块耗时：点击运行耗时旁的「!」后展开，逐项列出本轮评估各阶段墙钟耗时（秒）。
 *
 * 数据来自 ``result.stage_times``（后端 ``_result_summary`` 已随结果返回），
 * 模块名统一翻译成中文；未知模块名原样展示，展示顺序按 ``MODULE_ORDER`` 排。
 */
function ModuleTimes({ times }: { times: Record<string, number> }) {
  const entries = Object.entries(times);
  entries.sort((a, b) => {
    const ia = MODULE_ORDER.indexOf(a[0]);
    const ib = MODULE_ORDER.indexOf(b[0]);
    const ra = ia === -1 ? 999 : ia;
    const rb = ib === -1 ? 999 : ib;
    if (ra !== rb) return ra - rb;
    return a[0] < b[0] ? -1 : 1;
  });
  return (
    <div className="module-times">
      <div className="module-times-head">各模块耗时（秒）</div>
      {entries.map(([key, value]) => (
        <div className="module-time-row" key={key}>
          <span className="module-name">{MODULE_LABELS[key] ?? key}</span>
          <span className="module-val">{Number(value).toFixed(3)}s</span>
        </div>
      ))}
    </div>
  );
}

/**
 * 数字人职责卡片：每个数字人一张卡，展示统一编号、角色、职责说明、人设、平台。
 *
 * 数据来自 ``五维调度策略.目标对象.数字人分工``，后端已按其角色（首发 → 互动 →
 * 扩散 → 支持）分配好 ``数字人编号``，这里直接按列表渲染即可。
 */
function RoleView({
  humans,
}: {
  humans: Array<Record<string, unknown>>;
}) {
  if (!humans.length) {
    return <div className="empty-table">无数字人</div>;
  }
  return (
    <div className="role-cards">
      {humans.map((h) => {
        const no = String(h["数字人编号"] ?? "-");
        const id = String(h["数字人id"] ?? "-");
        return (
          <div className="role-card" key={`${no}-${id}`}>
            <div className="role-card-head">
              <span className="role-no">{no}</span>
              <span className="role-name">{String(h["角色"] ?? "-")}</span>
              <span className="role-platform">{String(h["平台"] ?? "-")}</span>
            </div>
            <div className="role-duty">{String(h["职责说明"] ?? "-")}</div>
            <div className="role-persona">{String(h["人设"] ?? "-")}</div>
            <div className="role-id">数字人id：{id}</div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * 各阶段统计：每个时期一行，展示窗口、参与数字人、动作数以及发帖/评论/点赞/转发分布。
 *
 * 数据来自 ``五维调度策略.时间.阶段统计``，后端已按四个时期顺序排好，前端直接渲染。
 */
function StageStatView({
  stats,
}: {
  stats: Array<Record<string, unknown>>;
}) {
  if (!stats.length) {
    return <div className="empty-table">无阶段统计</div>;
  }
  return (
    <div className="stage-stats">
      {stats.map((stat, index) => (
        <div className="stage-stat" key={String(stat["时期"] ?? index)}>
          <div className="stage-stat-head">
            <span className="stage-index">{index + 1}</span>
            <span className="stage-name">{String(stat["时期"] ?? "-")}</span>
            <span className="stage-range">{String(stat["窗口"] ?? "-")}</span>
          </div>
          <div className="stage-stat-body">
            <span className="stat-actors">参与：{String(stat["参与数字人"] ?? "-")}</span>
            <span className="stat-count">动作数：{String(stat["动作数"] ?? 0)}</span>
            <span className="stat-发帖">发帖 {String(stat["发帖"] ?? 0)}</span>
            <span className="stat-评论">评论 {String(stat["评论"] ?? 0)}</span>
            <span className="stat-点赞">点赞 {String(stat["点赞"] ?? 0)}</span>
            <span className="stat-转发">转发 {String(stat["转发"] ?? 0)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * 频率可视化（精简版）：每个数字人一行，直接给出「条/小时」速率。
 *
 * 后端已把动作数量换算成 ``条每小时``（多人模式 = 实际动作数 / 活动窗口小时，
 * 单人模式 = 计划频率 / 24）。多人模式顺带把动作类型分布作为次要信息附在后面。
 */
function FrequencyView({
  items,
}: {
  items: Array<Record<string, unknown>>;
}) {
  if (!items.length) {
    return <div className="empty-table">无频率信息</div>;
  }
  return (
    <div className="freq-compact">
      {items.map((item) => {
        const no = String(item["数字人编号"] ?? item["数字人id"] ?? "-");
        const perHour = item["条每小时"];
        const stats =
          (item["动作统计"] as Record<string, unknown> | undefined) ?? {};
        const detail = Object.entries(stats)
          .filter(([, v]) => Number(v) > 0)
          .map(([k, v]) => `${k}${v}`)
          .join(" ");
        return (
          <div className="freq-row" key={no}>
            <span className="freq-no">{no}</span>
            <span className="freq-rate">
              {perHour != null ? `${Number(perHour).toFixed(2)} 条/小时` : "-"}
            </span>
            {detail && <span className="freq-detail">（{detail}）</span>}
          </div>
        );
      })}
    </div>
  );
}

/**
 * 动作清单：按「动作分组」（多人模式即四个时期）分组，每组一个小标题，
 * 从而把同一时期的动作聚拢，明确体现首发期 → 互动期 → 扩散期 → 支持期。
 *
 * 后端已按序号（= 时间顺序）排好，这里只做分组展示。每行固定六列：
 * 动作编号 / 数字人 / 动作类型 / 依赖动作编号 / 依赖帖子编号 / 内容编号。
 */
function ActionListView({
  actions,
}: {
  actions: Array<Record<string, unknown>>;
}) {
  if (!actions.length) {
    return <div className="empty-table">无动作</div>;
  }
  const groups: Record<string, Array<Record<string, unknown>>> = {};
  const order: string[] = [];
  for (const action of actions) {
    const group = String(action["动作分组"] ?? "其他");
    if (!groups[group]) {
      groups[group] = [];
      order.push(group);
    }
    groups[group].push(action);
  }
  return (
    <div className="action-groups">
      {order.map((group) => (
        <div className="action-group" key={group}>
          <div className="action-group-head">
            <span className="action-stage">{group}</span>
            <span className="action-count">{groups[group].length} 个动作</span>
          </div>
          <div className="action-table">
            <div className="action-row action-row-head">
              <span>动作编号</span>
              <span>数字人</span>
              <span>动作类型</span>
              <span>依赖动作编号</span>
              <span>依赖帖子编号</span>
              <span>内容编号</span>
            </div>
            {groups[group].map((action, index) => (
              <div className="action-row" key={String(action["动作编号"] ?? index)}>
                <span>{String(action["动作编号"] ?? "-")}</span>
                <span>{String(action["数字人"] ?? "-")}</span>
                <span>{String(action["动作类型"] ?? "-")}</span>
                <span>{String(action["依赖动作编号"] ?? "-")}</span>
                <span>{String(action["依赖帖子编号"] ?? "-")}</span>
                <span>{String(action["内容编号"] ?? "-")}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * 跨平台分发总览：人数 / 缺口 / 安全等级 / 重复数字人 / 跨平台提示。
 *
 * 数据来自 ``五维调度策略.平台.跨平台总览``——与多人分发共用同一套输出结构，
 * 不再读取第二份顶层块。
 */
/**
 * 跨平台分发总览：人数 / 缺口 / 安全等级 / 重复数字人。
 *
 * 数据来自 ``五维调度策略.平台.跨平台总览``。渲染策略仿照多人分发的"目标对象"段：
 * 只保留结构化数据（摘要带 + id-row），跨平台提示 / 草案说明 / 范围说明等元
 * 信息从 UI 移除——这些字段仍保留在后端 payload 中，便于后续按需取回。
 */
function CrossPlatformOverview({ payload }: { payload: Record<string, unknown> }) {
  const strategy = payload["五维调度策略"] as Record<string, unknown> | undefined;
  const platform = strategy?.["平台"] as Record<string, unknown> | undefined;
  const overview =
    (platform?.["跨平台总览"] as Record<string, unknown> | undefined) ?? {};
  const tiers =
    (overview["安全等级分布"] as Record<string, unknown> | undefined) ?? {};
  const overlaps = (overview["跨平台重复数字人"] as string[] | undefined) ?? [];
  const target = strategy?.["目标对象"] as Record<string, unknown> | undefined;
  const division =
    (target?.["数字人分工"] as Array<Record<string, unknown>> | undefined) ?? [];
  const idMap = buildIdMap(division);

  return (
    <section className="result-section highlight">
      <h3>跨平台分发总览</h3>
      <div className="summary-band">
        <div>
          <span>分发的平台数</span>
          <strong>{Number(overview["平台数"] ?? 0)}</strong>
        </div>
        <div>
          <span>目标人数合计</span>
          <strong>{Number(overview["目标人数合计"] ?? 0)}</strong>
        </div>
        <div>
          <span>选取人数合计</span>
          <strong>{Number(overview["选取人数合计"] ?? 0)}</strong>
        </div>
        <div>
          <span>去重后人数</span>
          <strong>{Number(overview["去重数字人数"] ?? 0)}</strong>
        </div>
        <div>
          <span>合格缺口合计</span>
          <strong>{Number(overview["合格缺口合计"] ?? 0)}</strong>
        </div>
        <div>
          <span>安全等级分布</span>
          <strong>
            合格 {Number(tiers["合格"] ?? 0)} / 复核 {Number(tiers["待人工复核"] ?? 0)} /
            不合格 {Number(tiers["未通过资格筛选"] ?? 0)}
          </strong>
        </div>
      </div>
      {overlaps.length > 0 && (
        <>
          <p>以下数字人同时出现在多个平台的选取结果里：</p>
          <div className="id-row">
            {overlaps.map((id) => (
              <span key={id}>{idMap[id] ?? id}</span>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

/**
 * 单个平台的分发方案：人数概览 + 选人明细 + 内容计划 + 时间计划。
 *
 * 数据来自 ``五维调度策略.平台.各平台方案[i]``：与多人分发共用同一套五维输出，
 * 跨平台只是多出这一块平台专属明细。
 *
 * 显示策略仿照多人分发的"五维调度策略"：只保留结构化数据（DataTable + 摘要带 +
 * id-row），风险/护栏/免责/节奏建议等提示性文字已从 UI 移除——这些字段仍保留在
 * 后端 payload 中，便于后续按需取回。
 */
function PlatformSchemeCard({
  scheme,
  index,
  idMap,
}: {
  scheme: Record<string, unknown>;
  index: number;
  idMap: Record<string, string>;
}) {
  const platformKey = String(scheme["平台"] ?? index);
  const selectedIds = (scheme["选取数字人id组"] as string[] | undefined) ?? [];
  const tiers = (scheme["安全等级分布"] as Record<string, unknown> | undefined) ?? {};

  // 完整的逐节点选人与评分明细（可追溯依据），后端 _build_platform_schemes
  // 会以"选取节点明细"列出每个被选数字人的全部评分维度。
  const nodeDetailRows =
    (scheme["选取节点明细"] as Array<Record<string, unknown>> | undefined) ?? [];
  const guidanceRows =
    (scheme["写作指引"] as Array<Record<string, unknown>> | undefined) ?? [];
  const stageRows =
    (scheme["时间阶段"] as Array<Record<string, unknown>> | undefined) ?? [];

  return (
    <div className="sub-section">
      <h3>
        平台方案 · {platformLabel(platformKey)} · 目标{" "}
        {Number(scheme["目标人数"] ?? 0)} 人
      </h3>
      <div className="summary-band">
        <div>
          <span>实际选用</span>
          <strong>{Number(scheme["选取人数"] ?? 0)}</strong>
        </div>
        <div>
          <span>人数缺口</span>
          <strong>{Number(scheme["人数缺口"] ?? 0)}</strong>
        </div>
        <div>
          <span>合格缺口</span>
          <strong>{Number(scheme["合格缺口"] ?? 0)}</strong>
        </div>
        <div>
          <span>安全等级分布</span>
          <strong>
            合格 {Number(tiers["合格"] ?? 0)} / 复核{" "}
            {Number(tiers["待人工复核"] ?? 0)} / 不合格{" "}
            {Number(tiers["未通过资格筛选"] ?? 0)}
          </strong>
        </div>
      </div>

      <div className="stack">
        <div className="sub-section">
          <h3>选取节点</h3>
          <div className="id-row">
            {selectedIds.map((id) => (
              <span key={id}>{idMap[id] ?? id}</span>
            ))}
          </div>
          <DataTable items={nodeDetailRows} />
        </div>

        <div className="sub-section">
          <h3>内容计划</h3>
          <DataTable items={guidanceRows} />
        </div>

        <div className="sub-section">
          <h3>时间计划</h3>
          <DataTable items={stageRows} />
        </div>
      </div>
    </div>
  );
}

function TraceSection({ trace }: { trace: Record<string, unknown> | null }) {
  const files = ((trace?.files as Array<{ name: string; payload: unknown }> | undefined) ??
    []).slice(0, 8);
  if (!files.length) return null;
  return (
    <section className="result-section">
      <h3>Trace</h3>
      <div className="trace-list">
        {files.map((file) => (
          <details key={file.name}>
            <summary>{file.name}</summary>
            <pre>{JSON.stringify(file.payload, null, 2)}</pre>
          </details>
        ))}
      </div>
    </section>
  );
}
