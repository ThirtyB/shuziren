import { Download } from "lucide-react";
import type { RunResult } from "../types";
import { platformLabel, platformModeLabel } from "../lib/platforms";
import { DataTable } from "./DataTable";
import { SectionTitle } from "./SectionTitle";

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
  result,
  payload,
  trace,
  loading,
}: {
  runId: string | null;
  result: RunResult | null;
  payload: Record<string, unknown> | null;
  trace: Record<string, unknown> | null;
  loading: boolean;
}) {
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
  const platformSchemes =
    (platform?.["各平台方案"] as Array<Record<string, unknown>> | undefined) ?? [];

  return (
    <>
      <section className="result-section">
        <h3>目标对象</h3>
        <p>{String(target?.["传播目标"] ?? "-")}</p>
        <div className="id-row">
          {selectedIds.map((id) => (
            <span key={id}>{id}</span>
          ))}
        </div>
        <DataTable
          items={
            (target?.["数字人分工"] as Array<Record<string, unknown>> | undefined) ?? []
          }
        />
      </section>

      <section className="result-section">
        <h3>时间与频率</h3>
        <div className="split-grid">
          <DataTable
            items={
              (time?.["数字人时间安排"] as Array<Record<string, unknown>> | undefined) ?? []
            }
          />
          <DataTable
            items={
              (frequency?.["数字人频率"] as Array<Record<string, unknown>> | undefined) ?? []
            }
          />
        </div>
      </section>

      <section className="result-section">
        <h3>平台</h3>
        <p>{platformModeLabel(platform?.["平台模式"])}</p>
        <DataTable
          items={
            (platform?.["数字人平台分发"] as Array<Record<string, unknown>> | undefined) ?? []
          }
        />
        {platformSchemes.map((scheme, index) => (
          <PlatformSchemeCard
            key={`${String(scheme["平台"] ?? index)}-${index}`}
            scheme={scheme}
            index={index}
          />
        ))}
      </section>

      <section className="result-section">
        <h3>内容</h3>
        <DataTable
          items={
            (content?.["数字人发帖内容"] as Array<Record<string, unknown>> | undefined) ?? []
          }
        />
      </section>

      <section className="result-section">
        <h3>动作清单</h3>
        <DataTable
          items={
            (content?.["动作清单"] as Array<Record<string, unknown>> | undefined) ?? []
          }
        />
      </section>
    </>
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
              <span key={id}>{id}</span>
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
}: {
  scheme: Record<string, unknown>;
  index: number;
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
              <span key={id}>{id}</span>
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
