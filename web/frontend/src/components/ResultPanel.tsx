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

          {result.dispatch_mode === "multi_user" && (
            <LegacyFiveDimension payload={payload} />
          )}

          {result.dispatch_mode === "cross_platform" && (
            <CrossPlatformDetail payload={payload} result={result} />
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
      <LegacyFiveDimension payload={payload} />
    </>
  );
}

function LegacyFiveDimension({ payload }: { payload: Record<string, unknown> }) {
  const strategy = payload["五维调度策略"] as Record<string, unknown> | undefined;
  const target = strategy?.["目标对象"] as Record<string, unknown> | undefined;
  const time = strategy?.["时间"] as Record<string, unknown> | undefined;
  const frequency = strategy?.["频率"] as Record<string, unknown> | undefined;
  const platform = strategy?.["平台"] as Record<string, unknown> | undefined;
  const content = strategy?.["内容"] as Record<string, unknown> | undefined;
  const selectedIds = (target?.["选取数字人id组"] as string[] | undefined) ?? [];

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

function CrossPlatformDetail({
  payload,
  result,
}: {
  payload: Record<string, unknown>;
  result: RunResult;
}) {
  const block = (payload["多平台分发策略"] as Record<string, unknown> | undefined) ?? {};
  const platforms = (block.platforms as Array<Record<string, unknown>> | undefined) ?? [];
  const totals = (block["total_selected_safety_tiers"] as
    | Record<string, unknown>
    | undefined) ?? {};
  const summary = result.per_platform ?? [];
  return (
    <>
      <section className="result-section highlight">
        <h3>跨平台分发总览</h3>
        <div className="summary-band">
          <div>
            <span>分发的平台数</span>
            <strong>{platforms.length}</strong>
          </div>
          <div>
            <span>总选用数字人</span>
            <strong>{result.selected_digital_human_ids.length}</strong>
          </div>
          <div>
            <span>安全等级分布</span>
            <strong>
              safe {Number(totals.safe ?? 0)} / review {Number(totals.review_required ?? 0)} / no {Number(totals.not_eligible ?? 0)}
            </strong>
          </div>
        </div>
      </section>

      {(result.per_platform ?? []).map((item, index) => {
        const platformBlock = platforms[index] ?? {};
        const contentPlan = (platformBlock["content_plan"] as
          | Record<string, unknown>
          | undefined) ?? {};
        const nodes = (item.selected_user_ids ?? []) as string[];
        return (
          <section className="result-section" key={index}>
            <h3>
              平台 {platformLabel(item.platform_key)} · 目标 {Number(item.requested_count ?? 0)} 人
            </h3>
            <div className="summary-band">
              <div>
                <span>实际选用</span>
                <strong>{Number(item.selected_count ?? 0)}</strong>
              </div>
              <div>
                <span>人数缺口</span>
                <strong>{Number(item.shortfall_count ?? 0)}</strong>
              </div>
              <div>
                <span>安全缺口</span>
                <strong>{Number(item.safe_shortfall_count ?? 0)}</strong>
              </div>
            </div>
            <div className="id-row">
              {nodes.map((id) => (
                <span key={id}>{id}</span>
              ))}
            </div>
            <DataTable
              items={
                (contentPlan["段落指引"] as Array<Record<string, unknown>> | undefined) ??
                (contentPlan["paragraphs"] as Array<Record<string, unknown>> | undefined) ??
                []
              }
            />
            <DataTable
              items={
                (contentPlan["动作清单"] as Array<Record<string, unknown>> | undefined) ?? []
              }
            />
          </section>
        );
      })}

      {summary.length === 0 && (
        <section className="result-section">
          <h3>平台</h3>
          <div className="empty-table">未生成 per_platform 摘要</div>
        </section>
      )}
    </>
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
