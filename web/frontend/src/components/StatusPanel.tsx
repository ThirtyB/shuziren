import { useState } from "react";
import type { RunStatus, DispatchMode } from "../types";
import { platformLabel } from "../lib/platforms";
import { SectionTitle } from "./SectionTitle";
import { StatusBadge } from "./StatusBadge";

export function StatusPanel({
  run,
  notice,
}: {
  run: RunStatus | null;
  notice: string;
}) {
  if (!run) {
    return (
      <section className="panel status-panel">
        <SectionTitle title="任务状态" />
        {notice && <div className="message">{notice}</div>}
        <div className="empty-state">暂无正在跑的 run；点右上角「运行历史」查看或发起新任务。</div>
      </section>
    );
  }
  const progress =
    run.total > 0 ? Math.round(((run.completed + run.failed) / run.total) * 100) : 0;
  return (
    <section className="panel status-panel">
      <SectionTitle title="任务状态" extra={<DispatchModeChip mode={run.dispatch_mode} />} />
      {notice && <div className="message notice">{notice}</div>}
      <div className="status-row">
        <StatusBadge state={run.state} />
        <span className="run-id">{run.run_id}</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      {run.errors.length > 0 && (
        <div className="error-list">
          {run.errors.slice(-4).map((item, index) => (
            <div key={`${item.event_id}-${index}`} className="message error">
              {item.event_id || "任务"}：{item.message}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function DispatchModeChip({ mode }: { mode: DispatchMode }) {
  const map: Record<DispatchMode, string> = {
    single_target: "单人定向",
    multi_user: "多人协同",
    cross_platform: "跨平台并行",
  };
  return <span className={`dispatch-chip ${mode}`}>{map[mode]}</span>;
}
