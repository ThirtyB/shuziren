import { useState } from "react";
import type { RunStatus, DispatchMode } from "../types";
import { platformLabel } from "../lib/platforms";
import { SectionTitle } from "./SectionTitle";
import { Stat } from "./Stat";
import { StatusBadge } from "./StatusBadge";
import { RunHistoryPanel } from "./RunHistoryPanel";
import type { RunHistoryEntry } from "../lib/api";

export function StatusPanel({
  run,
  notice,
  selectedEventId,
  onSelect,
  history,
}: {
  run: RunStatus | null;
  notice: string;
  selectedEventId: string;
  onSelect: (eventId: string) => void;
  history: {
    activeRunId: string;
    refreshTick: number;
    onSelectRun: (entry: RunHistoryEntry) => void;
    onAfterDelete?: (deletedRunId: string, remaining: RunHistoryEntry[]) => void;
  };
}) {
  if (!run) {
    return (
      <section className="panel status-panel">
        <SectionTitle title="任务状态" />
        {notice && <div className="message">{notice}</div>}
        <RunHistoryPanel
          activeRunId={history.activeRunId}
          refreshTick={history.refreshTick}
          onSelectRun={history.onSelectRun}
          onAfterDelete={history.onAfterDelete}
        />
        <div className="empty-state">暂无正在跑的 run；选一条历史或返回上方发起新任务。</div>
      </section>
    );
  }
  const progress =
    run.total > 0 ? Math.round(((run.completed + run.failed) / run.total) * 100) : 0;
  return (
    <section className="panel status-panel">
      <SectionTitle title="任务状态" extra={<DispatchModeChip mode={run.dispatch_mode} />} />
      {notice && <div className="message">{notice}</div>}
      <RunHistoryPanel
        activeRunId={history.activeRunId}
        refreshTick={history.refreshTick}
        onSelectRun={history.onSelectRun}
        onAfterDelete={history.onAfterDelete}
      />
      <div className="status-row">
        <StatusBadge state={run.state} />
        <span className="run-id">{run.run_id}</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="stat-grid">
        <Stat label="完成" value={run.completed} />
        <Stat label="失败" value={run.failed} />
        <Stat label="总数" value={run.total} />
        <Stat label="进度" value={`${progress}%`} />
      </div>
      <div className="path-list">
        <div>
          <strong>输出</strong>
          <span>{run.output_dir}</span>
        </div>
        <div>
          <strong>Trace</strong>
          <span>{run.trace_dir}</span>
        </div>
        {run.current_event_id && (
          <div>
            <strong>当前</strong>
            <span>{run.current_event_id}</span>
          </div>
        )}
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
      <div className="result-list">
        {run.results.map((item) => (
          <ResultButton
            key={item.event_id}
            eventId={item.event_id}
            eventName={item.event_name}
            count={item.selected_digital_human_ids.length}
            dispatchMode={item.dispatch_mode}
            perPlatformCount={item.per_platform?.length ?? 0}
            active={selectedEventId === item.event_id}
            onSelect={onSelect}
          />
        ))}
      </div>
    </section>
  );
}

function ResultButton({
  eventId,
  eventName,
  count,
  dispatchMode,
  perPlatformCount,
  active,
  onSelect,
}: {
  eventId: string;
  eventName: string;
  count: number;
  dispatchMode: DispatchMode;
  perPlatformCount: number;
  active: boolean;
  onSelect: (eventId: string) => void;
}) {
  const rightText =
    dispatchMode === "cross_platform"
      ? `${perPlatformCount} 个平台`
      : `${count} 个数字人`;
  return (
    <button
      className={active ? "result-item active" : "result-item"}
      onClick={() => onSelect(eventId)}
    >
      <span className="result-event-id">{eventId}</span>
      <strong>{eventName}</strong>
      <small>{rightText}</small>
    </button>
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
