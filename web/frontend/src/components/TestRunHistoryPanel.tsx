import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, History, Loader2, RefreshCcw, Trash2 } from "lucide-react";
import type { TestRunHistoryItem, TestRunState } from "../types";
import { deleteTestRun, listTestRuns } from "../lib/api";
import { errorMessage } from "../lib/options";

const STATE_LABEL: Record<TestRunState, string> = {
  queued: "排队中",
  running: "测试中",
  completed: "完成",
  failed: "失败",
};

/**
 * 测试历史面板（总体测试专用列表，和主控制台的"运行历史"完全隔离）。
 * 只要有未结束的测试就每 2s 轮询，全部结束后自动停。
 */
export function TestRunHistoryPanel({
  activeId,
  onSelect,
  refreshTick,
  onAfterDelete,
}: {
  activeId: string;
  onSelect: (item: TestRunHistoryItem) => void;
  refreshTick: number;
  onAfterDelete?: (deletedId: string, remaining: TestRunHistoryItem[]) => void;
}) {
  const [items, setItems] = useState<TestRunHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(true);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [manualTick, setManualTick] = useState(0);
  const inFlightRef = useRef(false);

  const refreshNow = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setLoading(true);
    try {
      setItems(await listTestRuns());
      setError("");
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      inFlightRef.current = false;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshNow();
  }, [refreshTick, manualTick, refreshNow]);

  const hasActive = items.some((item) => item.state === "queued" || item.state === "running");

  useEffect(() => {
    if (!hasActive) return;
    const timer = window.setInterval(() => void refreshNow(), 2000);
    return () => window.clearInterval(timer);
  }, [hasActive, refreshNow]);

  async function handleDelete(entry: TestRunHistoryItem, event: React.MouseEvent) {
    event.stopPropagation();
    if (pendingDelete) return;
    const ok = window.confirm(`确认删除测试 ${entry.test_run_id}？`);
    if (!ok) return;
    setPendingDelete(entry.test_run_id);
    try {
      await deleteTestRun(entry.test_run_id);
      const remaining = items.filter((item) => item.test_run_id !== entry.test_run_id);
      setItems(remaining);
      onAfterDelete?.(entry.test_run_id, remaining);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setPendingDelete(null);
    }
  }

  const body = (
    <>
      {error && <div className="message error">{error}</div>}
      {items.length === 0 ? (
        <div className="empty-state small">
          {loading ? "加载中" : "暂无测试记录，点上方按钮发起第一次测试"}
        </div>
      ) : (
        <ul className="benchmark-history-list">
          {items.map((item) => {
            const active = item.test_run_id === activeId;
            const busy = item.state === "queued" || item.state === "running";
            return (
              <li key={item.test_run_id}>
                <div
                  role="option"
                  aria-selected={active}
                  className={active ? "benchmark-history-item active" : "benchmark-history-item"}
                  onClick={() => onSelect(item)}
                >
                  <span className={`state-dot state-${item.state}`} aria-hidden />
                  <span className="benchmark-history-main">
                    <span className="benchmark-history-label">{item.label || item.test_run_id}</span>
                    <span className="benchmark-history-sub mono">{item.test_run_id}</span>
                  </span>
                  <span className="benchmark-history-tags">
                    <span className="benchmark-history-progress">
                      {item.completed + item.failed}/{item.total_subtasks}
                    </span>
                  </span>
                  <span className="benchmark-history-time">{formatTime(item.created_at)}</span>
                  <span className={`benchmark-history-state state-${item.state}`}>
                    {busy && <Loader2 className="spin" size={12} />}
                    {STATE_LABEL[item.state]}
                  </span>
                  <button
                    type="button"
                    className="run-history-delete"
                    title={busy ? "测试中无法删除" : "删除这条测试记录"}
                    disabled={pendingDelete === item.test_run_id || busy}
                    onClick={(event) => handleDelete(item, event)}
                  >
                    {pendingDelete === item.test_run_id ? <Loader2 className="spin" size={14} /> : <Trash2 size={14} />}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </>
  );

  return (
    <div className="run-history">
      <button type="button" className="run-history-head" onClick={() => setOpen((value) => !value)}>
        <span className="run-history-title">
          <History size={16} />
          测试历史
          <span className="run-history-count">{items.length}</span>
        </span>
        <span className="run-history-meta">
          <span
            role="button"
            tabIndex={0}
            aria-label="刷新测试历史"
            title="刷新测试历史"
            className="run-history-refresh"
            onClick={(event) => {
              event.stopPropagation();
              setManualTick((value) => value + 1);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                event.stopPropagation();
                setManualTick((value) => value + 1);
              }
            }}
          >
            <RefreshCcw size={14} className={loading ? "spin" : ""} />
          </span>
          <ChevronDown size={14} className={open ? "chevron open" : "chevron"} />
        </span>
      </button>
      {open && body}
    </div>
  );
}

function formatTime(iso: string): string {
  if (!iso) return "—";
  const m = iso.match(/(\d{2}:\d{2})/);
  if (m) return iso.slice(0, 10) + " " + m[1];
  return iso;
}
