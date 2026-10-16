import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, History, Loader2, RefreshCcw, Trash2 } from "lucide-react";
import type { BenchmarkHistoryItem, BenchmarkState } from "../types";
import { deleteBenchmark, listBenchmarks } from "../lib/api";
import { errorMessage } from "../lib/options";

const STATE_LABEL: Record<BenchmarkState, string> = {
  queued: "排队中",
  running: "测试中",
  completed: "完成",
  failed: "失败",
};

/**
 * 测试历史面板（压测专用列表，和主控制台的"运行历史"完全隔离）。
 *
 * 与 ``RunHistoryPanel`` 的关键差别：**进行中会持续轮询**。
 *
 *  * 压测是异步任务：点一次按钮立刻落一条 ``queued`` 记录，历史列表里马上
 *    能看到"排队中 / 测试中"和 ``已完成轮数 / 总轮数``；
 *  * 只要列表里还有未结束的测试，就每 2s 拉一次 ``GET /api/benchmarks``，
 *    让进度与"测试中"字样实时更新；全部结束后自动停轮询；
 *  * 因此用户可以连续点多次"开始测试"（不锁按钮），新任务会排队并在列表里
 *    以"排队中"出现。
 */
export function BenchmarkHistoryPanel({
  activeId,
  onSelect,
  refreshTick,
  onAfterDelete,
}: {
  activeId: string;
  onSelect: (item: BenchmarkHistoryItem) => void;
  /** 父组件的"数据版本号"：提交新测试 / 手动刷新时 +1。 */
  refreshTick: number;
  /** 删完一条后回调；父组件用来更新 activeId（如果删的正是它）。 */
  onAfterDelete?: (deletedId: string, remaining: BenchmarkHistoryItem[]) => void;
}) {
  const [items, setItems] = useState<BenchmarkHistoryItem[]>([]);
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
      setItems(await listBenchmarks());
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

  const hasActive = items.some(
    (item) => item.state === "queued" || item.state === "running",
  );

  // 有测试在跑（或排队）时才轮询；全部结束自动停下。
  useEffect(() => {
    if (!hasActive) return;
    const timer = window.setInterval(() => void refreshNow(), 2000);
    return () => window.clearInterval(timer);
  }, [hasActive, refreshNow]);

  async function handleDelete(entry: BenchmarkHistoryItem, event: React.MouseEvent) {
    event.stopPropagation();
    if (pendingDelete) return;
    const ok = window.confirm(
      `确认删除测试 ${entry.benchmark_id}？\n\n将一并删除 outputs/web_runs/benchmark_runs/${entry.benchmark_id}/。`,
    );
    if (!ok) return;
    setPendingDelete(entry.benchmark_id);
    try {
      await deleteBenchmark(entry.benchmark_id);
      const remaining = items.filter((item) => item.benchmark_id !== entry.benchmark_id);
      setItems(remaining);
      onAfterDelete?.(entry.benchmark_id, remaining);
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
            const active = item.benchmark_id === activeId;
            const busy = item.state === "queued" || item.state === "running";
            return (
              <li key={item.benchmark_id}>
                <div
                  role="option"
                  aria-selected={active}
                  className={active ? "benchmark-history-item active" : "benchmark-history-item"}
                  onClick={() => onSelect(item)}
                >
                  <span className={`state-dot state-${item.state}`} aria-hidden />
                  <span className="benchmark-history-main">
                    <span className="benchmark-history-label">
                      {item.label || item.benchmark_id}
                    </span>
                    <span className="benchmark-history-sub mono">{item.benchmark_id}</span>
                  </span>
                  <span className="benchmark-history-tags">
                    <span className={item.use_llm ? "status-chip llm-yes" : "status-chip llm-no"}>
                      {item.use_llm ? "LLM" : "无LLM"}
                    </span>
                    <span className="benchmark-history-progress">
                      {item.completed + item.failed}/{item.iterations}
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
                    disabled={pendingDelete === item.benchmark_id || busy}
                    onClick={(event) => handleDelete(item, event)}
                  >
                    {pendingDelete === item.benchmark_id ? (
                      <Loader2 className="spin" size={14} />
                    ) : (
                      <Trash2 size={14} />
                    )}
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
      <button
        type="button"
        className="run-history-head"
        onClick={() => setOpen((value) => !value)}
      >
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
