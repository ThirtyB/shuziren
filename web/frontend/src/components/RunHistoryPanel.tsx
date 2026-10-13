import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, History, Loader2, RefreshCcw, Trash2 } from "lucide-react";
import { deleteRun, listRuns, type RunHistoryEntry } from "../lib/api";
import { errorMessage } from "../lib/options";

const DISPATCH_LABEL: Record<RunHistoryEntry["dispatch_mode"], string> = {
  single_target: "单人定向",
  multi_user: "多人协同",
  cross_platform: "跨平台并行",
};

const STATE_LABEL: Record<RunHistoryEntry["state"], string> = {
  queued: "排队",
  running: "运行中",
  completed: "完成",
  failed: "失败",
};

/** 历史列表的轮询间隔（毫秒）。足够短以让新 run 及时出现在列表上。 */
const POLL_INTERVAL_MS = 4000;

/**
 * 折叠式"历史运行"面板：拉取 ``/api/runs``，按 run_id 显示近 50 条。
 *
 * 设计目的：
 *  * 用户切场景 / 切事件来源后，已发起的 run 仍然存在（不丢）；
 *  * 用户可以从历史里点开任意一次旧 run 查看结果；
 *  * 用户可以删除历史 run：调 ``DELETE /api/runs/{run_id}`` 把整目录
 *    ``outputs/web_runs/{run_id}/`` 一起删掉。
 *
 * 刷新策略（修 Bug：历史不会自动刷新 / 刷新按钮无效）：
 *  * 头部 ``RefreshCcw`` 图标现在是一个**真正可点击**的按钮，点击会
 *    立即重新拉一次。
 *  * 组件内部用 ``setInterval`` 每 ``POLL_INTERVAL_MS`` 拉一次，
 *    这样新 run 提交后无需父组件"手动 +1 refreshTick"也能及时出现。
 *  * 监听 ``document.visibilityState``：切到后台时停止轮询避免空跑，
 *    切回前台时立即拉一次。
 *
 * ``activeRunId`` 是"当前正在显示的 run"——可以是正在跑的那条，也可以是
 * 用户从历史里挑出来的旧 run。
 */
export function RunHistoryPanel({
  activeRunId,
  onSelectRun,
  refreshTick,
  onAfterDelete,
  inline = false,
}: {
  activeRunId: string;
  onSelectRun: (entry: RunHistoryEntry) => void;
  /** 来自父组件的"数据版本号"，每次新发起 run 都会 +1，触发刷新。 */
  refreshTick: number;
  /** 删完一条后回调；父组件用来更新 activeRunId（如果删的正是它）。 */
  onAfterDelete?: (deletedRunId: string, remaining: RunHistoryEntry[]) => void;
  /** true 时始终展开（用于内嵌在 StatusPanel 顶部）。 */
  inline?: boolean;
}) {
  const [items, setItems] = useState<RunHistoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(true);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  // 内部"再拉一次"信号：刷新按钮和 visibility 切回前台都会触发它。
  const [manualTick, setManualTick] = useState(0);
  // in-flight 标记：避免上一次请求未返回时被新的请求覆盖。
  const inFlightRef = useRef(false);

  const refreshNow = useCallback(() => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setLoading(true);
    setError("");
    listRuns()
      .then((runs) => {
        setItems(runs);
      })
      .catch((err) => {
        setError(errorMessage(err));
      })
      .finally(() => {
        inFlightRef.current = false;
        setLoading(false);
      });
  }, []);

  // 触发条件：父组件 refreshTick 变化（提交新 run）、内部手动点击、
  // 页面从后台切回前台。轮询则由下方 setInterval 负责。
  useEffect(() => {
    refreshNow();
  }, [refreshTick, manualTick, refreshNow]);

  // 自动轮询：仅在前台时拉。
  useEffect(() => {
    if (typeof document === "undefined") return;
    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        setManualTick((value) => value + 1);
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") {
        refreshNow();
      }
    }, POLL_INTERVAL_MS);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibility);
      window.clearInterval(timer);
    };
  }, [refreshNow]);

  async function handleDelete(entry: RunHistoryEntry, event: React.MouseEvent) {
    event.stopPropagation();
    if (pendingDelete) return;
    const ok = window.confirm(
      `确认删除 run ${entry.run_id}？\n\n将一并删除 outputs/web_runs/${entry.run_id}/ 下的 status.json、output、trace。`,
    );
    if (!ok) return;
    setPendingDelete(entry.run_id);
    try {
      await deleteRun(entry.run_id);
      // 立即更新列表并通知父组件。
      const remaining = items.filter((item) => item.run_id !== entry.run_id);
      setItems(remaining);
      onAfterDelete?.(entry.run_id, remaining);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setPendingDelete(null);
    }
  }

  const body = (
    <>
      {error && <div className="message error">{error}</div>}
      {loading ? (
        <div className="empty-state small">加载中</div>
      ) : items.length === 0 ? (
        <div className="empty-state small">暂无历史运行</div>
      ) : (
        <ul className="run-history-list">
          {items.map((item) => (
            <li key={item.run_id}>
              <div
                role="option"
                aria-selected={activeRunId === item.run_id}
                className={
                  activeRunId === item.run_id
                    ? "run-history-item active"
                    : "run-history-item"
                }
                onClick={() => onSelectRun(item)}
              >
                <span className={`state-dot state-${item.state}`} aria-hidden />
                <span className="run-history-id">{item.run_id}</span>
                <span className="run-history-mode">{DISPATCH_LABEL[item.dispatch_mode]}</span>
                <span className="run-history-progress">
                  {item.completed + item.failed}/{item.total}
                </span>
                <span className="run-history-time">{formatTime(item.created_at)}</span>
                <span className="run-history-state">{STATE_LABEL[item.state]}</span>
                <button
                  type="button"
                  className="run-history-delete"
                  title={
                    item.state === "running" || item.state === "queued"
                      ? "运行中无法删除"
                      : "删除这条历史"
                  }
                  disabled={
                    pendingDelete === item.run_id ||
                    item.state === "running" ||
                    item.state === "queued"
                  }
                  onClick={(event) => handleDelete(item, event)}
                >
                  {pendingDelete === item.run_id ? (
                    <Loader2 className="spin" size={14} />
                  ) : (
                    <Trash2 size={14} />
                  )}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </>
  );

  if (inline) {
    return <div className="run-history run-history-inline">{body}</div>;
  }

  return (
    <div className="run-history">
      <button
        type="button"
        className="run-history-head"
        onClick={() => setOpen((value) => !value)}
      >
        <span className="run-history-title">
          <History size={16} />
          运行历史
          <span className="run-history-count">{items.length}</span>
        </span>
        <span className="run-history-meta">
          <span
            role="button"
            tabIndex={0}
            aria-label="刷新历史"
            title="刷新历史"
            className="run-history-refresh"
            onClick={(event) => {
              // 阻止冒泡到头部折叠按钮，避免点击刷新时同时展开/折叠。
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
          <ChevronDown
            size={14}
            className={open ? "chevron open" : "chevron"}
          />
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