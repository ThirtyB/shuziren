import { useEffect } from "react";
import { History, X } from "lucide-react";
import { RunHistoryPanel } from "./RunHistoryPanel";
import type { RunHistoryEntry } from "../lib/api";

/**
 * 运行历史侧边栏（常态隐藏的抽屉）：
 *
 *  * 默认不渲染（``open=false``），点击顶栏「运行历史」或浮层背景关闭；
 *  * 内部只放 ``RunHistoryPanel``（embedded 模式），即这里只承载"运行历史"信息；
 *  * 支持 Esc 关闭、点击背景关闭；点击面板内部不冒泡到背景。
 */
export function HistorySidebar({
  open,
  onClose,
  activeRunId,
  refreshTick,
  onSelectRun,
  onAfterDelete,
}: {
  open: boolean;
  onClose: () => void;
  activeRunId: string;
  refreshTick: number;
  onSelectRun: (entry: RunHistoryEntry) => void;
  onAfterDelete?: (deletedRunId: string, remaining: RunHistoryEntry[]) => void;
}) {
  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="history-sidebar-backdrop" onClick={onClose}>
      <aside
        className="history-sidebar-panel"
        role="dialog"
        aria-label="运行历史"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="history-sidebar-head">
          <span className="history-sidebar-title">
            <History size={16} />
            运行历史
          </span>
          <button
            type="button"
            className="icon-button ghost"
            onClick={onClose}
            aria-label="关闭运行历史"
            title="关闭"
          >
            <X size={18} />
          </button>
        </div>
        <div className="history-sidebar-body">
          <RunHistoryPanel
            activeRunId={activeRunId}
            refreshTick={refreshTick}
            onSelectRun={onSelectRun}
            onAfterDelete={onAfterDelete}
            embedded
          />
        </div>
      </aside>
    </div>
  );
}
