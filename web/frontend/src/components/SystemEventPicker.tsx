import { useEffect, useMemo, useState } from "react";
import { Loader2, Search } from "lucide-react";
import { listSystemEvents } from "../lib/api";
import type { SystemEventSummary } from "../types";
import { errorMessage } from "../lib/options";

export type SystemEventPickerProps = {
  /** 当前选中的系统事件 id（由父组件管理）。 */
  selectedEventId: string;
  onSelect: (summary: SystemEventSummary) => void;
  onError?: (message: string) => void;
};

/**
 * 系统事件选择器：从 ``/api/events`` 拉索引，支持关键字搜索 + 领域过滤 + 分页。
 *
 * 设计要点：
 *  * 只读视图，所选事件原封不动由后端返回，提交时由父组件拉取全字段
 *    ``/api/events/{event_id}``，避免在前端缓存 200 条完整正文。
 *  * 选用后会显示标题 / 概要 / 领域 / 合成标志 / 观点变体数。
 */
export function SystemEventPicker({
  selectedEventId,
  onSelect,
  onError,
}: SystemEventPickerProps) {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState("");
  const [items, setItems] = useState<SystemEventSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [domains, setDomains] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");

  const params = useMemo(
    () => ({
      limit: 50,
      q: query.trim() || undefined,
      domain: domain || undefined,
    }),
    [query, domain],
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setNotice("");
    listSystemEvents(params)
      .then((res) => {
        if (cancelled) return;
        setItems(res.items);
        setTotal(res.total);
        setDomains(res.domains);
      })
      .catch((err) => {
        if (cancelled) return;
        const message = errorMessage(err);
        setNotice(message);
        onError?.(message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [params, onError]);

  return (
    <div className="event-input">
      <div className="picker-head">
        <h3>系统事件库</h3>
        <span className="picker-count">
          {loading ? "加载中" : `共 ${total} 条，当前显示 ${items.length} 条`}
        </span>
      </div>
      <div className="picker-controls">
        <label className="picker-search">
          <Search size={16} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索 标题 / 概要 / 事件 ID"
          />
        </label>
        <label className="picker-domain">
          领域
          <select
            value={domain}
            onChange={(event) => setDomain(event.target.value)}
          >
            <option value="">全部</option>
            {domains.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>
      {notice && <div className="message error">{notice}</div>}
      {loading ? (
        <div className="empty-state">
          <Loader2 className="spin" size={18} />
          加载中
        </div>
      ) : items.length === 0 ? (
        <div className="empty-state">未找到匹配的系统事件</div>
      ) : (
        <div className="event-list" role="listbox" aria-label="系统事件列表">
          {items.map((item) => (
            <button
              key={item.event_id}
              type="button"
              role="option"
              aria-selected={selectedEventId === item.event_id}
              className={
                selectedEventId === item.event_id
                  ? "event-list-item active"
                  : "event-list-item"
              }
              onClick={() => onSelect(item)}
            >
              <div className="event-list-head">
                <span className="event-list-id">{item.event_id}</span>
                <span className="event-list-domain">{item.domain || "—"}</span>
                <span className="event-list-meta">
                  {item.opinion_variants_count} 个观点变体
                  {item.is_synthetic ? " · 合成" : ""}
                </span>
              </div>
              <strong>{item.event_title}</strong>
              <p>{item.event_summary}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}