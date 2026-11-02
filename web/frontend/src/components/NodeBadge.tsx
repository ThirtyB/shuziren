import {
  useEffect,
  useState,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from "react";

/** 数字人在底层画像库（数据库）中的原始记录（全部字段，未裁剪）。 */
export type RawNode = Record<string, unknown>;

// 进程级缓存：同一数字人只请求一次；值为 null 表示已确认库中查无此人；
// undefined 表示该 id 尚未发起过请求。
const nodeCache: Record<string, RawNode | null | undefined> = {};

/**
 * 按 user_id 取数字人原始画像的共享 hook：首次访问时请求 ``/api/nodes/{userId}``，
 * 后续命中进程级缓存。返回的 ``ensure`` 用于显式触发加载。
 */
export function useNodeProfile(userId: string) {
  const [data, setData] = useState<RawNode | null>(nodeCache[userId] ?? null);
  const [loading, setLoading] = useState(false);

  function ensure() {
    if (nodeCache[userId] !== undefined) {
      setData(nodeCache[userId] ?? null);
      return;
    }
    setLoading(true);
    fetch(`/api/nodes/${encodeURIComponent(userId)}`)
      .then((r) => (r.ok ? (r.json() as Promise<RawNode>) : null))
      .then((d) => {
        nodeCache[userId] = d;
        setData(d);
      })
      .catch(() => {
        nodeCache[userId] = null;
        setData(null);
      })
      .finally(() => setLoading(false));
  }

  return { data, loading, ensure };
}

/** 把任意原始字段值渲染成可读节点；对象 / 对象数组用可折叠的 JSON 展示。 */
function renderValue(value: unknown): ReactNode {
  if (value === null || value === undefined) return <span className="node-empty">—</span>;
  if (typeof value === "string" || typeof value === "boolean" || typeof value === "number") {
    return String(value);
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="node-empty">（空）</span>;
    const allPrimitive = value.every((v) => typeof v !== "object" || v === null);
    if (allPrimitive) return value.map((v) => String(v)).join("、");
    return (
      <details className="node-details">
        <summary>{value.length} 项</summary>
        <pre className="node-json">{JSON.stringify(value, null, 2)}</pre>
      </details>
    );
  }
  if (typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>);
    return (
      <details className="node-details" open>
        <summary>{keys.length} 个字段</summary>
        <pre className="node-json">{JSON.stringify(value, null, 2)}</pre>
      </details>
    );
  }
  return String(value);
}

/**
 * 数字人原始信息内容：标题（昵称 + 水军标记）+ 全字段列表。通用渲染，
 * 因此数据库里多出的任何字段都会被呈现出来。
 */
export function NodeRawContent({
  data,
  loading,
}: {
  data: RawNode | null;
  loading: boolean;
}) {
  if (loading && !data) return <div className="node-popover-loading">加载中…</div>;
  if (!data) return <div className="node-popover-loading">未找到该数字人原始数据</div>;
  return (
    <>
      <div className="node-popover-title">
        <span>{String(data.user_name ?? data.user_id ?? "数字人")}</span>
        {data.is_water_army ? <span className="node-tag water">水军</span> : null}
      </div>
      <div className="node-popover-body">
        {Object.entries(data).map(([key, value]) => (
          <div className="node-field" key={key}>
            <span className="node-field-label">{key}</span>
            <span className="node-field-value">{renderValue(value)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

/**
 * 数字人悬停卡片：包住任意编号 / id 块，鼠标移入时按 ``userId`` 拉取底层画像库
 * 原始记录，并以跟随触发元素的浮动卡片呈现（可滚动、可展开嵌套字段）。
 */
export function NodeBadge({
  userId,
  label,
  className,
}: {
  userId: string;
  label: ReactNode;
  className?: string;
}) {
  const { data, loading, ensure } = useNodeProfile(userId);
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState({ left: 0, top: 0 });

  function place(rect: DOMRect) {
    const width = 408;
    const estHeight = 340;
    let top = rect.bottom; // 紧贴触发元素底部，避免间隙导致鼠标移出
    if (top + estHeight > window.innerHeight - 8) {
      top = Math.max(8, rect.top - estHeight);
    }
    let left = rect.left;
    left = Math.min(left, window.innerWidth - width - 8);
    left = Math.max(8, left);
    setPos({ left, top });
  }

  function onEnter(e: ReactMouseEvent) {
    place(e.currentTarget.getBoundingClientRect());
    setShow(true);
    ensure();
  }

  return (
    <span className="node-badge-wrap" onMouseEnter={onEnter} onMouseLeave={() => setShow(false)}>
      <span className={`node-badge${className ? " " + className : ""}`}>{label}</span>
      {show && (
        <div className="node-popover" style={{ left: pos.left, top: pos.top }}>
          <NodeRawContent data={data} loading={loading} />
        </div>
      )}
    </span>
  );
}

/**
 * 点击数字人（职责卡 / 其他触发点）后弹出的模态框，展示该人在数据库中的原始信息。
 * 数据仅在弹窗打开（即用户点击）后才请求；弹窗出现瞬间先显示「加载中」，拿到后填充。
 */
export function NodeModal({
  userId,
  onClose,
}: {
  userId: string | null;
  onClose: () => void;
}) {
  const { data, loading, ensure } = useNodeProfile(userId ?? "");
  useEffect(() => {
    if (!userId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [userId, onClose]);
  // 仅在弹窗打开（用户点击）后才发起查询。
  useEffect(() => {
    if (userId) ensure();
  }, [userId, ensure]);

  if (!userId) return null;
  return (
    <div className="node-modal-overlay" onClick={onClose}>
      <div className="node-modal" onClick={(e) => e.stopPropagation()}>
        <button className="node-modal-close" onClick={onClose} aria-label="关闭">
          ×
        </button>
        <NodeRawContent data={data} loading={loading} />
      </div>
    </div>
  );
}
