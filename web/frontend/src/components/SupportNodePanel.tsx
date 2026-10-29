interface SupportNodeAction {
  动作编号?: string;
  动作分组?: string;
  动作类型?: string;
  动作名称?: string;
  执行阶段?: string;
  目标定位?: { post_ref?: string; owner_id?: string; label?: string };
  执行参数?: Record<string, unknown>;
}

interface SupportNodeInfo {
  user_id?: string;
  platform?: string;
  risk_level?: string;
  actions?: SupportNodeAction[];
}

interface SupportNodeBlock {
  enabled?: boolean;
  count?: number;
  platforms?: string[];
  core_actor_count?: number;
  selection?: SupportNodeInfo[];
  nodes?: SupportNodeInfo[];
  note?: string;
}

const ACTION_TYPE_LABEL: Record<string, string> = {
  like: "点赞",
  repost: "转发",
};

/**
 * 特殊支持节点面板：与「核心发布」完全分离地呈现其选人信息与各自行动清单。
 * 数据来自 payload["特殊支持节点"]（后端单独成块，不混入五维调度策略）。
 * 这些节点只做点赞 / 转发（无文本），不发布内容，故不显示任何正文。
 */
export default function SupportNodePanel({ payload }: { payload: Record<string, unknown> }) {
  const block = payload["特殊支持节点"] as SupportNodeBlock | undefined;
  if (!block || !block.nodes || block.nodes.length === 0) return null;

  const people = block.selection ?? block.nodes;

  return (
    <section className="result-section">
      <h3>特殊支持节点</h3>

      <div className="summary-band">
        <div>
          <span>启用</span>
          <strong>{block.enabled ? "是" : "否"}</strong>
        </div>
        <div>
          <span>支持节点数</span>
          <strong>{block.count ?? block.nodes.length}</strong>
        </div>
        <div>
          <span>核心发布数</span>
          <strong>{block.core_actor_count ?? "-"}</strong>
        </div>
        {block.platforms && block.platforms.length > 0 && (
          <div>
            <span>平台</span>
            <strong>{block.platforms.join("、")}</strong>
          </div>
        )}
      </div>

      <h4>选人信息（独立于核心发布）</h4>
      <table className="kv-table">
        <thead>
          <tr>
            <th>数字人 id</th>
            <th>平台</th>
            <th>风险等级</th>
          </tr>
        </thead>
        <tbody>
          {people.map((n) => (
            <tr key={n.user_id}>
              <td>{n.user_id}</td>
              <td>{n.platform ?? "-"}</td>
              <td>{n.risk_level || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>各节点行动清单（仅点赞 / 转发，无文本）</h4>
      {block.nodes.map((node) => (
        <div key={node.user_id} className="support-node-block">
          <div className="support-node-title">
            <strong>{node.user_id}</strong>
            <span className="muted">
              （{node.platform ?? "-"} · 风险 {node.risk_level || "-"}）
            </span>
          </div>
          <table className="kv-table">
            <thead>
              <tr>
                <th>时期</th>
                <th>动作</th>
                <th>目标帖子</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {(node.actions ?? []).map((a, i) => {
                const isRepost = a.动作类型 === "repost";
                const actionName =
                  a.动作名称 ?? ACTION_TYPE_LABEL[a.动作类型 ?? ""] ?? a.动作类型 ?? "-";
                const targetRef = a.目标定位?.post_ref ?? "-";
                const note = isRepost
                  ? "转发（无文本）"
                  : "点赞";
                return (
                  <tr key={a.动作编号 ?? i}>
                    <td>{a.动作分组 ?? a.执行阶段 ?? "-"}</td>
                    <td>{actionName}</td>
                    <td>{targetRef}</td>
                    <td>{note}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}

      {block.note && <p className="muted">{block.note}</p>}
    </section>
  );
}
