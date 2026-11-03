interface SupportNodeAction {
  动作编号?: string;
  数字人?: string;
  动作类型?: string;
  依赖动作编号?: string;
  依赖帖子编号?: string;
  内容编号?: string;
  动作分组?: string;
}

interface SupportNodeInfo {
  数字人编号?: string;
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

/**
 * 特殊支持节点面板：与「核心发布」完全分离地呈现其选人信息与各自行动清单。
 * 数据来自 payload["特殊支持节点"]（后端单独成块，不混入五维调度策略）。
 * 这些节点只做点赞 / 转发（无文本），但动作清单与核心节点同构：
 * 六列（动作编号 / 数字人 / 动作类型 / 依赖动作编号 / 依赖帖子编号 / 内容编号）。
 */
export default function SupportNodePanel({ payload }: { payload: Record<string, unknown> }) {
  const block = payload["特殊支持节点"] as SupportNodeBlock | undefined;
  if (!block || !block.nodes || block.nodes.length === 0) return null;

  const people = block.selection ?? block.nodes;

  // 把一个节点的动作按「动作分组」（时期）聚合，与核心节点的 ActionListView 同构。
  const groupActions = (actions: SupportNodeAction[] = []) => {
    const groups: Record<string, SupportNodeAction[]> = {};
    for (const a of actions) {
      const g = a.动作分组 || "其他";
      (groups[g] ||= []).push(a);
    }
    return groups;
  };

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

      <h4>选人信息（独立于核心发布，编号以 W 开头）</h4>
      <table className="kv-table">
        <thead>
          <tr>
            <th>数字人编号</th>
            <th>数字人 id</th>
            <th>平台</th>
            <th>风险等级</th>
          </tr>
        </thead>
        <tbody>
          {people.map((n) => (
            <tr key={n.user_id}>
              <td>{n.数字人编号 ?? "-"}</td>
              <td>{n.user_id}</td>
              <td>{n.platform ?? "-"}</td>
              <td>{n.risk_level || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>各节点行动清单</h4>
      {block.nodes.map((node) => {
        const groups = groupActions(node.actions);
        return (
          <div key={node.user_id} className="support-node-block">
            <div className="support-node-title">
              <strong>{node.数字人编号 ?? node.user_id}</strong>
              <span className="muted">
                （{node.user_id} · {node.platform ?? "-"} · 风险 {node.risk_level || "-"}）
              </span>
            </div>
            {Object.entries(groups).map(([group, acts]) => (
              <div key={group} className="action-group">
                <div className="action-group-head">
                  <span className="action-group-name">{group}</span>
                  <span className="action-count">{acts.length} 个动作</span>
                </div>
                <div className="action-table">
                  <div className="action-row action-row-head">
                    <span>动作编号</span>
                    <span>数字人</span>
                    <span>动作类型</span>
                    <span>依赖动作编号</span>
                    <span>依赖帖子编号</span>
                    <span>内容编号</span>
                  </div>
                  {acts.map((a, index) => (
                    <div className="action-row" key={String(a.动作编号 ?? index)}>
                      <span>{String(a.动作编号 ?? "-")}</span>
                      <span>{String(a.数字人 ?? "-")}</span>
                      <span>{String(a.动作类型 ?? "-")}</span>
                      <span>{String(a.依赖动作编号 ?? "-")}</span>
                      <span>{String(a.依赖帖子编号 ?? "-")}</span>
                      <span>{String(a.内容编号 ?? "-")}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        );
      })}

      {block.note && <p className="muted">{block.note}</p>}
    </section>
  );
}
