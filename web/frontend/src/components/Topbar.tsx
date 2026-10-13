import { NavLink } from "react-router-dom";
import { FlaskConical, LayoutGrid, RefreshCcw } from "lucide-react";

export function Topbar({
  useLlm,
  setUseLlm,
  refresh,
  hasRun,
  subtitle,
}: {
  useLlm: boolean;
  setUseLlm: (value: boolean) => void;
  refresh: () => void;
  hasRun: boolean;
  subtitle: string;
}) {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <h1>影响力事件分发控制台</h1>
        <p>{subtitle} · 输出目录 outputs/web_runs</p>
      </div>
      <div className="topbar-actions">
        <nav className="page-nav" aria-label="页面切换">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              isActive ? "page-nav-link active" : "page-nav-link"
            }
          >
            <LayoutGrid size={16} />
            主控制台
          </NavLink>
          <NavLink
            to="/testing"
            className={({ isActive }) =>
              isActive ? "page-nav-link active" : "page-nav-link"
            }
          >
            <FlaskConical size={16} />
            测试控制台
          </NavLink>
        </nav>
        <label className="toggle">
          <input
            type="checkbox"
            checked={useLlm}
            onChange={(event) => setUseLlm(event.target.checked)}
          />
          <span>{useLlm ? "LLM 开启" : "LLM 关闭"}</span>
        </label>
        <button className="icon-button" onClick={refresh} disabled={!hasRun}>
          <RefreshCcw size={18} />
          刷新
        </button>
      </div>
    </header>
  );
}