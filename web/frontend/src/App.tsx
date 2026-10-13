import { BrowserRouter, Route, Routes } from "react-router-dom";
import ConsoleRoute from "./routes/ConsoleRoute";
import TestingRoute from "./routes/TestingRoute";

/**
 * 应用壳：浏览器路由 + 两条顶层页面。
 *
 *  - ``/``         主控制台：单事件输入 + 场景（单/多/跨平台）三种 Tab；
 *  - ``/testing``  测试控制台：批量评测独立子页面，与主控制台不共享状态。
 *
 * 顶部 Topbar 内部有 ``<NavLink>`` 切换两条路由，因此两个页面的导航一致。
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ConsoleRoute />} />
        <Route path="/testing" element={<TestingRoute />} />
        <Route path="*" element={<ConsoleRoute />} />
      </Routes>
    </BrowserRouter>
  );
}