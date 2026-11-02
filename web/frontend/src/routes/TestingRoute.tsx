import { useEffect, useState } from "react";
import type { TestRunHistoryItem, TestRunResponse } from "../types";
import { getTestRun } from "../lib/api";
import { errorMessage } from "../lib/options";
import { Topbar } from "../components/Topbar";
import { TestRunHistoryPanel } from "../components/TestRunHistoryPanel";
import { TestRunDetailPanel } from "../components/TestRunDetailPanel";
import { TestRunScenario } from "../scenarios/TestRunScenario";

/**
 * 总体测试控制台：多子任务组合批量评测。
 *
 * 三栏布局，和主控制台彻底解耦：
 *
 *   1. 输入面板 —— 配置一组子任务（单人 / 多人 / 跨平台任意组合），点"开始总体测试"
 *      异步提交；事件由系统随机抽取，运行配置由用户指定；
 *   2. 测试历史 —— 总体测试专用列表（``/api/test-runs``），显示排队中 / 测试中
 *      与进度，点击任一条进入详情；提交后不锁按钮，可连续发起多次排队；
 *   3. 测试详情 —— 选中的那条测试：进度、总体阶段柱状图、每个子任务的计时明细。
 *      测试中会持续轮询直到结束。
 */
export default function TestingRoute() {
  const [useLlm, setUseLlm] = useState(true);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<TestRunResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [notice, setNotice] = useState("");
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoadingDetail(true);
    getTestRun(selectedId)
      .then((payload) => {
        if (!cancelled) setDetail(payload);
      })
      .catch((err) => {
        if (!cancelled) setNotice(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const detailActive = detail?.state === "queued" || detail?.state === "running";
  useEffect(() => {
    if (!selectedId || !detailActive) return;
    const timer = window.setInterval(() => {
      getTestRun(selectedId)
        .then((payload) => setDetail(payload))
        .catch(() => {
          /* 轮询失败静默，下一拍重试 */
        });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [selectedId, detailActive]);

  function handleStarted(run: { test_run_id: string }) {
    setSelectedId(run.test_run_id);
    setDetail(null);
    setNotice(`已提交测试 ${run.test_run_id}，正在排队执行。`);
    setRefreshTick((value) => value + 1);
  }

  function handleSelectItem(item: TestRunHistoryItem) {
    if (item.test_run_id === selectedId) return;
    setSelectedId(item.test_run_id);
    setDetail(null);
    setNotice("");
  }

  function handleAfterDelete(deletedId: string, remaining: TestRunHistoryItem[]) {
    if (selectedId !== deletedId) return;
    setSelectedId(remaining[0]?.test_run_id ?? "");
    setDetail(null);
  }

  return (
    <div className="app-shell">
      <Topbar
        useLlm={useLlm}
        setUseLlm={setUseLlm}
        refresh={() => setRefreshTick((value) => value + 1)}
        hasRun={!!selectedId}
        subtitle="总体测试：多子任务组合批量评测"
      />

      <main className="workspace-grid">
        <section className="panel input-panel">
          <div className="sub-section batch-eval-section">
            <h3>总体测试</h3>
            <TestRunScenario
              onStarted={handleStarted}
              onError={setNotice}
              useLlm={useLlm}
              setUseLlm={setUseLlm}
            />
          </div>
          {notice && <div className="message">{notice}</div>}
        </section>

        <section className="panel status-panel">
          <TestRunHistoryPanel
            activeId={selectedId}
            onSelect={handleSelectItem}
            refreshTick={refreshTick}
            onAfterDelete={handleAfterDelete}
          />
        </section>

        <TestRunDetailPanel
          run={detail}
          loading={loadingDetail}
          error={notice && !detail ? notice : ""}
        />
      </main>
    </div>
  );
}
