import { useEffect, useState } from "react";
import type { BenchmarkHistoryItem, BenchmarkResponse, RunOptions, RunStatus } from "../types";
import { getBenchmark, postJson } from "../lib/api";
import { errorMessage } from "../lib/options";
import { DEFAULT_ALLOWED_PLATFORMS } from "../lib/platforms";
import { Topbar } from "../components/Topbar";
import { BenchmarkHistoryPanel } from "../components/BenchmarkHistoryPanel";
import { BenchmarkDetailPanel } from "../components/BenchmarkDetailPanel";
import { BenchmarkScenario } from "../scenarios/BenchmarkScenario";

/**
 * 测试控制台：一键 N 轮压测（默认 5）。
 *
 * 三栏布局，和主控制台彻底解耦：
 *
 *   1. 输入面板 —— 轮数 / 标签 / 平台，点"开始测试"异步提交；
 *   2. 测试历史 —— 压测专用列表（``/api/benchmarks``），显示"排队中 / 测试中"
 *      与进度，点击任一条进入详情；提交后**不锁按钮**，可连续发起多次排队；
 *   3. 测试详情 —— 选中的那条测试：进度、每轮随机抽到的事件、阶段柱状图、
 *      每轮耗时明细。测试中会持续轮询直到结束。
 *
 * 注意这里不再复用主控制台的 ``StatusPanel`` / ``ResultPanel``：压测任务与
 * ``/api/runs`` 的运行历史是两套独立数据，混在一起会让"测试历史"被普通 run
 * 淹没。
 */
export default function TestingRoute() {
  // 基础运行参数
  const [profileLimit, setProfileLimit] = useState("");
  const [maxSelectedNodes, setMaxSelectedNodes] = useState(5);
  const [riskLevel, setRiskLevel] = useState("");
  const [campaignWindowHours, setCampaignWindowHours] = useState(24);
  const [maxFrequencyPerDay, setMaxFrequencyPerDay] = useState(3);
  const [allowedPlatforms, setAllowedPlatforms] = useState(DEFAULT_ALLOWED_PLATFORMS);
  const [useLlm, setUseLlm] = useState(true);

  // 测试历史 / 详情状态
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<BenchmarkResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [notice, setNotice] = useState("");
  const [refreshTick, setRefreshTick] = useState(0);

  // 选中哪条就拉哪条的详情。
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setLoadingDetail(true);
    getBenchmark(selectedId)
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

  // 选中的测试还在排队 / 测试中时，持续轮询详情；结束后自动停。
  const detailActive = detail?.state === "queued" || detail?.state === "running";
  useEffect(() => {
    if (!selectedId || !detailActive) return;
    const timer = window.setInterval(() => {
      getBenchmark(selectedId)
        .then((payload) => setDetail(payload))
        .catch(() => {
          /* 轮询失败静默，下一拍重试 */
        });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [selectedId, detailActive]);

  /** 新测试提交成功：立刻选中它并刷新历史列表（列表里会多一条"排队中"）。 */
  function handleStarted(benchmark: BenchmarkResponse) {
    setSelectedId(benchmark.benchmark_id);
    setDetail(benchmark);
    setNotice(`已提交测试 ${benchmark.benchmark_id}，正在排队执行。`);
    setRefreshTick((value) => value + 1);
  }

  function handleSelectItem(item: BenchmarkHistoryItem) {
    if (item.benchmark_id === selectedId) return;
    setSelectedId(item.benchmark_id);
    setDetail(null);
    setNotice("");
  }

  function handleAfterDelete(deletedId: string, remaining: BenchmarkHistoryItem[]) {
    if (selectedId !== deletedId) return;
    setSelectedId(remaining[0]?.benchmark_id ?? "");
    setDetail(null);
  }

  /**
   * 压测场景不使用共享的 ``submit``（它自己调 ``runBenchmark``），但
   * ``SharedScenarioProps`` 要求这个字段存在，这里保留一个不会被触发的实现。
   */
  async function submitEvalRun(
    _event: Record<string, unknown> | null,
    options: RunOptions,
  ): Promise<RunStatus> {
    return postJson<RunStatus>("/api/eval-runs", { options });
  }

  const sharedScenarioProps = {
    eventPayload: null as Record<string, unknown> | null,
    profileLimit,
    setProfileLimit,
    maxSelectedNodes,
    setMaxSelectedNodes,
    riskLevel,
    setRiskLevel,
    campaignWindowHours,
    setCampaignWindowHours,
    maxFrequencyPerDay,
    setMaxFrequencyPerDay,
    allowedPlatforms,
    setAllowedPlatforms,
    useLlm,
    // 压测是异步任务，按钮不由"是否有任务在跑"驱动 —— 允许连续提交排队。
    isRunning: false,
    submit: submitEvalRun,
    onStart: () => {
      /* 压测不走 RunStatus 流程，统一由 onStarted 处理。 */
    },
    onError: setNotice,
  };

  return (
    <div className="app-shell">
      <Topbar
        useLlm={useLlm}
        setUseLlm={setUseLlm}
        refresh={() => setRefreshTick((value) => value + 1)}
        hasRun={!!selectedId}
        subtitle="一键 N 轮压测：聚合用时 + 阶段可视化"
      />

      <main className="workspace-grid">
        <section className="panel input-panel">
          <div className="sub-section batch-eval-section">
            <h3>一键压测</h3>
            <BenchmarkScenario {...sharedScenarioProps} onStarted={handleStarted} />
          </div>
          {notice && <div className="message">{notice}</div>}
        </section>

        <section className="panel status-panel">
          <BenchmarkHistoryPanel
            activeId={selectedId}
            onSelect={handleSelectItem}
            refreshTick={refreshTick}
            onAfterDelete={handleAfterDelete}
          />
        </section>

        <BenchmarkDetailPanel
          benchmark={detail}
          loading={loadingDetail}
          error={notice && !detail ? notice : ""}
        />
      </main>
    </div>
  );
}
