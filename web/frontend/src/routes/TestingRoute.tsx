import { useEffect, useState } from "react";
import type { RunStatus } from "../types";
import { fetchJson, type RunHistoryEntry } from "../lib/api";
import { errorMessage } from "../lib/options";
import { DEFAULT_ALLOWED_PLATFORMS } from "../lib/platforms";
import { Topbar } from "../components/Topbar";
import { StatusPanel } from "../components/StatusPanel";
import { ResultPanel } from "../components/ResultPanel";
import { BatchEvalScenario } from "../scenarios/BatchEvalScenario";

/**
 * 测试控制台：批量评测入口。
 *
 * 全部事件来自 ``/api/eval-runs``，没有用户输入入口（避免与主控制台混淆）。
 * 状态、刷新、详情面板的代码与 ``ConsoleRoute`` 一致，但因为这是独立路由
 * 拥有自己的 run / detail / trace 状态。
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

  // 运行态
  const [run, setRun] = useState<RunStatus | null>(null);
  const [selectedEventId, setSelectedEventId] = useState("");
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [trace, setTrace] = useState<Record<string, unknown> | null>(null);
  const [notice, setNotice] = useState("");
  const [loadingDetail, setLoadingDetail] = useState(false);

  // 历史运行面板状态：与 ``ConsoleRoute`` 同样的设计。
  const [activeRunId, setActiveRunId] = useState("");
  const [latestRunId, setLatestRunId] = useState("");
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    if (run?.results?.length && !selectedEventId) {
      const first = run.results[0];
      setSelectedEventId(first.event_id);
      void loadDetail(run.run_id, first.event_id);
    }
  }, [run?.results, run?.run_id, selectedEventId]);

  const isRunning = run?.state === "queued" || run?.state === "running";

  async function handleStart(status: RunStatus) {
    setRun(status);
    setDetail(null);
    setTrace(null);
    setSelectedEventId("");
    setNotice("");
    setActiveRunId(status.run_id);
    setLatestRunId(status.run_id);
    setRefreshTick((value) => value + 1);
  }

  useEffect(() => {
    if (!activeRunId) {
      setRun(null);
      return;
    }
    let cancelled = false;
    fetchJson<RunStatus>(`/api/runs/${activeRunId}`)
      .then((payload) => {
        if (!cancelled) setRun(payload);
      })
      .catch((err) => {
        if (!cancelled) setNotice(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [activeRunId]);

  function handleSelectHistory(entry: RunHistoryEntry) {
    if (entry.run_id === activeRunId) return;
    setActiveRunId(entry.run_id);
    setSelectedEventId("");
    setDetail(null);
    setTrace(null);
    setNotice(`已切换到历史 run：${entry.run_id}`);
  }

  function handleAfterDelete(deletedRunId: string, remaining: RunHistoryEntry[]) {
    if (activeRunId === deletedRunId) {
      const next = remaining[0]?.run_id ?? "";
      setActiveRunId(next);
      setSelectedEventId("");
      setDetail(null);
      setTrace(null);
    }
    if (latestRunId === deletedRunId) {
      setLatestRunId(remaining[0]?.run_id ?? "");
    }
    setNotice(`已删除历史 run：${deletedRunId}`);
  }

  async function refreshRun() {
    if (!run?.run_id) return;
    try {
      const payload = await fetchJson<RunStatus>(`/api/runs/${run.run_id}`);
      setRun(payload);
    } catch (err) {
      setNotice(errorMessage(err));
    }
  }

  async function loadDetail(runId: string, eventId: string) {
    setLoadingDetail(true);
    setTrace(null);
    try {
      const [detailPayload, tracePayload] = await Promise.all([
        fetchJson<Record<string, unknown>>(`/api/runs/${runId}/results/${eventId}`),
        fetchJson<Record<string, unknown>>(`/api/runs/${runId}/traces/${eventId}`),
      ]);
      setDetail(detailPayload);
      setTrace(tracePayload);
    } catch (err) {
      setNotice(errorMessage(err));
    } finally {
      setLoadingDetail(false);
    }
  }

  useEffect(() => {
    // 只对"最近一次发起且尚未结束"的 run 自动轮询；历史 run 不会自刷新。
    if (!run?.run_id || run.state === "completed" || run.state === "failed") return;
    if (run.run_id !== latestRunId) return;
    const timer = window.setInterval(refreshRun, 1500);
    return () => window.clearInterval(timer);
  }, [run?.run_id, run?.state, latestRunId]);

  const selectedResult = run?.results.find((item) => item.event_id === selectedEventId) ?? null;

  // BatchEvalScenario 不需要 eventPayload（后端走 default_events_path）。
  // 为了让 SharedScenarioProps 类型对齐，这里传 null 是安全的：
  // buildOptions 不读 eventPayload；BatchEvalScenario.handleSubmit 也不会去 getPayload。
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
    isRunning,
    onStart: handleStart,
    onError: setNotice,
  };

  return (
    <div className="app-shell">
      <Topbar
        useLlm={useLlm}
        setUseLlm={setUseLlm}
        refresh={refreshRun}
        hasRun={!!run?.run_id}
        subtitle="批量评测：直接对内置 200 条 hot_event 跑当前分发场景"
      />

      <main className="workspace-grid">
        <section className="panel input-panel">
          <div className="sub-section batch-eval-section">
            <h3>批量评测（200 条 hot_event）</h3>
            <div className="scenario-hint small">
              全部事件来自后端 default_events_path（``eval/hot_event_opinion_variants_200.json``），
              输出写到 ``eval/output/&lt;mode&gt;/run_NNN/``，与主控制台 ``outputs/web_runs`` 隔离。
            </div>
            <BatchEvalScenario {...sharedScenarioProps} />
          </div>
        </section>

        <StatusPanel
          run={run}
          notice={notice}
          selectedEventId={selectedEventId}
          onSelect={(eventId) => {
            setSelectedEventId(eventId);
            if (run?.run_id) void loadDetail(run.run_id, eventId);
          }}
          history={{
            activeRunId,
            refreshTick,
            onSelectRun: handleSelectHistory,
            onAfterDelete: handleAfterDelete,
          }}
        />

        <ResultPanel
          runId={run?.run_id ?? null}
          result={selectedResult}
          payload={detail}
          trace={trace}
          loading={loadingDetail}
        />
      </main>
    </div>
  );
}