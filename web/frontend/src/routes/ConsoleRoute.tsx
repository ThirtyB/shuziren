import { useEffect, useRef, useState } from "react";
import { Layers, Target, Users } from "lucide-react";
import type {
  DispatchMode,
  EventSource,
  RunOptions,
  RunStatus,
  SystemEventSummary,
} from "../types";
import { SCENARIO_TABS } from "../types";
import { fetchJson, getSystemEvent, postJson, runTextImageEvent } from "../lib/api";
import { errorMessage } from "../lib/options";
import { DEFAULT_ALLOWED_PLATFORMS } from "../lib/platforms";
import { Topbar } from "../components/Topbar";
import { StatusPanel } from "../components/StatusPanel";
import { ResultPanel } from "../components/ResultPanel";
import { HistorySidebar } from "../components/HistorySidebar";
import { SystemEventPicker } from "../components/SystemEventPicker";
import { ImageEventForm } from "../components/ImageEventForm";
import { SingleTargetScenario } from "../scenarios/SingleTargetScenario";
import { MultiUserScenario } from "../scenarios/MultiUserScenario";
import { CrossPlatformScenario } from "../scenarios/CrossPlatformScenario";

const SCENARIO_ICONS: Record<DispatchMode, JSX.Element> = {
  single_target: <Target size={17} />,
  multi_user: <Users size={17} />,
  cross_platform: <Layers size={17} />,
};

/**
 * 主控制台：单事件入口。
 *
 * 关键设计：
 *  * 场景（单/多/跨平台）与事件来源（用户/系统）是正交的两条轴。
 *  * 事件来源由顶部 "事件来源" Tab 切换：
 *      - "用户输入"：显示一段事件描述文本（必填），并可附加图片（可选，
 *        本地 OCR 后与文字融合再解析为事件）；单人 / 多人 / 跨平台共用同一录入。
 *      - "系统事件"：显示 ``SystemEventPicker``，从 ``/api/events`` 选一条；
 *    两种模式互斥显示，避免一个页面出现两套事件输入。
 *  * ``eventPayload`` 在 ``ConsoleRoute`` 内部归一化后传入 Scenario，Scenario 内部
 *    不再关心事件来源。
 */
export default function ConsoleRoute() {
  // 基础运行参数
  const [profileLimit, setProfileLimit] = useState("");
  const [maxSelectedNodes, setMaxSelectedNodes] = useState(5);
  const [riskLevel, setRiskLevel] = useState("");
  const [campaignWindowHours, setCampaignWindowHours] = useState(24);
  const [maxFrequencyPerDay, setMaxFrequencyPerDay] = useState(3);
  const [allowedPlatforms, setAllowedPlatforms] = useState(DEFAULT_ALLOWED_PLATFORMS);
  // 特殊支持节点参与：放大参与人数（仅点赞/转发，不计入选人上限）。
  const [waterArmyEnabled, setWaterArmyEnabled] = useState(false);
  const [waterArmyCount, setWaterArmyCount] = useState(10);
  const [useLlm, setUseLlm] = useState(true);

  // 场景
  const [dispatchMode, setDispatchMode] = useState<DispatchMode>("multi_user");
  // 切场景/事件来源只清输入表单，不再清 run —— 切换只影响"下一次发起的事件"
  // 是什么；当前正在跑的 run 或用户想看历史 run 都不应被丢弃。

  // 事件来源
  const [eventSource, setEventSource] = useState<EventSource>("user");
  useEffect(() => {
    // 切源时清掉与上次系统事件/图像/文本输入相关的局部态（避免 picker 残留高亮、
    // 上一次的 File 对象或文本继续生效），但不要清 run / detail / trace，否则
    // 正在跑的 run 也会被一并隐藏。
    setSelectedEventId("");
    setSystemEvent(null);
    setImageFile(null);
    setUserText("");
  }, [eventSource]);

  useEffect(() => {
    // 切换场景时清空附带图片：图片在"用户输入"下作为可选补充，
    // 切到其它场景或切走再切回时不应残留上一次选中的文件。
    setImageFile(null);
  }, [dispatchMode]);

  // 用户文本输入（单人/多人/跨平台 的"用户输入"模式下使用）：一段事件描述
  // 文本，提交时发给后端 /api/runs/text 的 event_text 字段，由后端"事件解析"成结构化
  // 事件；下游选人 / 策略 / 输出流程与系统事件完全一致。
  const [userText, setUserText] = useState("");

  // 系统事件
  const [systemEvent, setSystemEvent] = useState<SystemEventSummary | null>(null);
  const [systemPayload, setSystemPayload] = useState<Record<string, unknown> | null>(null);

  // 多模态图像：在"用户输入 + 多人/跨平台"以及"单人定向的图文模式"下作为可选补充使用；
  // 切源或切场景时清空，避免上一次选中的 File 继续在闭包里"有效"，造成"看起来没选图却能跑"的假象。
  const [imageFile, setImageFile] = useState<File | null>(null);

  // 运行态
  const [run, setRun] = useState<RunStatus | null>(null);
  const [selectedEventId, setSelectedEventId] = useState("");
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [trace, setTrace] = useState<Record<string, unknown> | null>(null);
  const [notice, setNotice] = useState("");
  const [loadingDetail, setLoadingDetail] = useState(false);

  // 历史运行面板：
  //  - ``activeRunId``：当前 StatusPanel 显示的 run（最近一次发起的，或
  //    用户从历史列表里点开的那一条）。
  //  - ``latestRunId``：最近一次发起的 run，用于轮询进度（历史 run 不会自
  //    动刷新）。
  //  - ``refreshTick``：每次有 run 提交成功后 +1，触发历史面板重新拉列表。
  const [activeRunId, setActiveRunId] = useState("");
  const [latestRunId, setLatestRunId] = useState("");
  const [refreshTick, setRefreshTick] = useState(0);
  // 运行历史侧边栏：常态隐藏，点顶栏「运行历史」展开。
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // 自动选第一条结果加载详情；运行完成（state 变化）或切换 history 时都会补加载，
  // 避免「run 有结果但 detail 一直为空」导致面板只显示「选择一个结果查看详情」。
  useEffect(() => {
    if (!run?.run_id || !run.results?.length) return;
    const target = selectedEventId || run.results[0].event_id;
    if (!selectedEventId) setSelectedEventId(target);
    void loadDetail(run.run_id, target);
  }, [run?.run_id, run?.state, selectedEventId]);

  const isRunning = run?.state === "queued" || run?.state === "running";

  // 归一化 eventPayload：系统事件来源时为后端全字段；用户输入走图文路径，不使用该值。
  const eventPayload: Record<string, unknown> | null = systemPayload;

  async function pickSystemEvent(summary: SystemEventSummary) {
    setSystemEvent(summary);
    setNotice("");
    try {
      const payload = await getSystemEvent(summary.event_id);
      setSystemPayload(payload);
    } catch (err) {
      setNotice(errorMessage(err));
      setSystemEvent(null);
      setSystemPayload(null);
    }
  }

  async function handleStart(status: RunStatus) {
    setRun(status);
    setDetail(null);
    setTrace(null);
    setSelectedEventId("");
    setNotice("");
    // 新 run 提交后，设为最新与当前显示；历史面板会因 refreshTick 刷新。
    setActiveRunId(status.run_id);
    setLatestRunId(status.run_id);
    setRefreshTick((value) => value + 1);
  }

  /**
   * 提交 run 的统一入口：按事件来源分流。
   *  * 用户输入（单人 / 多人 / 跨平台 任一场景）：文字必填，可附加图片
   *    —— 走 ``/api/runs/text-image``，后端对图片做 OCR 并与文字融合后解析为事件；
   *    不带图片时退化为纯文本端点。
   *  * 系统事件：走 ``/api/runs/text`` 提交后端全字段事件。
   */
  async function runViaSource(
    event: Record<string, unknown> | null,
    options: RunOptions,
  ): Promise<RunStatus> {
    // 图文（文本+图片）提交：用户输入的任一场景均走此路径。
    if (eventSource === "user") {
      const text = userText.trim();
      if (!text) {
        throw new Error("请输入一段事件描述文本（图片为可选补充）。");
      }
      // 带图片：文字 + 图片一起发给后端，由后端 OCR 融合；否则走纯文本端点。
      if (imageFile) {
        return runTextImageEvent(text, imageFile, options);
      }
      return postJson<RunStatus>("/api/runs/text", { event_text: text, options });
    }
    if (!event) {
      throw new Error("请先在上方选择或填写事件。");
    }
    return postJson<RunStatus>("/api/runs/text", { event, options });
  }

  // 根据 activeRunId 拉取完整 run status（包含 results / errors）。
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

  // 历史列表项被点击：切换 activeRunId，详情面板会按新的 run 重新加载。
  function handleSelectHistory(entry: import("../lib/api").RunHistoryEntry) {
    if (entry.run_id === activeRunId) return;
    setActiveRunId(entry.run_id);
    setSelectedEventId("");
    setDetail(null);
    setTrace(null);
    setNotice(`已切换到历史 run：${entry.run_id}`);
  }

  // 删除一条历史后：如果删的正是当前显示的 run，回退到 remaining[0]（最新一条），
  // 如果删的也是 latestRunId 则一并清空；其它情况不动。
  function handleAfterDelete(
    deletedRunId: string,
    remaining: import("../lib/api").RunHistoryEntry[],
  ) {
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
      const detailPayload = await fetchJson<Record<string, unknown>>(
        `/api/runs/${runId}/results/${eventId}`,
      );
      setDetail(detailPayload);
    } catch (err) {
      setNotice(errorMessage(err));
    } finally {
      setLoadingDetail(false);
    }
    // trace 独立加载：跨平台模式的 trace 按平台分目录存于顶层之外，
    // 顶层 traces/{event_id} 会 404；此处失败仅置空 trace，不影响结果展示。
    fetchJson<Record<string, unknown>>(`/api/runs/${runId}/traces/${eventId}`)
      .then(setTrace)
      .catch(() => setTrace(null));
  }

  useEffect(() => {
    // 只对"最近一次发起且尚未结束"的 run 自动轮询；
    // 历史 run（activeRunId !== latestRunId）不会自刷新，避免噪声。
    if (!run?.run_id || run.state === "completed" || run.state === "failed") return;
    if (run.run_id !== latestRunId) return;
    const timer = window.setInterval(refreshRun, 1500);
    return () => window.clearInterval(timer);
  }, [run?.run_id, run?.state, latestRunId]);

  // 当前 run 从"进行中"变为结束（completed/failed）时，触发**一次**历史列表
  // 刷新，使历史条目里的状态与进度和后端对齐。
  // 任务进行中不做任何历史刷新：避免列表在跑的过程中持续抖动 / 图标一直转。
  const prevRunStateRef = useRef<{ id: string; state: RunStatus["state"] } | null>(null);
  useEffect(() => {
    const current = run ? { id: run.run_id, state: run.state } : null;
    const prev = prevRunStateRef.current;
    prevRunStateRef.current = current;
    if (!current || !prev) return;
    // 切换到别的 run 时不算"任务结束"，不触发刷新。
    if (prev.id !== current.id) return;
    if (prev.state === current.state) return;
    if (current.state === "completed" || current.state === "failed") {
      setRefreshTick((value) => value + 1);
    }
  }, [run?.run_id, run?.state]);

  const scenario = SCENARIO_TABS.find((item) => item.key === dispatchMode);
  const subtitle = scenario
    ? `当前场景：${scenario.label}`
    : "请选择一种分发场景";
  const selectedResult = run?.results.find((item) => item.event_id === selectedEventId) ?? null;

  const sharedScenarioProps = {
    eventPayload,
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
    waterArmyEnabled,
    setWaterArmyEnabled,
    waterArmyCount,
    setWaterArmyCount,
    useLlm,
    isRunning,
    submit: runViaSource,
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
        subtitle={subtitle}
        onToggleHistory={() => setSidebarOpen((value) => !value)}
        historyOpen={sidebarOpen}
      />

      <main className="workspace-grid">
        <section className="panel input-panel">
          <div className="scenario-tabs" role="tablist" aria-label="分发场景">
            {SCENARIO_TABS.map((item) => (
              <button
                key={item.key}
                role="tab"
                aria-selected={dispatchMode === item.key}
                className={
                  dispatchMode === item.key ? "scenario-tab active" : "scenario-tab"
                }
                onClick={() => setDispatchMode(item.key)}
              >
                <span className="scenario-tab-icon">{SCENARIO_ICONS[item.key]}</span>
                <span className="scenario-tab-label">{item.label}</span>
              </button>
            ))}
          </div>
          <ScenarioHint mode={dispatchMode} />

          <div className="sub-section">
            <div className="sub-section-head">
              <h3>事件来源</h3>
              <div className="segmented">
                <button
                  type="button"
                  className={eventSource === "user" ? "active" : ""}
                  onClick={() => setEventSource("user")}
                >
                  用户输入
                </button>
                <button
                  type="button"
                  className={eventSource === "system" ? "active" : ""}
                  onClick={() => setEventSource("system")}
                >
                  系统事件
                </button>
              </div>
            </div>
            {eventSource === "user" ? (
              <TextImageEventInput
                userText={userText}
                setUserText={setUserText}
                onFileChange={setImageFile}
              />
            ) : (
              <SystemEventPicker
                selectedEventId={systemEvent?.event_id ?? ""}
                onSelect={pickSystemEvent}
                onError={setNotice}
              />
            )}
          </div>

          {dispatchMode === "single_target" && (
            <SingleTargetScenario {...sharedScenarioProps} />
          )}
          {dispatchMode === "multi_user" && <MultiUserScenario {...sharedScenarioProps} />}
          {dispatchMode === "cross_platform" && (
            <CrossPlatformScenario {...sharedScenarioProps} />
          )}
        </section>

        <StatusPanel
          run={run}
          notice={notice}
          selectedEventId={selectedEventId}
          onSelect={(eventId) => {
            setSelectedEventId(eventId);
            if (run?.run_id) void loadDetail(run.run_id, eventId);
          }}
        />

        <ResultPanel
          runId={run?.run_id ?? null}
          run={run}
          result={selectedResult}
          payload={detail}
          trace={trace}
          loading={loadingDetail}
        />
      </main>

      <HistorySidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        activeRunId={activeRunId}
        refreshTick={refreshTick}
        onSelectRun={handleSelectHistory}
        onAfterDelete={handleAfterDelete}
      />
    </div>
  );
}

/**
 * 文本 + 可选图片的事件录入（图文形式）：一段事件描述文本（必填），
 * 可附加一张本地图片由后端 OCR 后与文字融合。被多人 / 跨平台 以及
 * 单人定向的"图文"模式共用。
 */
function TextImageEventInput({
  userText,
  setUserText,
  onFileChange,
}: {
  userText: string;
  setUserText: (value: string) => void;
  onFileChange: (file: File | null) => void;
}) {
  return (
    <div className="event-input">
      <label className="json-editor">
        事件描述（输入一段文本，系统自动解析为事件）
        <textarea
          value={userText}
          rows={12}
          onChange={(event) => setUserText(event.target.value)}
          placeholder="例如：某地突降暴雨引发城市内涝，多个路段积水严重，市民出行受阻，网上出现多种声音……"
          spellCheck={false}
        />
      </label>
      <p className="form-hint">
        与系统事件的区别仅在于事件由文本解析得到，其余分发逻辑（选人 / 策略 /
        输出）完全一致。
      </p>
      <div className="optional-image">
        <h4>可选：附加图片</h4>
        <p className="form-hint">
          图片将经过本地 OCR 识别其中文字，与上方文字融合后作为事件描述。
          不联网、不调用大模型；留空则仅用文字。
        </p>
        <ImageEventForm onFileChange={onFileChange} />
      </div>
    </div>
  );
}

function ScenarioHint({ mode }: { mode: DispatchMode }) {
  const item = SCENARIO_TABS.find((entry) => entry.key === mode);
  if (!item) return null;
  return <p className="scenario-summary">{item.hint}</p>;
}
