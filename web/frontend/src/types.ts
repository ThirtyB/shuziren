/**
 * 共享 TypeScript 类型：与后端 Pydantic 模型对齐。
 *
 * 新增的 ``DispatchMode`` 是本前端最重要的扩展：把"分发场景"
 * 从一个隐式行为提升为顶层 Tab 的强类型键，确保 UI 与请求体不漂移。
 */

export type TabKey = "text" | "image" | "eval";
export type RunState = "queued" | "running" | "completed" | "failed";

/** 三种分发场景 —— 与后端 ``RunOptions.dispatch_mode`` 一一对应。 */
export type DispatchMode = "single_target" | "multi_user" | "cross_platform";

export const SCENARIO_TABS: ReadonlyArray<{ key: DispatchMode; label: string; hint: string }> = [
  {
    key: "single_target",
    label: "单人定向分发",
    hint: "指定一个数字人，强制作为本次发布的唯一传播节点（常用于 VIP/KOL 测试或定向响应）。",
  },
  {
    key: "multi_user",
    label: "多人协同分发",
    hint: "在一个模拟平台上，按角色（首发/互动/扩散/支持）自动选人，组成多角色小队。",
  },
  {
    key: "cross_platform",
    label: "跨平台并行分发",
    hint: "为多个平台分别指定人数，每个平台独立选人，正文与动作按平台策略生成。",
  },
];

export type RunStatus = {
  run_id: string;
  mode: TabKey;
  dispatch_mode: DispatchMode;
  state: RunState;
  created_at: string;
  updated_at: string;
  total: number;
  completed: number;
  failed: number;
  current_event_id?: string | null;
  output_dir: string;
  trace_dir: string;
  results: RunResult[];
  errors: RunError[];
};

export type RunResult = {
  event_id: string;
  event_name: string;
  dispatch_mode: DispatchMode;
  selected_digital_human_ids: string[];
  content_llm?: Record<string, unknown>;
  json_output_path: string;
  /** 单人定向分发专属：目标诊断。 */
  target_dispatch?: {
    target_user_id?: unknown;
    target_platform?: unknown;
    target_role?: unknown;
    warnings?: string[];
  };
  /** 跨平台分发专属：每个平台的选取与缺口。 */
  per_platform?: Array<{
    platform_key?: unknown;
    requested_count?: unknown;
    selected_count?: unknown;
    shortfall_count?: unknown;
    safe_shortfall_count?: unknown;
    selected_user_ids?: string[];
  }>;
};

export type RunError = {
  event_id?: string | null;
  message: string;
};

export type EventForm = {
  domain: string;
  event_title: string;
  event_summary: string;
  target: string;
  opinion_variants: string;
};

/** 事件来源模式：用户输入（表单/JSON）、系统事件（从 ``/api/events`` 选）、图片输入（多模态识别）。 */
export type EventSource = "user" | "system" | "image";

/** 系统事件摘要：来自 ``/api/events`` 列表接口；不包含 opinion_variants 全文。 */
export type SystemEventSummary = {
  event_id: string;
  domain: string;
  event_title: string;
  event_summary: string;
  is_synthetic: boolean;
  opinion_variants_count: number;
};

export type SystemEventListResponse = {
  total: number;
  limit: number;
  offset: number;
  domains: string[];
  items: SystemEventSummary[];
};

export type PlatformSpecPayload = {
  platform_key: string;
  requested_count: number;
};

export type RunOptions = {
  profile_limit?: number | null;
  max_selected_nodes: number;
  risk_level?: "low" | "medium" | "high" | null;
  campaign_window_hours: number;
  max_frequency_per_day: number;
  allowed_platforms: string[];
  dispatch_mode: DispatchMode;
  target_user_id?: string | null;
  dispatch_platform?: string | null;
  platform_specs: PlatformSpecPayload[];
  use_llm: boolean;
  event_limit?: number | null;
  event_id?: string | null;
};

/** 压测任务的执行状态：与后端 benchmark.json 的 ``state`` 对齐。 */
export type BenchmarkState = "queued" | "running" | "completed" | "failed";

/** 压测里单个测试点（一轮）使用的事件；一次测试内每个测试点事件都不同。 */
export type BenchmarkEventEntry = {
  iteration?: number;
  event_id: string;
  event_title?: string | null;
  domain?: string | null;
};

/** 后端压测 /api/benchmarks 的耗时数据形状，前端只读这一份做柱状图与明细展示。 */
export type BenchmarkStageTimes = Record<string, number>;

/** 压测请求体：与后端 ``BenchmarkRequest`` 对齐。 */
export type BenchmarkRequest = {
  options: RunOptions;
  iterations?: number;
  event_id?: string | null;
  label?: string | null;
};

export type BenchmarkRun = {
  iteration: number;
  run_id: string;
  ok: boolean;
  error?: string;
  elapsed_seconds: number;
  stage_times: BenchmarkStageTimes;
  /** 本轮随机抽到的事件（异步压测里每轮都不一样）。 */
  event?: BenchmarkEventEntry | null;
  summary: RunResult | null;
};

export type BenchmarkSummary = {
  iterations: number;
  successful_iterations: number;
  failed_iterations: number;
  total_seconds: number;
  average_seconds: number;
  min_seconds: number;
  max_seconds: number;
  stage_averages: Record<string, number>;
  stage_min_max: Record<string, { min: number; max: number; samples: number[] }>;
  per_iteration: Array<{ iteration: number; seconds: number }>;
};

/** 单条压测的完整状态：``POST`` 立即返回 queued，前端凭 benchmark_id 轮询。 */
export type BenchmarkResponse = {
  benchmark_id: string;
  label: string | null;
  state: BenchmarkState;
  iterations: number;
  completed: number;
  failed: number;
  successful_iterations: number;
  failed_iterations: number;
  options: RunOptions;
  /** 每个测试点用的事件（随着轮次推进逐步追加）。 */
  events: BenchmarkEventEntry[];
  runs: BenchmarkRun[];
  /** 刚开始跑时为 null；至少成功一轮后才有汇总数据。 */
  summary: BenchmarkSummary | null;
  output_dir: string;
  created_at: string;
  updated_at: string;
  error: string | null;
};

/** 测试历史列表条目：``GET /api/benchmarks`` 返回的轻量结构。 */
export type BenchmarkHistoryItem = {
  benchmark_id: string;
  label: string | null;
  state: BenchmarkState;
  iterations: number;
  completed: number;
  failed: number;
  successful_iterations: number;
  failed_iterations: number;
  use_llm: boolean;
  dispatch_mode: DispatchMode;
  event_count: number;
  events: BenchmarkEventEntry[];
  created_at: string;
  updated_at: string;
  total_seconds: number | null;
  average_seconds: number | null;
  output_dir: string;
  error: string | null;
};
