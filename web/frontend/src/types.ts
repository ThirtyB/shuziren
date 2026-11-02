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
    label: "单人定向",
    hint: "",
  },
  {
    key: "multi_user",
    label: "多人协同",
    hint: "",
  },
  {
    key: "cross_platform",
    label: "跨平台并行",
    hint: "",
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
  /** 端到端耗时（秒）：从收到请求到本次运行产出结果。进行中为 null。 */
  elapsed_seconds: number | null;
};

export type RunResult = {
  event_id: string;
  event_name: string;
  dispatch_mode: DispatchMode;
  selected_digital_human_ids: string[];
  content_llm?: Record<string, unknown>;
  /** 单次事件的完整五维调度策略产物，前端「具体结果」卡片展开查看。 */
  strategy_output?: Record<string, unknown>;
  /** 特殊支持节点（水军）汇总信息。 */
  support_nodes?: {
    enabled?: boolean;
    count?: number;
    platforms?: unknown;
    core_actor_count?: number;
  };
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
  /** 本轮评估各模块耗时（秒）：键为模块名，值为墙钟耗时。用于「!」展开查看。 */
  stage_times?: Record<string, number>;
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

/** 事件来源模式：用户输入（文字必填、图片可选）、系统事件（从 ``/api/events`` 选）。 */
export type EventSource = "user" | "system";

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
  // 单人定向两种打法：self_operate=思路一（自运营闭环）/ attention=思路二（多数字人 @ 指定路人）。
  single_target_mode?: "self_operate" | "attention";
  // 思路二中 @ 路人引用的旧帖编号（可选；后端默认 POST-0001）。
  target_old_post_id?: string | null;
  // 特殊支持节点参与：放大参与人数（仅点赞/转发，无文本），不计入选人上限。
  water_army_enabled?: boolean;
  water_army_count?: number;
  platform_specs: PlatformSpecPayload[];
  use_llm: boolean;
  event_limit?: number | null;
  event_id?: string | null;
};

/* ---------------- 总体测试（test-run）：多子任务组合批量评测 ---------------- */

export type TestRunState = "queued" | "running" | "completed" | "failed";

export type PlatformSpec = {
  platform_key: string;
  requested_count: number;
};

export type SubTaskKind = "single_target" | "multi_user" | "cross_platform";
export type SingleTargetMode = "self_operate" | "attention";

/** 一个子任务：单人（定向 / 注意）、多人、跨平台，均可指定是否含图与跑几条随机事件。 */
export type SubTaskSpec = {
  kind: SubTaskKind;
  single_target_mode?: SingleTargetMode;
  /** 单人定向必须手动指定目标数字人。 */
  target_user_id?: string | null;
  /** 数字人注意（attention）模式下引用的"原帖编号"，仅用于 @ 点名，不读取内容。 */
  target_old_post_id?: string | null;
  /** 多人 / 跨平台的人数与平台配置。 */
  platform_specs: PlatformSpec[];
  /** 是否含图：true 时系统随机抽一张图 OCR 成事件再分发。 */
  with_image?: boolean;
  /** 本子任务跑几条随机事件（每条独立计时）。 */
  event_count?: number;
  /** 时间窗口：该子任务的投放窗口（小时）。 */
  campaign_window_hours?: number;
  /** 特殊支持节点（水军）：仅点赞 / 转发以放大参与人数。 */
  water_army_enabled?: boolean;
  water_army_count?: number;
  /** 是否启用 LLM（由顶栏开关统一控制，按子任务透传）。 */
  use_llm?: boolean;
};

export type TestRunStageTimes = Record<string, number>;

export type TestRunPerEvent = {
  event_id: string;
  event_title?: string | null;
  stage_times: TestRunStageTimes;
  /** 该事件真实墙钟耗时（秒）。 */
  elapsed_seconds?: number | null;
  summary: RunResult | null;
  error?: string | null;
};

export type TestRunSubTaskResult = {
  index: number;
  kind: SubTaskKind;
  single_target_mode?: SingleTargetMode;
  with_image: boolean;
  target_user_id?: string | null;
  platform_specs: PlatformSpec[];
  event_count: number;
  ok: boolean;
  error?: string | null;
  stage_times_list: TestRunStageTimes[];
  per_event: TestRunPerEvent[];
  subtask_summary: TestRunSummary | null;
};

/** 一组运行结果的阶段耗时汇总（与后端 _aggregate_stage_timings 输出对齐）。 */
export type TestRunSummary = {
  iterations: number;
  successful_iterations: number;
  failed_iterations: number;
  total_seconds: number;
  average_seconds: number;
  min_seconds: number;
  max_seconds: number;
  stage_averages: TestRunStageTimes;
  stage_min_max: Record<string, { min: number; max: number; samples: number[] }>;
  per_iteration: Array<{ iteration: number; seconds: number }>;
};

/** 单次总体测试请求的完整状态：POST 立即返回 queued，前端凭 test_run_id 轮询。 */
export type TestRunResponse = {
  test_run_id: string;
  label: string | null;
  state: TestRunState;
  total_subtasks: number;
  completed: number;
  failed: number;
  subtasks: TestRunSubTaskResult[];
  summary: TestRunSummary | null;
  output_dir: string;
  created_at: string;
  updated_at: string;
  error: string | null;
};

/** 测试历史列表条目：GET /api/test-runs 返回的轻量结构。 */
export type TestRunHistoryItem = {
  test_run_id: string;
  label: string | null;
  state: TestRunState;
  total_subtasks: number;
  completed: number;
  failed: number;
  created_at: string;
  updated_at: string;
  total_seconds: number | null;
  average_seconds: number | null;
  output_dir: string | null;
  error: string | null;
};

export type TestRunRequest = {
  subtasks: SubTaskSpec[];
  label?: string | null;
};
