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
