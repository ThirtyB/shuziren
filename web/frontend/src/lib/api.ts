/** HTTP 工具：把 ``fetch`` 封装为强类型的 ``fetchJson/postJson``。 */

import type {
  BenchmarkHistoryItem,
  BenchmarkRequest,
  BenchmarkResponse,
  RunStatus,
  SystemEventListResponse,
} from "../types";

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    return JSON.stringify(payload);
  } catch {
    return response.statusText;
  }
}

export { readError };

export type ListSystemEventsParams = {
  limit?: number;
  offset?: number;
  domain?: string;
  q?: string;
};

export async function listSystemEvents(
  params: ListSystemEventsParams = {},
): Promise<SystemEventListResponse> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  if (params.domain) query.set("domain", params.domain);
  if (params.q) query.set("q", params.q);
  const path = query.toString() ? `/api/events?${query.toString()}` : "/api/events";
  return fetchJson<SystemEventListResponse>(path);
}

export async function getSystemEvent(eventId: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(`/api/events/${encodeURIComponent(eventId)}`);
}

export type RunHistoryEntry = {
  run_id: string;
  mode: "text" | "image" | "eval";
  dispatch_mode: "single_target" | "multi_user" | "cross_platform";
  state: "queued" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  total: number;
  completed: number;
  failed: number;
  output_dir: string;
  trace_dir: string;
};

export async function listRuns(): Promise<RunHistoryEntry[]> {
  const payload = await fetchJson<{ runs: RunHistoryEntry[] }>("/api/runs");
  return payload.runs ?? [];
}

export async function deleteRun(runId: string): Promise<void> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
}

/**
 * 多模态图像事件输入：把图片上传到 ``/api/runs/image``，由后端视觉 LLM
 * （``VISION_LLM_MODEL``，默认 ``qwen3-vl-flash``）识别出 ``event_title /
 * event_summary / domain / keywords / target / opinion_variants``，再走
 * 与文本事件完全相同的策略生成管线。
 *
 * 视觉模型不可用时后端会回退到 ``image_event_loader._fallback_result``，
 * 前端这边只负责提交文件；识别结果（match_score / fallback_used）由详情
 * 面板展示，不需要前端做二次判断。
 */
export async function runImageEvent(
  image: File,
  options: import("../types").RunOptions,
): Promise<RunStatus> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("options", JSON.stringify(options));
  const response = await fetch("/api/runs/image", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as RunStatus;
}

/**
 * 提交一次压测：后端**异步**执行，这里只会拿到 state=queued 的初始状态与
 * ``benchmark_id``；真正的进度与结果要靠 ``getBenchmark`` 轮询。
 *
 * 因为 POST 立即返回，前端点完按钮就能解锁，用户可以连续提交多次测试
 * （后端线程池串行排队执行）。
 */
export async function runBenchmark(request: BenchmarkRequest): Promise<BenchmarkResponse> {
  return postJson<BenchmarkResponse>("/api/benchmarks", request);
}

/** 轮询单条压测的完整状态（每轮随机事件 / 阶段耗时 / 汇总）。 */
export async function getBenchmark(benchmarkId: string): Promise<BenchmarkResponse> {
  return fetchJson<BenchmarkResponse>(`/api/benchmarks/${encodeURIComponent(benchmarkId)}`);
}

/** 测试历史列表（压测专用，与 ``/api/runs`` 的运行历史隔离）。 */
export async function listBenchmarks(): Promise<BenchmarkHistoryItem[]> {
  const payload = await fetchJson<{ benchmarks: BenchmarkHistoryItem[] }>(
    "/api/benchmarks",
  );
  return payload.benchmarks ?? [];
}

export async function deleteBenchmark(benchmarkId: string): Promise<void> {
  const response = await fetch(`/api/benchmarks/${encodeURIComponent(benchmarkId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
}
