/** HTTP 工具：把 ``fetch`` 封装为强类型的 ``fetchJson/postJson``。 */

import type {
  RunStatus,
  SystemEventListResponse,
  TestRunHistoryItem,
  TestRunRequest,
  TestRunResponse,
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
 * 用户输入（文字必填 + 图片可选）：把 ``event_text`` 与可选的图片一起上传到
 * ``/api/runs/text-image``。后端对图片做**本地 OCR**（RapidOCR）读出文字，
 * 再把 OCR 文字与用户输入文字融合成一段事件描述，交给与纯文本完全相同的规则
 * 事件解析链路。全程不调用大模型；下游选人 / 策略 / 输出与系统事件完全一致。
 *
 * 图片为空时后端退化为纯文本事件；图片里读不出文字时后端只保留用户输入文字，
 * 不会编造内容。
 */
export async function runTextImageEvent(
  eventText: string,
  image: File | null,
  options: import("../types").RunOptions,
): Promise<RunStatus> {
  const formData = new FormData();
  formData.append("event_text", eventText);
  if (image) {
    formData.append("image", image);
  }
  formData.append("options", JSON.stringify(options));
  const response = await fetch("/api/runs/text-image", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as RunStatus;
}

/**
 * 提交一次"总体测试"：后端**异步**执行，这里只会拿到 state=queued 的初始状态与
 * ``test_run_id``；真正的进度与结果要靠 ``getTestRun`` 轮询。
 *
 * 因为 POST 立即返回，前端可连续提交多次（后端串行排队执行）。
 */
export async function createTestRun(request: TestRunRequest): Promise<TestRunResponse> {
  return postJson<TestRunResponse>("/api/test-runs", request);
}

/** 轮询单条总体测试的完整状态（每个子任务 / 事件 / 阶段耗时 / 汇总）。 */
export async function getTestRun(testRunId: string): Promise<TestRunResponse> {
  return fetchJson<TestRunResponse>(`/api/test-runs/${encodeURIComponent(testRunId)}`);
}

/** 测试历史列表（总体测试专用，与 ``/api/runs`` 的运行历史隔离）。 */
export async function listTestRuns(): Promise<TestRunHistoryItem[]> {
  const payload = await fetchJson<{ test_runs: TestRunHistoryItem[] }>("/api/test-runs");
  return payload.test_runs ?? [];
}

export async function deleteTestRun(testRunId: string): Promise<void> {
  const response = await fetch(`/api/test-runs/${encodeURIComponent(testRunId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
}
