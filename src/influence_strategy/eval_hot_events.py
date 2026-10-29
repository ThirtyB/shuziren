from __future__ import annotations

import itertools
import json
import logging
import random
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Literal, Sequence

from .content_arsenal import (
    AmmoVectorIndex,
    ContentArsenal,
    _arsenal_fingerprint,
    _stable_seed,
)
from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import (
    FeatureBuildSummary,
    NodeScore,
    ScoreResult,
    StrategyNodePlan,
    StrategyResult,
)
from .pipeline import PipelineArtifacts, StrategyPipeline
from .platform_dispatcher import PlatformSpec
from .platform_profiles import (
    DATASET_PLATFORM_VALUES,
    dataset_platform_value,
    is_valid_dataset_platform,
    normalize_platform_key,
)

TRACE_PREVIEW_LIMIT = 25

#: 文案生成是否调用大模型。默认关闭 —— 本地部署不依赖第三方 API，正文改用规则
#: 生成的占位文本（``_fallback_node_content``）。占位文本已带话题、角色与表达
#: 风格，足够撑起动作清单与下游结构，只是不具备真正的创作性。
#:
#: 需要恢复时把它改回 True 并配置 API Key；关闭状态下 ``_build_llm_node_texts``
#: 会立刻返回空 dict（``fallback_reason="llm_disabled"``），不产生任何网络请求。
CONTENT_LLM_ENABLED = False

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# ④ 内容生成 LLM 的可观测性
# --------------------------------------------------------------------------- #
# 与 ``feature_builder._augment_with_llm``（②）同款设计：**只做统计与日志，
# 不改变任何业务行为**。这里单独拆一套指标是因为 ④ 是"核心策略已出、只差文案"
# 的最后一步，第二轮评估"能否把 ④ 挪出实时链路"必须有它自己的耗时/失败率基线。
#
# 记录字段：
#   * ``success`` / ``failure``：本次是否真的拿到 LLM 文案
#   * ``fallback_reason``：降级原因（llm_disabled / no_eligible_nodes /
#     llm_not_configured / llm_request_failed:<异常类型> / invalid_llm_response /
#     empty_llm_content）
#   * ``attempted``：是否真的发起了远端请求（用于区分"没请求"和"请求失败"）
#   * ``elapsed_sec``：本次调用墙钟耗时
#   * ``prompt_chars``：请求体字符数（用于排查是否与请求体过大相关）
_CONTENT_LLM_METRICS: dict[str, Any] = {
    "calls": 0,
    "success": 0,
    "failure": 0,
    "attempted": 0,
    "fallback_by_reason": {},
    "last_call": None,
}
_CONTENT_LLM_METRICS_LOCK = threading.Lock()


def _record_content_llm_outcome(
    *,
    success: bool,
    fallback_reason: str | None,
    attempted: bool,
    prompt_chars: int = 0,
    elapsed_sec: float = 0.0,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """记录一次内容生成 LLM 调用的结果。仅在 ``_build_llm_node_texts`` 内部调用。"""
    with _CONTENT_LLM_METRICS_LOCK:
        _CONTENT_LLM_METRICS["calls"] += 1
        if success:
            _CONTENT_LLM_METRICS["success"] += 1
        else:
            _CONTENT_LLM_METRICS["failure"] += 1
        if attempted:
            _CONTENT_LLM_METRICS["attempted"] += 1
        if fallback_reason:
            bucket = _CONTENT_LLM_METRICS["fallback_by_reason"]
            bucket[fallback_reason] = bucket.get(fallback_reason, 0) + 1
        _CONTENT_LLM_METRICS["last_call"] = {
            "success": success,
            "fallback_reason": fallback_reason,
            "attempted": attempted,
            "prompt_chars": prompt_chars,
            "elapsed_sec": elapsed_sec,
            "error_type": error_type,
            "error_message": error_message,
        }


def get_content_llm_metrics() -> dict[str, Any]:
    """获取内容生成 LLM（④）的统计快照（线程安全）。"""
    with _CONTENT_LLM_METRICS_LOCK:
        return {
            "calls": _CONTENT_LLM_METRICS["calls"],
            "success": _CONTENT_LLM_METRICS["success"],
            "failure": _CONTENT_LLM_METRICS["failure"],
            "attempted": _CONTENT_LLM_METRICS["attempted"],
            "fallback_by_reason": dict(_CONTENT_LLM_METRICS["fallback_by_reason"]),
            "last_call": (
                dict(_CONTENT_LLM_METRICS["last_call"])
                if _CONTENT_LLM_METRICS["last_call"] is not None
                else None
            ),
        }


def reset_content_llm_metrics() -> None:
    """清零内容生成 LLM（④）的统计。仅供测试使用。"""
    with _CONTENT_LLM_METRICS_LOCK:
        _CONTENT_LLM_METRICS["calls"] = 0
        _CONTENT_LLM_METRICS["success"] = 0
        _CONTENT_LLM_METRICS["failure"] = 0
        _CONTENT_LLM_METRICS["attempted"] = 0
        _CONTENT_LLM_METRICS["fallback_by_reason"] = {}
        _CONTENT_LLM_METRICS["last_call"] = None

# 同一个 event 的不同运行（automatic / targeted / 多平台组合）会写入同一个
# ``{trace_dir}/{event_id}/``。为避免后一次运行静默覆盖前一次的证据，
# 每次写入前先把上一次的顶层步骤文件快照到 ``{event_id}/runs/<timestamp>_<label>/``。
TRACE_RUNS_DIRNAME = "runs"

# --------------------------------------------------------------------------- #
# 每次评估运行独占一个目录
# --------------------------------------------------------------------------- #
# 目录结构恒为 ``<输出分类根目录>/<mode>/run_NNN/``：
#
#   eval/output/
#   ├── single_target/run_001/
#   ├── multi_user/run_001/
#   └── multi_platform/run_001/
#
# 本次运行的 JSON、摘要、trace 与错误日志全部写在同一个 ``run_NNN/`` 内，
# 因此不同运行之间不会互相覆盖，也不再需要把结果复制到别处。
EVAL_MODE_SINGLE_TARGET = "single_target"
EVAL_MODE_MULTI_USER = "multi_user"
EVAL_MODE_MULTI_PLATFORM = "multi_platform"
#: 三种运行模式，同时也是 ``eval/output/`` 下的三个分类目录名。
EVAL_MODES: tuple[str, ...] = (
    EVAL_MODE_SINGLE_TARGET,
    EVAL_MODE_MULTI_USER,
    EVAL_MODE_MULTI_PLATFORM,
)

EVAL_DIRNAME = "eval"
EVAL_OUTPUT_SUBDIRNAME = "output"
RUN_DIR_PREFIX = "run_"
RUN_DIR_DIGITS = 3
TRACE_DIRNAME = "trace"
#: 本次运行的摘要（与打印到 stdout 的内容一致）。
RUN_SUMMARY_FILENAME = "run_summary.json"
#: 运行失败时写入当前运行目录的错误日志。
RUN_ERROR_FILENAME = "run_error.json"

# --------------------------------------------------------------------------- #
# 跨平台分发的输出契约（与多人分发共用五维结构）
# --------------------------------------------------------------------------- #
# 多平台分发（cross_platform）与多人分发共用**同一套**最终业务 JSON 契约：
# 顶层都是 ``事件名称`` / ``输出格式版本`` / ``五维调度策略``，五个维度的字段名
# 完全一致，因此前端可以复用同一套渲染组件，执行者也能用同一套读法消费。
#
# 差异只落在「平台」维度：多平台模式在 ``五维调度策略.平台`` 下额外输出
# ``跨平台总览`` 与 ``各平台方案``（每个平台各自的人数、缺口、安全等级、内容
# 计划与时间计划），把平台差异集中到一处，而不是散在顶层的说明文本里。
FIVE_DIMENSION_STRATEGY_KEY = "五维调度策略"
#: 多平台最终业务 JSON 的输出版本标识（区别于旧版 ``action_schema_v5_...``）。
MULTIPLATFORM_OUTPUT_FORMAT_VERSION = "action_schema_v6_multi_platform_reuse"
MULTIPLATFORM_DRAFT_NOTE = (
    "以下内容为各平台分别复用多人协同（Selector + StrategyGenerator）管线生成的结构化"
    "内容草案与动作草案，尚未在任何真实平台发布；每个平台都是一次独立的多人协同分发，"
    "按各自的 requested_count 选人并产出与单平台完全同构的五维调度策略。"
)
MULTIPLATFORM_SCOPE_NOTE = (
    "本文件是多平台分发模式的最终业务结果：每个平台的五维调度策略结构与单人/多人分发"
    "完全一致（复用同一套协同逻辑），因此前端可逐个平台原样复用多人协同的渲染；"
    "「跨平台总览」仅对各平台的人数、安全等级与跨平台重复数字人做整合汇总，不另起选人。"
)

#: ``NodeScore.priority_tier`` → 动作优先级，与 ``Selector._dispatch_priority`` 同源。
PRIORITY_TIER_TO_DISPATCH_PRIORITY: dict[str, str] = {
    "high": "p1",
    "medium": "p2",
    "low": "p3",
}


def load_hot_events(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict):
        events = [payload]
    else:
        raise ValueError("Hot event eval input must be a JSON object or a JSON array.")

    normalized_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each hot event item must be a JSON object.")
        normalized_events.append(event)
    return normalized_events


def select_hot_event(
    events: list[dict[str, Any]],
    event_id: str | None = None,
) -> dict[str, Any]:
    if not events:
        raise ValueError("Hot event eval input is empty.")
    if event_id is None:
        return events[0]

    for event in events:
        if str(event.get("event_id", "")) == event_id:
            return event
    raise ValueError(f"Hot event not found: {event_id}")


def select_hot_events(
    events: list[dict[str, Any]],
    *,
    event_id: str | None = None,
    event_limit: int | None = 10,
) -> list[dict[str, Any]]:
    if event_id is not None:
        return [select_hot_event(events, event_id=event_id)]
    if not events:
        raise ValueError("Hot event eval input is empty.")
    if event_limit is None:
        return events
    if event_limit < 1:
        raise ValueError("event_limit must be at least 1.")
    return events[:event_limit]


def hot_event_to_pipeline_payload(
    hot_event: dict[str, Any],
    *,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
) -> dict[str, Any]:
    event_id = str(hot_event.get("event_id") or "hot_event")
    event_title = str(hot_event.get("event_title") or event_id)
    event_summary = str(hot_event.get("event_summary") or "").strip()
    target = str(hot_event.get("target") or "扩大热点信息触达并引导理性讨论").strip()
    domain = str(hot_event.get("domain") or "general").strip() or "general"
    variants = _normalize_variants(hot_event.get("opinion_variants", []))

    variant_lines = "\n".join(
        f"{index}. {variant}" for index, variant in enumerate(variants, start=1)
    )
    event_description = "\n".join(
        item
        for item in (
            event_title,
            event_summary,
            f"热点领域: {domain}",
            f"传播目标: {target}",
            "任务约束: 输出影响力事件分发策略，不要按照商品推荐任务理解。",
            "热点叙述变体:",
            variant_lines,
        )
        if item
    )

    resolved_risk_level = risk_level or _infer_hot_event_risk_level(domain)
    return {
        "event_id": event_id,
        "product_name": "abc_reading",
        "event_title": event_title,
        "event_description": event_description,
        "target_goal": target,
        "target_audience": ["general_public"],
        "constraints": {
            "risk_level": resolved_risk_level,
            "max_selected_nodes": max_selected_nodes,
            "max_frequency_per_day": max_frequency_per_day,
            "campaign_window_hours": campaign_window_hours,
            "allowed_platforms": allowed_platforms
            or list(DATASET_PLATFORM_VALUES),
        },
    }


@dataclass(frozen=True, slots=True)
class EvalRunDirs:
    """一次评估运行的自有目录。

    ``run_dir`` 是本次运行的唯一根目录：策略 JSON、摘要与错误日志写在它内部，
    trace 默认也写在它内部的 ``trace/`` 子目录里。
    """

    mode: str
    run_id: str
    run_dir: Path
    output_dir: Path
    trace_dir: Path


def default_output_root(workspace_root: str | Path) -> Path:
    """默认的输出分类根目录：``<workspace_root>/eval/output``。"""

    return Path(workspace_root) / EVAL_DIRNAME / EVAL_OUTPUT_SUBDIRNAME


def run_dir_number(path: Path) -> int | None:
    """解析 ``run_NNN`` 目录名中的编号；不符合命名规则时返回 ``None``。"""

    name = path.name
    if not name.startswith(RUN_DIR_PREFIX):
        return None
    digits = name[len(RUN_DIR_PREFIX) :]
    if not digits.isdigit():
        return None
    return int(digits)


def _highest_run_number(category_roots: Sequence[Path]) -> int:
    highest = 0
    for category_root in category_roots:
        if not category_root.is_dir():
            continue
        for child in category_root.iterdir():
            if not child.is_dir():
                continue
            number = run_dir_number(child)
            if number is not None and number > highest:
                highest = number
    return highest


def allocate_run_dir(
    category_root: Path,
    *,
    reserved_roots: Sequence[Path] = (),
) -> Path:
    """在 ``category_root`` 下原子地分配下一个 ``run_NNN`` 目录。

    编号规则：取 ``category_root``（以及 ``reserved_roots`` 中同名的分类目录）
    里已有 ``run_NNN`` 的最大编号 + 1。

    * 空目录 → ``run_001``；
    * 已有 ``run_001``/``run_002`` → ``run_003``；
    * 编号不连续（例如只有 ``run_005``）→ ``run_006``，不会填补空洞而撞上历史结果；
    * 并发启动时目录用 ``mkdir(exist_ok=False)`` 抢占，同一编号只会有一个进程成功，
      失败的一方递增到下一个编号后再试，因此不会出现两个进程写入同一个 ``run_NNN``。
    """

    category_root.mkdir(parents=True, exist_ok=True)
    number = _highest_run_number((category_root, *reserved_roots)) + 1
    while True:
        candidate = category_root / f"{RUN_DIR_PREFIX}{number:0{RUN_DIR_DIGITS}d}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            # 编号已被占用：可能是并发进程刚建好的目录，也可能是同名非目录文件。
            # 这里只递增编号，不再回头读取最大编号，保证循环单调推进且不会死循环。
            number += 1
            continue
        return candidate


def resolve_eval_run_dirs(
    *,
    workspace_root: str | Path,
    mode: str,
    output_root: str | Path | None = None,
    trace_root: str | Path | None = None,
) -> EvalRunDirs:
    """为一次评估运行创建独立目录，并返回本次运行的输出/trace 目录。

    * ``mode`` 必须是 :data:`EVAL_MODES` 之一，它同时是分类目录名。
    * ``output_root`` 默认 ``<workspace_root>/eval/output``；运行目录恒为
      ``<output_root>/<mode>/run_NNN/``。
    * ``trace_root`` 为 ``None`` 时 trace 落在 ``<运行目录>/trace/``，
      保证一次运行的所有产物都在同一个目录内。
    * 显式给出 ``trace_root`` 时 trace 落在
      ``<trace_root>/<mode>/<run_id>/trace/``：仍然是一次运行独占的目录，
      编号与结果目录保持一致（分配编号时会同时查看两个分类目录），
      既不与其它运行混用，也不产生重复副本。
    """

    if mode not in EVAL_MODES:
        raise ValueError(f"未知的评估运行模式：{mode!r}，可选 {EVAL_MODES}。")

    resolved_output_root = (
        Path(output_root)
        if output_root is not None
        else default_output_root(workspace_root)
    )
    category_root = resolved_output_root / mode

    trace_category_root: Path | None = None
    if trace_root is not None:
        trace_category_root = Path(trace_root) / mode

    reserved: tuple[Path, ...] = (
        (trace_category_root,) if trace_category_root is not None else ()
    )
    run_dir = allocate_run_dir(category_root, reserved_roots=reserved)

    if trace_category_root is None:
        trace_path = run_dir / TRACE_DIRNAME
    else:
        trace_path = trace_category_root / run_dir.name / TRACE_DIRNAME
    trace_path.mkdir(parents=True, exist_ok=True)

    return EvalRunDirs(
        mode=mode,
        run_id=run_dir.name,
        run_dir=run_dir,
        output_dir=run_dir,
        trace_dir=trace_path,
    )


def _resolve_run_dirs(
    *,
    workspace_root: str | Path,
    mode: str,
    output_dir: str | Path | None,
    trace_dir: str | Path | None,
) -> EvalRunDirs:
    """把入口参数解析成本次运行的输出/trace 目录。

    注意区分两层语义：``run_eval.py`` 的 ``--output-dir`` / ``--trace-dir`` 是
    **分类根目录**，由入口先调用 :func:`resolve_eval_run_dirs` 完成分类与编号；
    而本函数的 ``output_dir`` / ``trace_dir`` 指的是**最终目录**。

    * ``output_dir`` 与 ``trace_dir`` 都为 ``None``：按统一结构自动分配
      ``<workspace_root>/eval/output/<mode>/run_NNN/``，trace 放在其中的 ``trace/``。
    * 显式给出 ``output_dir``：该目录即本次运行的输出目录，trace 默认放在它内部的
      ``trace/``；只有显式给出 ``trace_dir`` 时才改用该路径。
    * 只给出 ``trace_dir``：仍自动分配运行目录，trace 使用
      ``<trace_dir>/<mode>/<run_id>/trace/``，编号与运行目录一致。
    """

    if output_dir is None:
        return resolve_eval_run_dirs(
            workspace_root=workspace_root,
            mode=mode,
            output_root=None,
            trace_root=trace_dir,
        )

    run_path = Path(output_dir)
    trace_path = (
        Path(trace_dir) if trace_dir is not None else run_path / TRACE_DIRNAME
    )
    return EvalRunDirs(
        mode=mode,
        run_id=run_path.name,
        run_dir=run_path,
        output_dir=run_path,
        trace_dir=trace_path,
    )


def mode_for_legacy_run(target_user_id: str | None) -> str:
    """旧入口（非多平台）的运行模式：指定目标数字人即单人定向，否则多人分发。"""

    return EVAL_MODE_SINGLE_TARGET if target_user_id else EVAL_MODE_MULTI_USER


def write_eval_run_summary(run_dir: Path, payload: dict[str, Any]) -> Path:
    """把本次运行的摘要写入运行目录，避免摘要只存在于 stdout。"""

    path = Path(run_dir) / RUN_SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_eval_run_error(run_dir: Path, payload: dict[str, Any]) -> Path:
    """把运行失败信息写入运行目录，保证失败日志归属于本次运行。"""

    path = Path(run_dir) / RUN_ERROR_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_hot_event_evaluation(
    *,
    workspace_root: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    event_id: str | None = None,
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
    dispatch_platform: str | None = None,
    recall_size: int = 200,
    recall_mode: Literal["platform_aware"] = "platform_aware",
) -> tuple[Path, dict[str, Any]]:
    events = load_hot_events(input_path)
    hot_event = select_hot_event(events, event_id=event_id)
    run_dirs = _resolve_run_dirs(
        workspace_root=workspace_root,
        mode=EVAL_MODE_MULTI_USER,
        output_dir=output_dir,
        trace_dir=trace_dir,
    )
    output_path, output_payload, _stage_times = _run_hot_event_evaluation(
        workspace_root=workspace_root,
        output_dir=run_dirs.output_dir,
        hot_event=hot_event,
        profile_limit=profile_limit,
        max_selected_nodes=max_selected_nodes,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
        use_llm=use_llm,
        llm_client=llm_client,
        trace_dir=run_dirs.trace_dir,
        dispatch_platform=dispatch_platform,
        recall_size=recall_size,
        recall_mode=recall_mode,
    )
    return output_path, output_payload


def run_hot_event_evaluations(
    *,
    workspace_root: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    event_id: str | None = None,
    event_limit: int | None = 10,
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
    target_user_id: str | None = None,
    dispatch_platform: str | None = None,
    recall_size: int = 200,
    recall_mode: Literal["platform_aware"] = "platform_aware",
) -> list[tuple[Path, dict[str, Any]]]:
    events = load_hot_events(input_path)
    selected_events = select_hot_events(
        events,
        event_id=event_id,
        event_limit=event_limit,
    )
    run_dirs = _resolve_run_dirs(
        workspace_root=workspace_root,
        mode=mode_for_legacy_run(target_user_id),
        output_dir=output_dir,
        trace_dir=trace_dir,
    )

    results: list[tuple[Path, dict[str, Any]]] = []
    for hot_event in selected_events:
        output_path, output_payload, _stage_times = _run_hot_event_evaluation(
            workspace_root=workspace_root,
            output_dir=run_dirs.output_dir,
            hot_event=hot_event,
            profile_limit=profile_limit,
            max_selected_nodes=max_selected_nodes,
            risk_level=risk_level,
            campaign_window_hours=campaign_window_hours,
            max_frequency_per_day=max_frequency_per_day,
            allowed_platforms=allowed_platforms,
            use_llm=use_llm,
            llm_client=llm_client,
            trace_dir=run_dirs.trace_dir,
            target_user_id=target_user_id,
            dispatch_platform=dispatch_platform,
            recall_size=recall_size,
            recall_mode=recall_mode,
        )
        results.append((output_path, output_payload))

    return results


def run_hot_event_dict_evaluations(
    *,
    workspace_root: str | Path,
    events: list[dict[str, Any]],
    output_dir: str | Path | None = None,
    event_id: str | None = None,
    event_limit: int | None = None,
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
    target_user_id: str | None = None,
    dispatch_platform: str | None = None,
    recall_size: int = 200,
    recall_mode: Literal["platform_aware"] = "platform_aware",
) -> list[tuple[Path, dict[str, Any]]]:
    selected_events = select_hot_events(
        events,
        event_id=event_id,
        event_limit=event_limit,
    )
    run_dirs = _resolve_run_dirs(
        workspace_root=workspace_root,
        mode=mode_for_legacy_run(target_user_id),
        output_dir=output_dir,
        trace_dir=trace_dir,
    )

    results: list[tuple[Path, dict[str, Any]]] = []
    for hot_event in selected_events:
        output_path, output_payload, _stage_times = _run_hot_event_evaluation(
            workspace_root=workspace_root,
            output_dir=run_dirs.output_dir,
            hot_event=hot_event,
            profile_limit=profile_limit,
            max_selected_nodes=max_selected_nodes,
            risk_level=risk_level,
            campaign_window_hours=campaign_window_hours,
            max_frequency_per_day=max_frequency_per_day,
            allowed_platforms=allowed_platforms,
            use_llm=use_llm,
            llm_client=llm_client,
            trace_dir=run_dirs.trace_dir,
            target_user_id=target_user_id,
            dispatch_platform=dispatch_platform,
            recall_size=recall_size,
            recall_mode=recall_mode,
        )
        results.append((output_path, output_payload))
    return results


def run_hot_event_multiplatform_evaluations(
    *,
    workspace_root: str | Path,
    input_path: str | Path,
    platform_specs: list[PlatformSpec],
    output_dir: str | Path | None = None,
    event_id: str | None = None,
    event_limit: int | None = 10,
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
    recall_size: int = 200,
    recall_mode: Literal["platform_aware"] = "platform_aware",
) -> list[tuple[Path, dict[str, Any]]]:
    """多平台并行分发入口。

    沿用原有的事件解析 / 数据加载 / FeatureBuilder / Scorer 流程拿到真实
    ``ScoreResult``，再按每个平台自己的 ``requested_count`` 独立选取数字人。
    旧模式（automatic / targeted）的输出结构完全不变；多平台模式由
    :func:`build_multiplatform_output` 独立构造最终业务 JSON，旧版五维调度
    策略不会出现在多平台输出里。
    """

    if not platform_specs:
        raise ValueError("platform_specs 不能为空，请至少指定一个平台。")

    events = load_hot_events(input_path)
    selected_events = select_hot_events(
        events,
        event_id=event_id,
        event_limit=event_limit,
    )
    run_dirs = _resolve_run_dirs(
        workspace_root=workspace_root,
        mode=EVAL_MODE_MULTI_PLATFORM,
        output_dir=output_dir,
        trace_dir=trace_dir,
    )

    results: list[tuple[Path, dict[str, Any]]] = []
    for hot_event in selected_events:
        output_path, output_payload, _stage_times = _run_hot_event_multiplatform_evaluation(
            workspace_root=workspace_root,
            output_dir=run_dirs.output_dir,
            hot_event=hot_event,
            platform_specs=platform_specs,
            profile_limit=profile_limit,
            max_selected_nodes=max_selected_nodes,
            risk_level=risk_level,
            campaign_window_hours=campaign_window_hours,
            max_frequency_per_day=max_frequency_per_day,
            allowed_platforms=allowed_platforms,
            use_llm=use_llm,
            llm_client=llm_client,
            trace_dir=run_dirs.trace_dir,
            recall_size=recall_size,
            recall_mode=recall_mode,
        )
        results.append((output_path, output_payload))
    return results


def _run_single_platform_evaluation(
    *,
    workspace_root: Path,
    output_dir: Path,
    trace_dir: Path,
    hot_event: dict[str, Any],
    spec: PlatformSpec,
    profile_limit: int | None,
    risk_level: str | None,
    campaign_window_hours: int,
    max_frequency_per_day: int,
    allowed_platforms: list[str] | None,
    use_llm: bool,
    recall_size: int,
    recall_mode: str,
) -> dict[str, Any]:
    """单条平台任务：一次独立的多人协同分发（在独立线程中执行）。

    直接复用 :func:`_run_hot_event_evaluation`，把输入拆成该平台的「多人协同」任务：
    ``dispatch_platform=平台``、``max_selected_nodes=该平台目标人数``。产出与单平台
    多人协同**完全同构** 的五维调度策略（含动作清单），供上层整合成多平台结果。
    """
    root = Path(workspace_root)
    client = None
    if use_llm:
        try:
            client = OpenAICompatibleLLMClient.from_env_files(root)
        except Exception:
            client = None

    _, platform_payload, per_stage = _run_hot_event_evaluation(
        workspace_root=root,
        output_dir=output_dir,
        hot_event=hot_event,
        profile_limit=profile_limit,
        max_selected_nodes=spec.requested_count,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
        use_llm=use_llm,
        llm_client=client,
        trace_dir=trace_dir,
        dispatch_platform=spec.platform_key,
        recall_size=recall_size,
        recall_mode=recall_mode,
    )

    strategy = platform_payload.get("五维调度策略", {})
    target = strategy.get("目标对象", {}) if isinstance(strategy, dict) else {}
    selected_ids = (
        target.get("选取数字人id组", []) if isinstance(target, dict) else []
    )
    return {
        "平台": spec.platform_key,
        "目标人数": spec.requested_count,
        "选取人数": len(selected_ids),
        "安全等级分布": _aggregate_safety(strategy),
        "stage_times": per_stage,
        "json_output_path": str(_),
        "五维调度策略": strategy,
    }


def _run_hot_event_multiplatform_evaluation(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    hot_event: dict[str, Any],
    platform_specs: list[PlatformSpec],
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path,
    recall_size: int = 200,
    recall_mode: Literal["platform_aware"] = "platform_aware",
) -> tuple[Path, dict[str, Any], dict[str, float]]:
    """多平台分发 = 把输入拆成各平台的独立多人任务，**并行**复用多人协同管线。

    输入是「时间窗口 + 系统事件 + 各平台人数」。每个 ``platform_spec`` 被拆成一条
    独立的「多人协同分发」任务，直接复用 :func:`_run_hot_event_evaluation`
    （Selector + StrategyGenerator 全链路），传入 ``dispatch_platform=平台`` 与
    ``max_selected_nodes=该平台目标人数``。所有平台任务用线程池**并行**执行以加快
    效率，每个平台使用独立的 ``output_dir`` / ``trace_dir`` 子目录互不覆盖。

    每个平台产出的就是与单平台多人协同**完全同构** 的五维调度策略（含动作清单、
    时间/频率/内容计划），最后由 :func:`build_multiplatform_output` 把各平台结果
    **整合** 为「各平台结果 + 跨平台总览」，不另起一套选人逻辑。
    """
    root = Path(workspace_root)
    stage_times: dict[str, float] = {}
    total_start = time.perf_counter()
    platform_tag = "-".join(spec.platform_key for spec in platform_specs)

    common = dict(
        workspace_root=root,
        hot_event=hot_event,
        profile_limit=profile_limit,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
        use_llm=use_llm,
        recall_size=recall_size,
        recall_mode=recall_mode,
    )

    # 并行执行：每个平台一条独立多人任务，线程池并发跑完整管线以加快效率。
    platform_results: list[dict[str, Any] | None] = [None] * len(platform_specs)
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(platform_specs))) as ex:
        future_to_index: dict[Any, int] = {}
        for i, spec in enumerate(platform_specs):
            future_to_index[
                ex.submit(
                    _run_single_platform_evaluation,
                    output_dir=Path(output_dir) / spec.platform_key,
                    trace_dir=Path(trace_dir) / spec.platform_key,
                    spec=spec,
                    **common,
                )
            ] = i
        for fut in as_completed(future_to_index):
            i = future_to_index[fut]
            spec = platform_specs[i]
            try:
                platform_results[i] = fut.result()
            except Exception as exc:  # 单平台失败不拖垮整轮
                errors[spec.platform_key] = str(exc)
                platform_results[i] = {
                    "平台": spec.platform_key,
                    "目标人数": spec.requested_count,
                    "选取人数": 0,
                    "安全等级分布": {
                        "合格": 0,
                        "待人工复核": 0,
                        "未通过资格筛选": 0,
                    },
                    "stage_times": {},
                    "json_output_path": None,
                    "五维调度策略": {},
                    "error": str(exc),
                }

    t0 = time.perf_counter()
    output_payload = build_multiplatform_output(
        hot_event=hot_event,
        platform_results=platform_results,
    )
    if errors:
        overview = output_payload.get("跨平台总览")
        if isinstance(overview, dict):
            overview["平台级错误"] = errors
    stage_times["aggregate"] = time.perf_counter() - t0

    target_dir = Path(output_dir)
    output_path = (
        target_dir
        / f"{hot_event.get('event_id')}_multiplatform_{platform_tag}_strategy_output.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stage_times["event_total"] = time.perf_counter() - total_start
    return output_path, output_payload, stage_times


def _aggregate_safety(strategy: Any) -> dict[str, int]:
    """从单平台五维调度策略里统计安全等级分布（合格 / 待人工复核 / 未通过资格筛选）。

    单平台 ``五维调度策略.平台.数字人平台分发`` 每行带 ``安全等级`` 字段（取值为
    ``safe`` / ``review_required`` / ``not_eligible`` 或其中文别名），这里统一归并
    成三档中文计数，供跨平台总览汇总。
    """
    safety_agg = {"合格": 0, "待人工复核": 0, "未通过资格筛选": 0}
    if not isinstance(strategy, dict):
        return safety_agg
    platform_dim = strategy.get("平台") or {}
    rows = (
        platform_dim.get("数字人平台分发")
        if isinstance(platform_dim, dict)
        else None
    )
    if not isinstance(rows, list):
        return safety_agg
    mapping = {
        "safe": "合格",
        "合格": "合格",
        "review_required": "待人工复核",
        "待人工复核": "待人工复核",
        "not_eligible": "未通过资格筛选",
        "未通过资格筛选": "未通过资格筛选",
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        tier = mapping.get(str(row.get("安全等级", "")), "合格")
        safety_agg[tier] = safety_agg.get(tier, 0) + 1
    return safety_agg


def build_multiplatform_output(
    *,
    hot_event: dict[str, Any],
    platform_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """把逐平台复用的多人协同结果整合为多平台分发最终输出。

    每个 ``platform_results[i]`` 已经是一次完整多人协同产出的五维调度策略
    （与单平台模式**完全同构**）。这里只做「整合」，不做新的选人逻辑：

    * ``各平台结果`` 原样保留每个平台的五维策略，前端据此**复用多人协同的渲染组件**
      逐个平台呈现，保证「各个平台与单平台完全一样」；
    * ``跨平台总览`` 对各平台的人数、安全等级、重复数字人做汇总，作为总览呈现。
    """
    all_selected: list[str] = []
    safety_agg = {"合格": 0, "待人工复核": 0, "未通过资格筛选": 0}
    overlaps: set[str] = set()
    prev_ids: set[str] | None = None

    for pr in platform_results:
        ids = [
            str(uid)
            for uid in (
                pr.get("五维调度策略", {})
                .get("目标对象", {})
                .get("选取数字人id组", []) or []
            )
        ]
        all_selected.extend(ids)
        sd = pr.get("安全等级分布", {}) or {}
        safety_agg["合格"] += int(sd.get("合格", 0))
        safety_agg["待人工复核"] += int(sd.get("待人工复核", 0))
        safety_agg["未通过资格筛选"] += int(sd.get("未通过资格筛选", 0))
        id_set = set(ids)
        if prev_ids is not None:
            overlaps |= prev_ids & id_set
        prev_ids = id_set

    total_requested = sum(int(pr.get("目标人数", 0)) for pr in platform_results)
    total_selected = sum(int(pr.get("选取人数", 0)) for pr in platform_results)

    overview = {
        "调度模式": "cross_platform",
        "平台数": len(platform_results),
        "目标人数合计": total_requested,
        "选取人数合计": total_selected,
        "去重数字人数": len(set(all_selected)),
        "合格缺口合计": max(0, total_requested - total_selected),
        "安全等级分布": safety_agg,
        "跨平台重复数字人": sorted(overlaps),
        "内容草案说明": MULTIPLATFORM_DRAFT_NOTE,
        "结果范围说明": MULTIPLATFORM_SCOPE_NOTE,
    }

    return {
        "事件名称": str(hot_event.get("event_title") or ""),
        "输出格式版本": MULTIPLATFORM_OUTPUT_FORMAT_VERSION,
        "各平台结果": platform_results,
        "跨平台总览": overview,
    }


def build_eval_output(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    workspace_root: str | Path | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_nodes = strategy_result.selected_nodes
    selected_ids = [node.user_id for node in selected_nodes]
    platform = strategy_result.strategy.platform_plan.primary_platform
    # 对外输出统一使用数据集口径（如 instagram_simulated），避免同一个 JSON 里
    # 同时出现 "instagram" 与 "instagram_simulated" 两套平台写法。
    output_platform = (
        dataset_platform_value(platform)
        if is_valid_dataset_platform(platform)
        else platform
    )
    secondary_platforms = strategy_result.strategy.platform_plan.secondary_platforms
    target = str(
        hot_event.get("target") or strategy_result.event.target_goal or "扩大热点信息触达并引导理性讨论"
    ).strip()
    variants = _normalize_variants(hot_event.get("opinion_variants", []))

    llm_overrides, content_llm_meta = _build_llm_node_texts(
        hot_event=hot_event,
        strategy_result=strategy_result,
        platform=platform,
        target=target,
        workspace_root=workspace_root,
        use_llm=use_llm,
        llm_client=llm_client,
    )

    selection_summary = strategy_result.selection_summary
    if selection_summary.selection_mode == "targeted":
        # 单人分发：唯一数字人作为主发布节点，按四时期做自运营闭环
        # （首发期发布→互动期自赞→扩散期自转发→支持期自赞并再发承接帖）。
        node_outputs, post_refs, _ = _build_single_user_node_outputs(
            strategy_result=strategy_result,
            selected_nodes=selected_nodes,
            platform=output_platform,
            hot_event=hot_event,
            variants=variants,
            llm_overrides=llm_overrides,
            workspace_root=workspace_root,
        )
    else:
        # 多人 / 跨平台协同：按「首发期 → 互动期 → 扩散期 → 支持期」编排，动作只有
        # 发帖 / 评论 / 点赞 / 转发 四种，文案一律从弹药库取最相关的一条。
        node_outputs, post_refs, _ = _build_staged_node_outputs(
            strategy_result=strategy_result,
            selected_nodes=selected_nodes,
            platform=output_platform,
            hot_event=hot_event,
            variants=variants,
            llm_overrides=llm_overrides,
            workspace_root=workspace_root,
        )

    five_dimension_strategy = _build_five_dimension_strategy(
        strategy_result=strategy_result,
        selected_nodes=selected_nodes,
        selected_ids=selected_ids,
        platform=output_platform,
        target=target,
        post_refs=post_refs,
        node_outputs=node_outputs,
        max_actions_per_node=None,
    )

    five_dimension_strategy["内容"]["内容生成诊断"] = _compact_content_llm_meta(content_llm_meta)
    payload = {
        "事件名称": str(hot_event.get("event_title") or strategy_result.event.event_title),
        "输出格式版本": "action_schema_v5_five_dimensions_minimal",
        "五维调度策略": five_dimension_strategy,
    }
    selection_summary = strategy_result.selection_summary
    if selection_summary.selection_mode == "targeted":
        payload["定向分发诊断"] = {
            "selection_mode": "targeted",
            "target_user_id": selection_summary.target_user_id,
            "target_role": selection_summary.target_role,
            "target_platform": selection_summary.target_platform,
            "target_final_score": selection_summary.target_final_score,
            "target_topic_match_score": selection_summary.target_topic_match_score,
            "target_warning": list(selection_summary.target_warning),
        }

    # 主选为空时，把策略整体状态写到 payload 顶层，避免消费者误以为
    # ``数字人平台分发: []`` + ``传播目标: ...`` 表示"已规划好但还没分配"。
    strategy_summary = strategy_result.summary
    if strategy_summary.selected_count == 0:
        payload["状态"] = {
            "status": strategy_summary.status,
            "status_reason": strategy_summary.status_reason,
            "review_required": strategy_summary.review_required,
            "fallback_candidates": [
                {
                    "user_id": node.user_id,
                    "user_name": node.user_name,
                    "platform": node.platform,
                    "selection_rank": node.selection_rank,
                    "final_score": node.final_score,
                    "risk_level": node.risk_level,
                    "risk_flags": list(node.risk_flags),
                    "manual_review_required": node.manual_review_required,
                    "rationale": list(node.rationale),
                }
                for node in strategy_result.fallback_nodes
            ],
            "note": "以上 fallback_candidates 仅作为人工挑选首发账号的候补名单，不会被自动执行。",
        }
    return payload, content_llm_meta


# --------------------------------------------------------------------------- #
# 多人协同（单平台）：按时期编排的动作生成
# --------------------------------------------------------------------------- #
# 动作类型收敛为四种 —— 发帖 / 评论 / 点赞 / 转发；
# 动作按**时期**组织，而不是"每个节点四个"：后一时期的评论、点赞、转发
# 都通过「依赖动作」指向前一时期已经产生的动作，形成真正的传播链。

#: 四种允许的动作。
ACTION_PUBLISH = ("publish_post", "发布主帖")
ACTION_COMMENT = ("comment_post", "评论帖子")
ACTION_LIKE = ("like_post", "点赞帖子")
ACTION_REPOST = ("repost_post", "转发帖子")

#: 动作类型 → 统计用的中文名。
ACTION_TYPE_LABELS = {
    "publish_post": "发帖",
    "comment_post": "评论",
    "like_post": "点赞",
    "repost_post": "转发",
}

#: 弹药语义分下限。低于此值视为"没有合适内容"，只转发、不附文案。
#: 实测同类弹药 p50≈0.45、最低≈0.26；0.30 与项目既有的 BGE 语义门槛一致。
ARSENAL_MIN_SCORE = 0.30

#: 取消点赞的概率：风险越高越倾向放弃这一次点赞（保守发声）。
LIKE_SKIP_PROBABILITY = {"low": 0.10, "medium": 0.20, "high": 0.35}

#: 执行阶段 → 时期显示名（与 ``StrategyGenerator._build_time_plan`` 四段对应）。
STAGE_LABELS = {
    "stage_1_launch": "首发期",
    "stage_2_engage": "互动期",
    "stage_3_amplify": "扩散期",
    "stage_2_support": "支持期",
}

#: 节点角色 → 中文显示名（数字人职责）。
ROLE_LABELS = {
    "core_publish_node": "首发节点",
    "interaction_response_node": "互动节点",
    "amplification_node": "扩散节点",
    "support_node": "支持节点",
}

#: 节点角色 → 职责说明（一句话讲清这个人在传播链里干什么）。
ROLE_RESPONSIBILITY = {
    "core_publish_node": "首发期发布主帖，奠定舆论基调与核心叙事",
    "interaction_response_node": "互动期评论 / 点赞首发帖，引导并承接讨论",
    "amplification_node": "扩散期转发扩声量，推动二次传播",
    "support_node": "支持期补充背景、转发扩散，承接并延续热度",
}

#: 节点角色排序：给数字人分配统一编号（首发 → 互动 → 扩散 → 支持）时按此序。
ROLE_ORDER = {
    "core_publish_node": 0,
    "interaction_response_node": 1,
    "amplification_node": 2,
    "support_node": 3,
}

#: 动作类型 → 中文名（阶段统计用）。
ACTION_TYPE_LABELS_FALLBACK = {
    "publish_post": "发帖",
    "comment_post": "评论",
    "like_post": "点赞",
    "repost_post": "转发",
}

#: 需要展示正文的动作类型（发帖 / 评论 / 转发）。点赞无正文，不进入内容清单。
_CONTENT_ACTION_TYPES = ("publish_post", "comment_post", "repost_post")


def _encode_corpus(encoder: Any, texts: list[str], *, is_query: bool) -> Any:
    """批量编码，兼容 BGE（有 ``_encode_batch``）与 SVD（只有单条 ``encode``）。"""

    if hasattr(encoder, "_encode_batch"):
        return encoder._encode_batch(texts, is_query=is_query)
    import numpy as np

    return np.stack([encoder.encode(text) for text in texts])


def _arsenal_semantics(
    workspace_root: str | Path | None,
    event_text: str,
) -> tuple[Any, Any] | None:
    """返回 ``(编码器, 事件向量)``，无本地模型时返回 None。

    事件文本只编码 1 条（毫秒级）；弹药向量走离线预计算索引，不再实时编码。
    """

    if not workspace_root or not event_text.strip():
        return None
    try:
        from .local_models import load_local_models
    except ImportError:
        return None

    local_models = load_local_models(workspace_root)
    if local_models is None:
        return None
    try:
        event_vector = _encode_corpus(local_models.encoder, [event_text], is_query=True)[0]
    except Exception:  # pragma: no cover - 模型损坏时退化为无语义信号
        logger.warning("弹药语义打分不可用，退化为按类目选取", exc_info=True)
        return None
    return local_models.encoder, event_vector


def _ammo_scores(
    texts: list[str],
    *,
    semantics: tuple[Any, Any] | None,
    arsenal: ContentArsenal,
) -> list[float]:
    """给一批弹药文本打语义分。

    优先用**离线预计算**的向量（``ammo_vectors.npz``）：只做一次矩阵乘法，
    毫秒级。文件缺失或指纹与弹药库不一致时，才回退到实时编码整个池
    （BGE on CPU 约 8 秒）。
    """

    if semantics is None:
        return [0.0] * len(texts)

    encoder, event_vector = semantics
    if arsenal.source_dir is not None:
        index = AmmoVectorIndex.load(
            arsenal.source_dir,
            expected_fingerprint=_arsenal_fingerprint(
                item["text"] for item in arsenal.items
            ),
        )
        if index is not None:
            position = {text: i for i, text in enumerate(index.texts)}
            if all(text in position for text in texts):
                all_scores = index.scores(event_vector)
                return [all_scores[position[text]] for text in texts]

    docs = _encode_corpus(encoder, texts, is_query=False)
    return (docs @ event_vector).tolist()


def _build_ammo_ranking(
    arsenal: ContentArsenal,
    *,
    event_type: str,
    semantics: tuple[Any, Any] | None,
) -> list[tuple[str, float]]:
    """对整个弹药池一次性打分，返回 ``(文本, 语义分)`` 的降序列表。

    池子的分数只取决于事件本身，跟"取第几条"无关，所以这里**只算一次**；
    之后每次取文案都在这份排序结果上顺序扫描。早期实现是每次取文案都把整个
    池子重新编码一遍——10 人的编排大约会重复编码 7000 条，是主要性能瓶颈。
    """

    pool = [item for item in arsenal.items if event_type in (item.get("event_types") or ())]
    if not pool:
        pool = arsenal.items
    if not pool:
        return []

    texts = [item["text"] for item in pool]
    scores = _ammo_scores(texts, semantics=semantics, arsenal=arsenal)
    return sorted(zip(texts, scores), key=lambda pair: pair[1], reverse=True)


def _pick_ammo_texts(
    ranking: list[tuple[str, float]],
    *,
    count: int,
    exclude: set[str],
    min_score: float | None,
) -> list[tuple[str, float]]:
    """从已排好序的弹药里取 ``count`` 条最相关且未使用过的。

    ``min_score`` 非 None 时，语义分低于它的视为没有合适内容，返回空串
    （转发文案留空）。ranking 已降序，遇到第一条低于门槛即可停止。
    """

    if count <= 0:
        return []

    picked: list[tuple[str, float]] = []
    for text, score in ranking:
        if text in exclude:
            continue
        if min_score is not None and score < min_score:
            break
        exclude.add(text)
        picked.append((text, float(score)))
        if len(picked) >= count:
            break
    while len(picked) < count:
        picked.append(("", 0.0))
    return picked


def _count_actions(actions: list[dict[str, Any]]) -> dict[str, int]:
    """统计一个节点各类动作的数量。

    这是动作清单编排**完成之后**的实际统计，不是预置的频率上限——频率不再
    用于限制动作生成，只是把每个人实际做了多少事如实反映出来。
    """

    counts = {label: 0 for label in ACTION_TYPE_LABELS.values()}
    for action in actions:
        label = ACTION_TYPE_LABELS.get(str(action.get("动作类型", "")))
        if label:
            counts[label] += 1
    counts["合计"] = len(actions)
    return counts


def _should_skip_like(actor_id: str, target_ref: str, risk_level: str) -> bool:
    """按风险等级概率性取消一次点赞。

    种子取 ``(执行主体, 目标帖子)``，保证同一份输入每次运行结果一致。
    """

    probability = LIKE_SKIP_PROBABILITY.get(risk_level, LIKE_SKIP_PROBABILITY["medium"])
    roll = random.Random(_stable_seed("like", actor_id, target_ref)).random()
    return roll < probability


def _arsenal_event_text(strategy_result: StrategyResult) -> str:
    """用于弹药语义打分的事件文本。"""

    event = strategy_result.event
    return " ".join(
        part
        for part in (event.event_title, event.event_description, *event.extracted_keywords)
        if part
    )


def _build_staged_node_outputs(
    *,
    strategy_result: StrategyResult,
    selected_nodes: list[StrategyNodePlan],
    platform: str,
    hot_event: dict[str, Any] | None = None,
    variants: list[str] | None = None,
    llm_overrides: dict[str, dict[str, str]] | None = None,
    workspace_root: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """多人协同（单平台）按四个时期编排动作。

    ``(首发期 → 互动期 → 扩散期 → 支持期)``：

    * 首发期：首发节点各发一次帖，文案取弹药库最相关的一条；
    * 互动期：非首发节点点赞**全部**首发帖；互动节点对每个首发帖评论，
      文案互不重复；
    * 扩散期：扩散节点转发（一次扩散），其余节点对转发点赞；随后支持节点
      转发一次扩散的结果（二次扩散），其余节点再点赞；
    * 支持期：互动节点以自己的评论内容发帖，扩散 / 支持节点取弹药发帖。
    """

    by_role: dict[str, list[StrategyNodePlan]] = {}
    for node in selected_nodes:
        by_role.setdefault(node.selected_role, []).append(node)
    launch_nodes = by_role.get("core_publish_node", [])
    interact_nodes = by_role.get("interaction_response_node", [])
    amplify_nodes = by_role.get("amplification_node", [])
    support_nodes = by_role.get("support_node", [])

    event_type = strategy_result.event.event_type
    arsenal = ContentArsenal.load(workspace_root)
    semantics = _arsenal_semantics(workspace_root, _arsenal_event_text(strategy_result))
    # 弹药池只在这里打分一次；后面每次取文案都复用这份排序结果。
    ammo_ranking = _build_ammo_ranking(arsenal, event_type=event_type, semantics=semantics)
    ammo_min_score = ARSENAL_MIN_SCORE if semantics is not None else None
    time_plan = strategy_result.strategy.time_plan

    actions_by_node: dict[str, list[dict[str, Any]]] = {
        node.user_id: [] for node in selected_nodes
    }
    post_refs: dict[str, str] = {}
    publish_action_ids: dict[str, str] = {}
    used_texts: set[str] = set()
    counter = itertools.count(1)

    def emit(
        node: StrategyNodePlan,
        kind: tuple[str, str],
        stage: str,
        trigger: str,
        target: dict[str, Any],
        depends_on: list[str],
        creates: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        seq = next(counter)
        label = STAGE_LABELS.get(stage, stage)
        action = _compact_action(
            action_id=f"A{seq:03d}",
            seq=seq,
            actor_id=node.user_id,
            action_group=label,
            action_type=kind[0],
            action_name=kind[1],
            platform=platform,
            stage=f"{label} {time_plan.get(stage, '')}".strip(),
            trigger_condition=trigger,
            target=target,
            depends_on=list(depends_on),
            creates=creates,
            payload=payload,
        )
        actions_by_node[node.user_id].append(action)
        return action

    variant_pool = list(variants or [])
    event_title = str(
        (hot_event or {}).get("event_title") or strategy_result.event.event_title or ""
    )
    event_summary = str(
        (hot_event or {}).get("event_summary") or strategy_result.event.event_description or ""
    )

    def ammo(count: int = 1) -> list[tuple[str, float]]:
        return _pick_ammo_texts(
            ammo_ranking,
            count=count,
            exclude=used_texts,
            min_score=ammo_min_score,
        )

    def text_or_fallback(picked: str, node: StrategyNodePlan, node_index: int) -> str:
        """文案优先级：弹药库 → LLM 生成 → 规则占位文本。

        弹药库取不到（库为空或语义分过低）时依次后退，保证发帖/评论的文案
        不为空。转发是唯一允许留空的动作（语义分过低就只转不附文案），因此
        只有发帖与评论走这里。
        """

        if picked:
            return picked
        llm_text = str((llm_overrides or {}).get(node.user_id, {}).get("post_content") or "").strip()
        if llm_text:
            return llm_text
        return _post_content_text(
            node=node,
            node_index=node_index,
            event_title=event_title,
            event_summary=event_summary,
            variants=variant_pool,
        )

    # ① 首发期：每个首发节点发一次帖
    launch_posts: list[tuple[str, str, str, float]] = []
    for index, node in enumerate(launch_nodes, start=1):
        text, score = ammo()[0]
        text = text_or_fallback(text, node, index)
        ref = f"P{index:03d}"
        post_refs[node.user_id] = ref
        action = emit(
            node,
            ACTION_PUBLISH,
            "stage_1_launch",
            trigger="进入首发期后立即执行",
            target=_post_target(ref, node.user_id, "本节点首发帖"),
            depends_on=[],
            creates={"post_ref": ref, "post_type": "anchor_post"},
            payload={"正文草案": text, "弹药语义分": round(score, 4)},
        )
        publish_action_ids[node.user_id] = action["动作编号"]
        launch_posts.append((node.user_id, ref, action["动作编号"], score))

    # ② 互动期：非首发节点点赞全部首发帖
    for owner_id, ref, action_id, _score in launch_posts:
        for node in selected_nodes:
            if node.selected_role == "core_publish_node":
                continue
            if _should_skip_like(node.user_id, ref, node.risk_level):
                continue
            emit(
                node,
                ACTION_LIKE,
                "stage_2_engage",
                trigger="首发帖出现后执行",
                target=_post_target(ref, owner_id, f"数字人 {owner_id} 的首发帖"),
                depends_on=[action_id],
                creates={},
                payload={"跳过原因": ""},
            )

    # ② 互动期：互动节点对每个首发帖评论（文案互不重复）
    comment_text_by_node: dict[str, str] = {}
    for owner_id, ref, action_id, _score in launch_posts:
        for c_index, node in enumerate(interact_nodes, start=1):
            text, score = ammo()[0]
            text = text_or_fallback(text, node, c_index)
            comment_text_by_node.setdefault(node.user_id, text)
            emit(
                node,
                ACTION_COMMENT,
                "stage_2_engage",
                trigger="首发帖发布后跟进评论",
                target=_post_target(ref, owner_id, f"数字人 {owner_id} 的首发帖"),
                depends_on=[action_id],
                creates={"comment_ref": f"C{len(actions_by_node[node.user_id]):03d}"},
                payload={"评论内容": text, "弹药语义分": round(score, 4)},
            )

    # ③ 扩散期：扩散节点转发（一次扩散），优先转发语义分最高的首发帖
    ranked_launch = sorted(launch_posts, key=lambda item: item[3], reverse=True)
    repost1: list[tuple[str, str, str]] = []
    for index, node in enumerate(amplify_nodes, start=1):
        if not ranked_launch:
            break
        owner_id, src_ref, src_action, _score = ranked_launch[(index - 1) % len(ranked_launch)]
        text, score = ammo()[0]
        ref = f"R{index:03d}"
        action = emit(
            node,
            ACTION_REPOST,
            "stage_3_amplify",
            trigger="进入扩散期后执行第一次扩散",
            target=_post_target(src_ref, owner_id, f"数字人 {owner_id} 的首发帖"),
            depends_on=[src_action],
            creates={"post_ref": ref, "post_type": "repost"},
            payload={
                "转发文案": text,
                "原文post_ref": src_ref,
                "文案留空": not text,
                "弹药语义分": round(score, 4),
            },
        )
        post_refs.setdefault(node.user_id, ref)
        repost1.append((node.user_id, ref, action["动作编号"]))

    # ③ 扩散期：其余节点对一次扩散的转发点赞
    for owner_id, ref, action_id in repost1:
        for node in launch_nodes + interact_nodes + support_nodes:
            if _should_skip_like(node.user_id, ref, node.risk_level):
                continue
            emit(
                node,
                ACTION_LIKE,
                "stage_3_amplify",
                trigger="一次扩散转发出现后执行",
                target=_post_target(ref, owner_id, f"数字人 {owner_id} 的转发帖"),
                depends_on=[action_id],
                creates={},
                payload={"跳过原因": ""},
            )

    # ③ 二次扩散：支持节点转发最相关的一次扩散结果
    repost2: list[tuple[str, str, str]] = []
    for index, node in enumerate(support_nodes, start=1):
        if not repost1:
            break
        owner_id, src_ref, src_action = repost1[(index - 1) % len(repost1)]
        ref = f"S{index:03d}"
        action = emit(
            node,
            ACTION_REPOST,
            "stage_3_amplify",
            trigger="扩散期内完成第二次扩散",
            target=_post_target(src_ref, owner_id, f"数字人 {owner_id} 的转发帖"),
            depends_on=[src_action],
            creates={"post_ref": ref, "post_type": "repost"},
            payload={"转发文案": "", "原文post_ref": src_ref, "文案留空": True},
        )
        post_refs.setdefault(node.user_id, ref)
        repost2.append((node.user_id, ref, action["动作编号"]))

    # ③ 二次扩散后：首发 / 互动 / 扩散节点点赞
    for owner_id, ref, action_id in repost2:
        for node in launch_nodes + interact_nodes + amplify_nodes:
            if _should_skip_like(node.user_id, ref, node.risk_level):
                continue
            emit(
                node,
                ACTION_LIKE,
                "stage_3_amplify",
                trigger="二次扩散转发出现后执行",
                target=_post_target(ref, owner_id, f"数字人 {owner_id} 的转发帖"),
                depends_on=[action_id],
                creates={},
                payload={"跳过原因": ""},
            )

    # ④ 支持期：互动节点以评论内容发帖，扩散 / 支持节点取弹药发帖
    support_publishers = (
        [(node, comment_text_by_node.get(node.user_id, "")) for node in interact_nodes]
        + [(node, ammo()[0][0]) for node in amplify_nodes]
        + [(node, ammo()[0][0]) for node in support_nodes]
    )
    for index, (node, text) in enumerate(support_publishers, start=1):
        text = text_or_fallback(text, node, index)
        ref = f"Q{index:03d}"
        action = emit(
            node,
            ACTION_PUBLISH,
            "stage_2_support",
            trigger="进入支持期后发布承接帖",
            target=_post_target(ref, node.user_id, "本节点支持期发帖"),
            depends_on=[],
            creates={"post_ref": ref, "post_type": "support_post"},
            payload={"正文草案": text, "来源": "互动评论" if node in interact_nodes else "弹药库"},
        )
        post_refs.setdefault(node.user_id, ref)
        publish_action_ids.setdefault(node.user_id, action["动作编号"])

    node_outputs: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(selected_nodes, start=1):
        actions = actions_by_node[node.user_id]
        own_ref = post_refs.get(node.user_id, "")
        node_outputs[f"id{node.user_id}"] = {
            "时间阶段": STAGE_LABELS.get(node.dispatch_stage, node.dispatch_stage),
            "发帖平台": platform,
            "发帖内容": next(
                (
                    action["执行参数"].get("正文草案") or action["执行参数"].get("转发文案", "")
                    for action in actions
                    if action["动作类型"] == "publish_post"
                ),
                "",
            ),
            "动作清单": actions,
            "动作统计": _count_actions(actions),
            "执行约束": {
                "节点角色": node.selected_role,
                "执行阶段": node.dispatch_stage,
                "执行优先级": node.dispatch_priority,
                "节点得分": round(node.final_score, 6),
                "风险等级": node.risk_level,
                "人工复核": node.manual_review_required,
                "风险标签": node.risk_flags,
            },
            "post_ref": own_ref,
        }
    return node_outputs, post_refs, publish_action_ids


def _build_single_user_node_outputs(
    *,
    strategy_result: StrategyResult,
    selected_nodes: list[StrategyNodePlan],
    platform: str,
    hot_event: dict[str, Any] | None = None,
    variants: list[str] | None = None,
    llm_overrides: dict[str, dict[str, str]] | None = None,
    workspace_root: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """单人分发：唯一数字人作为主发布节点，按四时期做自运营闭环。

    编排（与多人协同共享同一套时期语义，输出结构完全一致，前端无需分支）：

    * 首发期（第一个时期）：本人发布主帖（anchor_post）；
    * 互动期：给自己的主帖点赞；
    * 扩散期（第一个传播期）：转发自己的主帖；
    * 支持期（第二个传播期 + 最后一个时期）：给自己的转发点赞，并再发一篇承接帖。

    这样单人也形成「发布 → 互动 → 扩散 → 再发布承接」的完整链路。动作类型仍是
    发帖 / 点赞 / 转发 三种，文案走与多人协同相同的弹药库 / 规则占位路径；图文
    （图片）事件走同一条 OCR→文本→内容生成链路，因此与多人协同对图文的处理一致。
    """

    if not selected_nodes:
        return {}, {}, {}
    node = selected_nodes[0]
    event_type = strategy_result.event.event_type
    arsenal = ContentArsenal.load(workspace_root)
    semantics = _arsenal_semantics(workspace_root, _arsenal_event_text(strategy_result))
    # 弹药池只在这里打分一次；后面每次取文案都复用这份排序结果。
    ammo_ranking = _build_ammo_ranking(arsenal, event_type=event_type, semantics=semantics)
    ammo_min_score = ARSENAL_MIN_SCORE if semantics is not None else None
    time_plan = strategy_result.strategy.time_plan

    variant_pool = list(variants or [])
    event_title = str(
        (hot_event or {}).get("event_title") or strategy_result.event.event_title or ""
    )
    event_summary = str(
        (hot_event or {}).get("event_summary") or strategy_result.event.event_description or ""
    )

    used_texts: set[str] = set()
    counter = itertools.count(1)
    actions: list[dict[str, Any]] = []

    def ammo(count: int = 1) -> list[tuple[str, float]]:
        return _pick_ammo_texts(
            ammo_ranking,
            count=count,
            exclude=used_texts,
            min_score=ammo_min_score,
        )

    def text_or_fallback(picked: str, node_index: int) -> str:
        """文案优先级：弹药库 → LLM 生成 → 规则占位文本（与多人协同一致）。"""

        if picked:
            return picked
        llm_text = str((llm_overrides or {}).get(node.user_id, {}).get("post_content") or "").strip()
        if llm_text:
            return llm_text
        return _post_content_text(
            node=node,
            node_index=node_index,
            event_title=event_title,
            event_summary=event_summary,
            variants=variant_pool,
        )

    def emit(
        kind: tuple[str, str],
        stage: str,
        trigger: str,
        target: dict[str, Any],
        depends_on: list[str],
        creates: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        seq = next(counter)
        label = STAGE_LABELS.get(stage, stage)
        action = _compact_action(
            action_id=f"A{seq:03d}",
            seq=seq,
            actor_id=node.user_id,
            action_group=label,
            action_type=kind[0],
            action_name=kind[1],
            platform=platform,
            stage=f"{label} {time_plan.get(stage, '')}".strip(),
            trigger_condition=trigger,
            target=target,
            depends_on=list(depends_on),
            creates=creates,
            payload=payload,
        )
        actions.append(action)
        return action

    # ① 首发期（第一个时期）：本人发布主帖
    p_text, p_score = ammo()[0]
    p_text = text_or_fallback(p_text, 1)
    p_ref = "P001"
    pa = emit(
        ACTION_PUBLISH,
        "stage_1_launch",
        "进入首发期后立即执行",
        _publish_target(node=node, own_post_ref=p_ref, target_core_id=node.user_id, target_core_ref=p_ref),
        [],
        {"post_ref": p_ref, "post_type": "anchor_post"},
        {"正文草案": p_text, "弹药语义分": round(p_score, 4)},
    )

    # ② 互动期：给自己的主帖点赞
    emit(
        ACTION_LIKE,
        "stage_2_engage",
        "本人主帖发布后执行",
        _post_target(p_ref, node.user_id, "本人首发帖"),
        [pa["动作编号"]],
        {},
        {"跳过原因": ""},
    )

    # ③ 扩散期（第一个传播期）：转发自己的主帖
    r_text, r_score = ammo()[0]
    r_ref = "R001"
    ra = emit(
        ACTION_REPOST,
        "stage_3_amplify",
        "进入扩散期后转发本人主帖",
        _post_target(p_ref, node.user_id, "本人首发帖"),
        [pa["动作编号"]],
        {"post_ref": r_ref, "post_type": "repost"},
        {
            "转发文案": r_text,
            "原文post_ref": p_ref,
            "文案留空": not r_text,
            "弹药语义分": round(r_score, 4),
        },
    )

    # ④ 支持期（第二个传播期）：给自己的转发点赞；并再发一篇承接帖（最后一个时期）
    emit(
        ACTION_LIKE,
        "stage_2_support",
        "本人转发出现后执行",
        _post_target(r_ref, node.user_id, "本人转发帖"),
        [ra["动作编号"]],
        {},
        {"跳过原因": ""},
    )
    q_text, q_score = ammo()[0]
    q_text = text_or_fallback(q_text, 2)
    q_ref = "Q001"
    emit(
        ACTION_PUBLISH,
        "stage_2_support",
        "进入支持期后发布承接帖",
        _publish_target(node=node, own_post_ref=q_ref, target_core_id=node.user_id, target_core_ref=q_ref),
        [],
        {"post_ref": q_ref, "post_type": "support_post"},
        {"正文草案": q_text, "来源": "弹药库"},
    )

    node_outputs: dict[str, dict[str, Any]] = {}
    own_ref = p_ref
    node_outputs[f"id{node.user_id}"] = {
        "时间阶段": "首发期 → 支持期（全程自运营）",
        "发帖平台": platform,
        "发帖内容": next(
            (
                action["执行参数"].get("正文草案") or action["执行参数"].get("转发文案", "")
                for action in actions
                if action["动作类型"] == "publish_post"
            ),
            "",
        ),
        "动作清单": actions,
        "动作统计": _count_actions(actions),
        "执行约束": {
            "节点角色": node.selected_role,
            "执行阶段": node.dispatch_stage,
            "执行优先级": node.dispatch_priority,
            "节点得分": round(node.final_score, 6),
            "风险等级": node.risk_level,
            "人工复核": node.manual_review_required,
            "风险标签": node.risk_flags,
        },
        "post_ref": own_ref,
    }
    post_refs: dict[str, str] = {node.user_id: p_ref}
    publish_action_ids: dict[str, str] = {node.user_id: pa["动作编号"]}
    return node_outputs, post_refs, publish_action_ids


#: 四个时期的先后顺序——后端与前端都按这个顺序排，不用各自猜。
STAGE_ORDER = ("stage_1_launch", "stage_2_engage", "stage_3_amplify", "stage_2_support")


def _stage_windows(time_plan: dict[str, Any]) -> list[dict[str, str]]:
    """把 ``time_plan``（英文 key）翻成「时期 + 窗口」的中文列表。

    返回顺序就是执行先后顺序，前端按顺序渲染即可，不必再自己排序或翻译。
    原样保留 ``传播窗口`` 那份机读字典，这里只补一份给人看的对照表。
    """

    windows: list[dict[str, str]] = []
    for stage in STAGE_ORDER:
        window = time_plan.get(stage)
        if window:
            windows.append({"时期": STAGE_LABELS.get(stage, stage), "窗口": str(window)})
    # 兜底：time_plan 里若出现四个时期之外的 key，也列出来，避免静默丢信息。
    for stage, window in time_plan.items():
        if stage in STAGE_ORDER or not window:
            continue
        windows.append({"时期": STAGE_LABELS.get(stage, str(stage)), "窗口": str(window)})
    return windows


def _stage_statistics(
    all_actions: list[dict[str, Any]],
    time_plan: dict[str, Any],
    node_number: dict[str, str],
) -> list[dict[str, Any]]:
    """每个时期的统计：参与数字人、动作数、发帖/评论/点赞/转发分布。

    数据来自已经编排好的全量动作（``all_actions``），``动作分组`` 是时期中文名、
    ``执行主体`` 是数字人 id、``动作类型`` 是英文枚举。前端按这份结构直接渲染，
    不必再自己统计或翻译。
    """

    stats: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        label = STAGE_LABELS.get(stage, stage)
        stage_actions = [a for a in all_actions if a.get("动作分组") == label]
        actors = sorted(
            {
                node_number.get(a.get("执行主体", ""), a.get("执行主体", ""))
                for a in stage_actions
            }
        )
        counts = {"发帖": 0, "评论": 0, "点赞": 0, "转发": 0}
        for action in stage_actions:
            name = ACTION_TYPE_LABELS_FALLBACK.get(action.get("动作类型", ""))
            if name:
                counts[name] += 1
        stats.append(
            {
                "时期": label,
                "窗口": str(time_plan.get(stage, "")),
                "参与数字人": "、".join(actors) or "—",
                "动作数": len(stage_actions),
                "发帖": counts["发帖"],
                "评论": counts["评论"],
                "点赞": counts["点赞"],
                "转发": counts["转发"],
            }
        )
    return stats


def _build_five_dimension_strategy(
    *,
    strategy_result: StrategyResult,
    selected_nodes: list[StrategyNodePlan],
    selected_ids: list[str],
    platform: str,
    target: str,
    post_refs: dict[str, str],
    node_outputs: dict[str, dict[str, Any]],
    platform_item_fields: dict[str, dict[str, Any]] | None = None,
    platform_extra: dict[str, Any] | None = None,
    max_actions_per_node: int | None = 3,
) -> dict[str, Any]:
    """``max_actions_per_node`` 为 None 时保留该节点的全部动作。

    多人协同按时期编排后，单人的点赞/评论动辄十几条，截断到 3 条会把
    传播链砍没；跨平台分发仍用默认值，保持原行为不变。
    """

    target_objects = strategy_result.strategy.target_object or ["泛公众"]
    frequency_plan = strategy_result.strategy.frequency_plan

    # 平台模式由选中数字人的实际平台归属推导，不再硬编码单一平台。
    selected_platform_keys = sorted(
        {
            normalize_platform_key(node.platform)
            for node in selected_nodes
            if normalize_platform_key(node.platform)
        }
    )
    if not selected_platform_keys:
        platform_mode = "unspecified"
    elif len(selected_platform_keys) == 1:
        platform_mode = f"{selected_platform_keys[0]}_only"
    else:
        platform_mode = "cross_platform"

    target_plan: list[dict[str, Any]] = []
    time_plan: list[dict[str, Any]] = []
    frequency_items: list[dict[str, Any]] = []
    platform_items: list[dict[str, Any]] = []
    content_items: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    # 给每个数字人分配统一编号（按角色顺序 + id 升序），所有维度共用，
    # 避免各表用各的 id、看不出谁是谁。
    numbered_nodes = sorted(
        selected_nodes,
        key=lambda n: (ROLE_ORDER.get(n.selected_role, 9), str(n.user_id)),
    )
    node_number = {node.user_id: f"D{i + 1:02d}" for i, node in enumerate(numbered_nodes)}
    node_role_label = {
        node.user_id: ROLE_LABELS.get(node.selected_role, node.selected_role)
        for node in selected_nodes
    }
    # 全量动作索引（含完整字段），用于把「依赖动作」解析成可读的指向对象。
    all_actions: list[dict[str, Any]] = []
    for node_output in node_outputs.values():
        all_actions.extend(node_output.get("动作清单", []) or [])
    action_index = {a.get("动作编号"): a for a in all_actions}

    # 活动窗口小时数：用于把动作数量换算成"条/天"。优先取事件约束里的窗口，
    # 取不到再退回默认的 24 小时，保证除零安全。
    window_hours = 24
    _evt = getattr(strategy_result, "event", None)
    _cons = getattr(_evt, "constraints", None) if _evt is not None else None
    if _cons is not None:
        _wh = getattr(_cons, "campaign_window_hours", None)
        if _wh:
            window_hours = int(_wh)

    for node in selected_nodes:
        node_key = f"id{node.user_id}"
        node_output = node_outputs.get(node_key, {})
        post_ref = post_refs.get(node.user_id, "")
        node_actions = node_output.get("动作清单", [])
        # 至少保留 content_publish + audience_interaction + digital_human_coordination
        # 三组。``digital_human_coordination`` 是数字人之间的真正协同（点赞/评论/转发
        # 其他人的帖子），原先 ``max_actions=2`` 把它和 ``comment_monitoring`` 一起
        # 砍掉了，导致整张动作清单看起来都是"自发自评"。
        selected_actions = _minimal_actions_for_node(
            node_actions,
            max_actions=max_actions_per_node,
            action_index=action_index,
            node_number=node_number,
            node_role_label=node_role_label,
        )

        target_plan.append(
            {
                "数字人编号": node_number[node.user_id],
                "数字人id": node.user_id,
                "角色": node_role_label[node.user_id],
                "职责说明": ROLE_RESPONSIBILITY.get(node.selected_role, ""),
                "人设": _persona_for_node(node=node, topic_labels=[]),
                "平台": node.platform or platform,
                "节点角色": node.selected_role,
                "目标受众": target_objects,
            }
        )
        time_plan.append(
            {
                "数字人编号": node_number[node.user_id],
                "数字人id": node.user_id,
                "时间阶段": node_output.get("时间阶段", _stage_text(node)),
            }
        )
        action_stats = node_output.get("动作统计")
        if action_stats:
            # 多人协同：频率不再是预置上限，而是动作编排完成后的实际统计。
            total_actions = int(action_stats.get("合计", 0) or 0)
            # 直接由「动作总数 / 活动窗口天数」反算条/天，避免先 round 到 per_hour 再
            # ×24 带来的二次取整误差（窗口=1 天时条/天 应等于动作合计，为整数）。
            per_day = round(total_actions * 24 / window_hours, 2) if window_hours else 0
            frequency_items.append(
                {
                    "数字人编号": node_number[node.user_id],
                    "数字人id": node.user_id,
                    "动作统计": action_stats,
                    "条每天": per_day,
                }
            )
        else:
            # 单人模式：计划频率按"条/天"给出，直接以"条/天"展示。
            freq_per_day = node.frequency_per_day or 0
            frequency_items.append(
                {
                    "数字人编号": node_number[node.user_id],
                    "数字人id": node.user_id,
                    "发帖频率": node_output.get(
                        "发帖频率", _frequency_text(node.frequency_per_day)
                    ),
                    "频率上限": _frequency_text(frequency_plan.global_cap_per_day),
                    "条每天": freq_per_day,
                }
            )
        platform_item: dict[str, Any] = {
            "数字人编号": node_number[node.user_id],
            "数字人id": node.user_id,
            # 每个数字人只归属唯一模拟平台，优先输出其自身的平台归属。
            "分发平台": node.platform or platform,
        }
        if platform_item_fields:
            # 跨平台模式在这里补上"平台角色 / 安全等级 / 平台适配分"等平台专属依据；
            # 多人分发不传该参数，因此不会凭空多出字段。
            platform_item.update(platform_item_fields.get(node.user_id, {}))
        platform_items.append(platform_item)
        # 内容维度：把该数字人「所有要发布的内容」都列出来——发帖 / 评论 / 转发
        # 各占一行，配统一的内容编号（P/C/R/S/Q 系列）。转发允许无文案（标「为空」），
        # 发帖 / 评论的文案由兜底逻辑保证非空，但万一为空也明确标「为空」。
        node_content_actions = [
            a
            for a in all_actions
            if a.get("执行主体") == node.user_id
            and a.get("动作类型") in _CONTENT_ACTION_TYPES
        ]
        for action in node_content_actions:
            _creates = action.get("生成对象", {}) or {}
            _payload = action.get("执行参数", {}) or {}
            content_id = _creates.get("post_ref") or _creates.get("comment_ref") or ""
            content_text = (
                _payload.get("正文草案")
                or _payload.get("评论内容")
                or _payload.get("转发文案")
                or ""
            )
            content_text = (content_text or "").strip()
            content_items.append(
                {
                    "数字人编号": node_number[node.user_id],
                    "数字人id": node.user_id,
                    "动作类型": ACTION_TYPE_LABELS.get(
                        action.get("动作类型", ""), action.get("动作类型", "")
                    ),
                    "内容编号": content_id,
                    "内容": content_text or "为空",
                }
            )
        # 单节点模式的主帖可能只挂在节点 output 上、不在动作清单里，兜底补一行；
        # 按文案去重，避免与动作清单里的发帖重复显示。
        _main_post = (node_output.get("发帖内容") or "").strip()
        if _main_post and not any(
            c.get("内容") == _main_post for c in content_items
        ):
            content_items.append(
                {
                    "数字人编号": node_number[node.user_id],
                    "数字人id": node.user_id,
                    "动作类型": "发帖",
                    "内容编号": post_ref or "—",
                    "内容": _main_post,
                }
            )
        actions.extend(selected_actions)

    # 动作清单按编排顺序输出（首发期 → 互动期 → 扩散期 → 支持期）。
    # 原先是逐个节点拼接的，同一节点的动作聚成一堆，读不出传播推进。
    actions.sort(key=lambda item: item.get("序号", 0) or 0)

    platform_dimension: dict[str, Any] = {
        "平台模式": platform_mode,
        "数字人平台分发": platform_items,
    }
    if platform_extra:
        # 跨平台模式把「跨平台总览」与「各平台方案」挂在这里；多人分发不传该参数，
        # 因此两个模式的「平台」维度只在跨平台时多出这两块，其余字段完全一致。
        platform_dimension.update(platform_extra)

    return {
        "目标对象": {
            "传播目标": target,
            "选取数字人id组": selected_ids,
            "数字人分工": target_plan,
        },
        "时间": {
            "传播窗口": strategy_result.strategy.time_plan,
            "时期窗口": _stage_windows(strategy_result.strategy.time_plan),
            "阶段统计": _stage_statistics(all_actions, strategy_result.strategy.time_plan, node_number),
            "数字人时间安排": time_plan,
        },
        "频率": {
            "数字人频率": frequency_items,
        },
        "平台": platform_dimension,
        "内容": {
            "数字人发帖内容": content_items,
            "动作清单": actions,
        },
    }


def _minimal_actions_for_node(
    actions: list[dict[str, Any]],
    *,
    max_actions: int | None,
    action_index: dict[str, Any] | None = None,
    node_number: dict[str, str] | None = None,
    node_role_label: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """``max_actions`` 为 None 时保留全部动作（多人协同的时期编排需要）。"""

    preferred_groups = ("content_publish", "audience_interaction", "digital_human_coordination")
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for group in preferred_groups:
        for action in actions:
            action_id = str(action.get("动作编号", ""))
            if action_id in used_ids or action.get("动作分组") != group:
                continue
            selected.append(
                _minimal_action(
                    action,
                    action_index=action_index,
                    node_number=node_number,
                    node_role_label=node_role_label,
                )
            )
            used_ids.add(action_id)
            break
        if max_actions is not None and len(selected) >= max_actions:
            return selected

    for action in actions:
        action_id = str(action.get("动作编号", ""))
        if action_id in used_ids:
            continue
        selected.append(
            _minimal_action(
                action,
                action_index=action_index,
                node_number=node_number,
                node_role_label=node_role_label,
            )
        )
        used_ids.add(action_id)
        if max_actions is not None and len(selected) >= max_actions:
            break
    return selected


def _minimal_action(
    action: dict[str, Any],
    *,
    action_index: dict[str, Any] | None = None,
    node_number: dict[str, str] | None = None,
    node_role_label: dict[str, str] | None = None,
) -> dict[str, Any]:
    actor = action.get("执行主体", "")
    creates = action.get("生成对象", {}) or {}
    # 内容编号 = 本动作生成的内容 id：发帖/转发用 post_ref，评论用 comment_ref。
    content_id = creates.get("post_ref") or creates.get("comment_ref") or ""
    result = {
        "动作编号": action.get("动作编号", ""),
        # 序号 = 编排时的生成顺序，前端据此把动作清单按时间铺开。
        "序号": action.get("序号", 0),
        # 多人模式是时期名（首发期 / 互动期 / 扩散期 / 支持期），跨平台模式是
        # 动作组；两种模式都用它给动作清单分组，否则看不出传播推进。
        "动作分组": action.get("动作分组", ""),
        # 数字人统一用 D01 / D02 …… 编号，和「数字人分工」里的编号一致。
        "数字人": (node_number or {}).get(actor, actor) if node_number else actor,
        # 动作类型中文名（发帖 / 评论 / 点赞 / 转发）。
        "动作类型": ACTION_TYPE_LABELS.get(
            action.get("动作类型", ""), action.get("动作类型", "")
        ),
        # 本动作生成的内容编号（P/C/R/S/Q 系列）；点赞等无内容则为空。
        "内容编号": content_id,
    }
    depends_on = action.get("依赖动作", []) or []
    if depends_on:
        # 依赖动作编号：本次动作所依赖的前置动作。
        result["依赖动作编号"] = "、".join(str(d) for d in depends_on)
        # 依赖帖子编号：前置动作所针对的帖子内容编号（即「被依赖的帖子」）。
        dep_refs: list[str] = []
        if action_index is not None:
            for dep_id in depends_on:
                dep = action_index.get(dep_id)
                if not dep:
                    continue
                dep_ref = (
                    (dep.get("目标定位", {}) or {}).get("post_ref")
                    or (dep.get("生成对象", {}) or {}).get("post_ref")
                    or ""
                )
                if dep_ref:
                    dep_refs.append(str(dep_ref))
        result["依赖帖子编号"] = "、".join(dep_refs) if dep_refs else "—"
    else:
        result["依赖动作编号"] = ""
        result["依赖帖子编号"] = ""
    return result


def _compact_action(
    *,
    action_id: str,
    seq: int,
    actor_id: str,
    action_group: str,
    action_type: str,
    action_name: str,
    platform: str,
    stage: str,
    trigger_condition: str,
    target: dict[str, Any],
    depends_on: list[str],
    creates: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "动作编号": action_id,
        "序号": seq,
        "动作分组": action_group,
        "动作类型": action_type,
        "动作名称": action_name,
        "执行主体": actor_id,
        "执行平台": platform,
        "执行阶段": stage,
        "触发条件": trigger_condition,
        "目标定位": target,
        "依赖动作": depends_on,
        "生成对象": creates,
        "执行参数": payload,
    }


def _post_target(post_ref: str, owner_id: str, target_label: str) -> dict[str, Any]:
    return {
        "目标类型": "post",
        "post_ref": post_ref,
        "目标数字人id": owner_id,
        "目标说明": target_label,
    }


def _persona_for_node(*, node: StrategyNodePlan, topic_labels: list[str]) -> str:
    topic_hint = "、".join(topic_labels[:2])
    suffix = f"（侧重：{topic_hint}）" if topic_hint else ""
    mapping = {
        "core_publish_node": "理性框架型信息发布者",
        "interaction_response_node": "答疑承接型互动账号",
        "amplification_node": "摘要扩散型转述账号",
        "support_node": "背景补充型说明账号",
    }
    return f"{mapping.get(node.selected_role, '通用协同型数字人')}{suffix}"


def _publish_target(
    *,
    node: StrategyNodePlan,
    own_post_ref: str,
    target_core_id: str,
    target_core_ref: str,
) -> dict[str, Any]:
    if node.selected_role == "core_publish_node" or target_core_id == node.user_id:
        return {
            "目标类型": "public_feed",
            "post_ref": own_post_ref,
            "目标说明": "公共信息流首发",
        }
    return _post_target(target_core_ref, target_core_id, f"数字人 {target_core_id} 的主帖")


def _build_platform_coordination(
    *,
    primary_platform: str,
    secondary_platforms: list[str],
) -> list[dict[str, Any]]:
    plans = [
        {
            "平台": primary_platform,
            "角色": "主平台",
            "协同方式": "所有主动作先在此闭环执行，先发帖、再监测、再协同互动。",
        }
    ]
    for platform in secondary_platforms:
        plans.append(
            {
                "平台": platform,
                "角色": "协同平台",
                "协同方式": "当前仅保留占位，不单独生成动作；后续接入该平台数据集后再展开同步发布与转述。",
            }
        )
    return plans


def _make_action(
    *,
    action_type: str,
    action_name: str,
    execute_window: str,
    trigger_condition: str,
    target_object: str,
    parameters: dict[str, Any],
    target_result: str,
) -> dict[str, Any]:
    return {
        "动作类型": action_type,
        "动作名称": action_name,
        "执行窗口": execute_window,
        "触发条件": trigger_condition,
        "目标对象": target_object,
        "执行参数": parameters,
        "目标结果": target_result,
    }


def _build_content_publish_actions(
    *,
    node: StrategyNodePlan,
    platform: str,
    post_content: str,
    topic_labels: list[str],
    role_bucket_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    role_action_map = {
        "core_publish_node": ("publish_post", "发布主帖"),
        "interaction_response_node": ("publish_followup_post", "发布跟进帖"),
        "amplification_node": ("quote_repost", "转发并附评"),
        "support_node": ("publish_support_post", "发布补充帖"),
    }
    action_type, action_name = role_action_map.get(
        node.selected_role,
        ("publish_post", "发布帖子"),
    )

    reference_ids = role_bucket_ids.get("core_publish_node", [])
    trigger_condition = "进入对应时间阶段后立即执行"
    target_object = "公共信息流"
    if node.selected_role in {"interaction_response_node", "amplification_node", "support_node"} and reference_ids:
        trigger_condition = f"核心节点 {reference_ids[0]} 完成首发后 10-30 分钟内执行"
        target_object = f"围绕数字人 {reference_ids[0]} 的核心信息展开"

    return [
        _make_action(
            action_type=action_type,
            action_name=action_name,
            execute_window=_stage_text(node),
            trigger_condition=trigger_condition,
            target_object=target_object,
            parameters={
                "执行平台": platform,
                "执行频次": _frequency_text(node.frequency_per_day),
                "内容风格": node.suggested_content_style,
                "建议话题标签": topic_labels,
                "正文草案": post_content,
                "关联数字人id": reference_ids[:2],
            },
            target_result="完成节点自身的阶段性内容发布任务，并为后续互动或扩散提供锚点。",
        )
    ]


def _build_audience_action_plan(
    *,
    node: StrategyNodePlan,
    topic_labels: list[str],
    interaction_focus: str,
    role_bucket_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        _make_action(
            action_type="monitor_comments",
            action_name="监测评论",
            execute_window=_stage_text(node),
            trigger_condition="帖子发出后持续监测评论区反馈",
            target_object="评论区与转评区",
            parameters={
                "优先关键词": topic_labels[:4],
                "关注类型": ["高频疑问", "事实补充", "明显误读"],
            },
            target_result="识别需要优先回应的问题与高价值讨论线索。",
        )
    ]

    if node.selected_role == "core_publish_node":
        actions.extend(
            [
                _make_action(
                    action_type="like_comment",
                    action_name="点赞评论",
                    execute_window=_stage_text(node),
                    trigger_condition="出现理性提问或有效补充信息的评论时执行",
                    target_object="用户评论",
                    parameters={
                        "数量上限": 5,
                        "优先评论类型": ["理性提问", "事实补充"],
                    },
                    target_result="提高优质评论的可见度，稳定讨论方向。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="前 3 个高频问题出现后执行",
                    target_object="高频问题评论",
                    parameters={
                        "数量上限": 3,
                        "回复风格": "解释型",
                        "回复要点": topic_labels[:3],
                    },
                    target_result="快速建立统一口径，避免首轮误读扩散。",
                ),
            ]
        )
    elif node.selected_role == "interaction_response_node":
        target_core = (role_bucket_ids.get("core_publish_node") or ["-"])[0]
        actions.extend(
            [
                _make_action(
                    action_type="comment_on_post",
                    action_name="评论主帖",
                    execute_window=_stage_text(node),
                    trigger_condition="核心主帖发布后立即跟进",
                    target_object=f"数字人 {target_core} 的主帖",
                    parameters={
                        "评论数量": 1,
                        "评论目的": "补充常见问题入口或追问锚点",
                        "互动关注点": interaction_focus,
                    },
                    target_result="把评论区讨论引导到可承接的问题上。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="评论区出现高频疑问、误读或追问时执行",
                    target_object="主帖评论区",
                    parameters={
                        "数量上限": 5,
                        "回复风格": "答疑型",
                        "引用核心节点": target_core,
                    },
                    target_result="集中消化高频问题，降低核心节点的回复压力。",
                ),
            ]
        )
    elif node.selected_role == "amplification_node":
        actions.extend(
            [
                _make_action(
                    action_type="like_post",
                    action_name="点赞帖子",
                    execute_window=_stage_text(node),
                    trigger_condition="核心节点完成首发后执行",
                    target_object=f"数字人 {(role_bucket_ids.get('core_publish_node') or ['-'])[0]} 的主帖",
                    parameters={
                        "执行次数": 1,
                    },
                    target_result="建立扩散节点与核心信息之间的可见连接。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="二次扩散帖下出现转化为事实问题的评论时执行",
                    target_object="扩散帖评论区",
                    parameters={
                        "数量上限": 3,
                        "回复风格": "摘要型",
                        "回复要点": topic_labels[:2],
                    },
                    target_result="让扩散环节保持一致口径，不偏离核心叙事。",
                ),
            ]
        )
    else:
        actions.extend(
            [
                _make_action(
                    action_type="like_comment",
                    action_name="点赞评论",
                    execute_window=_stage_text(node),
                    trigger_condition="出现有代表性的长尾问题或补充观点时执行",
                    target_object="长尾评论",
                    parameters={
                        "数量上限": 4,
                        "优先评论类型": ["背景追问", "案例补充"],
                    },
                    target_result="扶持更深入的讨论线索，避免讨论过快收缩。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="需要补充背景信息时执行",
                    target_object="长尾问题评论",
                    parameters={
                        "数量上限": 2,
                        "回复风格": "补充说明型",
                        "互动关注点": interaction_focus,
                    },
                    target_result="补全背景信息，提升讨论完整度。",
                ),
            ]
        )

    return actions


def _build_coordination_action_plan(
    *,
    node: StrategyNodePlan,
    coordination_focus: str,
    role_bucket_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    core_ids = role_bucket_ids.get("core_publish_node", [])
    interaction_ids = role_bucket_ids.get("interaction_response_node", [])
    amplification_ids = role_bucket_ids.get("amplification_node", [])
    support_ids = role_bucket_ids.get("support_node", [])

    if node.selected_role == "core_publish_node":
        return [
            _make_action(
                action_type="reply_agent_comment",
                action_name="回复数字人评论",
                execute_window=_stage_text(node),
                trigger_condition="互动节点在主帖下完成第一轮评论后执行",
                target_object=f"数字人 {','.join(interaction_ids[:2]) or '-'} 的评论",
                parameters={
                    "动作目的": "确认关键信息并锁定统一口径",
                    "涉及数字人id": interaction_ids[:2],
                },
                target_result="形成核心节点与互动节点之间的可见协同。",
            ),
            _make_action(
                action_type="like_post",
                action_name="点赞数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="扩散或支持节点发布有效补充内容后执行",
                target_object=f"数字人 {','.join((amplification_ids + support_ids)[:2]) or '-'} 的帖子",
                parameters={
                    "数量上限": 2,
                    "协同关注点": coordination_focus,
                },
                target_result="放大协同节点的有效补充内容。",
            ),
        ]

    if node.selected_role == "interaction_response_node":
        return [
            _make_action(
                action_type="comment_on_post",
                action_name="评论数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="核心主帖发布后 5-15 分钟内执行",
                target_object=f"数字人 {','.join(core_ids[:1]) or '-'} 的主帖",
                parameters={
                    "评论数量": 1,
                    "评论目的": "承接问答和补充追问入口",
                    "涉及数字人id": core_ids[:1],
                },
                target_result="把互动节点挂接到核心帖下的主要讨论流。",
            ),
            _make_action(
                action_type="like_post",
                action_name="点赞数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="支持节点发布补充信息后执行",
                target_object=f"数字人 {','.join(support_ids[:2]) or '-'} 的补充帖",
                parameters={
                    "数量上限": 2,
                    "协同关注点": coordination_focus,
                },
                target_result="提高补充帖的可见度并形成协同反馈。",
            ),
        ]

    if node.selected_role == "amplification_node":
        return [
            _make_action(
                action_type="like_post",
                action_name="点赞数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="核心节点完成首发后执行",
                target_object=f"数字人 {','.join(core_ids[:1]) or '-'} 的主帖",
                parameters={
                    "执行次数": 1,
                    "涉及数字人id": core_ids[:1],
                },
                target_result="建立扩散节点与核心节点的显式关联。",
            ),
            _make_action(
                action_type="comment_on_post",
                action_name="评论数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="支持节点或互动节点发布补充信息后执行",
                target_object=f"数字人 {','.join((interaction_ids + support_ids)[:2]) or '-'} 的帖子",
                parameters={
                    "评论数量": 1,
                    "评论目的": "提炼亮点并导流回核心信息",
                    "协同关注点": coordination_focus,
                },
                target_result="让扩散内容和补充内容形成互相导流。",
            ),
        ]

    return [
        _make_action(
            action_type="comment_on_post",
            action_name="评论数字人帖子",
            execute_window=_stage_text(node),
            trigger_condition="核心帖或扩散帖出现需要补充背景的节点时执行",
            target_object=f"数字人 {','.join((core_ids + amplification_ids)[:2]) or '-'} 的帖子",
            parameters={
                "评论数量": 1,
                "评论目的": "补充背景、案例或上下文",
                "协同关注点": coordination_focus,
            },
            target_result="让支持节点承担背景补完角色，而不是重复主信息。",
        ),
        _make_action(
            action_type="like_comment",
            action_name="点赞数字人评论",
            execute_window=_stage_text(node),
            trigger_condition="互动节点完成高质量答疑后执行",
            target_object=f"数字人 {','.join(interaction_ids[:2]) or '-'} 的评论",
            parameters={
                "数量上限": 2,
                "涉及数字人id": interaction_ids[:2],
            },
            target_result="提高高质量答疑评论的曝光，稳定讨论节奏。",
        ),
    ]


def _build_risk_action_plan(node: StrategyNodePlan) -> list[dict[str, Any]]:
    return [
        _make_action(
            action_type="monitor_sentiment",
            action_name="监测舆情反馈",
            execute_window=_stage_text(node),
            trigger_condition="节点动作执行后持续观察反馈",
            target_object="评论区、转评区、协同节点反馈",
            parameters={
                "重点风险": node.risk_flags or ["内容同质化", "误读扩散"],
                "人工复核": node.manual_review_required,
            },
            target_result="尽快发现误读、负向升级或异常扩散。",
        ),
        _make_action(
            action_type="pause_dispatch_if_needed",
            action_name="必要时暂停动作",
            execute_window=_stage_text(node),
            trigger_condition="出现明显误读升级、敏感争议或与主口径冲突时执行",
            target_object="当前节点后续发布与互动动作",
            parameters={
                "暂停条件": ["负向评论快速上升", "敏感话题偏离", "与核心口径冲突"],
                "后续处理": "切换为人工复核或仅保留事实回应",
            },
            target_result="避免错误扩散和协同失真。",
        ),
    ]


def _run_hot_event_evaluation(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    hot_event: dict[str, Any],
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path,
    target_user_id: str | None = None,
    dispatch_platform: str | None = None,
    recall_size: int = 200,
    recall_mode: Literal["platform_aware"] = "platform_aware",
) -> tuple[Path, dict[str, Any], dict[str, float]]:
    root = Path(workspace_root)
    stage_times: dict[str, float] = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    client = llm_client
    if client is None and use_llm:
        client = OpenAICompatibleLLMClient.from_env_files(root)
    stage_times["llm_client_init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipeline_payload = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=max_selected_nodes,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
    )
    stage_times["hot_event_to_payload"] = time.perf_counter() - t0

    # 召回开关归一化：
    #   1) 单人 target_user_id 是「强制唯一节点」，不存在候选池概念 → 关掉；
    #   2) recall_size <= 0 → 视作关闭，统一归一成 None，
    #      避免下游拿到 0 这种「半开」状态。
    effective_recall_size = (
        recall_size
        if target_user_id is None and recall_mode and recall_size > 0
        else None
    )
    effective_recall_mode = (
        recall_mode if effective_recall_size is not None else None
    )

    pipeline = StrategyPipeline(product_name="abc_reading")
    artifacts = pipeline.run_with_artifacts(
        workspace_root=root,
        event_input=pipeline_payload,
        profile_limit=profile_limit,
        use_llm=use_llm,
        llm_client=client,
        target_user_id=target_user_id,
        dispatch_platform=dispatch_platform,
        recall_size=effective_recall_size,
        recall_mode=effective_recall_mode,
    )
    # 把 pipeline 子键展开成顶层键，便于压测面板直接拿 event_parser/feature_builder/...
    for key, value in artifacts.stage_times.items():
        stage_times[key] = value

    t0 = time.perf_counter()
    output_payload, content_llm_meta = build_eval_output(
        hot_event=hot_event,
        strategy_result=artifacts.strategy_result,
        workspace_root=root,
        # 文案生成不再调用大模型（见 CONTENT_LLM_ENABLED），改走规则占位文本。
        use_llm=CONTENT_LLM_ENABLED,
        llm_client=client,
    )
    stage_times["content_generation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    target_dir = Path(output_dir)
    output_path = target_dir / f"{artifacts.strategy_result.event.event_id}_strategy_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage_times["write_output"] = time.perf_counter() - t0

    # run_label 只表示**运行模式**（automatic / targeted），不再复用它透传选人结果；
    # 选人状态单独写进 selection_status，二者职责分开。
    run_label_value = "targeted" if target_user_id else "automatic"
    selection_status = artifacts.strategy_result.summary.status
    t0 = time.perf_counter()
    _write_pipeline_trace(
        trace_dir=trace_dir,
        hot_event=hot_event,
        pipeline_payload=pipeline_payload,
        artifacts=artifacts,
        final_output=output_payload,
        output_path=output_path,
        content_llm_meta=content_llm_meta,
        run_label=run_label_value,
        selection_status=selection_status,
    )
    stage_times["write_trace"] = time.perf_counter() - t0

    stage_times["event_total"] = time.perf_counter() - total_start
    return output_path, output_payload, stage_times


def _build_llm_node_texts(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    platform: str,
    target: str,
    workspace_root: str | Path | None,
    use_llm: bool,
    llm_client: Any | None,
    platform_for: Callable[[StrategyNodePlan], str] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "requested": bool(use_llm),
        "configured": False,
        "attempted": False,
        "used": False,
        "provider": None,
        "model": None,
        "base_url": None,
        "fallback_reason": None,
    }
    started = time.monotonic()

    if not use_llm:
        metadata["fallback_reason"] = "llm_disabled"
        _record_content_llm_outcome(
            success=False,
            fallback_reason="llm_disabled",
            attempted=False,
            elapsed_sec=time.monotonic() - started,
        )
        return {}, metadata

    if strategy_result.selection_summary.selection_mode == "targeted":
        # Targeted mode: the user has explicitly chosen the node in Selector.
        # Skip the LLM-side eligibility filter so the chosen node can still
        # produce content; risk flags / manual_review_required are preserved
        # on the node and surface through risk_control downstream.
        eligible_nodes = list(strategy_result.selected_nodes)
        metadata["targeted_bypass_applied"] = True
        metadata["bypass_reason"] = "targeted_user_explicitly_chosen"
    elif hot_event.get("source_type") == "image":
        eligible_nodes = list(strategy_result.selected_nodes)
    else:
        eligible_nodes = [
            node for node in strategy_result.selected_nodes if _can_connect_llm(node)
        ]
    if not eligible_nodes:
        metadata["fallback_reason"] = "no_eligible_nodes"
        _record_content_llm_outcome(
            success=False,
            fallback_reason="no_eligible_nodes",
            attempted=False,
            elapsed_sec=time.monotonic() - started,
        )
        return {}, metadata

    client = llm_client
    if client is None and workspace_root is not None:
        client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
    if client is None:
        metadata["fallback_reason"] = "llm_not_configured"
        _record_content_llm_outcome(
            success=False,
            fallback_reason="llm_not_configured",
            attempted=False,
            elapsed_sec=time.monotonic() - started,
        )
        return {}, metadata

    if hasattr(client, "describe"):
        details = client.describe()
        metadata["provider"] = details.get("provider")
        metadata["model"] = details.get("model")
        metadata["base_url"] = details.get("base_url")
    else:
        metadata["provider"] = getattr(client, "provider", None)
        metadata["model"] = getattr(client, "model", None)
        metadata["base_url"] = getattr(client, "base_url", None)
    metadata["configured"] = True
    metadata["attempted"] = True

    system_prompt = (
        "你是影响力事件分发策略的内容生成助手。"
        "请仅为已选出的匿名数字人生成中文执行文案。"
        "必须返回严格 JSON，不要输出 Markdown。"
    )
    user_prompt = _llm_prompt(
        hot_event=hot_event,
        strategy_result=strategy_result,
        eligible_nodes=eligible_nodes,
        platform=platform,
        target=target,
        platform_for=platform_for,
    )

    prompt_chars = len(system_prompt) + len(user_prompt)
    call_started = time.monotonic()

    try:
        response = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (LLMClientError, OSError, ValueError, TypeError) as exc:
        metadata["fallback_reason"] = f"llm_request_failed:{type(exc).__name__}"
        _record_content_llm_outcome(
            success=False,
            fallback_reason=metadata["fallback_reason"],
            attempted=True,
            prompt_chars=prompt_chars,
            elapsed_sec=time.monotonic() - call_started,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        logger.warning(
            "eval_hot_events._build_llm_node_texts failed: "
            "error_type=%s prompt_chars=%d elapsed_sec=%.3f error=%r",
            type(exc).__name__,
            prompt_chars,
            time.monotonic() - call_started,
            exc,
        )
        return {}, metadata

    llm_nodes = _extract_llm_node_mapping(
        response=response,
        expected_ids=[node.user_id for node in eligible_nodes],
    )
    if not llm_nodes:
        metadata["fallback_reason"] = "invalid_llm_response"
        _record_content_llm_outcome(
            success=False,
            fallback_reason="invalid_llm_response",
            attempted=True,
            prompt_chars=prompt_chars,
            elapsed_sec=time.monotonic() - call_started,
        )
        return {}, metadata

    overrides: dict[str, dict[str, str]] = {}
    for node in eligible_nodes:
        raw_node = llm_nodes.get(node.user_id)
        if not isinstance(raw_node, dict):
            continue
        cleaned = _clean_llm_node_fields(raw_node)
        if cleaned:
            overrides[node.user_id] = cleaned

    elapsed_sec = time.monotonic() - call_started
    if overrides:
        metadata["used"] = True
        _record_content_llm_outcome(
            success=True,
            fallback_reason=None,
            attempted=True,
            prompt_chars=prompt_chars,
            elapsed_sec=elapsed_sec,
        )
        logger.info(
            "eval_hot_events._build_llm_node_texts success: "
            "prompt_chars=%d elapsed_sec=%.3f nodes=%d",
            prompt_chars,
            elapsed_sec,
            len(overrides),
        )
    else:
        metadata["fallback_reason"] = "empty_llm_content"
        _record_content_llm_outcome(
            success=False,
            fallback_reason="empty_llm_content",
            attempted=True,
            prompt_chars=prompt_chars,
            elapsed_sec=elapsed_sec,
        )
    return overrides, metadata


def _compact_content_llm_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """精简的内容生成 LLM 诊断，内嵌到策略输出里，便于一眼看出帖子是 LLM 生成还是兜底。"""
    return {
        "used": bool(meta.get("used")),
        "fallback_reason": meta.get("fallback_reason"),
        "provider": meta.get("provider"),
        "model": meta.get("model"),
    }


def _fallback_content_bundle(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    node: StrategyNodePlan,
    node_index: int,
    target: str,
    variants: list[str],
    selected_ids: list[str],
) -> dict[str, Any]:
    return {
        "post_content": _post_content_text(
            node=node,
            node_index=node_index,
            event_title=str(hot_event.get("event_title") or ""),
            event_summary=str(hot_event.get("event_summary") or ""),
            variants=variants,
        ),
        "audience_profile": _audience_profile_text(
            hot_event=hot_event,
            strategy_result=strategy_result,
            node=node,
        ),
        "audience_interaction_strategy": _audience_interaction_text(node),
        "cross_digital_human_ids": [item for item in selected_ids if item != node.user_id],
        "cross_digital_human_strategy": _cross_digital_human_strategy(
            node,
            strategy_result.selected_nodes,
        ),
    }


def _can_connect_llm(node: StrategyNodePlan) -> bool:
    return (
        not node.manual_review_required
        and node.risk_level != "high"
        and node.final_score >= 0.30
        and node.stability_score >= 0.20
    )


def _llm_prompt(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    eligible_nodes: list[StrategyNodePlan],
    platform: str,
    target: str,
    platform_for: Callable[[StrategyNodePlan], str] | None = None,
) -> str:
    variants = _normalize_variants(hot_event.get("opinion_variants", []))
    nodes_payload = []
    for node in eligible_nodes:
        # 跨平台模式下每个节点属于不同平台，把同平台协作对象一并交给 LLM，
        # 否则模型会以为"其它平台的人"也会来挂接当前平台的帖子。
        node_platform = platform_for(node) if platform_for is not None else platform
        other_ids = [
            item.user_id
            for item in strategy_result.selected_nodes
            if item.user_id != node.user_id
            and (platform_for(item) if platform_for is not None else platform)
            == node_platform
        ]
        nodes_payload.append(
            {
                "user_id": node.user_id,
                "role": node.selected_role,
                "stage": _stage_text(node),
                "frequency": _frequency_text(node.frequency_per_day),
                "platform": node_platform,
                "other_digital_human_ids": other_ids,
                "recommended_action": node.recommended_action,
                "content_style_hint": node.suggested_content_style,
            }
        )

    example_node_key = (
        f"id{eligible_nodes[0].user_id}" if eligible_nodes else "id_example"
    )

    prompt_payload = {
        "task": "为每个候选数字人生成结构化分发文案。",
        "event": {
            "event_id": hot_event.get("event_id"),
            "domain": hot_event.get("domain"),
            "event_title": hot_event.get("event_title"),
            "event_summary": hot_event.get("event_summary"),
            "target": target,
            "opinion_variants": variants,
        },
        "constraints": [
            "保持客观克制，适合公共热点信息传播。",
            "不要编造未给出的外部事实或真实平台权限。",
            "每个字段使用 1 到 3 句中文自然语言。",
            "高粉丝量不能替代主题相关性。",
            "返回 nodes 对象时，key 使用 id+user_id，例如 id81584。",
        ],
        "eligible_nodes": nodes_payload,
        "required_json_schema": {
            "nodes": {
                example_node_key: {
                    "post_content": "发帖内容",
                    "audience_profile": "目标群体画像",
                    "audience_interaction_strategy": "目标群体交互策略",
                    "cross_digital_human_strategy": "与其他数字人的互动策略",
                }
            }
        },
    }
    return json.dumps(prompt_payload, ensure_ascii=False, indent=2)


def _clean_llm_node_fields(raw_node: dict[str, Any]) -> dict[str, str]:
    field_aliases = {
        "post_content": ("post_content", "发帖内容"),
        "audience_profile": ("audience_profile", "目标群体画像"),
        "audience_interaction_strategy": (
            "audience_interaction_strategy",
            "目标群体交互策略",
        ),
        "cross_digital_human_strategy": (
            "cross_digital_human_strategy",
            "互动策略",
            "与其他数字人互动策略",
        ),
    }

    cleaned: dict[str, str] = {}
    for normalized_name, aliases in field_aliases.items():
        for alias in aliases:
            value = raw_node.get(alias)
            if isinstance(value, str) and value.strip():
                cleaned[normalized_name] = value.strip()
                break
    return cleaned


def _extract_llm_node_mapping(
    *,
    response: dict[str, Any],
    expected_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(response, dict):
        return {}

    for key in ("nodes", "candidates", "results", "items", "data"):
        normalized = _normalize_llm_container(response.get(key), expected_ids)
        if normalized:
            return normalized

    normalized = _normalize_llm_container(response, expected_ids)
    if normalized:
        return normalized

    for value in response.values():
        normalized = _normalize_llm_container(value, expected_ids)
        if normalized:
            return normalized
    return {}


def _normalize_llm_container(
    container: Any,
    expected_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if isinstance(container, list):
        normalized: dict[str, dict[str, Any]] = {}
        for item in container:
            if not isinstance(item, dict):
                continue
            user_id = str(
                item.get("user_id")
                or item.get("id")
                or item.get("node_id")
                or item.get("digital_human_id")
                or ""
            ).strip()
            if user_id:
                normalized[user_id] = item
        return normalized

    if not isinstance(container, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for expected_id in expected_ids:
        raw_node = (
            container.get(expected_id)
            or container.get(str(expected_id))
            or container.get(f"id{expected_id}")
            or container.get(f"digital_human_{expected_id}")
        )
        if isinstance(raw_node, dict):
            normalized[str(expected_id)] = raw_node
    if normalized:
        return normalized

    for key, value in container.items():
        if not isinstance(value, dict):
            continue
        candidate_id = str(key).strip().removeprefix("id").removeprefix("digital_human_")
        if candidate_id in expected_ids:
            normalized[candidate_id] = value
    return normalized


def _stage_text(node: StrategyNodePlan) -> str:
    stage_map = {
        "stage_1_launch": "启动期",
        "stage_2_engage": "互动期",
        "stage_2_support": "支持期",
        "stage_3_amplify": "扩散期",
    }
    stage_label = stage_map.get(node.dispatch_stage, node.dispatch_stage)
    if node.timing_window:
        return f"{stage_label}（{node.timing_window}）"
    return stage_label


def _frequency_text(frequency_per_day: int) -> str:
    return f"{frequency_per_day}条/天"


def _post_content_text(
    *,
    node: StrategyNodePlan,
    node_index: int,
    event_title: str,
    event_summary: str,
    variants: list[str],
) -> str:
    """生成可发布的自然叙述帖子正文（兜底路径，不依赖 LLM）。

    优先复用事件自带的 opinion_variants——它们本身就是完整中文句，按节点顺序轮转，
    保证不同数字人拿到不同正文。无变体时基于事件摘要合成陈述。
    不再拼接「传播目标/内容风格/执行动作」等内部标签：那些是策略元数据，
    已经存在于动作清单的执行参数里，不应进入对外帖子。
    """
    if variants:
        return variants[(node_index - 1) % len(variants)]

    summary = (event_summary or event_title or "").strip()
    if not summary:
        summary = "事件信息更新中，请以官方通报为准。"
    if node.selected_role == "interaction_response_node":
        return f"{summary} 你怎么看这件事？"
    return summary


def _audience_profile_text(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    node: StrategyNodePlan,
) -> str:
    target_objects = "、".join(strategy_result.strategy.target_object) or "泛公众"
    domain = str(hot_event.get("domain") or "general")
    return (
        f"重点面向 {target_objects}，优先覆盖关注 {domain} 议题、"
        "需要事件解释与阶段性信息更新的人群。"
        f"该节点 final_score={node.final_score:.3f}。"
    )


def _audience_interaction_text(node: StrategyNodePlan) -> str:
    if node.selected_role == "core_publish_node":
        return "优先承接首轮评论，围绕核心事实、影响范围和后续行动建议引导讨论。"
    if node.selected_role == "interaction_response_node":
        return "重点回复高频问题，筛选理性追问并做二次解释，必要时回引核心发布内容。"
    if node.selected_role == "amplification_node":
        return "优先转述核心信息，面向相邻圈层用户做摘要式扩散和补充互动。"
    return "补充背景信息，回应长尾问题，避免与其他数字人重复表达。"


def _cross_digital_human_strategy(
    node: StrategyNodePlan,
    selected_nodes: list[StrategyNodePlan],
) -> str:
    core_ids = [
        item.user_id for item in selected_nodes
        if item.selected_role == "core_publish_node" and item.user_id != node.user_id
    ]
    engage_ids = [
        item.user_id for item in selected_nodes
        if item.selected_role == "interaction_response_node" and item.user_id != node.user_id
    ]
    amplify_ids = [
        item.user_id for item in selected_nodes
        if item.selected_role == "amplification_node" and item.user_id != node.user_id
    ]

    if node.selected_role == "core_publish_node":
        return (
            f"发布首帖后，由互动节点 {','.join(engage_ids) or '-'} 承接评论，"
            f"由扩散节点 {','.join(amplify_ids) or '-'} 在后续阶段做转述放大。"
        )
    if node.selected_role == "interaction_response_node":
        return (
            f"优先评论核心发布节点 {','.join(core_ids) or '-'} 的主帖，"
            "集中回复评论区高频问题，并将讨论回收到统一口径。"
        )
    if node.selected_role == "amplification_node":
        return (
            f"在扩散期转发核心节点 {','.join(core_ids) or '-'} 的内容，"
            f"必要时补充互动节点 {','.join(engage_ids) or '-'} 的答疑信息。"
        )
    return "对其他主选数字人的关键信息做补充说明和长尾维护。"


def _normalize_variants(raw_variants: Any) -> list[str]:
    if not isinstance(raw_variants, list):
        return []
    return [str(item).strip() for item in raw_variants if str(item).strip()]


def _infer_hot_event_risk_level(domain: str) -> str:
    if domain in {"military", "politics", "international_relations", "cybersecurity"}:
        return "high"
    if domain in {"finance", "financial_market", "energy", "public_policy"}:
        return "medium"
    return "low"


def _slugify_run_label(run_label: str | None) -> str:
    """Turn a free-form run label into a filesystem-safe short token."""

    if not run_label:
        return ""
    cleaned = "".join(
        char if (char.isalnum() or char in "-_") else "-" for char in str(run_label)
    )
    return cleaned.strip("-_")[:60]


def _previous_run_label(event_dir: Path) -> str:
    """Read the ``run_label`` recorded by the previous trace writer, if any.

    Older traces predate this field, in which case an empty label is returned
    and the archive folder falls back to the bare timestamp.
    """

    try:
        payload = json.loads(
            (event_dir / "07_final_output.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return _slugify_run_label(payload.get("run_label"))


def _archive_previous_trace(*, event_dir: Path) -> Path | None:
    """Snapshot the current top-level trace files into ``event_dir/runs/<stamp>/``.

    ``{trace_dir}/{event_id}/`` is keyed by event only, so a later run of a
    different mode would otherwise silently overwrite the earlier run's
    ``07_final_output.json``. Copying the previous run aside keeps every run
    reproducible. Canonical ``{event_id}/*.json`` paths keep pointing at the
    latest run, so existing readers are unaffected.

    The archive folder is named ``<timestamp>_<label of the run it holds>`` so
    the directory name describes its own contents rather than the new run.

    Returns the archive directory, or ``None`` when there was nothing to keep.
    """

    if not event_dir.exists():
        return None

    step_files = sorted(
        path
        for path in event_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    )
    if not step_files:
        return None

    runs_dir = event_dir / TRACE_RUNS_DIRNAME
    base = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    archived_label = _previous_run_label(event_dir)
    if archived_label:
        base = f"{base}_{archived_label}"

    archive_dir = runs_dir / base
    suffix = 0
    while archive_dir.exists():
        suffix += 1
        archive_dir = runs_dir / f"{base}_{suffix}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for path in step_files:
        shutil.copy2(path, archive_dir / path.name)
    return archive_dir


def _candidate_counts_view(summary: FeatureBuildSummary) -> dict[str, Any]:
    """把召回口径翻译成人类可读的「召回 N / 全量 M」视图。

    ``summary.node_count`` 在召回开启时是**召回人数**（如 200），不是全量
    候选人数。展示 / 报告一律用这个视图，不要直接把 node_count 当全量用。
    """
    total = summary.total_candidate_count
    if total is None:
        # 老数据 / 直连 build_features 未填：退回 node_count，口径标记 unknown。
        total = summary.node_count
    recalled = (
        summary.recall_size if summary.recall_mode is not None else summary.node_count
    )
    recall_enabled = summary.recall_mode is not None
    return {
        "candidate_pool_total": total,
        "candidate_pool_used": recalled,
        "recall_enabled": recall_enabled,
        "recall_mode": summary.recall_mode,
        # 人类可读口径，例如 "召回 200 / 全量 35636"
        "display_text": (
            f"召回 {recalled} / 全量 {total}"
            if recall_enabled and recalled != total
            else f"全量 {total}"
        ),
    }


def _write_pipeline_trace(
    *,
    trace_dir: str | Path,
    hot_event: dict[str, Any],
    pipeline_payload: dict[str, Any],
    artifacts: PipelineArtifacts,
    final_output: dict[str, Any],
    output_path: Path,
    content_llm_meta: dict[str, Any] | None = None,
    run_label: str | None = None,
    selection_status: str | None = None,
) -> None:
    # trace_dir 由调用方（评估入口 / 运行目录分配）显式给出，恒为本次运行的
    # trace 目录（默认 ``<运行目录>/trace/``），因此不会再写入独立的旧目录。
    target_root = Path(trace_dir)
    event_dir = target_root / artifacts.strategy_result.event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    archived_dir = _archive_previous_trace(event_dir=event_dir)

    if hot_event.get("source_type") == "image":
        _write_json(
            event_dir / "00_image_input.json",
            {
                "event_id": hot_event.get("event_id"),
                "source_type": hot_event.get("source_type"),
                "source_image": hot_event.get("source_image"),
            },
        )
        _write_json(
            event_dir / "00_image_recognition.json",
            hot_event.get("image_recognition", {}),
        )

    _write_json(
        event_dir / "00_hot_event_input.json",
        hot_event,
    )
    _write_json(
        event_dir / "01_pipeline_payload.json",
        pipeline_payload,
    )
    _write_json(
        event_dir / "02_event_parser_output.json",
        artifacts.event.model_dump(mode="json"),
    )
    _write_json(
        event_dir / "03_feature_builder_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.feature_result.summary.model_dump(mode="json"),
            # 显式展示口径：summary.node_count 在召回开启时是「召回人数」，
            # 不是全量人数。这里单独给出两个数字，避免读 trace 的人把 200
            # 误读成全量候选规模。
            "candidate_counts": _candidate_counts_view(
                artifacts.feature_result.summary
            ),
            "event_keywords": {
                "event_title": artifacts.event.event_title,
                "extracted_keywords": artifacts.event.extracted_keywords,
                "semantic_tags": artifacts.event.semantic_tags,
                "target_audience": artifacts.event.target_audience,
            },
            "llm_feature_used_count": sum(
                1 for node in artifacts.feature_result.node_features if node.llm_feature_used
            ),
            "preview_top_features": [
                node.model_dump(mode="json")
                for node in artifacts.feature_result.node_features[:TRACE_PREVIEW_LIMIT]
            ],
        },
    )
    _write_json(
        event_dir / "04_scorer_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.score_result.summary.model_dump(mode="json"),
            "eligible_count": sum(1 for node in artifacts.score_result.node_scores if node.eligible),
            "preview_top_scores": [
                node.model_dump(mode="json")
                for node in artifacts.score_result.node_scores[:TRACE_PREVIEW_LIMIT]
            ],
        },
    )
    _write_json(
        event_dir / "05_selector_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.selection_result.summary.model_dump(mode="json"),
            "selected_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.selection_result.selected_nodes
            ],
            "fallback_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.selection_result.fallback_nodes[:TRACE_PREVIEW_LIMIT]
            ],
        },
    )
    _write_json(
        event_dir / "06_strategy_generator_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.strategy_result.summary.model_dump(mode="json"),
            "stage_plans": [
                item.model_dump(mode="json")
                for item in artifacts.strategy_result.stage_plans
            ],
            "selected_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.strategy_result.selected_nodes
            ],
            "fallback_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.strategy_result.fallback_nodes[:TRACE_PREVIEW_LIMIT]
            ],
            "strategy": artifacts.strategy_result.strategy.model_dump(mode="json"),
        },
    )
    _write_json(
        event_dir / "07_final_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_label": run_label,
            "selection_status": selection_status,
            "final_output_path": str(output_path),
            "previous_run_archive": archived_dir.name if archived_dir else None,
            "payload": final_output,
        },
    )
    if content_llm_meta is not None:
        _write_json(
            event_dir / "08_content_generation_meta.json",
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "content_generation": content_llm_meta,
                "post_content_source": (
                    "llm" if content_llm_meta.get("used") else "rule_fallback"
                ),
            },
        )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
