from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import NodeScore, ScoreResult, SelectedNode, StrategyNodePlan, StrategyResult
from .pipeline import PipelineArtifacts, StrategyPipeline
from .platform_dispatcher import (
    MultiPlatformDispatchResult,
    PlatformDispatcher,
    PlatformSpec,
)
from .platform_profiles import (
    DATASET_PLATFORM_VALUES,
    dataset_platform_value,
    is_valid_dataset_platform,
    normalize_platform_key,
)
from .selector import (
    DEFAULT_DISPATCH_STAGE,
    DEFAULT_SELECTED_ROLE,
    ROLE_HINT_TO_DISPATCH_STAGE,
    ROLE_HINT_TO_SELECTED_ROLE,
    rebalance_selected_roles,
)
from .strategy_generator import StrategyGenerator

TRACE_PREVIEW_LIMIT = 25

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
MULTIPLATFORM_OUTPUT_FORMAT_VERSION = "action_schema_v6_multi_platform_minimal"
MULTIPLATFORM_DRAFT_NOTE = (
    "以下内容为 PlatformDispatcher 生成的结构化内容草案与动作草案，尚未在任何真实平台发布；"
    "各平台的数字人选取由 PlatformDispatcher 按平台自身的 requested_count 独立评估得出，"
    "不是由一次整体选取结果拆分而来，也未使用旧 Selector 的 max_selected_nodes 作为平台人数。"
)
MULTIPLATFORM_SCOPE_NOTE = (
    "本文件是多平台分发模式的最终业务结果：五维调度策略的结构与多人分发一致，"
    "但数字人由 PlatformDispatcher 按平台分别选取，不再经过旧版单平台模拟分发的整体选人；"
    "各平台的选人依据、安全等级、内容计划与时间计划集中在「平台」维度，便于逐平台核对。"
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
) -> tuple[Path, dict[str, Any]]:
    events = load_hot_events(input_path)
    hot_event = select_hot_event(events, event_id=event_id)
    run_dirs = _resolve_run_dirs(
        workspace_root=workspace_root,
        mode=EVAL_MODE_MULTI_USER,
        output_dir=output_dir,
        trace_dir=trace_dir,
    )
    return _run_hot_event_evaluation(
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
    )


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
        output_path, output_payload = _run_hot_event_evaluation(
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
        output_path, output_payload = _run_hot_event_evaluation(
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
        output_path, output_payload = _run_hot_event_multiplatform_evaluation(
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
        )
        results.append((output_path, output_payload))
    return results


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
) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace_root)
    client = llm_client
    if client is None and use_llm:
        client = OpenAICompatibleLLMClient.from_env_files(root)

    pipeline_payload = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=max_selected_nodes,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
    )

    pipeline = StrategyPipeline(product_name="abc_reading")
    artifacts = pipeline.run_with_artifacts(
        workspace_root=root,
        event_input=pipeline_payload,
        profile_limit=profile_limit,
        use_llm=use_llm,
        llm_client=client,
        target_user_id=None,
    )

    # 跨平台模式同样产出五维调度策略：先由 PlatformDispatcher 按平台独立选人，
    # 再把选人结果翻译回策略节点，复用 build_eval_output 同一套节点规划与动作生成，
    # 保证两种模式的动作口径、阶段口径与内容来源完全一致。
    dispatch_result = PlatformDispatcher().dispatch(
        score_result=artifacts.score_result,
        platform_specs=platform_specs,
        campaign_window_hours=campaign_window_hours,
    )
    output_payload, content_llm_meta = build_multiplatform_output(
        hot_event=hot_event,
        strategy_result=artifacts.strategy_result,
        dispatch_result=dispatch_result,
        score_result=artifacts.score_result,
        workspace_root=root,
        use_llm=use_llm,
        llm_client=client,
    )

    target_dir = Path(output_dir)
    # 文件名同时带事件 ID、多平台标记与平台集合，
    # 保证既不覆盖旧的 {event_id}_strategy_output.json，也不会让不同平台组合互相覆盖。
    platform_tag = "-".join(
        platform.platform_key for platform in dispatch_result.platforms
    )
    output_path = (
        target_dir
        / f"{artifacts.strategy_result.event.event_id}_multiplatform_{platform_tag}_strategy_output.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_pipeline_trace(
        trace_dir=trace_dir,
        hot_event=hot_event,
        pipeline_payload=pipeline_payload,
        artifacts=artifacts,
        final_output=output_payload,
        output_path=output_path,
        # 跨平台模式现在也会为选中节点生成正文，因此同样写入内容生成诊断，
        # 让 trace 能区分正文来自 LLM 还是规则兜底。
        content_llm_meta=content_llm_meta,
        run_label=f"multiplatform_{platform_tag}",
    )
    return output_path, output_payload


def build_multiplatform_output(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    dispatch_result: MultiPlatformDispatchResult,
    score_result: ScoreResult,
    workspace_root: str | Path | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构造跨平台分发的最终业务输出（与多人分发共用五维结构）。

    输出契约与 :func:`build_eval_output` 一致：顶层为事件名称、输出版本标识与
    ``五维调度策略``，五个维度（目标对象 / 时间 / 频率 / 平台 / 内容）的字段名
    一一对应，因此前端可以复用同一套渲染组件。

    差异只落在「平台」维度：额外输出 ``跨平台总览`` 与 ``各平台方案``，把每个平台
    独立选人的人数、缺口、安全等级、内容计划与时间计划集中到一处。

    节点级内容与多人分发同源：``PlatformDispatcher`` 选出的数字人先按
    ``source_role_hint`` 翻译回策略角色（见 :mod:`selector` 的映射表），再由
    ``StrategyGenerator`` 补齐阶段/频率/内容风格，最后复用同一个动作生成器产出
    动作清单，避免两种模式各写一套而慢慢漂移。
    """

    selected_nodes = _selected_nodes_from_dispatch(
        dispatch_result=dispatch_result,
        score_result=score_result,
        strategy_result=strategy_result,
    )
    selected_ids = [node.user_id for node in selected_nodes]
    target = str(
        hot_event.get("target")
        or strategy_result.event.target_goal
        or "扩大热点信息触达并引导理性讨论"
    ).strip()
    variants = _normalize_variants(hot_event.get("opinion_variants", []))

    # LLM 只认 strategy_result.selected_nodes；跨平台选的人与整体选人不同，因此传
    # 一份"选中节点已替换为平台选人结果"的副本，其余策略上下文保持原样。
    llm_strategy_result = strategy_result.model_copy(
        update={"selected_nodes": selected_nodes}
    )
    llm_overrides, content_llm_meta = _build_llm_node_texts(
        hot_event=hot_event,
        strategy_result=llm_strategy_result,
        platform="、".join(item.display_name for item in dispatch_result.platforms)
        or dispatch_result.dispatch_mode,
        target=target,
        workspace_root=workspace_root,
        use_llm=use_llm,
        llm_client=llm_client,
        platform_for=_dispatch_platform_of_node,
    )

    # 协同对象只在**同一平台内**寻找：各平台独立排期执行，跨平台挂接动作既不
    # 符合这一前提，也会让动作指向另一个平台的帖子。
    peers_by_platform: dict[str, list[StrategyNodePlan]] = {}
    for node in selected_nodes:
        peers_by_platform.setdefault(_dispatch_platform_of_node(node), []).append(node)

    node_outputs, post_refs, _ = _build_node_outputs(
        hot_event=hot_event,
        strategy_result=strategy_result,
        selected_nodes=selected_nodes,
        target=target,
        variants=variants,
        llm_overrides=llm_overrides,
        platform_for=_dispatch_platform_of_node,
        peer_nodes_for=lambda node: peers_by_platform.get(
            _dispatch_platform_of_node(node), [node]
        ),
    )

    # 节点角色以 rebalance 后的 selected_role 为准：跨平台选人时 source_role_hint
    # 来自 PlatformDispatcher._rank_candidates，不保证每平台都有 core_publish_node。
    # _selected_nodes_from_dispatch 已经按平台做过角色再平衡（见
    # rebalance_selected_roles），这里输出的"节点角色"必须跟动作清单里实际使用的
    # 角色保持一致，避免 UI 看到 amplification 但动作像 core 的割裂感。
    selected_node_by_id = {node.user_id: node for node in selected_nodes}
    platform_item_fields = {
        item.user_id: {
            "平台角色": item.platform_role,
            "节点角色": selected_node_by_id[item.user_id].selected_role,
            "选取名次": item.selection_rank,
            "安全等级": item.safety_tier,
            "平台适配分": item.platform_fit_score,
            "平台调度分": item.platform_dispatch_score,
        }
        for platform_result in dispatch_result.platforms
        for item in platform_result.selected_nodes
    }

    five_dimension_strategy = _build_five_dimension_strategy(
        strategy_result=strategy_result,
        selected_nodes=selected_nodes,
        selected_ids=selected_ids,
        platform=dispatch_result.dispatch_mode,
        target=target,
        post_refs=post_refs,
        node_outputs=node_outputs,
        platform_item_fields=platform_item_fields,
        platform_extra={
            "跨平台总览": _build_cross_platform_overview(dispatch_result),
            "各平台方案": _build_platform_schemes(dispatch_result),
        },
    )
    five_dimension_strategy["内容"]["内容生成诊断"] = _compact_content_llm_meta(
        content_llm_meta
    )

    payload = {
        "事件名称": str(
            hot_event.get("event_title") or strategy_result.event.event_title
        ),
        "输出格式版本": MULTIPLATFORM_OUTPUT_FORMAT_VERSION,
        FIVE_DIMENSION_STRATEGY_KEY: five_dimension_strategy,
    }
    return payload, content_llm_meta


def _selected_nodes_from_dispatch(
    *,
    dispatch_result: MultiPlatformDispatchResult,
    score_result: ScoreResult,
    strategy_result: StrategyResult,
) -> list[StrategyNodePlan]:
    """把各平台的选人结果翻译成多人分发使用的 ``StrategyNodePlan``。

    ``PlatformNodeAssignment`` 只携带排序与安全字段，缺少行动级字段
    （阶段 / 频率 / 内容风格）。这里按 ``source_role_hint`` 恢复角色与阶段，再交给
    ``StrategyGenerator`` 用**同一个策略上下文**补齐其余字段，使跨平台输出的动作
    清单与多人分发逐字段同源，而不是各推一套。

    按平台分别做角色再平衡（``rebalance_selected_roles``），保证每个平台至少有一
    个 ``core_publish_node``。否则下游 ``_coordination_target_id`` /
    ``_interaction_target`` 找不到协同目标，整张动作清单会退化成"自发自评"——
    多人分发的 ``Selector`` 路径自带这个再平衡，跨平台分发的
    ``PlatformDispatcher`` 路径没有，这里手动补齐。
    """

    node_score_by_id = {node.user_id: node for node in score_result.node_scores}
    generator = StrategyGenerator()
    plans: list[StrategyNodePlan] = []
    for platform_result in dispatch_result.platforms:
        # Step 1：把该平台内选中的 assignment 转成 SelectedNode，保留 source_role_hint
        # 推导出的初始 selected_role。
        platform_selections: list[SelectedNode] = []
        for assignment in platform_result.selected_nodes:
            base = node_score_by_id.get(assignment.user_id)
            if base is None:
                # PlatformDispatcher 只从 score_result 的候选池里选人，正常不会发生；
                # 显式报错优于静默产出一个缺画像、缺策略依据的节点。
                raise ValueError(
                    f"跨平台调度结果中的数字人 {assignment.user_id} 不在本次评分结果内，"
                    "无法复用多人分发的节点规划逻辑。"
                )
            role_hint = assignment.source_role_hint or base.role_hint
            platform_selections.append(
                SelectedNode(
                    **base.model_dump(),
                    selection_rank=assignment.selection_rank,
                    selected_role=ROLE_HINT_TO_SELECTED_ROLE.get(
                        role_hint, DEFAULT_SELECTED_ROLE
                    ),
                    dispatch_stage=ROLE_HINT_TO_DISPATCH_STAGE.get(
                        role_hint, DEFAULT_DISPATCH_STAGE
                    ),
                    dispatch_priority=PRIORITY_TIER_TO_DISPATCH_PRIORITY.get(
                        str(getattr(base, "priority_tier", "") or ""), "p3"
                    ),
                    selection_bucket="primary",
                )
            )
        # Step 2：per-platform 角色再平衡。多人分发走的是
        # ``Selector._rebalance_selected_roles``，跨平台分发走的是
        # ``PlatformDispatcher``，没有内置这个保障；这里手动补齐，避免
        # amplification/support 全部聚集在同一平台导致动作生成阶段全部"自指"。
        rebalanced_selections = rebalance_selected_roles(platform_selections)
        # Step 3：由 ``StrategyGenerator`` 用同一套策略上下文补齐阶段/频率/内容风格。
        for selected_node in rebalanced_selections:
            plans.append(
                generator.build_node_plan_from_strategy(
                    node=selected_node,
                    strategy=strategy_result.strategy,
                    bucket="primary",
                )
            )
    return plans


def _dispatch_platform_of_node(node: StrategyNodePlan) -> str:
    """输出用的平台口径：统一取数据集写法（如 ``facebook_simulated``）。

    与 :func:`build_eval_output` 相同，避免同一份 JSON 里同时出现
    ``facebook`` 与 ``facebook_simulated`` 两套平台写法。
    """

    platform = node.platform or ""
    if is_valid_dataset_platform(platform):
        return dataset_platform_value(platform)
    return platform


def _safety_tier_distribution(tiers: Any) -> dict[str, int]:
    """把安全层级计数渲染成中文键，与五维调度策略的其余字段保持同一语言。"""

    return {
        "合格": int(getattr(tiers, "safe", 0)),
        "待人工复核": int(getattr(tiers, "review_required", 0)),
        "未通过资格筛选": int(getattr(tiers, "not_eligible", 0)),
    }


def _build_cross_platform_overview(
    dispatch_result: MultiPlatformDispatchResult,
) -> dict[str, Any]:
    """跨平台级汇总：人数、缺口、安全等级与跨平台告警。"""

    # PlatformDispatcher 会把每平台告警加 ``[平台]`` 前缀后聚合到顶层；不带前缀的
    # 才是真正的跨平台告警（合格缺口、重复数字人等），据此区分避免重复展示。
    cross_platform_warnings = [
        str(warning)
        for warning in dispatch_result.warnings
        if not str(warning).startswith("[")
    ]
    return {
        "调度模式": dispatch_result.dispatch_mode,
        "活动窗口小时": dispatch_result.campaign_window_hours,
        "平台数": len(dispatch_result.platforms),
        "目标人数合计": dispatch_result.total_requested_count,
        "选取人数合计": dispatch_result.total_selected_count,
        "去重数字人数": dispatch_result.unique_user_count,
        "合格缺口合计": dispatch_result.total_safe_shortfall_count,
        "安全等级分布": _safety_tier_distribution(
            dispatch_result.total_selected_safety_tiers
        ),
        "跨平台重复数字人": list(dispatch_result.overlapping_user_ids),
        "跨平台提示": cross_platform_warnings,
        "内容草案说明": MULTIPLATFORM_DRAFT_NOTE,
        "结果范围说明": MULTIPLATFORM_SCOPE_NOTE,
    }


def _build_platform_schemes(
    dispatch_result: MultiPlatformDispatchResult,
) -> list[dict[str, Any]]:
    """逐平台方案：人数与安全等级、内容计划、时间计划与平台级提示。"""

    schemes: list[dict[str, Any]] = []
    for platform_result in dispatch_result.platforms:
        content_plan = platform_result.content_plan
        time_plan = platform_result.time_plan
        schemes.append(
            {
                "平台": platform_result.platform_key,
                "平台名称": platform_result.display_name,
                "目标人数": platform_result.requested_count,
                "选取人数": platform_result.selected_count,
                "人数缺口": platform_result.shortfall_count,
                "合格缺口": platform_result.safe_shortfall_count,
                "安全等级分布": _safety_tier_distribution(
                    platform_result.selected_safety_tiers
                ),
                "活动窗口小时": time_plan.campaign_window_hours,
                "选取数字人id组": [
                    item.user_id for item in platform_result.selected_nodes
                ],
                "平台角色配置": list(platform_result.platform_roles),
                # 逐节点保留完整的选人与安全依据：五维表里给的是精简概览，
                # 这里是可追溯的明细，避免对齐结构时丢掉评分依据。
                "选取节点明细": [
                    {
                        "数字人id": item.user_id,
                        "数字人名称": item.user_name,
                        "分发平台": item.platform,
                        "选取名次": item.selection_rank,
                        "平台角色": item.platform_role,
                        "节点角色": ROLE_HINT_TO_SELECTED_ROLE.get(
                            item.source_role_hint, DEFAULT_SELECTED_ROLE
                        ),
                        "原始角色提示": item.source_role_hint,
                        "安全等级": item.safety_tier,
                        "综合分": item.final_score,
                        "主题匹配分": item.topic_match_score,
                        "影响力分": item.influence_score,
                        "扩散分": item.diffusion_score,
                        "稳定分": item.stability_score,
                        "活跃分": item.activity_score,
                        "平台适配分": item.platform_fit_score,
                        "平台调度分": item.platform_dispatch_score,
                        "适配依据": list(item.fit_basis),
                        "未声明输入": list(item.undeclared_inputs),
                        "风险等级": item.risk_level,
                        "资格通过": item.eligible,
                        "人工复核": item.manual_review_required,
                        "风险标记": list(item.risk_flags),
                    }
                    for item in platform_result.selected_nodes
                ],
                "内容形态": list(content_plan.content_modes),
                "写作指引": [
                    {
                        "环节标识": section.section_key,
                        "环节": section.label,
                        "写作指引": section.guidance,
                    }
                    for section in content_plan.sections
                ],
                "内容护栏": list(content_plan.guardrails),
                "时间阶段": [
                    {
                        "阶段标识": stage.stage_key,
                        "阶段": stage.label,
                        "时间窗": stage.window,
                        "开始小时": stage.start_hour,
                        "结束小时": stage.end_hour,
                        "重点": stage.focus,
                    }
                    for stage in time_plan.stages
                ],
                "节奏建议": list(time_plan.cadence_notes),
                "阶段说明": time_plan.stage_schedule_note,
                "平台提示": list(platform_result.warnings),
                "免责声明": platform_result.policy_disclaimer,
                "适配限制": list(platform_result.fit_limitations),
                "适配方法": platform_result.fit_method,
            }
        )
    return schemes


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

    node_outputs, post_refs, _ = _build_node_outputs(
        hot_event=hot_event,
        strategy_result=strategy_result,
        selected_nodes=selected_nodes,
        target=target,
        variants=variants,
        llm_overrides=llm_overrides,
        platform_for=lambda node: output_platform,
    )

    five_dimension_strategy = _build_five_dimension_strategy(
        strategy_result=strategy_result,
        selected_nodes=selected_nodes,
        selected_ids=selected_ids,
        platform=output_platform,
        target=target,
        post_refs=post_refs,
        node_outputs=node_outputs,
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


def _build_node_outputs(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    selected_nodes: list[StrategyNodePlan],
    target: str,
    variants: list[str],
    llm_overrides: dict[str, dict[str, str]],
    platform_for: Callable[[StrategyNodePlan], str],
    peer_nodes_for: Callable[[StrategyNodePlan], list[StrategyNodePlan]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """为每个选中节点生成正文与动作清单。

    多人分发与跨平台分发共用这段组装逻辑，差异只由两个回调表达：

    * ``platform_for``：节点执行落地的平台。多人分发是全局主平台，跨平台分发是
      该数字人自己的平台。
    * ``peer_nodes_for``：协同对象（``digital_human_coordination``）的候选范围。
      多人分发是全部选中节点；跨平台分发只在同平台内寻找——各平台独立排期执行，
      跨平台挂接动作既不成立，也会让动作指向另一个平台的帖子。

    返回 ``(node_outputs, post_refs, publish_action_ids)``。
    """

    post_refs = {
        node.user_id: f"P{index:03d}"
        for index, node in enumerate(selected_nodes, start=1)
    }
    publish_action_ids = {
        node.user_id: f"A{((index - 1) * 4) + 1:03d}"
        for index, node in enumerate(selected_nodes, start=1)
    }

    node_outputs: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(selected_nodes, start=1):
        peers = peer_nodes_for(node) if peer_nodes_for is not None else selected_nodes
        peer_ids = [item.user_id for item in peers]
        node_platform = platform_for(node)
        fallback_content = _fallback_content_bundle(
            hot_event=hot_event,
            strategy_result=strategy_result,
            node=node,
            node_index=index,
            target=target,
            variants=variants,
            selected_ids=peer_ids,
        )
        override_content = llm_overrides.get(node.user_id, {})
        post_content = (
            override_content.get("post_content") or fallback_content["post_content"]
        )
        audience_profile = (
            override_content.get("audience_profile")
            or fallback_content["audience_profile"]
        )
        interaction_focus = (
            override_content.get("audience_interaction_strategy")
            or fallback_content["audience_interaction_strategy"]
        )
        coordination_focus = (
            override_content.get("cross_digital_human_strategy")
            or fallback_content["cross_digital_human_strategy"]
        )
        topic_labels = _topic_labels(node, strategy_result)
        role_bucket_ids = _role_bucket_ids(peers, node.user_id)
        actions = _build_compact_action_plan(
            node=node,
            node_index=index,
            platform=node_platform,
            post_content=post_content,
            topic_labels=topic_labels,
            interaction_focus=interaction_focus,
            coordination_focus=coordination_focus,
            role_bucket_ids=role_bucket_ids,
            post_refs=post_refs,
            publish_action_ids=publish_action_ids,
        )
        coordination_action_ids = [
            action["动作编号"]
            for action in actions
            if action.get("动作分组") == "digital_human_coordination"
        ]

        node_outputs[f"id{node.user_id}"] = {
            "时间阶段": _stage_text(node),
            "发帖频率": _frequency_text(node.frequency_per_day),
            "发帖平台": node_platform,
            "数字人人设": _persona_for_node(node=node, topic_labels=topic_labels),
            "发帖内容": post_content,
            "目标受众": {
                "目标群体画像": audience_profile,
                "目标群体标签": strategy_result.strategy.target_object,
                "主题关键词": topic_labels,
            },
            "动作清单": actions,
            "与其他数字人互动策略": {
                "互动数字人id集合": [
                    item for item in peer_ids if item != node.user_id
                ],
                "协同动作编号": coordination_action_ids,
                "协同说明": coordination_focus,
            },
            "执行约束": {
                "节点角色": node.selected_role,
                "执行阶段": node.dispatch_stage,
                "执行优先级": node.dispatch_priority,
                "节点得分": round(node.final_score, 6),
                "风险等级": node.risk_level,
                "人工复核": node.manual_review_required,
                "风险标签": node.risk_flags,
            },
        }
    return node_outputs, post_refs, publish_action_ids


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
) -> dict[str, Any]:
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

    for node in selected_nodes:
        node_key = f"id{node.user_id}"
        node_output = node_outputs.get(node_key, {})
        post_ref = post_refs.get(node.user_id, "")
        node_actions = node_output.get("动作清单", [])
        # 至少保留 content_publish + audience_interaction + digital_human_coordination
        # 三组。``digital_human_coordination`` 是数字人之间的真正协同（点赞/评论/转发
        # 其他人的帖子），原先 ``max_actions=2`` 把它和 ``comment_monitoring`` 一起
        # 砍掉了，导致整张动作清单看起来都是"自发自评"。
        selected_actions = _minimal_actions_for_node(node_actions, max_actions=3)

        target_plan.append(
            {
                "数字人id": node.user_id,
                "节点角色": node.selected_role,
                "目标受众": target_objects,
            }
        )
        time_plan.append(
            {
                "数字人id": node.user_id,
                "时间阶段": node_output.get("时间阶段", _stage_text(node)),
            }
        )
        frequency_items.append(
            {
                "数字人id": node.user_id,
                "发帖频率": node_output.get("发帖频率", _frequency_text(node.frequency_per_day)),
                "频率上限": _frequency_text(frequency_plan.global_cap_per_day),
            }
        )
        platform_item: dict[str, Any] = {
            "数字人id": node.user_id,
            # 每个数字人只归属唯一模拟平台，优先输出其自身的平台归属。
            "分发平台": node.platform or platform,
        }
        if platform_item_fields:
            # 跨平台模式在这里补上"平台角色 / 安全等级 / 平台适配分"等平台专属依据；
            # 多人分发不传该参数，因此不会凭空多出字段。
            platform_item.update(platform_item_fields.get(node.user_id, {}))
        platform_items.append(platform_item)
        content_items.append(
            {
                "数字人id": node.user_id,
                "post_ref": post_ref,
                "发帖内容": node_output.get("发帖内容", ""),
            }
        )
        actions.extend(selected_actions)

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
    max_actions: int,
) -> list[dict[str, Any]]:
    preferred_groups = ("content_publish", "audience_interaction", "digital_human_coordination")
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for group in preferred_groups:
        for action in actions:
            action_id = str(action.get("动作编号", ""))
            if action_id in used_ids or action.get("动作分组") != group:
                continue
            selected.append(_minimal_action(action))
            used_ids.add(action_id)
            break
        if len(selected) >= max_actions:
            return selected

    for action in actions:
        action_id = str(action.get("动作编号", ""))
        if action_id in used_ids:
            continue
        selected.append(_minimal_action(action))
        used_ids.add(action_id)
        if len(selected) >= max_actions:
            break
    return selected


def _minimal_action(action: dict[str, Any]) -> dict[str, Any]:
    target = action.get("目标定位", {})
    result = {
        "动作编号": action.get("动作编号", ""),
        "执行主体": action.get("执行主体", ""),
        "动作类型": action.get("动作类型", ""),
        "执行平台": action.get("执行平台", ""),
        "目标帖子": target.get("post_ref", ""),
    }
    depends_on = action.get("依赖动作", [])
    if depends_on:
        result["依赖动作"] = depends_on
    return result

def _topic_labels(node: StrategyNodePlan, strategy_result: StrategyResult) -> list[str]:
    labels: list[str] = []
    for item in [
        *node.matched_keywords,
        *node.semantic_tags,
        *strategy_result.event.extracted_keywords[:3],
    ]:
        text = str(item).strip()
        if text and text not in labels:
            labels.append(text)
    return labels[:6]


def _role_bucket_ids(
    selected_nodes: list[StrategyNodePlan],
    current_user_id: str,
) -> dict[str, list[str]]:
    role_map = {
        "core_publish_node": [],
        "interaction_response_node": [],
        "amplification_node": [],
        "support_node": [],
    }
    for node in selected_nodes:
        if node.user_id == current_user_id:
            continue
        role_map.setdefault(node.selected_role, []).append(node.user_id)
    return role_map


def _build_compact_action_plan(
    *,
    node: StrategyNodePlan,
    node_index: int,
    platform: str,
    post_content: str,
    topic_labels: list[str],
    interaction_focus: str,
    coordination_focus: str,
    role_bucket_ids: dict[str, list[str]],
    post_refs: dict[str, str],
    publish_action_ids: dict[str, str],
) -> list[dict[str, Any]]:
    own_post_ref = post_refs.get(node.user_id, f"P{node_index:03d}")
    own_publish_action_id = publish_action_ids.get(node.user_id, f"A{((node_index - 1) * 4) + 1:03d}")
    core_ids = role_bucket_ids.get("core_publish_node", [])
    target_core_id = _first_other_id(core_ids, node.user_id) or (core_ids[0] if core_ids else node.user_id)
    target_core_ref = post_refs.get(target_core_id, own_post_ref)
    has_external_core = target_core_id != node.user_id
    target_core_publish_action_id = publish_action_ids.get(target_core_id) if has_external_core else None
    coordination_target_id = _coordination_target_id(node=node, role_bucket_ids=role_bucket_ids) or target_core_id
    coordination_target_ref = post_refs.get(coordination_target_id, own_post_ref)
    coordination_publish_action_id = publish_action_ids.get(coordination_target_id) if coordination_target_id != node.user_id else None
    coordination_seq = ((node_index - 1) * 4) + 4
    if (
        coordination_publish_action_id
        and _action_number(coordination_publish_action_id) >= coordination_seq
    ):
        coordination_target_id = node.user_id
        coordination_target_ref = own_post_ref
        coordination_publish_action_id = None
    coordination_self_monitor = coordination_target_id == node.user_id
    publish_depends_on = []
    if has_external_core and node.selected_role != "core_publish_node" and target_core_publish_action_id:
        publish_depends_on.append(target_core_publish_action_id)
    publish_action_type = _publish_action_type(node.selected_role) if has_external_core else "publish_post"
    publish_action_name = _publish_action_name(node.selected_role) if has_external_core else "发布主帖"
    publish_post_type = _post_type_for_role(node.selected_role) if has_external_core else "anchor_post"

    actions: list[dict[str, Any]] = [
        _compact_action(
            action_id=own_publish_action_id,
            seq=((node_index - 1) * 4) + 1,
            actor_id=node.user_id,
            action_group="content_publish",
            action_type=publish_action_type,
            action_name=publish_action_name,
            platform=platform,
            stage=_stage_text(node),
            trigger_condition="进入对应时间阶段后立即执行",
            target=_publish_target(node=node, own_post_ref=own_post_ref, target_core_id=target_core_id, target_core_ref=target_core_ref),
            depends_on=publish_depends_on,
            creates={"post_ref": own_post_ref, "post_type": publish_post_type},
            payload={
                "正文草案": post_content,
                "话题标签": topic_labels,
                "内容风格": node.suggested_content_style,
                "执行频次": _frequency_text(node.frequency_per_day),
            },
        ),
        _compact_action(
            action_id=f"A{((node_index - 1) * 4) + 2:03d}",
            seq=((node_index - 1) * 4) + 2,
            actor_id=node.user_id,
            action_group="comment_monitoring",
            action_type="monitor_comments",
            action_name="监测评论",
            platform=platform,
            stage=_stage_text(node),
            trigger_condition="帖子发布后持续监测评论区反馈",
            target=_post_target(own_post_ref, node.user_id, "自身帖子"),
            depends_on=[own_publish_action_id],
            creates={},
            payload={
                "优先关键词": topic_labels[:4],
                "关注类型": ["高频疑问", "事实补充", "明显误读"],
            },
        ),
        _compact_action(
            action_id=f"A{((node_index - 1) * 4) + 3:03d}",
            seq=((node_index - 1) * 4) + 3,
            actor_id=node.user_id,
            action_group="audience_interaction",
            action_type=_interaction_action_type(node.selected_role),
            action_name=_interaction_action_name(node.selected_role),
            platform=platform,
            stage=_stage_text(node),
            trigger_condition=_interaction_trigger(node.selected_role, own_publish_action_id, target_core_publish_action_id),
            target=_interaction_target(
                node=node,
                own_post_ref=own_post_ref,
                target_core_id=target_core_id,
                target_core_ref=target_core_ref,
            ),
            depends_on=_unique_action_ids([own_publish_action_id, target_core_publish_action_id]),
            creates={},
            payload=_interaction_payload(
                node=node,
                interaction_focus=interaction_focus,
                topic_labels=topic_labels,
                target_core_id=target_core_id,
                target_core_ref=target_core_ref,
            ),
        ),
        _compact_action(
            action_id=f"A{((node_index - 1) * 4) + 4:03d}",
            seq=coordination_seq,
            actor_id=node.user_id,
            action_group="digital_human_coordination",
            action_type="monitor_coordination" if coordination_self_monitor else _coordination_action_type(node.selected_role),
            action_name="监测协同反馈" if coordination_self_monitor else _coordination_action_name(node.selected_role),
            platform=platform,
            stage=_stage_text(node),
            trigger_condition=(
                "自身帖子发布后，等待其它数字人挂接评论、点赞或转发"
                if coordination_self_monitor
                else _coordination_trigger(node.selected_role, coordination_publish_action_id)
            ),
            target=_coordination_target(
                node=node,
                own_post_ref=own_post_ref,
                target_owner_id=coordination_target_id,
                target_post_ref=coordination_target_ref,
            ),
            depends_on=_unique_action_ids([own_publish_action_id, coordination_publish_action_id]),
            creates={},
            payload={
                "协同说明": coordination_focus,
                "协同对象数字人id": [
                    item for item in [coordination_target_id] if item and item != node.user_id
                ],
                "协同目标": (
                    "等待后续节点挂接到当前帖子"
                    if coordination_self_monitor
                    else _coordination_goal(node.selected_role)
                ),
            },
        ),
    ]
    return actions


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


def _first_other_id(candidates: list[str], current_user_id: str) -> str | None:
    for user_id in candidates:
        if user_id != current_user_id:
            return user_id
    return None


def _unique_action_ids(action_ids: list[str | None]) -> list[str]:
    result: list[str] = []
    for action_id in action_ids:
        if action_id and action_id not in result:
            result.append(action_id)
    return result


def _action_number(action_id: str) -> int:
    try:
        return int(str(action_id).lstrip("A"))
    except ValueError:
        return 0


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


def _publish_action_type(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "publish_post",
        "interaction_response_node": "publish_followup_post",
        "amplification_node": "quote_repost",
        "support_node": "publish_support_post",
    }
    return mapping.get(selected_role, "publish_post")


def _publish_action_name(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "发布主帖",
        "interaction_response_node": "发布跟进帖",
        "amplification_node": "转发并附评",
        "support_node": "发布补充帖",
    }
    return mapping.get(selected_role, "发布帖子")


def _post_type_for_role(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "core_post",
        "interaction_response_node": "followup_post",
        "amplification_node": "amplification_post",
        "support_node": "support_post",
    }
    return mapping.get(selected_role, "post")


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


def _interaction_action_type(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "reply_comment",
        "interaction_response_node": "reply_comment",
        "amplification_node": "like_post",
        "support_node": "reply_comment",
    }
    return mapping.get(selected_role, "reply_comment")


def _interaction_action_name(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "回复评论",
        "interaction_response_node": "回复评论",
        "amplification_node": "点赞帖子",
        "support_node": "回复评论",
    }
    return mapping.get(selected_role, "回复评论")


def _interaction_trigger(selected_role: str, own_publish_action_id: str, target_core_publish_action_id: str | None) -> str:
    if selected_role == "core_publish_node":
        return f"在 {own_publish_action_id} 发布后，出现高频提问或有效补充时执行"
    if target_core_publish_action_id:
        return f"在 {target_core_publish_action_id} 发布后进入互动窗口时执行"
    return f"在 {own_publish_action_id} 发布后进入互动窗口时执行"


def _interaction_target(
    *,
    node: StrategyNodePlan,
    own_post_ref: str,
    target_core_id: str,
    target_core_ref: str,
) -> dict[str, Any]:
    if node.selected_role == "core_publish_node" or target_core_id == node.user_id:
        return {
            "目标类型": "post_comments",
            "post_ref": own_post_ref,
            "目标数字人id": node.user_id,
            "目标说明": "自身主帖评论区",
            "评论筛选": ["高频问题", "理性追问", "事实补充"],
        }
    if node.selected_role == "amplification_node":
        return _post_target(target_core_ref, target_core_id, f"数字人 {target_core_id} 的主帖")
    return _post_target(target_core_ref, target_core_id, f"数字人 {target_core_id} 的主帖")


def _interaction_payload(
    *,
    node: StrategyNodePlan,
    interaction_focus: str,
    topic_labels: list[str],
    target_core_id: str,
    target_core_ref: str,
) -> dict[str, Any]:
    if node.selected_role == "core_publish_node" or target_core_id == node.user_id:
        return {
            "回复数量上限": 3,
            "回复要点": topic_labels[:3],
            "回复提示": interaction_focus,
        }
    if node.selected_role == "amplification_node":
        return {
            "点赞数量": 1,
            "点赞对象": target_core_ref,
            "点赞对象数字人id": target_core_id,
            "协同目的": "为核心信息建立显式连接",
        }
    return {
        "评论数量": 1,
        "评论对象": target_core_ref,
        "评论对象数字人id": target_core_id,
        "评论目的": interaction_focus,
    }


def _coordination_action_type(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "like_post",
        "interaction_response_node": "comment_post",
        "amplification_node": "like_post",
        "support_node": "comment_post",
    }
    return mapping.get(selected_role, "comment_post")


def _coordination_action_name(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "点赞数字人帖子",
        "interaction_response_node": "评论数字人帖子",
        "amplification_node": "点赞数字人帖子",
        "support_node": "评论数字人帖子",
    }
    return mapping.get(selected_role, "协同动作")


def _coordination_target_id(
    *,
    node: StrategyNodePlan,
    role_bucket_ids: dict[str, list[str]],
) -> str | None:
    if node.selected_role == "core_publish_node":
        for role in ("interaction_response_node", "amplification_node", "support_node"):
            target_id = _first_other_id(role_bucket_ids.get(role, []), node.user_id)
            if target_id:
                return target_id
        return None
    core_target = _first_other_id(role_bucket_ids.get("core_publish_node", []), node.user_id)
    if core_target:
        return core_target
    for role in ("interaction_response_node", "support_node", "amplification_node"):
        target_id = _first_other_id(role_bucket_ids.get(role, []), node.user_id)
        if target_id:
            return target_id
    return None


def _coordination_trigger(selected_role: str, target_core_publish_action_id: str | None) -> str:
    if selected_role == "core_publish_node":
        return "互动节点和扩散节点完成首轮动作后执行"
    if target_core_publish_action_id:
        return f"在 {target_core_publish_action_id} 执行后进入协同窗口时执行"
    return "在核心帖发布后进入协同窗口时执行"


def _coordination_target(
    *,
    node: StrategyNodePlan,
    own_post_ref: str,
    target_owner_id: str,
    target_post_ref: str,
) -> dict[str, Any]:
    if target_owner_id == node.user_id:
        return _post_target(own_post_ref, node.user_id, "自身帖子")
    if node.selected_role == "core_publish_node":
        return _post_target(target_post_ref, target_owner_id, f"数字人 {target_owner_id} 的协同帖")
    return _post_target(target_post_ref, target_owner_id, f"数字人 {target_owner_id} 的主帖")


def _coordination_goal(selected_role: str) -> str:
    mapping = {
        "core_publish_node": "保持首发节点和后续节点节奏一致",
        "interaction_response_node": "承接核心节点并把讨论拉回主帖",
        "amplification_node": "放大核心信息并保持统一口径",
        "support_node": "补充背景并避免与其他节点重复",
    }
    return mapping.get(selected_role, "维持协同")


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
) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace_root)
    client = llm_client
    if client is None and use_llm:
        client = OpenAICompatibleLLMClient.from_env_files(root)

    pipeline_payload = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=max_selected_nodes,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
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
    )
    output_payload, content_llm_meta = build_eval_output(
        hot_event=hot_event,
        strategy_result=artifacts.strategy_result,
        workspace_root=root,
        use_llm=use_llm,
        llm_client=client,
    )

    target_dir = Path(output_dir)
    output_path = target_dir / f"{artifacts.strategy_result.event.event_id}_strategy_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # run_label 同时透传选人状态：targeted / ok / manual_required / no_eligible_nodes
    if target_user_id:
        run_label_value = "targeted"
    else:
        run_label_value = artifacts.strategy_result.summary.status
    _write_pipeline_trace(
        trace_dir=trace_dir,
        hot_event=hot_event,
        pipeline_payload=pipeline_payload,
        artifacts=artifacts,
        final_output=output_payload,
        output_path=output_path,
        content_llm_meta=content_llm_meta,
        run_label=run_label_value,
    )
    return output_path, output_payload


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

    if not use_llm:
        metadata["fallback_reason"] = "llm_disabled"
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
        return {}, metadata

    client = llm_client
    if client is None and workspace_root is not None:
        client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
    if client is None:
        metadata["fallback_reason"] = "llm_not_configured"
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

    try:
        response = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (LLMClientError, OSError, ValueError, TypeError) as exc:
        metadata["fallback_reason"] = f"llm_request_failed:{type(exc).__name__}"
        return {}, metadata

    llm_nodes = _extract_llm_node_mapping(
        response=response,
        expected_ids=[node.user_id for node in eligible_nodes],
    )
    if not llm_nodes:
        metadata["fallback_reason"] = "invalid_llm_response"
        return {}, metadata

    overrides: dict[str, dict[str, str]] = {}
    for node in eligible_nodes:
        raw_node = llm_nodes.get(node.user_id)
        if not isinstance(raw_node, dict):
            continue
        cleaned = _clean_llm_node_fields(raw_node)
        if cleaned:
            overrides[node.user_id] = cleaned

    if overrides:
        metadata["used"] = True
    else:
        metadata["fallback_reason"] = "empty_llm_content"
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
    return f"{frequency_per_day}/day"


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
