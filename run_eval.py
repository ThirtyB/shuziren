from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from influence_strategy.eval_hot_events import (
    EVAL_MODE_MULTI_PLATFORM,
    EVAL_MODE_MULTI_USER,
    EVAL_MODE_SINGLE_TARGET,
    FIVE_DIMENSION_STRATEGY_KEY,
    RUN_SUMMARY_FILENAME,
    load_hot_events,
    resolve_eval_run_dirs,
    run_hot_event_dict_evaluations,
    run_hot_event_evaluations,
    run_hot_event_multiplatform_evaluations,
    write_eval_run_error,
    write_eval_run_summary,
)
from influence_strategy.image_event_loader import load_image_events
from influence_strategy.platform_dispatcher import (
    PlatformSpec,
    UnsupportedPlatformError,
    get_platform_profile,
)
from influence_strategy.platform_profiles import (
    DATASET_PLATFORM_VALUES,
    normalize_platform_key,
)


class PlatformArgumentError(ValueError):
    """命令行平台参数错误（--platforms / --platform-counts）。"""


def parse_platform_keys(raw: str | None) -> list[str]:
    """解析逗号分隔的平台 key，返回规范化（小写去空格）后的列表。"""

    if raw is None:
        return []
    keys = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not keys:
        raise PlatformArgumentError(
            "--platforms 不能为空，请提供逗号分隔的平台 key，例如 facebook,instagram,twitter。"
        )
    return keys


def parse_platform_counts(raw: str | None) -> dict[str, int]:
    """解析逗号分隔的 platform=count 配置。"""

    if raw is None:
        return {}
    items = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not items:
        raise PlatformArgumentError(
            "--platform-counts 不能为空，请提供平台人数配置，例如 facebook=3,twitter=5。"
        )

    counts: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise PlatformArgumentError(
                f"--platform-counts 配置 '{item}' 格式错误，应为 platform=count，例如 facebook=3。"
            )
        key_part, value_part = item.split("=", 1)
        platform_key = key_part.strip().lower()
        count_text = value_part.strip()
        if not platform_key:
            raise PlatformArgumentError(
                f"--platform-counts 配置 '{item}' 缺少平台名称，应为 platform=count。"
            )
        if platform_key in counts:
            raise PlatformArgumentError(
                f"--platform-counts 中平台 '{platform_key}' 重复配置人数，每个平台只能配置一次。"
            )
        if not count_text:
            raise PlatformArgumentError(
                f"--platform-counts 中平台 '{platform_key}' 缺少人数，应为 {platform_key}=<正整数>。"
            )
        try:
            count = int(count_text)
        except ValueError:
            raise PlatformArgumentError(
                f"--platform-counts 中平台 '{platform_key}' 的人数 '{count_text}' 不是整数，必须是正整数。"
            ) from None
        if count < 1:
            raise PlatformArgumentError(
                f"--platform-counts 中平台 '{platform_key}' 的人数 {count} 非法，必须是正整数（>=1）。"
            )
        counts[platform_key] = count
    return counts


def build_platform_specs(
    platforms_raw: str | None,
    counts_raw: str | None,
) -> list[PlatformSpec]:
    """把命令行参数转换成 ``PlatformSpec`` 列表，并做完整校验。

    ``platforms_raw`` 为 None 表示未启用多平台模式，此时返回空列表；
    若只提供了 ``--platform-counts``，视为参数错误而不是静默忽略。
    """

    if platforms_raw is None:
        if counts_raw is not None:
            raise PlatformArgumentError(
                "--platform-counts 必须与 --platforms 一起使用；"
                "--platforms 未指定时不会推断要运行哪些平台。"
            )
        return []

    platform_keys = parse_platform_keys(platforms_raw)
    counts = parse_platform_counts(counts_raw)

    ordered_keys: list[str] = []
    seen: set[str] = set()
    for key in platform_keys:
        try:
            profile = get_platform_profile(key)
        except UnsupportedPlatformError as exc:
            raise PlatformArgumentError(str(exc)) from None
        if profile.platform_key in seen:
            raise PlatformArgumentError(
                f"--platforms 中平台 '{profile.platform_key}' 重复出现，每个平台只能指定一次。"
            )
        seen.add(profile.platform_key)
        ordered_keys.append(profile.platform_key)

    if counts_raw is None:
        example = ",".join(f"{key}=3" for key in ordered_keys)
        raise PlatformArgumentError(
            "缺少 --platform-counts：多平台模式必须为每个平台配置正整数人数，"
            f"例如 --platform-counts {example}。"
        )

    extra_keys = sorted(key for key in counts if key not in seen)
    if extra_keys:
        raise PlatformArgumentError(
            "--platform-counts 中存在 --platforms 未声明的平台："
            + ", ".join(extra_keys)
            + f"。请只配置 {', '.join(ordered_keys)}。"
        )

    missing_keys = [key for key in ordered_keys if key not in counts]
    if missing_keys:
        raise PlatformArgumentError(
            "--platform-counts 缺少以下平台的人数配置："
            + ", ".join(missing_keys)
            + "。每个平台必须且只能有一个有效人数配置。"
        )

    return [
        PlatformSpec(platform_key=key, requested_count=counts[key])
        for key in ordered_keys
    ]


def _event_id_from_output_path(output_path: Path) -> str:
    """从输出文件名还原事件 ID。

    兼容三种文件名：
    ``{event_id}_strategy_output.json``（旧 automatic / targeted）、
    ``{event_id}_multiplatform_{platform_tag}_strategy_output.json``（多平台）。
    """

    name = output_path.name
    suffix = "_strategy_output.json"
    if not name.endswith(suffix):
        return output_path.stem
    stem = name[: -len(suffix)]
    marker = "_multiplatform"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem


#: 五维调度策略里的中文安全等级 → 运行摘要对外沿用的英文 tier 键。
_TIER_KEY_BY_LABEL: dict[str, str] = {
    "合格": "safe",
    "待人工复核": "review_required",
    "未通过资格筛选": "not_eligible",
}


def _tier_counts(value: object) -> dict:
    """把五维里的中文安全等级计数还原成英文 tier 键。

    业务 JSON 已统一使用中文标签，但运行摘要（``run_summary.json``）的对外字段名
    保持不变，避免下游脚本因一次结构对齐而全部失效。
    """

    if not isinstance(value, dict):
        return {english: 0 for english in _TIER_KEY_BY_LABEL.values()}
    return {
        english: int(value.get(chinese, 0) or 0)
        for chinese, english in _TIER_KEY_BY_LABEL.items()
    }


def _five_dimension_platform_dimension(payload: dict) -> dict | None:
    """取出五维调度策略的「平台」维度；只有跨平台模式才有「各平台方案」。"""

    strategy = payload.get(FIVE_DIMENSION_STRATEGY_KEY)
    if not isinstance(strategy, dict):
        return None
    platform_dimension = strategy.get("平台")
    if not isinstance(platform_dimension, dict):
        return None
    if not isinstance(platform_dimension.get("各平台方案"), list):
        return None
    return platform_dimension


def _multiplatform_summary_from_payload(payload: dict) -> dict | None:
    """提取跨平台分发摘要；非跨平台输出返回 None（不改变旧输出）。

    跨平台输出与多人分发共用五维调度策略结构，逐平台的选人结果集中在
    ``五维调度策略.平台.各平台方案``；这里只把运行摘要需要的计数字段窄化出来。
    """

    platform_dimension = _five_dimension_platform_dimension(payload)
    if platform_dimension is None:
        return None

    schemes = platform_dimension["各平台方案"]
    overview = platform_dimension.get("跨平台总览")
    overview = overview if isinstance(overview, dict) else {}

    return {
        "dispatch_mode": overview.get("调度模式"),
        "platforms": [
            {
                "platform_key": item.get("平台"),
                "requested_count": item.get("目标人数"),
                "selected_count": item.get("选取人数"),
                "shortfall_count": item.get("人数缺口"),
                # 人数够不够 ≠ 合格的人够不够，两个缺口都要暴露给操作者。
                "safe_shortfall_count": item.get("合格缺口"),
                "selected_safety_tiers": _tier_counts(item.get("安全等级分布")),
                "selected_user_ids": list(item.get("选取数字人id组") or []),
            }
            for item in schemes
            if isinstance(item, dict)
        ],
        "total_selected_safety_tiers": _tier_counts(overview.get("安全等级分布")),
        "total_safe_shortfall_count": overview.get("合格缺口合计"),
        "unique_user_count": overview.get("去重数字人数"),
        "overlapping_user_ids": overview.get("跨平台重复数字人"),
    }


def _selected_ids_from_payload(payload: dict) -> list[str]:
    """读取本次运行选取的数字人 id（保持首次出现顺序并去重）。

    三种模式现在共用五维调度策略结构，选人信息统一取自
    ``五维调度策略.目标对象.选取数字人id组``，不再需要按模式分支汇总。
    """

    strategy = payload.get(FIVE_DIMENSION_STRATEGY_KEY, {})
    target_object = strategy.get("目标对象", {}) if isinstance(strategy, dict) else {}
    selected_ids = target_object.get("选取数字人id组", [])
    if selected_ids:
        return selected_ids
    legacy_ids = payload.get("选取数字人id组", [])
    if legacy_ids:
        return legacy_ids
    return []


def _content_llm_status_from_payload(payload: dict) -> dict:
    """读取内容生成诊断：帖子正文是 LLM 生成还是规则兜底，以及兜底原因。

    三种模式共用五维调度策略结构，诊断统一取自
    ``五维调度策略.内容.内容生成诊断``。
    """

    strategy = payload.get(FIVE_DIMENSION_STRATEGY_KEY, {})
    content = strategy.get("内容", {}) if isinstance(strategy, dict) else {}
    diagnostic = content.get("内容生成诊断", {})
    if isinstance(diagnostic, dict) and diagnostic:
        return diagnostic
    return {}


def default_eval_input_path() -> Path:
    expanded_input = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
    if expanded_input.exists():
        return expanded_input
    return PROJECT_ROOT / "eval" / "hot_event_opinion_variants.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 hot_event 测试集并生成结构化 JSON 分发策略。",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT,
        help="项目工作目录，默认是当前项目根目录。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_eval_input_path(),
        help="hot_event 输入文件，可以是单个对象或事件数组；默认优先使用 200 条扩展事件集。",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Single image input; the image is recognized as a hot_event before evaluation.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Image directory input; reads .png/.jpg/.jpeg/.webp files by name order.",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        help="指定只运行某个 hot_event ID。",
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=10,
        help="未指定 --event-id 时，默认只运行前 N 个事件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "输出分类根目录。每次运行都会在其中创建 <模式>/run_NNN/ 独立目录"
            "（模式为 single_target / multi_user / multi_platform），"
            "本次运行的全部结果都写在该目录内。"
            f"默认使用 {PROJECT_ROOT / 'eval' / 'output'}。"
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help=(
            "流水线中间结果（trace）根目录。默认写入本次运行目录内部的 trace/ 子目录，"
            "保证一次运行的结果集中在同一个目录；显式指定时写入 "
            "<trace-dir>/<模式>/run_NNN/trace/，编号与结果目录保持一致。"
        ),
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        help="仅加载前 N 个 profile，便于快速调试。",
    )
    parser.add_argument(
        "--max-selected-nodes",
        type=int,
        default=5,
        help="主选数字人数量上限，默认 5。",
    )
    parser.add_argument(
        "--risk-level",
        choices=["low", "medium", "high"],
        help="覆盖事件风险等级；默认按 hot_event domain 推断。",
    )
    parser.add_argument(
        "--campaign-window-hours",
        type=int,
        default=24,
        help="传播窗口时长，单位小时，默认 24。",
    )
    parser.add_argument(
        "--max-frequency-per-day",
        type=int,
        default=3,
        help="单个数字人每天最多发帖次数，默认 3。",
    )
    parser.add_argument(
        "--allowed-platforms",
        type=str,
        help=(
            "允许的平台列表，逗号分隔，例如 "
            "facebook_simulated,instagram_simulated,twitter_simulated。"
        ),
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="关闭大模型增强，仅运行规则流水线。",
    )
    # 第十轮：事件解析阶段的 LLM 默认 off（与生产默认一致，走规则解析）。
    # 评测要复现"事件解析也用 LLM"的旧结果时显式给 on。选人阶段的 LLM
    # 不受此开关影响，仍由 --disable-llm 控制。
    parser.add_argument(
        "--event-parser-llm",
        choices=["on", "off"],
        default="off",
        help="事件解析阶段是否调用大模型，默认 off（走规则解析）。",
    )
    parser.add_argument(
        "--target-user-id",
        type=str,
        default=None,
        help="指定单一数字人进行定向分发（targeted 模式），未指定时走 automatic 模式。",
    )
    parser.add_argument(
        "--platforms",
        type=str,
        default=None,
        help=(
            "多平台并行分发模式：逗号分隔的平台 key，例如 facebook,instagram,twitter。"
            "未指定时完全保持原有运行逻辑；与 --target-user-id 互斥。"
        ),
    )
    parser.add_argument(
        "--platform-counts",
        type=str,
        default=None,
        help=(
            "多平台人数配置：逗号分隔的 platform=count，例如 facebook=3,twitter=5,instagram=2；"
            "每个平台必须且只能配置一次，必须与 --platforms 一起使用。"
        ),
    )
    parser.add_argument(
        "--dispatch-platform",
        type=str,
        default=None,
        help=(
            "多人分发指定平台：多人分发只从该模拟平台已有的数字人中选人，"
            f"可选 {', '.join(DATASET_PLATFORM_VALUES)}；未指定时默认 facebook_simulated。"
            "单人分发（--target-user-id）自动取目标所属平台，多平台模式（--platforms）"
            "按各平台自身归属选人，两者都不使用该参数。"
        ),
    )
    parser.add_argument(
        "--recall-size",
        type=int,
        default=200,
        help=(
            "分层召回层目标人数（35,636 → N）；默认 200。"
            "传 0 或负数则关闭召回，恢复全量 35,636。"
        ),
    )
    parser.add_argument(
        "--recall-mode",
        type=str,
        default="platform_aware",
        choices=["platform_aware", "off"],
        help=(
            "分层召回模式：默认 platform_aware。"
            "off 表示关闭召回，等价于 --recall-size 0。"
        ),
    )
    return parser


def resolve_eval_mode(
    args: argparse.Namespace,
    platform_specs: list[PlatformSpec] | None,
) -> str:
    """按实际执行流程判定本次运行属于三种模式中的哪一种。

    判定依据是命令行参数与实际调用分支，而不是输出 JSON 里的旧字段：

    * 实际启用 ``--platforms`` 并执行多平台调度 → ``multi_platform``；
    * 使用 ``--target-user-id`` 指定目标数字人 → ``single_target``；
    * 其余情况（自动选人、多人单平台分发、图片输入）→ ``multi_user``。
    """

    if platform_specs is not None:
        return EVAL_MODE_MULTI_PLATFORM
    if args.target_user_id:
        return EVAL_MODE_SINGLE_TARGET
    return EVAL_MODE_MULTI_USER


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    allowed_platforms = None
    if args.allowed_platforms:
        allowed_platforms = [
            item.strip()
            for item in args.allowed_platforms.split(",")
            if item.strip()
        ]

    platform_mode_requested = args.platforms is not None or args.platform_counts is not None

    if platform_mode_requested and args.target_user_id:
        parser.error(
            "--platforms/--platform-counts 与 --target-user-id 不能同时使用："
            "多平台并行分发模式与单人定向模式互斥，请只保留其中一个参数。"
        )

    # 多人分发平台参数：显式提供时校验取值，并与单人 / 多平台模式互斥。
    dispatch_platform: str | None = None
    if args.dispatch_platform is not None:
        platform_key = normalize_platform_key(args.dispatch_platform)
        if platform_key not in {
            normalize_platform_key(value) for value in DATASET_PLATFORM_VALUES
        }:
            parser.error(
                f"--dispatch-platform '{args.dispatch_platform}' 不是有效平台，"
                f"可选：{', '.join(DATASET_PLATFORM_VALUES)}。"
            )
        if platform_mode_requested:
            parser.error(
                "--dispatch-platform 与 --platforms/--platform-counts 不能同时使用："
                "多平台模式按各平台自身归属选人，不需要指定单一分发平台。"
            )
        if args.target_user_id:
            parser.error(
                "--dispatch-platform 与 --target-user-id 不能同时使用："
                "单人分发会自动使用目标数字人所属平台。"
            )
        dispatch_platform = platform_key

    platform_specs: list[PlatformSpec] | None = None
    if args.platforms is not None:
        try:
            platform_specs = build_platform_specs(args.platforms, args.platform_counts)
        except ValueError as exc:
            parser.error(str(exc))
    elif platform_mode_requested:
        parser.error(
            "--platform-counts 必须与 --platforms 一起使用："
            "--platforms 未指定时不会推断要运行哪些平台。"
        )

    if platform_specs is not None and (
        args.image is not None or args.image_dir is not None
    ):
        parser.error(
            "--platforms/--platform-counts 目前不支持与 --image/--image-dir 同时使用。"
        )

    workspace_root = args.workspace_root.resolve()
    mode = resolve_eval_mode(args, platform_specs)
    # 多人分发必须落在唯一模拟平台上：未显式指定时使用默认平台 facebook_simulated。
    effective_dispatch_platform: str | None = None
    if mode == EVAL_MODE_MULTI_USER:
        effective_dispatch_platform = dispatch_platform or normalize_platform_key(
            DATASET_PLATFORM_VALUES[0]
        )
    # 召回层：off / size≤0 → 关闭；其它 → 按 (size, mode) 透传到入口。
    # 单人 target_user_id 强制唯一节点，召回无意义，下游 _run_hot_event_evaluation
    # 内部会再判一次把它关掉。
    effective_recall_mode: Literal["platform_aware"] | None = (
        None if args.recall_mode == "off" else "platform_aware"
    )
    effective_recall_size: int = (
        0 if args.recall_mode == "off" or args.recall_size <= 0 else args.recall_size
    )
    # 参数校验全部通过后才创建本次运行目录，避免非法参数留下空目录。
    run_dirs = resolve_eval_run_dirs(
        workspace_root=workspace_root,
        mode=mode,
        output_root=args.output_dir.resolve() if args.output_dir is not None else None,
        trace_root=args.trace_dir.resolve() if args.trace_dir is not None else None,
    )

    try:
        if platform_specs is not None:
            results = run_hot_event_multiplatform_evaluations(
                workspace_root=workspace_root,
                input_path=args.input.resolve(),
                platform_specs=platform_specs,
                output_dir=run_dirs.output_dir,
                event_id=args.event_id,
                event_limit=args.event_limit,
                profile_limit=args.profile_limit,
                max_selected_nodes=args.max_selected_nodes,
                risk_level=args.risk_level,
                campaign_window_hours=args.campaign_window_hours,
                max_frequency_per_day=args.max_frequency_per_day,
                allowed_platforms=allowed_platforms,
                use_llm=not args.disable_llm,
                # 第十轮：评测入口显式传值，默认与生产一致（关闭事件解析 LLM）。
                use_llm_event_parser=args.event_parser_llm == "on",
                trace_dir=run_dirs.trace_dir,
                recall_size=effective_recall_size,
                recall_mode=effective_recall_mode,
            )
        elif args.image is not None or args.image_dir is not None:
            reference_events = load_hot_events(args.input.resolve())
            image_events = load_image_events(
                image=args.image.resolve() if args.image is not None else None,
                image_dir=args.image_dir.resolve() if args.image_dir is not None else None,
                reference_events=reference_events,
                workspace_root=workspace_root,
                event_limit=args.event_limit,
                event_id=args.event_id,
                use_llm=not args.disable_llm,
            )
            results = run_hot_event_dict_evaluations(
                workspace_root=workspace_root,
                events=image_events,
                output_dir=run_dirs.output_dir,
                event_id=None,
                event_limit=None,
                profile_limit=args.profile_limit,
                max_selected_nodes=args.max_selected_nodes,
                risk_level=args.risk_level,
                campaign_window_hours=args.campaign_window_hours,
                max_frequency_per_day=args.max_frequency_per_day,
                allowed_platforms=allowed_platforms,
                use_llm=not args.disable_llm,
                use_llm_event_parser=args.event_parser_llm == "on",
                trace_dir=run_dirs.trace_dir,
                target_user_id=args.target_user_id,
                dispatch_platform=effective_dispatch_platform,
                recall_size=effective_recall_size,
                recall_mode=effective_recall_mode,
            )
        else:
            results = run_hot_event_evaluations(
                workspace_root=workspace_root,
                input_path=args.input.resolve(),
                output_dir=run_dirs.output_dir,
                event_id=args.event_id,
                event_limit=args.event_limit,
                profile_limit=args.profile_limit,
                max_selected_nodes=args.max_selected_nodes,
                risk_level=args.risk_level,
                campaign_window_hours=args.campaign_window_hours,
                max_frequency_per_day=args.max_frequency_per_day,
                allowed_platforms=allowed_platforms,
                use_llm=not args.disable_llm,
                use_llm_event_parser=args.event_parser_llm == "on",
                trace_dir=run_dirs.trace_dir,
                target_user_id=args.target_user_id,
                dispatch_platform=effective_dispatch_platform,
                recall_size=effective_recall_size,
                recall_mode=effective_recall_mode,
            )
    except Exception as exc:
        # 失败信息必须留在本次运行目录内，而不是散落到旧路径或只打印到终端。
        write_eval_run_error(
            run_dirs.run_dir,
            {
                "generated_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "run_id": run_dirs.run_id,
                "mode": run_dirs.mode,
                "failed_stage": "evaluation",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "output_dir": str(run_dirs.output_dir),
                "trace_dir": str(run_dirs.trace_dir),
                "workspace_root": str(workspace_root),
                "argv": list(sys.argv),
            },
        )
        raise

    console_outputs: list[dict] = []
    for output_path, payload in results:
        event_key = _event_id_from_output_path(output_path)
        output_item = {
            "event_id": event_key,
            "event_name": payload.get("事件名称", ""),
            "selected_digital_human_ids": _selected_ids_from_payload(payload),
            "content_llm": _content_llm_status_from_payload(payload),
            "json_output_path": str(output_path),
            "trace_output_dir": str(run_dirs.trace_dir / event_key),
        }
        multiplatform_summary = _multiplatform_summary_from_payload(payload)
        if multiplatform_summary is not None:
            output_item["multi_platform"] = multiplatform_summary
        console_outputs.append(output_item)

    console_summary = {
        "run": {
            "mode": run_dirs.mode,
            "run_id": run_dirs.run_id,
            "run_dir": str(run_dirs.run_dir),
            "output_dir": str(run_dirs.output_dir),
            "trace_dir": str(run_dirs.trace_dir),
            "summary_path": str(run_dirs.run_dir / RUN_SUMMARY_FILENAME),
        },
        "event_count": len(results),
        "outputs": console_outputs,
    }
    # 摘要同时落盘到本次运行目录，避免结果只存在于 stdout。
    write_eval_run_summary(run_dirs.run_dir, console_summary)
    print(json.dumps(console_summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
