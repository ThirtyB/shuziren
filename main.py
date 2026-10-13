from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from influence_strategy.pipeline import StrategyPipeline
from influence_strategy.platform_profiles import (
    DATASET_PLATFORM_VALUES,
    is_valid_dataset_platform,
)
from influence_strategy.reporting import (
    render_strategy_markdown,
    write_markdown,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="输入一个影响力事件，输出结构化传播策略。",
    )

    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT,
        help="项目工作目录，默认是当前项目根目录。",
    )

    parser.add_argument(
        "--event-text",
        type=str,
        help="直接输入事件描述文本。",
    )

    parser.add_argument(
        "--event-file",
        type=Path,
        help="输入 JSON 文件路径，文件内容可以是事件对象或纯文本字符串。",
    )

    parser.add_argument(
        "--risk-level",
        choices=["low", "medium", "high"],
        help="覆盖事件风险等级。",
    )

    parser.add_argument(
        "--max-selected-nodes",
        type=int,
        help="覆盖最多选择的主节点数量。",
    )

    parser.add_argument(
        "--max-frequency-per-day",
        type=int,
        help="覆盖单节点每天最大触发次数。",
    )

    parser.add_argument(
        "--campaign-window-hours",
        type=int,
        help="覆盖传播窗口时长，单位小时。",
    )

    parser.add_argument(
        "--allowed-platforms",
        type=str,
        help=(
            "允许的平台列表，使用逗号分隔，"
            "例如 facebook_simulated,instagram_simulated,twitter_simulated。"
        ),
    )

    parser.add_argument(
        "--dispatch-platform",
        type=str,
        help=(
            "多人分发指定平台：能且只能从该平台已有的数字人中选人，"
            "可选值 facebook_simulated / instagram_simulated / twitter_simulated。"
        ),
    )

    parser.add_argument(
        "--profile-limit",
        type=int,
        help="仅加载前 N 个 profile，便于快速调试。",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "输出 JSON 文件路径。"
            "默认保存到 outputs/strategy/strategy_<event_id>.json"
        ),
    )

    # 新增：是否启用大语言模型
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="启用大语言模型增强能力。",
    )

    return parser


def load_event_input(
    args: argparse.Namespace,
) -> str | dict[str, Any]:
    if args.event_text and args.event_file:
        raise SystemExit(
            "`--event-text` 和 `--event-file` 只能二选一。"
        )

    if args.event_file is not None:
        with args.event_file.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:
            payload = json.load(handle)

    elif args.event_text:
        payload = args.event_text.strip()

    else:
        payload = input("请输入事件描述：").strip()

    if not payload:
        raise SystemExit("事件输入不能为空。")

    return apply_overrides(payload, args)


def apply_overrides(
    payload: str | dict[str, Any],
    args: argparse.Namespace,
) -> str | dict[str, Any]:
    has_override = any(
        value is not None
        for value in (
            args.risk_level,
            args.max_selected_nodes,
            args.max_frequency_per_day,
            args.campaign_window_hours,
            args.allowed_platforms,
        )
    )

    if not has_override:
        return payload

    if isinstance(payload, dict):
        updated_payload = dict(payload)
    else:
        updated_payload = {
            "event_description": str(payload)
        }

    constraints = dict(
        updated_payload.get("constraints", {})
    )

    if args.risk_level is not None:
        constraints["risk_level"] = args.risk_level

    if args.max_selected_nodes is not None:
        constraints["max_selected_nodes"] = (
            args.max_selected_nodes
        )

    if args.max_frequency_per_day is not None:
        constraints["max_frequency_per_day"] = (
            args.max_frequency_per_day
        )

    if args.campaign_window_hours is not None:
        constraints["campaign_window_hours"] = (
            args.campaign_window_hours
        )

    if args.allowed_platforms is not None:
        constraints["allowed_platforms"] = [
            item.strip()
            for item in args.allowed_platforms.split(",")
            if item.strip()
        ]

    updated_payload["constraints"] = constraints

    return updated_payload


def default_output_path(
    workspace_root: Path,
    event_id: str,
) -> Path:
    return (
        workspace_root
        / "outputs"
        / "strategy"
        / f"strategy_{event_id}.json"
    )


def default_markdown_path(
    json_path: Path,
) -> Path:
    return json_path.with_suffix(".md")


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def render_json_for_console(
    payload: dict[str, Any],
) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    workspace_root = (
        args.workspace_root.resolve()
    )

    if args.dispatch_platform is not None and not is_valid_dataset_platform(
        args.dispatch_platform
    ):
        raise SystemExit(
            "--dispatch-platform 只支持："
            + " / ".join(DATASET_PLATFORM_VALUES)
            + "。"
        )

    event_input = load_event_input(args)

    pipeline = StrategyPipeline(
        product_name="abc_reading"
    )

    strategy_result = pipeline.run(
        workspace_root=workspace_root,
        event_input=event_input,
        profile_limit=args.profile_limit,
        dispatch_platform=args.dispatch_platform,

        # 新增：将命令行参数传给流水线
        use_llm=args.use_llm,
    )

    output_path = (
        args.output.resolve()
        if args.output
        else default_output_path(
            workspace_root,
            strategy_result.event.event_id,
        )
    )

    markdown_path = default_markdown_path(
        output_path
    )

    payload = strategy_result.model_dump(
        mode="json"
    )

    write_json(
        output_path,
        payload,
    )

    write_markdown(
        markdown_path,
        render_strategy_markdown(
            strategy_result
        ),
    )

    console_summary = {
        "event_id": strategy_result.event.event_id,
        "event_title": strategy_result.event.event_title,
        "selected_count": (
            strategy_result.summary.selected_count
        ),
        "fallback_count": (
            strategy_result.summary.fallback_count
        ),
        "primary_platform": (
            strategy_result.summary.primary_platform
        ),
        "dispatch_platform": (
            strategy_result.summary.dispatch_platform
        ),
        "selected_nodes": [
            {
                "user_id": node.user_id,
                "platform": node.platform or "unknown",
                "role": node.selected_role,
            }
            for node in strategy_result.selected_nodes
        ],
        "json_output_path": str(output_path),
        "markdown_output_path": str(markdown_path),
    }

    print(
        render_json_for_console(
            console_summary
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())