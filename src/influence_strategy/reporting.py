from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import StrategyResult


def render_strategy_markdown(strategy_result: StrategyResult) -> str:
    event = strategy_result.event
    summary = strategy_result.summary
    strategy = strategy_result.strategy

    lines: list[str] = [
        f"# 影响力事件传播策略报告：{event.event_title}",
        "",
        "## 事件概览",
        f"- `event_id`: `{event.event_id}`",
        f"- `event_type`: `{event.event_type}`",
        f"- `target_goal`: `{event.target_goal}`",
        f"- `risk_level`: `{event.constraints.risk_level}`",
        f"- `campaign_window_hours`: `{event.constraints.campaign_window_hours}`",
        f"- `allowed_platforms`: `{', '.join(event.constraints.allowed_platforms)}`",
        "",
        "## 总体摘要",
        f"- 主选节点数：`{summary.selected_count}`",
        f"- 备选节点数：`{summary.fallback_count}`",
        f"- 主平台：`{summary.primary_platform}`",
        f"- 分发平台（多人分发指定）：`{summary.dispatch_platform or '不限'}`",
        f"- 目标平台（单人分发）：`{summary.target_platform or '不适用'}`",
        f"- 预计总触发次数：`{summary.estimated_total_dispatches}`",
        f"- 主选节点平均分：`{summary.avg_selected_final_score:.3f}`",
        f"- 人工复核需求：`{summary.review_required}`",
        "",
        "## 五个维度",
        f"- 分发对象：{_join_or_dash(strategy.target_object)}",
        f"- 时间安排：{_compact_mapping(strategy.time_plan)}",
        f"- 频率安排：{_compact_mapping(strategy.frequency_plan.model_dump())}",
        f"- 平台安排：{_compact_mapping(strategy.platform_plan.model_dump())}",
        f"- 内容安排：{_compact_mapping(strategy.content_plan.model_dump())}",
        "",
        "## 阶段计划",
        _markdown_table(
            headers=["阶段", "时间窗口", "目标", "节点数", "角色"],
            rows=[
                [
                    stage.stage_label,
                    stage.time_window,
                    stage.objective,
                    str(stage.node_count),
                    _join_or_dash(stage.selected_roles),
                ]
                for stage in strategy_result.stage_plans
            ],
        ),
        "",
        "## 主选节点",
        _markdown_table(
            headers=["排名", "user_id", "角色", "平台", "阶段", "频次", "final_score", "risk_level"],
            rows=[
                [
                    str(node.selection_rank),
                    node.user_id,
                    node.selected_role,
                    node.platform or "-",
                    node.dispatch_stage,
                    f"{node.frequency_per_day}条/天",
                    f"{node.final_score:.3f}",
                    node.risk_level,
                ]
                for node in strategy_result.selected_nodes
            ],
        ),
        "",
        "## 备选节点",
        _markdown_table(
            headers=["排名", "user_id", "角色", "平台", "阶段", "final_score", "risk_level"],
            rows=[
                [
                    str(node.selection_rank),
                    node.user_id,
                    node.selected_role,
                    node.platform or "-",
                    node.dispatch_stage,
                    f"{node.final_score:.3f}",
                    node.risk_level,
                ]
                for node in strategy_result.fallback_nodes
            ],
        ),
        "",
        "## 风险控制",
        f"- 风险等级：`{strategy.risk_control.risk_level}`",
        f"- 人工复核：`{strategy.risk_control.review_required}`",
        f"- 复核节点：{_join_or_dash(strategy.risk_control.manual_review_node_ids)}",
        f"- 触发条件：{_join_or_dash(strategy.risk_control.fallback_trigger)}",
        f"- 备注：{_join_or_dash(strategy.risk_control.notes)}",
        "",
        "## 可解释性说明",
    ]

    for item in strategy.explainability:
        lines.append(f"- {item}")

    return "\n".join(lines).strip() + "\n"


def render_eval_markdown(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {})
    event = payload.get("event", {})
    summary = payload.get("summary", {})
    dimensions = payload.get("five_dimensions", {})
    selected_nodes = payload.get("selected_digital_humans", [])
    fallback_nodes = payload.get("fallback_digital_humans", [])
    risk_control = payload.get("risk_control", {})
    explainability = payload.get("explainability", [])

    lines: list[str] = [
        f"# 热点事件评测报告：{event.get('event_title', '-')}",
        "",
        "## 生成元信息",
        f"- schema_version: `{meta.get('schema_version', '-')}`",
        f"- generated_at: `{meta.get('generated_at', '-')}`",
        f"- generator_mode: `{meta.get('generator_mode', '-')}`",
        f"- llm_used: `{meta.get('llm', {}).get('used', False)}`",
        f"- llm_provider: `{meta.get('llm', {}).get('provider', '-')}`",
        f"- llm_model: `{meta.get('llm', {}).get('model', '-')}`",
        "",
        "## 事件概览",
        f"- `event_id`: `{event.get('event_id', '-')}`",
        f"- `domain`: `{event.get('domain', '-')}`",
        f"- `target`: {event.get('target', '-')}",
        f"- `risk_level`: `{summary.get('risk_level', '-')}`",
        f"- `primary_platform`: `{summary.get('primary_platform', '-')}`",
        "",
        "## 五个维度",
        f"- 分发对象：{_join_or_dash(dimensions.get('target_object', []))}",
        f"- 时间安排：{_compact_mapping(dimensions.get('time_plan', {}))}",
        f"- 频率安排：{_compact_mapping(dimensions.get('frequency_plan', {}))}",
        f"- 平台安排：{_compact_mapping(dimensions.get('platform_plan', {}))}",
        f"- 内容安排：{_compact_mapping(dimensions.get('content_plan', {}))}",
        "",
        "## 主选数字人",
        _markdown_table(
            headers=["排名", "user_id", "角色", "阶段", "频次", "平台", "分数", "风险"],
            rows=[
                [
                    str(node.get("selection_rank", "-")),
                    str(node.get("user_id", "-")),
                    str(node.get("selected_role", "-")),
                    str(node.get("stage_text", "-")),
                    str(node.get("frequency_text", "-")),
                    str(node.get("platform", "-")),
                    f"{float(node.get('final_score', 0.0)):.3f}",
                    str(node.get("risk_level", "-")),
                ]
                for node in selected_nodes
            ],
        ),
        "",
        "## 备选数字人",
        _markdown_table(
            headers=["排名", "user_id", "角色", "阶段", "分数", "风险"],
            rows=[
                [
                    str(node.get("selection_rank", "-")),
                    str(node.get("user_id", "-")),
                    str(node.get("selected_role", "-")),
                    str(node.get("stage_text", "-")),
                    f"{float(node.get('final_score', 0.0)):.3f}",
                    str(node.get("risk_level", "-")),
                ]
                for node in fallback_nodes
            ],
        ),
        "",
        "## 风险控制",
        f"- review_required: `{risk_control.get('review_required', False)}`",
        f"- manual_review_node_ids: { _join_or_dash(risk_control.get('manual_review_node_ids', [])) }",
        f"- notes: { _join_or_dash(risk_control.get('notes', [])) }",
        "",
        "## 可解释性说明",
    ]

    for item in explainability:
        lines.append(f"- {item}")

    lines.extend(["", "## 主选数字人执行细节"])
    for node in selected_nodes:
        content_output = node.get("content_output", {})
        lines.extend(
            [
                "",
                f"### `{node.get('user_id', '-')}` · {node.get('selected_role', '-')}",
                f"- 时间阶段：{node.get('stage_text', '-')}",
                f"- 发帖频率：{node.get('frequency_text', '-')}",
                f"- 发帖平台：{node.get('platform', '-')}",
                f"- 发帖内容：{content_output.get('post_content', '-')}",
                f"- 目标群体画像：{content_output.get('audience_profile', '-')}",
                f"- 目标群体互动策略：{content_output.get('audience_interaction_strategy', '-')}",
                f"- 与其他数字人互动策略：{content_output.get('cross_digital_human_strategy', '-')}",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def render_eval_batch_markdown(results: list[tuple[Path, dict[str, Any]]]) -> str:
    lines: list[str] = [
        "# 热点事件评测批量结果",
        "",
        _markdown_table(
            headers=["event_id", "event_title", "generator_mode", "selected_count", "json_output"],
            rows=[
                [
                    str(payload.get("summary", {}).get("event_id", "-")),
                    str(payload.get("summary", {}).get("event_title", "-")),
                    str(payload.get("meta", {}).get("generator_mode", "-")),
                    str(payload.get("summary", {}).get("selected_count", "-")),
                    str(output_path.name),
                ]
                for output_path, payload in results
            ],
        ),
    ]
    return "\n".join(lines).strip() + "\n"


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_无数据_"
    header_line = "| " + " | ".join(_escape_cell(item) for item in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = []
    for row in rows:
        row_lines.append("| " + " | ".join(_escape_cell(item) for item in row) + " |")
    return "\n".join([header_line, separator_line, *row_lines])


def _escape_cell(value: Any) -> str:
    text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def _join_or_dash(values: list[Any]) -> str:
    if not values:
        return "-"
    return "；".join(str(item) for item in values)


def _compact_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "-"
    parts: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, list):
            rendered = _join_or_dash(value)
        elif isinstance(value, dict):
            rendered = ", ".join(f"{sub_key}={sub_value}" for sub_key, sub_value in value.items())
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return "；".join(parts)
