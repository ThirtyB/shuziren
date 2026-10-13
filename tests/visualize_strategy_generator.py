from __future__ import annotations

import json
from pathlib import Path

from influence_strategy.pipeline import StrategyPipeline
from influence_strategy.strategy_generator import StrategyGenerator


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = workspace_root / "outputs" / "strategy"
    output_dir.mkdir(parents=True, exist_ok=True)

    event_input = {
        "event_title": "亲子阅读影响力传播测试",
        "event_description": "希望围绕亲子阅读与英语启蒙做一次传播活动，提升讨论度，并控制集中刷屏风险。",
        "target_goal": "engagement",
        "target_audience": ["parent_child", "english_learning"],
        "constraints": {
            "risk_level": "medium",
            "max_selected_nodes": 10,
            "max_frequency_per_day": 3,
            "campaign_window_hours": 24,
        },
    }

    result = StrategyPipeline().run(workspace_root=workspace_root, event_input=event_input)
    generator = StrategyGenerator()

    (output_dir / "strategy_preview.json").write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    generator.to_frame(result).to_csv(output_dir / "selected_node_plan.csv", index=False, encoding="utf-8-sig")
    generator.to_frame(result, bucket="fallback").to_csv(
        output_dir / "fallback_node_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    generator.stage_frame(result).to_csv(output_dir / "stage_plan.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
