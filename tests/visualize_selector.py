from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from influence_strategy.data_loader import DataLoader
from influence_strategy.event_parser import RuleBasedEventParser
from influence_strategy.feature_builder import FeatureBuilder
from influence_strategy.scorer import Scorer
from influence_strategy.selector import Selector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize selected nodes for abc_reading.")
    parser.add_argument("--workspace-root", default=".", help="Workspace root path.")
    parser.add_argument(
        "--event",
        default="希望围绕亲子阅读和英语启蒙做一次传播活动，重点提升讨论度，并控制刷屏争议。",
        help="Natural language event description.",
    )
    parser.add_argument("--output-dir", default="outputs/selector", help="Directory used to store preview files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = Path(args.workspace_root)
    output_dir = workspace_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(workspace_root)
    event = RuleBasedEventParser().parse(args.event)
    feature_result = FeatureBuilder().build_features(
        product_context=loader.load_product_context(),
        profiles=loader.load_profiles(),
        event=event,
        enriched_profiles=loader.load_enriched_profiles(),
        source_user_ids=set(loader.load_interactions(limit_records_per_source=0).keys()),
    )
    score_result = Scorer().score(feature_result)
    selection_result = Selector().select(score_result)

    selected_frame = Selector().to_frame(selection_result)
    fallback_frame = Selector().to_frame(selection_result, bucket="fallback")

    selected_csv = output_dir / "selected_nodes.csv"
    fallback_csv = output_dir / "fallback_nodes.csv"
    summary_json = output_dir / "selection_summary.json"
    score_png = output_dir / "selected_node_final_score.png"
    role_png = output_dir / "selected_role_distribution.png"

    selected_frame.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    fallback_frame.to_csv(fallback_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(
            {
                "summary": selection_result.summary.model_dump(),
                "event": selection_result.event.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_scores(selected_frame, score_png)
    _plot_roles(selected_frame, role_png)

    print(f"saved: {selected_csv}")
    print(f"saved: {fallback_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {score_png}")
    print(f"saved: {role_png}")
    print(selected_frame[[
        "selection_rank",
        "user_id",
        "selected_role",
        "dispatch_stage",
        "final_score",
        "risk_score",
        "selection_reasons",
    ]].to_string(index=False))
    return 0


def _plot_scores(frame, output_path: Path) -> None:
    if frame.empty:
        return
    labels = [f"{row.selection_rank}:{row.user_id}" for row in frame.itertuples()]
    scores = frame["final_score"].tolist()

    plt.figure(figsize=(10, 5))
    plt.barh(labels[::-1], scores[::-1], color="#E15759")
    plt.xlabel("final_score")
    plt.ylabel("selected node")
    plt.title("Selected Node Final Scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_roles(frame, output_path: Path) -> None:
    if frame.empty:
        return
    role_counts = frame["selected_role"].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(role_counts.index.tolist(), role_counts.values.tolist(), color="#76B7B2")
    plt.ylabel("count")
    plt.title("Selected Role Distribution")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
