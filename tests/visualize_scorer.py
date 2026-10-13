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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize first-pass scorer output for abc_reading.")
    parser.add_argument("--workspace-root", default=".", help="Workspace root path.")
    parser.add_argument(
        "--event",
        default="希望围绕亲子阅读和英语启蒙做一次传播活动，重点提升讨论度，并控制刷屏争议。",
        help="Natural language event description.",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Number of top nodes to show.")
    parser.add_argument("--output-dir", default="outputs/scorer", help="Directory used to store preview files.")
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
    frame = Scorer().to_frame(score_result)
    preview = frame.head(args.top_k).copy()

    csv_path = output_dir / "top_node_scores.csv"
    json_path = output_dir / "score_summary.json"
    bar_path = output_dir / "top_node_final_score.png"
    scatter_path = output_dir / "score_risk_scatter.png"

    preview.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "summary": score_result.summary.model_dump(),
                "event": score_result.event.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_bar(preview, bar_path)
    _plot_scatter(frame.head(500), scatter_path)

    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {bar_path}")
    print(f"saved: {scatter_path}")
    print(preview[[
        "user_id",
        "role_hint",
        "priority_tier",
        "eligible",
        "final_score",
        "risk_score",
        "topic_match_score",
        "selection_reasons",
    ]].to_string(index=False))
    return 0


def _plot_bar(frame, output_path: Path) -> None:
    labels = [f"{row.user_id}:{row.priority_tier}" for row in frame.itertuples()]
    scores = frame["final_score"].tolist()

    plt.figure(figsize=(12, 6))
    plt.barh(labels[::-1], scores[::-1], color="#59A14F")
    plt.xlabel("final_score")
    plt.ylabel("node")
    plt.title("Top Node Final Scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_scatter(frame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(
        frame["final_score"],
        frame["risk_score"],
        s=(frame["influence_score"] * 180) + 10,
        c=frame["topic_match_score"],
        cmap="plasma",
        alpha=0.75,
    )
    plt.xlabel("final_score")
    plt.ylabel("risk_score")
    plt.title("Score vs Risk Preview")
    plt.colorbar(label="topic_match_score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
