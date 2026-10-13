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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize first-pass node features for abc_reading.")
    parser.add_argument("--workspace-root", default=".", help="Workspace root path.")
    parser.add_argument(
        "--event",
        default="希望围绕亲子阅读和英语启蒙做一次传播活动，重点提升讨论度，并控制刷屏争议。",
        help="Natural language event description.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top nodes to show in the ranking chart.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/feature_builder",
        help="Directory used to store preview files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = Path(args.workspace_root)
    output_dir = workspace_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(workspace_root)
    event = RuleBasedEventParser().parse(args.event)
    builder = FeatureBuilder()
    result = builder.build_features(
        product_context=loader.load_product_context(),
        profiles=loader.load_profiles(),
        event=event,
        enriched_profiles=loader.load_enriched_profiles(),
        source_user_ids=set(loader.load_interactions(limit_records_per_source=0).keys()),
    )
    frame = builder.to_frame(result)
    preview = frame.head(args.top_k).copy()

    preview_csv = output_dir / "top_node_features.csv"
    preview_json = output_dir / "feature_summary.json"
    ranking_png = output_dir / "top_node_feature_ready_score.png"
    scatter_png = output_dir / "feature_scatter.png"

    preview.to_csv(preview_csv, index=False, encoding="utf-8-sig")
    preview_json.write_text(
        json.dumps(
            {
                "summary": result.summary.model_dump(),
                "event": result.event.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_ranking(preview, ranking_png)
    _plot_scatter(frame, scatter_png)

    print(f"saved: {preview_csv}")
    print(f"saved: {preview_json}")
    print(f"saved: {ranking_png}")
    print(f"saved: {scatter_png}")
    print(preview[[
        "user_id",
        "role_hint",
        "feature_ready_score",
        "topic_match_score",
        "influence_score",
        "diffusion_score",
        "matched_keywords",
    ]].to_string(index=False))
    return 0


def _plot_ranking(frame, output_path: Path) -> None:
    labels = [f"{row.user_id}:{row.role_hint}" for row in frame.itertuples()]
    scores = frame["feature_ready_score"].tolist()

    plt.figure(figsize=(12, 6))
    plt.barh(labels[::-1], scores[::-1], color="#4C78A8")
    plt.xlabel("feature_ready_score")
    plt.ylabel("node")
    plt.title("Top Node Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_scatter(frame, output_path: Path) -> None:
    sample = frame.head(500)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        sample["influence_score"],
        sample["topic_match_score"],
        s=(sample["feature_ready_score"] * 180) + 10,
        c=sample["stability_score"],
        cmap="viridis",
        alpha=0.75,
    )
    plt.xlabel("influence_score")
    plt.ylabel("topic_match_score")
    plt.title("Node Feature Scatter Preview")
    plt.colorbar(label="stability_score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
