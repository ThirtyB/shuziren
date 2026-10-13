from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


KEPT_TOP_LEVEL_FIELDS = (
    "user_id",
    "user_name",
    "user_followers",
    "user_friends",
    "user_interests",
    "user_description",
)

KEPT_GRAPH_ATTRIBUTE_FIELDS = (
    "neighbor_count",
    "mutual_neighbor_count",
    "self_interaction_count",
    "received_interaction_count",
    "received_comment_count",
    "received_repost_count",
    "made_interaction_count",
    "made_comment_count",
    "made_repost_count",
    "isolated",
)


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def compact_profile_record(user_id: str, raw_record: dict[str, Any]) -> dict[str, Any]:
    compact_record = {
        "user_id": str(raw_record.get("user_id") or user_id),
        "user_name": raw_record.get("user_name") or f"user_{user_id}",
        "user_followers": int(raw_record.get("user_followers") or 0),
        "user_friends": int(raw_record.get("user_friends") or 0),
        "user_interests": [
            str(item).strip()
            for item in raw_record.get("user_interests", [])
            if str(item).strip()
        ],
        "user_description": str(raw_record.get("user_description") or "").strip(),
    }

    raw_graph_attributes = raw_record.get("graph_attributes", {})
    compact_record["graph_attributes"] = {
        field: raw_graph_attributes.get(field, 0 if field != "isolated" else True)
        for field in KEPT_GRAPH_ATTRIBUTE_FIELDS
    }
    return compact_record


def build_compact_dataset(input_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    compact_payload: dict[str, dict[str, Any]] = {}
    for user_id, raw_record in input_payload.items():
        if not isinstance(raw_record, dict):
            continue
        compact_payload[str(user_id)] = compact_profile_record(str(user_id), raw_record)
    return compact_payload


def summarize_dataset(
    *,
    compact_payload: dict[str, dict[str, Any]],
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    node_count = len(compact_payload)
    description_lengths = [len(item["user_description"]) for item in compact_payload.values()]
    interest_counts = [len(item["user_interests"]) for item in compact_payload.values()]
    isolated_count = sum(
        1 for item in compact_payload.values()
        if bool(item["graph_attributes"].get("isolated", False))
    )
    nonempty_description_count = sum(1 for length in description_lengths if length > 0)
    total_interest_count = sum(interest_counts)

    input_size = input_path.stat().st_size if input_path.exists() else 0
    output_size = output_path.stat().st_size if output_path.exists() else 0
    reduction_ratio = 0.0
    if input_size > 0:
        reduction_ratio = round(1 - (output_size / input_size), 6)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "node_count": node_count,
        "isolated_node_count": isolated_count,
        "connected_node_count": node_count - isolated_count,
        "nonempty_description_ratio": round(nonempty_description_count / max(node_count, 1), 6),
        "avg_description_length": round(mean(description_lengths), 4) if description_lengths else 0.0,
        "avg_interest_count": round(mean(interest_counts), 4) if interest_counts else 0.0,
        "total_interest_count": total_interest_count,
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "size_reduction_ratio": reduction_ratio,
        "kept_top_level_fields": list(KEPT_TOP_LEVEL_FIELDS) + ["graph_attributes"],
        "kept_graph_attribute_fields": list(KEPT_GRAPH_ATTRIBUTE_FIELDS),
        "removed_large_fields": [
            "neighbors",
            "engaged_by_neighbor_count",
            "engaged_to_neighbor_count",
            "self_comment_count",
            "self_repost_count",
        ],
        "notes": [
            "all nodes are preserved",
            "original dataset is not modified",
            "compact dataset keeps only fields used by the current pipeline or likely useful soon",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact abc_reading profile dataset without modifying the original files."
    )
    parser.add_argument(
        "--input",
        default="data/derived/abc_reading_profile_with_neighbors.graph.anon",
        help="Path to the enriched profile dataset.",
    )
    parser.add_argument(
        "--output",
        default="data/derived/abc_reading_profile_compact.graph.anon",
        help="Path to the compact dataset output.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/derived/abc_reading_profile_compact_summary.json",
        help="Path to the summary JSON output.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation size. Use 0 for compact output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    indent = None if args.indent <= 0 else args.indent

    input_payload = load_json_object(input_path)
    compact_payload = build_compact_dataset(input_payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(compact_payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )

    summary_payload = summarize_dataset(
        compact_payload=compact_payload,
        input_path=input_path,
        output_path=output_path,
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved compact dataset: {output_path}")
    print(f"saved summary: {summary_output_path}")
    print(f"node_count: {summary_payload['node_count']}")
    print(f"size_reduction_ratio: {summary_payload['size_reduction_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
