from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def make_neighbor_stats() -> dict[str, int]:
    return {
        "received_comment_count": 0,
        "received_repost_count": 0,
        "made_comment_count": 0,
        "made_repost_count": 0,
    }


def make_node_stats() -> dict[str, int | bool]:
    return {
        "neighbor_count": 0,
        "engaged_by_neighbor_count": 0,
        "engaged_to_neighbor_count": 0,
        "mutual_neighbor_count": 0,
        "self_interaction_count": 0,
        "self_comment_count": 0,
        "self_repost_count": 0,
        "received_interaction_count": 0,
        "received_comment_count": 0,
        "received_repost_count": 0,
        "made_interaction_count": 0,
        "made_comment_count": 0,
        "made_repost_count": 0,
        "isolated": True,
    }


def relation_label(stats: dict[str, int]) -> str:
    received_total = stats["received_comment_count"] + stats["received_repost_count"]
    made_total = stats["made_comment_count"] + stats["made_repost_count"]
    if received_total and made_total:
        return "mutual"
    if received_total:
        return "engaged_by"
    if made_total:
        return "engaged_to"
    return "isolated"


def build_enriched_profiles(profile_path: Path, interaction_path: Path) -> dict[str, dict[str, Any]]:
    profiles = load_json(profile_path)
    interactions = load_json(interaction_path)

    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(make_neighbor_stats)
    )
    node_stats: defaultdict[str, dict[str, int | bool]] = defaultdict(make_node_stats)

    for source_id, records in interactions.items():
        source_id = str(source_id)
        for record in records:
            target_id = str(record.get("interact_id"))
            interact_type = record.get("interact_type")
            if not target_id or target_id == "None":
                continue

            is_self_loop = source_id == target_id
            source_to_target = relation_map[source_id][target_id]
            target_to_source = relation_map[target_id][source_id]

            node_stats[source_id]["isolated"] = False
            node_stats[target_id]["isolated"] = False
            node_stats[source_id]["received_interaction_count"] += 1
            node_stats[target_id]["made_interaction_count"] += 1
            if is_self_loop:
                node_stats[source_id]["self_interaction_count"] += 1

            if interact_type == "comment":
                source_to_target["received_comment_count"] += 1
                target_to_source["made_comment_count"] += 1
                node_stats[source_id]["received_comment_count"] += 1
                node_stats[target_id]["made_comment_count"] += 1
                if is_self_loop:
                    node_stats[source_id]["self_comment_count"] += 1
            elif interact_type == "reposts":
                source_to_target["received_repost_count"] += 1
                target_to_source["made_repost_count"] += 1
                node_stats[source_id]["received_repost_count"] += 1
                node_stats[target_id]["made_repost_count"] += 1
                if is_self_loop:
                    node_stats[source_id]["self_repost_count"] += 1

    enriched_profiles: dict[str, dict[str, Any]] = {}
    all_ids = set(profiles) | set(relation_map)

    for user_id in all_ids:
        profile = profiles.get(user_id)
        if profile is None:
            profile = {
                "user_id": int(user_id),
                "user_name": f"user_{user_id}",
                "user_followers": 0,
                "user_friends": 0,
                "user_interests": [],
                "user_description": "",
            }

        user_neighbors: list[dict[str, Any]] = []
        engaged_by_neighbor_count = 0
        engaged_to_neighbor_count = 0
        mutual_neighbor_count = 0

        for neighbor_id, stats in relation_map.get(user_id, {}).items():
            if neighbor_id == user_id:
                continue

            relation = relation_label(stats)
            if relation == "engaged_by":
                engaged_by_neighbor_count += 1
            elif relation == "engaged_to":
                engaged_to_neighbor_count += 1
            elif relation == "mutual":
                mutual_neighbor_count += 1

            total_received = stats["received_comment_count"] + stats["received_repost_count"]
            total_made = stats["made_comment_count"] + stats["made_repost_count"]
            user_neighbors.append(
                {
                    "neighbor_id": neighbor_id,
                    "relation": relation,
                    "received_comment_count": stats["received_comment_count"],
                    "received_repost_count": stats["received_repost_count"],
                    "made_comment_count": stats["made_comment_count"],
                    "made_repost_count": stats["made_repost_count"],
                    "received_interaction_count": total_received,
                    "made_interaction_count": total_made,
                    "total_interaction_count": total_received + total_made,
                }
            )

        user_neighbors.sort(
            key=lambda item: (
                item["total_interaction_count"],
                item["received_interaction_count"],
                item["made_interaction_count"],
                item["neighbor_id"],
            ),
            reverse=True,
        )

        stats = node_stats[user_id]
        stats["neighbor_count"] = len(user_neighbors)
        stats["engaged_by_neighbor_count"] = engaged_by_neighbor_count
        stats["engaged_to_neighbor_count"] = engaged_to_neighbor_count
        stats["mutual_neighbor_count"] = mutual_neighbor_count
        stats["isolated"] = len(user_neighbors) == 0

        enriched_profiles[user_id] = {
            **profile,
            "graph_attributes": stats,
            "neighbors": user_neighbors,
        }

    return enriched_profiles


def summarize(payload: dict[str, dict[str, Any]]) -> dict[str, int]:
    node_count = len(payload)
    isolated_count = 0
    neighbor_links = 0
    mutual_nodes = 0

    for record in payload.values():
        graph_attributes = record["graph_attributes"]
        neighbor_links += graph_attributes["neighbor_count"]
        isolated_count += int(bool(graph_attributes["isolated"]))
        mutual_nodes += int(graph_attributes["mutual_neighbor_count"] > 0)

    return {
        "node_count": node_count,
        "isolated_node_count": isolated_count,
        "connected_node_count": node_count - isolated_count,
        "neighbor_link_count": neighbor_links,
        "nodes_with_mutual_neighbors": mutual_nodes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build neighbor-enriched abc_reading profile data.")
    parser.add_argument(
        "--profile",
        default="data/raw/abc_reading_profile.graph.anon",
        help="Path to the profile file.",
    )
    parser.add_argument(
        "--interaction",
        default="data/raw/abc_reading_interaction.graph.anon",
        help="Path to the interaction file.",
    )
    parser.add_argument(
        "--output",
        default="data/derived/abc_reading_profile_with_neighbors.graph.anon",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation size. Use 0 for compact output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    profile_path = Path(args.profile)
    interaction_path = Path(args.interaction)
    output_path = Path(args.output)
    indent = None if args.indent <= 0 else args.indent

    payload = build_enriched_profiles(profile_path, interaction_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)

    print(f"saved: {output_path}")
    for key, value in summarize(payload).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
