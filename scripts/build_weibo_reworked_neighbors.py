
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NEIGHBOR_STAT_FIELDS = (
    "received_comment_count",
    "received_repost_count",
    "made_comment_count",
    "made_repost_count",
)

GRAPH_ATTRIBUTE_FIELDS = (
    "neighbor_count",
    "engaged_by_neighbor_count",
    "engaged_to_neighbor_count",
    "mutual_neighbor_count",
    "self_interaction_count",
    "self_comment_count",
    "self_repost_count",
    "received_interaction_count",
    "received_comment_count",
    "received_repost_count",
    "made_interaction_count",
    "made_comment_count",
    "made_repost_count",
    "isolated",
)

GENERIC_PUBLIC_TERMS = [
    "\u65f6\u653f",      # politics/current affairs
    "\u653f\u7b56",      # policy
    "\u793e\u4f1a",      # society
    "\u516c\u5171",      # public
    "\u8d22\u7ecf",      # finance
    "\u56fd\u9645",      # international
    "\u6cd5\u5f8b",      # law
    "\u6559\u80b2",      # education
    "\u79d1\u6280",      # technology
    "\u6c11\u751f",      # livelihood
    "\u5a92\u4f53",      # media
    "\u57ce\u5e02",      # city
]


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                items.append(item)
    return items


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_neighbor_stats() -> dict[str, int]:
    return {field: 0 for field in NEIGHBOR_STAT_FIELDS}


def stable_hash_int(text: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


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


def dedupe_texts(items: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalize_profile_text(record: dict[str, Any]) -> str:
    interests = " ".join(str(item).strip() for item in record.get("user_interests", []) if str(item).strip())
    description = str(record.get("user_description") or "").strip()
    return f"{interests} {description}".lower()


def add_neighbor_stats(
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    user_id: str,
    neighbor_id: str,
    stats: dict[str, Any],
) -> None:
    if user_id == neighbor_id:
        return
    target = relation_map[user_id][neighbor_id]
    for field in NEIGHBOR_STAT_FIELDS:
        target[field] += safe_int(stats.get(field, 0))


def add_interaction(
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    post_owner_id: str,
    actor_id: str,
    interact_type: str,
    count: int,
) -> None:
    if post_owner_id == actor_id or count <= 0:
        return
    owner_stats = relation_map[post_owner_id][actor_id]
    actor_stats = relation_map[actor_id][post_owner_id]
    if interact_type == "comment":
        owner_stats["received_comment_count"] += count
        actor_stats["made_comment_count"] += count
    elif interact_type == "reposts":
        owner_stats["received_repost_count"] += count
        actor_stats["made_repost_count"] += count


def preserve_original_relations(
    *,
    original_enriched: dict[str, Any],
    kept_ids: set[str],
    rewritten_ids: set[str],
) -> tuple[defaultdict[str, defaultdict[str, dict[str, int]]], int]:
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(make_neighbor_stats)
    )
    preserved_entries = 0
    for user_id, record in original_enriched.items():
        user_id = str(user_id)
        if user_id not in kept_ids or user_id in rewritten_ids:
            continue
        if not isinstance(record, dict):
            continue
        for neighbor in record.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            neighbor_id = str(neighbor.get("neighbor_id") or "").strip()
            if not neighbor_id or neighbor_id not in kept_ids or neighbor_id in rewritten_ids:
                continue
            add_neighbor_stats(relation_map, user_id, neighbor_id, neighbor)
            preserved_entries += 1
    return relation_map, preserved_entries


def build_category_terms(
    *,
    records: dict[str, dict[str, Any]],
    rewritten_manifest: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    terms_by_category: dict[str, list[str]] = defaultdict(list)
    for user_id, item in rewritten_manifest.items():
        category = str(item.get("assigned_category") or "general")
        record = records.get(user_id) or {}
        terms_by_category[category].extend(record.get("user_interests", []))
        rewritten = item.get("rewritten") if isinstance(item.get("rewritten"), dict) else {}
        terms_by_category[category].extend(rewritten.get("user_interests", []))
    return {
        category: dedupe_texts([*terms, *GENERIC_PUBLIC_TERMS])
        for category, terms in terms_by_category.items()
    }


def score_original_candidate(
    *,
    category_terms: list[str],
    candidate_id: str,
    record: dict[str, Any],
) -> float:
    text = normalize_profile_text(record)
    exact_hits = sum(1 for term in category_terms if term and term.lower() in text)
    graph = record.get("graph_attributes") or {}
    follower_score = math.log1p(max(0, safe_int(record.get("user_followers"))))
    activity_score = math.log1p(
        max(0, safe_int(graph.get("received_interaction_count")))
        + max(0, safe_int(graph.get("made_interaction_count")))
    )
    description_bonus = 0.4 if str(record.get("user_description") or "").strip() else 0.0
    interest_bonus = min(len(record.get("user_interests", [])), 5) * 0.12
    jitter = (stable_hash_int(candidate_id, 17) % 1000) / 10000
    return exact_hits * 5.0 + follower_score * 0.20 + activity_score * 0.25 + description_bonus + interest_bonus + jitter


def build_category_rankings(
    *,
    records: dict[str, dict[str, Any]],
    candidate_ids: list[str],
    category_terms: dict[str, list[str]],
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for category, terms in category_terms.items():
        scored = [
            (
                score_original_candidate(
                    category_terms=terms,
                    candidate_id=user_id,
                    record=records[user_id],
                ),
                user_id,
            )
            for user_id in candidate_ids
        ]
        scored.sort(reverse=True)
        rankings[category] = [user_id for _, user_id in scored]
    return rankings


def pick_original_neighbors(
    *,
    rewritten_id: str,
    category: str,
    rankings: dict[str, list[str]],
    count: int,
    seed: int,
) -> list[str]:
    if count <= 0:
        return []
    ranking = rankings.get(category) or []
    if not ranking:
        return []
    rng = random.Random(stable_hash_int(f"original:{rewritten_id}:{category}", seed))
    pool_size = min(len(ranking), max(600, count * 50))
    pool = list(ranking[:pool_size])
    rng.shuffle(pool)
    selected = pool[:count]
    if len(selected) < count:
        for user_id in ranking:
            if user_id not in selected:
                selected.append(user_id)
                if len(selected) >= count:
                    break
    return selected[:count]


def pick_rewritten_peer_neighbors(
    *,
    rewritten_id: str,
    category: str,
    category_by_rewritten_id: dict[str, str],
    count: int,
    seed: int,
) -> list[str]:
    if count <= 0:
        return []
    same_category = [
        user_id
        for user_id, candidate_category in category_by_rewritten_id.items()
        if user_id != rewritten_id and candidate_category == category
    ]
    other_category = [
        user_id
        for user_id, candidate_category in category_by_rewritten_id.items()
        if user_id != rewritten_id and candidate_category != category
    ]
    same_category.sort(key=lambda item: stable_hash_int(f"peer-same:{rewritten_id}:{item}", seed))
    other_category.sort(key=lambda item: stable_hash_int(f"peer-other:{rewritten_id}:{item}", seed))
    return [*same_category, *other_category][:count]


def generated_counts_for_pair(
    *,
    rewritten_record: dict[str, Any],
    target_record: dict[str, Any],
    seed_text: str,
    seed: int,
    peer_link: bool,
) -> dict[str, int]:
    rng = random.Random(stable_hash_int(seed_text, seed))
    rewritten_followers = safe_int(rewritten_record.get("user_followers"))
    target_followers = safe_int(target_record.get("user_followers"))
    scale = 1
    if rewritten_followers >= 300_000 or target_followers >= 300_000:
        scale = 3
    elif rewritten_followers >= 50_000 or target_followers >= 50_000:
        scale = 2
    if peer_link:
        scale += 1
    incoming_comment = rng.randint(1, 2 + scale)
    incoming_repost = rng.randint(0, scale)
    outgoing_comment = rng.randint(0, 1 + scale)
    outgoing_repost = rng.randint(0, max(1, scale - 1))
    if outgoing_comment + outgoing_repost == 0:
        outgoing_comment = 1
    return {
        "incoming_comment": incoming_comment,
        "incoming_repost": incoming_repost,
        "outgoing_comment": outgoing_comment,
        "outgoing_repost": outgoing_repost,
    }


def add_generated_relationship(
    *,
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    records: dict[str, dict[str, Any]],
    rewritten_id: str,
    target_id: str,
    seed: int,
    peer_link: bool,
) -> None:
    if rewritten_id == target_id:
        return
    counts = generated_counts_for_pair(
        rewritten_record=records[rewritten_id],
        target_record=records[target_id],
        seed_text=f"{rewritten_id}:{target_id}:{peer_link}",
        seed=seed,
        peer_link=peer_link,
    )
    add_interaction(relation_map, rewritten_id, target_id, "comment", counts["incoming_comment"])
    add_interaction(relation_map, rewritten_id, target_id, "reposts", counts["incoming_repost"])
    add_interaction(relation_map, target_id, rewritten_id, "comment", counts["outgoing_comment"])
    add_interaction(relation_map, target_id, rewritten_id, "reposts", counts["outgoing_repost"])


def desired_neighbor_count(record: dict[str, Any], default: int = 12) -> int:
    graph = record.get("graph_attributes") or {}
    value = safe_int(graph.get("neighbor_count"), default)
    followers = safe_int(record.get("user_followers"))
    if followers >= 300_000:
        return max(70, min(value, 220))
    if followers >= 50_000:
        return max(25, min(value, 120))
    if followers >= 5_000:
        return max(10, min(value, 60))
    return max(6, min(value, 24))


def inject_rewritten_relations(
    *,
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    records: dict[str, dict[str, Any]],
    rewritten_manifest: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rewritten_ids = set(rewritten_manifest)
    original_candidate_ids = [user_id for user_id in records if user_id not in rewritten_ids]
    category_by_rewritten_id = {
        user_id: str(item.get("assigned_category") or "general")
        for user_id, item in rewritten_manifest.items()
    }
    category_terms = build_category_terms(records=records, rewritten_manifest=rewritten_manifest)
    rankings = build_category_rankings(
        records=records,
        candidate_ids=original_candidate_ids,
        category_terms=category_terms,
    )

    manifest_items: list[dict[str, Any]] = []
    for rewritten_id in sorted(rewritten_ids, key=lambda item: stable_hash_int(item, seed)):
        if rewritten_id not in records:
            continue
        category = category_by_rewritten_id.get(rewritten_id, "general")
        desired = desired_neighbor_count(records[rewritten_id])
        peer_count = min(12, max(1, desired // 7)) if desired >= 8 else 0
        original_count = max(0, desired - peer_count)
        original_neighbors = pick_original_neighbors(
            rewritten_id=rewritten_id,
            category=category,
            rankings=rankings,
            count=original_count,
            seed=seed,
        )
        peer_neighbors = pick_rewritten_peer_neighbors(
            rewritten_id=rewritten_id,
            category=category,
            category_by_rewritten_id=category_by_rewritten_id,
            count=peer_count,
            seed=seed,
        )
        for target_id in original_neighbors:
            add_generated_relationship(
                relation_map=relation_map,
                records=records,
                rewritten_id=rewritten_id,
                target_id=target_id,
                seed=seed,
                peer_link=False,
            )
        for target_id in peer_neighbors:
            add_generated_relationship(
                relation_map=relation_map,
                records=records,
                rewritten_id=rewritten_id,
                target_id=target_id,
                seed=seed,
                peer_link=True,
            )
        manifest_items.append(
            {
                "user_id": rewritten_id,
                "assigned_category": category,
                "target_neighbor_count": desired,
                "generated_original_neighbor_count": len(original_neighbors),
                "generated_rewritten_peer_count": len(peer_neighbors),
                "generated_original_neighbor_ids": original_neighbors,
                "generated_rewritten_peer_ids": peer_neighbors,
            }
        )
    return manifest_items


def build_profile_with_recomputed_graph(
    *,
    user_id: str,
    record: dict[str, Any],
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
) -> dict[str, Any]:
    neighbors: list[dict[str, Any]] = []
    engaged_by_neighbor_count = 0
    engaged_to_neighbor_count = 0
    mutual_neighbor_count = 0
    totals = {field: 0 for field in GRAPH_ATTRIBUTE_FIELDS if field != "isolated"}

    for neighbor_id, stats in relation_map.get(user_id, {}).items():
        if neighbor_id == user_id:
            continue
        received_interaction_count = stats["received_comment_count"] + stats["received_repost_count"]
        made_interaction_count = stats["made_comment_count"] + stats["made_repost_count"]
        total_interaction_count = received_interaction_count + made_interaction_count
        if total_interaction_count <= 0:
            continue
        relation = relation_label(stats)
        if relation == "engaged_by":
            engaged_by_neighbor_count += 1
        elif relation == "engaged_to":
            engaged_to_neighbor_count += 1
        elif relation == "mutual":
            mutual_neighbor_count += 1
        totals["received_comment_count"] += stats["received_comment_count"]
        totals["received_repost_count"] += stats["received_repost_count"]
        totals["made_comment_count"] += stats["made_comment_count"]
        totals["made_repost_count"] += stats["made_repost_count"]
        neighbors.append(
            {
                "neighbor_id": neighbor_id,
                "relation": relation,
                "received_comment_count": stats["received_comment_count"],
                "received_repost_count": stats["received_repost_count"],
                "made_comment_count": stats["made_comment_count"],
                "made_repost_count": stats["made_repost_count"],
                "received_interaction_count": received_interaction_count,
                "made_interaction_count": made_interaction_count,
                "total_interaction_count": total_interaction_count,
            }
        )

    neighbors.sort(
        key=lambda item: (
            item["total_interaction_count"],
            item["received_interaction_count"],
            item["made_interaction_count"],
            item["neighbor_id"],
        ),
        reverse=True,
    )
    graph_attributes = {
        "neighbor_count": len(neighbors),
        "engaged_by_neighbor_count": engaged_by_neighbor_count,
        "engaged_to_neighbor_count": engaged_to_neighbor_count,
        "mutual_neighbor_count": mutual_neighbor_count,
        "self_interaction_count": 0,
        "self_comment_count": 0,
        "self_repost_count": 0,
        "received_interaction_count": totals["received_comment_count"] + totals["received_repost_count"],
        "received_comment_count": totals["received_comment_count"],
        "received_repost_count": totals["received_repost_count"],
        "made_interaction_count": totals["made_comment_count"] + totals["made_repost_count"],
        "made_comment_count": totals["made_comment_count"],
        "made_repost_count": totals["made_repost_count"],
        "isolated": len(neighbors) == 0,
    }
    return {
        "user_id": str(record.get("user_id") or user_id),
        "user_name": str(record.get("user_name") or f"user_{user_id}"),
        "user_followers": safe_int(record.get("user_followers")),
        "user_friends": safe_int(record.get("user_friends")),
        "user_interests": [str(item).strip() for item in record.get("user_interests", []) if str(item).strip()],
        "user_description": str(record.get("user_description") or "").strip(),
        "graph_attributes": graph_attributes,
        "neighbors": neighbors,
    }


def build_summary(
    *,
    output_payload: dict[str, dict[str, Any]],
    rewritten_ids: set[str],
    preserved_neighbor_entries: int,
    generated_manifest: list[dict[str, Any]],
    input_reworked_path: Path,
    input_original_enriched_path: Path,
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    neighbor_counts = [item["graph_attributes"]["neighbor_count"] for item in output_payload.values()]
    rewritten_neighbor_counts = [
        output_payload[user_id]["graph_attributes"]["neighbor_count"]
        for user_id in rewritten_ids
        if user_id in output_payload
    ]
    relation_counter = Counter()
    for item in output_payload.values():
        for neighbor in item.get("neighbors", []):
            relation_counter[neighbor.get("relation", "unknown")] += 1
    generated_relationship_count = sum(
        item["generated_original_neighbor_count"] + item["generated_rewritten_peer_count"]
        for item in generated_manifest
    )
    return {
        "input_reworked_path": str(input_reworked_path),
        "input_original_enriched_path": str(input_original_enriched_path),
        "output_path": str(output_path),
        "node_count": len(output_payload),
        "rewritten_node_count": len(rewritten_ids),
        "isolated_node_count": sum(1 for item in output_payload.values() if item["graph_attributes"]["isolated"]),
        "neighbor_entry_count": sum(neighbor_counts),
        "avg_neighbor_count": round(sum(neighbor_counts) / max(len(neighbor_counts), 1), 4),
        "avg_rewritten_neighbor_count": round(sum(rewritten_neighbor_counts) / max(len(rewritten_neighbor_counts), 1), 4),
        "preserved_original_neighbor_entries": preserved_neighbor_entries,
        "generated_undirected_relationship_count": generated_relationship_count,
        "generated_neighbor_entries": generated_relationship_count * 2,
        "relation_distribution": dict(relation_counter),
        "seed": seed,
        "notes": [
            "original enriched graph is not modified",
            "relations between retained non-rewritten original nodes are preserved",
            "old relations touching rewritten nodes are replaced",
            "rewritten nodes are connected to semantically related original nodes and a small number of compatible rewritten peers",
            "graph_attributes are recomputed from explicit neighbors",
        ],
    }


def build_reworked_neighbor_graph(
    *,
    reworked_profiles: dict[str, Any],
    original_enriched: dict[str, Any],
    rewrite_manifest_items: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = {str(user_id): dict(record) for user_id, record in reworked_profiles.items() if isinstance(record, dict)}
    kept_ids = set(records)
    rewrite_manifest = {
        str(item.get("user_id")): item
        for item in rewrite_manifest_items
        if item.get("user_id") is not None and str(item.get("user_id")) in kept_ids
    }
    rewritten_ids = set(rewrite_manifest)
    relation_map, preserved_entries = preserve_original_relations(
        original_enriched=original_enriched,
        kept_ids=kept_ids,
        rewritten_ids=rewritten_ids,
    )
    generated_manifest = inject_rewritten_relations(
        relation_map=relation_map,
        records=records,
        rewritten_manifest=rewrite_manifest,
        seed=seed,
    )
    output_payload = {
        user_id: build_profile_with_recomputed_graph(
            user_id=user_id,
            record=records[user_id],
            relation_map=relation_map,
        )
        for user_id in records
    }
    helper_summary = {
        "preserved_entries": preserved_entries,
        "rewritten_ids": sorted(rewritten_ids),
    }
    return output_payload, generated_manifest, helper_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build explicit neighbors for the reworked Weibo hot-event simulation dataset."
    )
    parser.add_argument(
        "--reworked-profile",
        default="data/derived/weibo_profile_reworked.graph.anon",
        help="Input reworked profile dataset without explicit neighbors.",
    )
    parser.add_argument(
        "--original-enriched-profile",
        default="data/derived/abc_reading_profile_with_neighbors.graph.anon",
        help="Original neighbor-enriched profile dataset used as the background graph.",
    )
    parser.add_argument(
        "--rewrite-manifest",
        default="data/derived/weibo_profile_rewrite_manifest.jsonl",
        help="Manifest produced by build_weibo_reworked_dataset.py.",
    )
    parser.add_argument(
        "--output",
        default="data/derived/weibo_profile_reworked_with_neighbors.graph.anon",
        help="Output reworked dataset with explicit neighbors.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/derived/weibo_profile_reworked_graph_summary.json",
        help="Output graph reconstruction summary JSON.",
    )
    parser.add_argument(
        "--graph-manifest-output",
        default="data/derived/weibo_graph_rewrite_manifest.jsonl",
        help="Output graph rewrite manifest JSONL.",
    )
    parser.add_argument("--seed", type=int, default=20260624, help="Deterministic seed.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation. Use 0 for compact output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reworked_path = Path(args.reworked_profile)
    original_enriched_path = Path(args.original_enriched_profile)
    rewrite_manifest_path = Path(args.rewrite_manifest)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)
    graph_manifest_output_path = Path(args.graph_manifest_output)
    indent = None if args.indent <= 0 else args.indent

    reworked_profiles = load_json_object(reworked_path)
    original_enriched = load_json_object(original_enriched_path)
    rewrite_manifest_items = load_jsonl(rewrite_manifest_path)

    output_payload, generated_manifest, helper_summary = build_reworked_neighbor_graph(
        reworked_profiles=reworked_profiles,
        original_enriched=original_enriched,
        rewrite_manifest_items=rewrite_manifest_items,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )

    graph_manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_manifest_output_path.open("w", encoding="utf-8") as handle:
        for item in generated_manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = build_summary(
        output_payload=output_payload,
        rewritten_ids=set(helper_summary["rewritten_ids"]),
        preserved_neighbor_entries=helper_summary["preserved_entries"],
        generated_manifest=generated_manifest,
        input_reworked_path=reworked_path,
        input_original_enriched_path=original_enriched_path,
        output_path=output_path,
        seed=args.seed,
    )
    summary.update(
        {
            "graph_manifest_output_path": str(graph_manifest_output_path),
            "output_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        }
    )
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
