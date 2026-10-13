"""Build a human-readable slice of the current working dataset.

Why this script exists
----------------------
The active dataset
(``data/derived/weibo_profile_reworked_with_neighbors_with_platform.graph.anon``)
is a single-line JSON object that is tens of MB. Even ``cat``/``grep`` cannot
produce a useful read. This script rewrites the same profiles as **JSON Lines
(JSONL)** with the bulky ``neighbors`` array stripped out, so:

* every user becomes one line that an editor, ``grep`` or ``head`` can scan;
* aggregate statistics (platform counts, follower / interaction segments,
  top interests) are dumped to a small summary JSON;
* a tiny ``_preview.json`` of the first N users is pretty-printed for direct
  eyeballing — exactly the experience a small "旧版" sample used to give.

The script never mutates the source dataset; it is purely a **read-only**
projection that writes new files next to it. Idempotent: re-running overwrites
the same output set with identical bytes.

Usage::

    python scripts/build_readable_dataset.py
    python scripts/build_readable_dataset.py --input <other.graph.anon> \
        --jsonl data/derived/_scratch.jsonl --preview-count 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# 保留的兴趣标签上限；超过的会在 JSONL 里被截断，原始字段不受影响。
_INTEREST_KEEP = 8

# 影响力段位边界（粉丝数 → 段位名）。
_FOLLOWER_TIERS: tuple[tuple[int, str], ...] = (
    (1, "dormant(<1)"),
    (100, "low(1-99)"),
    (1_000, "mid(100-999)"),
    (10_000, "high(1k-9.9k)"),
    (10**9, "head(10k+)"),
)

# 互动度段位边界（received + made）。
_INTERACTION_TIERS: tuple[tuple[int, str], ...] = (
    (0, "silent(0)"),
    (5, "light(1-4)"),
    (20, "active(5-19)"),
    (100, "busy(20-99)"),
    (10**9, "viral(100+)"),
)


def _load_profiles(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object keyed by user_id.")
    return payload


def _tier(value: int, bands: tuple[tuple[int, str], ...]) -> str:
    for upper, label in bands:
        if value < upper:
            return label
    return bands[-1][1]


def _project_user(record: dict[str, Any]) -> dict[str, Any]:
    """Strip the bulky neighbors array but keep every human-useful field."""

    interests = list(record.get("user_interests") or [])
    interests = [str(item) for item in interests if str(item).strip()]

    graph = dict(record.get("graph_attributes") or {})

    return {
        "user_id": str(record.get("user_id", "")),
        "user_name": str(record.get("user_name", "") or ""),
        "platform": str(record.get("platform", "") or ""),
        "user_followers": int(record.get("user_followers", 0) or 0),
        "user_friends": int(record.get("user_friends", 0) or 0),
        "user_interests": interests[:_INTEREST_KEEP],
        "user_interests_truncated": len(interests) > _INTEREST_KEEP,
        "user_description": str(record.get("user_description", "") or ""),
        "neighbor_count": int(graph.get("neighbor_count", 0) or 0),
        "mutual_neighbor_count": int(graph.get("mutual_neighbor_count", 0) or 0),
        "engaged_by_neighbor_count": int(graph.get("engaged_by_neighbor_count", 0) or 0),
        "engaged_to_neighbor_count": int(graph.get("engaged_to_neighbor_count", 0) or 0),
        "received_interaction_count": int(graph.get("received_interaction_count", 0) or 0),
        "received_comment_count": int(graph.get("received_comment_count", 0) or 0),
        "received_repost_count": int(graph.get("received_repost_count", 0) or 0),
        "made_interaction_count": int(graph.get("made_interaction_count", 0) or 0),
        "made_comment_count": int(graph.get("made_comment_count", 0) or 0),
        "made_repost_count": int(graph.get("made_repost_count", 0) or 0),
        "isolated": bool(graph.get("isolated", False)),
        "engagement_total": (
            int(graph.get("received_interaction_count", 0) or 0)
            + int(graph.get("made_interaction_count", 0) or 0)
        ),
        "follower_tier": _tier(int(record.get("user_followers", 0) or 0), _FOLLOWER_TIERS),
    }


def _empty_user_index() -> dict[str, int]:
    return {
        "social": 0,
        "interests_empty": 0,
        "isolated": 0,
        "has_interactions": 0,
        "no_neighbor": 0,
    }


def _build_summary(
    profiles: dict[str, Any],
    jsonl_records: list[dict[str, Any]],
) -> dict[str, Any]:
    platform_counts: Counter[str] = Counter()
    follower_tier_counts: Counter[str] = Counter()
    interaction_tier_counts: Counter[str] = Counter()
    interest_freq: Counter[str] = Counter()
    user_idx = _empty_user_index()

    for proj in jsonl_records:
        platform_counts[proj["platform"] or "<unknown>"] += 1
        follower_tier_counts[proj["follower_tier"]] += 1
        interaction_tier_counts[_tier(proj["engagement_total"], _INTERACTION_TIERS)] += 1

        if proj["user_interests"]:
            user_idx["social"] += 1
        else:
            user_idx["interests_empty"] += 1
        if proj["isolated"]:
            user_idx["isolated"] += 1
        if proj["engagement_total"] > 0:
            user_idx["has_interactions"] += 1
        if proj["neighbor_count"] == 0:
            user_idx["no_neighbor"] += 1

        for tag in proj["user_interests"]:
            interest_freq[tag] += 1

    top_interests = interest_freq.most_common(50)

    # 前 20 个高粉丝用户：仅依赖已被投影进 jsonl_records 的字段，重新排序。
    top_influencers = sorted(
        jsonl_records,
        key=lambda r: (r["user_followers"], r["user_id"]),
        reverse=True,
    )[:20]

    return {
        "source_count": len(profiles),
        "projected_count": len(jsonl_records),
        "platform_counts": dict(platform_counts.most_common()),
        "platform_ratios": {
            key: round(count / max(len(jsonl_records), 1), 6)
            for key, count in platform_counts.items()
        },
        "follower_tier_counts": dict(follower_tier_counts.most_common()),
        "interaction_tier_counts": dict(interaction_tier_counts.most_common()),
        "user_index": user_idx,
        "unique_interest_tags": len(interest_freq),
        "top_interests": [
            {"interest": tag, "count": count} for tag, count in top_interests
        ],
        "top_influencers": [
            {
                "user_id": proj["user_id"],
                "user_name": proj["user_name"],
                "platform": proj["platform"],
                "user_followers": proj["user_followers"],
                "user_interests": proj["user_interests"][:5],
            }
            for proj in top_influencers
        ],
        "projection_notes": [
            "neighbors 数组已剥离，需要细粒度关系数据请直接看原文件",
            f"每个用户的 user_interests 仅保留前 {_INTEREST_KEEP} 个标签",
            "follower_tier 段位：dormant(<1) / low(1-99) / mid(100-999) / high(1k-9.9k) / head(10k+)",
            "interaction_tier 段位：silent(0) / light(1-4) / active(5-19) / busy(20-99) / viral(100+)",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把当前使用的数字人画像数据转换为可读的 JSONL + 摘要文件，"
            "不修改原始数据集。"
        ),
    )
    parser.add_argument(
        "--input",
        default="data/derived/weibo_profile_reworked_with_neighbors_with_platform.graph.anon",
        help="输入画像数据集（JSON 对象，键为 user_id）。",
    )
    parser.add_argument(
        "--jsonl",
        default="data/derived/weibo_profile_reworked_readable.jsonl",
        help="输出的 JSONL 路径（每行一个用户的精简视图）。",
    )
    parser.add_argument(
        "--summary",
        default="data/derived/weibo_profile_reworked_readable_summary.json",
        help="输出的统计摘要 JSON 路径。",
    )
    parser.add_argument(
        "--preview",
        default="data/derived/weibo_profile_reworked_readable_preview.json",
        help="输出的 pretty-print 预览 JSON 路径（前 N 条用户的完整视图）。",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=20,
        help="预览文件保留的用户条数，默认 20。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    jsonl_path = Path(args.jsonl)
    summary_path = Path(args.summary)
    preview_path = Path(args.preview)

    profiles = _load_profiles(input_path)

    jsonl_records: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for raw_user_id, record in profiles.items():
            if not isinstance(record, dict):
                continue
            proj = _project_user(record)
            handle.write(json.dumps(proj, ensure_ascii=False))
            handle.write("\n")
            jsonl_records.append(proj)

    summary = _build_summary(profiles, jsonl_records)
    summary["input_path"] = str(input_path)
    summary["jsonl_path"] = str(jsonl_path)
    summary["preview_path"] = str(preview_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_records = jsonl_records[: max(0, int(args.preview_count))]
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(
        json.dumps(preview_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "input_path": str(input_path),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "preview_path": str(preview_path),
        "preview_count": len(preview_records),
        "projected_count": len(jsonl_records),
        "jsonl_size_mb": round(jsonl_path.stat().st_size / 1024 / 1024, 3),
        "summary_size_mb": round(summary_path.stat().st_size / 1024 / 1024, 3),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
