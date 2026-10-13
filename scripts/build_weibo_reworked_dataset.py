from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


GRAPH_ATTRIBUTE_FIELDS = (
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

PRODUCT_NOISE_TERMS = {
    "购物",
    "电商",
    "促销",
    "折扣",
    "返利",
    "导购",
    "带货",
    "直播带货",
    "好物推荐",
    "淘宝",
    "京东",
    "拼多多",
}

PUBLIC_AFFAIRS_TEMPLATES = [
    {
        "category": "政策解读",
        "interests": ["时政", "政策解读", "公共治理", "民生政策", "法律法规", "社会观察"],
        "description_patterns": [
            "关注公共政策和民生议题，习惯把复杂信息整理成清楚的背景和影响。",
            "长期记录政策变化、公共服务和城市治理信息，偏好理性讨论。",
        ],
    },
    {
        "category": "国际观察",
        "interests": ["时政", "国际关系", "地缘政治", "国际贸易", "能源安全", "全球供应链"],
        "description_patterns": [
            "关注国际局势、贸易通道和能源变化，常做资料整理和背景解释。",
            "记录国际新闻和区域冲突动态，重视信息来源和多角度分析。",
        ],
    },
    {
        "category": "财经产业",
        "interests": ["宏观经济", "产业政策", "供应链", "金融市场", "企业经营", "消费观察"],
        "description_patterns": [
            "关注宏观经济、产业链和企业经营变化，喜欢用数据解释趋势。",
            "长期观察产业政策和市场反应，偏好简洁清楚的财经分析。",
        ],
    },
    {
        "category": "法律规则",
        "interests": ["法律法规", "公共规则", "合规风险", "消费者权益", "社会事件", "事实核查"],
        "description_patterns": [
            "关注公共事件中的法律边界和责任划分，强调事实、证据和规则。",
            "记录社会事件与法律常识，习惯先核实信息再发表观点。",
        ],
    },
    {
        "category": "民生服务",
        "interests": ["民生", "交通出行", "教育", "医疗", "就业", "公共服务"],
        "description_patterns": [
            "关注交通、教育、医疗和就业等民生信息，喜欢整理实用提醒。",
            "记录公共服务和生活政策变化，希望信息表达更清楚、更好懂。",
        ],
    },
    {
        "category": "媒体核查",
        "interests": ["事实核查", "媒体观察", "舆情分析", "信息源", "谣言澄清", "公共议题"],
        "description_patterns": [
            "关注热点事件的信息源和传播路径，倾向于做事实核查和澄清。",
            "记录公共议题中的争议点，习惯区分事实、观点和推测。",
        ],
    },
    {
        "category": "地方治理",
        "interests": ["城市治理", "区域发展", "本地新闻", "基层服务", "公共安全", "社区观察"],
        "description_patterns": [
            "关注本地公共事务、城市治理和区域发展，记录身边变化。",
            "长期观察城市公共服务和社区议题，偏好具体问题具体分析。",
        ],
    },
    {
        "category": "科技产业",
        "interests": ["科技政策", "人工智能", "芯片产业", "新能源", "数字经济", "平台治理"],
        "description_patterns": [
            "关注科技政策、AI 和产业升级，喜欢解释技术变化背后的社会影响。",
            "记录数字经济、新能源和平台治理信息，偏好理性讨论。",
        ],
    },
    {
        "category": "能源航运",
        "interests": ["能源价格", "航运物流", "国际贸易", "供应链", "港口运输", "全球市场"],
        "description_patterns": [
            "关注能源、航运和全球供应链变化，习惯把热点放回产业背景里看。",
            "记录港口、航线、能源价格和贸易通道信息，偏好数据化表达。",
        ],
    },
    {
        "category": "教育青年",
        "interests": ["教育政策", "青年就业", "高校", "职业发展", "社会流动", "公共讨论"],
        "description_patterns": [
            "关注教育政策、青年就业和职业发展，喜欢整理有用信息和现实影响。",
            "记录高校、就业和社会流动议题，偏好温和但有依据的表达。",
        ],
    },
]

INFLUENCE_TIERS = {
    "ordinary": {
        "followers": (800, 5_000),
        "friends": (80, 600),
        "neighbors": (3, 15),
        "received_rate": (0.012, 0.035),
        "made_rate": (0.05, 0.18),
    },
    "vertical": {
        "followers": (5_000, 50_000),
        "friends": (150, 1_200),
        "neighbors": (10, 45),
        "received_rate": (0.01, 0.03),
        "made_rate": (0.04, 0.14),
    },
    "medium": {
        "followers": (50_000, 300_000),
        "friends": (300, 2_000),
        "neighbors": (25, 100),
        "received_rate": (0.006, 0.018),
        "made_rate": (0.025, 0.10),
    },
    "high": {
        "followers": (300_000, 900_000),
        "friends": (600, 3_000),
        "neighbors": (70, 220),
        "received_rate": (0.004, 0.012),
        "made_rate": (0.015, 0.06),
    },
}


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


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


def clean_interests(items: list[Any]) -> tuple[list[str], int]:
    cleaned: list[str] = []
    removed_count = 0
    for item in dedupe_texts(items):
        if item in PRODUCT_NOISE_TERMS:
            removed_count += 1
            continue
        cleaned.append(item)
    return cleaned, removed_count


def normalize_graph_attributes(raw_graph_attributes: Any) -> dict[str, Any]:
    raw = raw_graph_attributes if isinstance(raw_graph_attributes, dict) else {}
    graph_attributes: dict[str, Any] = {}
    for field in GRAPH_ATTRIBUTE_FIELDS:
        if field == "isolated":
            graph_attributes[field] = safe_bool(raw.get(field, True))
        else:
            graph_attributes[field] = safe_int(raw.get(field, 0))
    return graph_attributes


def normalize_profile_record(user_id: str, raw_record: dict[str, Any]) -> tuple[dict[str, Any], int]:
    interests, removed_count = clean_interests(raw_record.get("user_interests", []))
    normalized = {
        "user_id": str(raw_record.get("user_id") or user_id),
        "user_name": str(raw_record.get("user_name") or f"user_{user_id}"),
        "user_followers": safe_int(raw_record.get("user_followers", 0)),
        "user_friends": safe_int(raw_record.get("user_friends", 0)),
        "user_interests": interests,
        "user_description": str(raw_record.get("user_description") or "").strip(),
        "graph_attributes": normalize_graph_attributes(raw_record.get("graph_attributes", {})),
    }
    return normalized, removed_count


def has_profile_signal(record: dict[str, Any]) -> bool:
    return bool(record.get("user_interests")) or bool(str(record.get("user_description") or "").strip())


def stable_hash_int(text: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{text}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return int(digest, 16)


def low_value_score(record: dict[str, Any]) -> float:
    graph = record.get("graph_attributes", {})
    return (
        safe_int(record.get("user_followers")) * 1.0
        + safe_int(record.get("user_friends")) * 0.4
        + safe_int(graph.get("neighbor_count")) * 25.0
        + safe_int(graph.get("received_interaction_count")) * 2.0
        + safe_int(graph.get("made_interaction_count")) * 1.5
    )


def select_rewrite_ids(
    normalized_records: dict[str, dict[str, Any]],
    *,
    rewrite_count: int,
    seed: int,
) -> set[str]:
    candidates = [
        user_id
        for user_id, record in normalized_records.items()
        if not has_profile_signal(record)
    ]
    candidates.sort(
        key=lambda user_id: (
            low_value_score(normalized_records[user_id]),
            stable_hash_int(user_id, seed),
        )
    )
    window_size = min(len(candidates), max(rewrite_count * 5, rewrite_count))
    rng = random.Random(seed)
    rewrite_pool = candidates[:window_size]
    rng.shuffle(rewrite_pool)
    return set(rewrite_pool[:rewrite_count])


def pick_influence_tier(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.54:
        return "ordinary"
    if roll < 0.86:
        return "vertical"
    if roll < 0.98:
        return "medium"
    return "high"


def rand_range(rng: random.Random, bounds: tuple[int, int]) -> int:
    low, high = bounds
    return rng.randint(low, high)


def rand_float_range(rng: random.Random, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return rng.uniform(low, high)


def build_rewritten_profile(
    user_id: str,
    original: dict[str, Any],
    *,
    rewrite_index: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(stable_hash_int(user_id, seed))
    template = PUBLIC_AFFAIRS_TEMPLATES[rewrite_index % len(PUBLIC_AFFAIRS_TEMPLATES)]
    tier = pick_influence_tier(rng)
    tier_config = INFLUENCE_TIERS[tier]

    follower_count = max(
        safe_int(original.get("user_followers")),
        rand_range(rng, tier_config["followers"]),
    )
    friend_count = max(
        safe_int(original.get("user_friends")),
        rand_range(rng, tier_config["friends"]),
    )
    received_interaction_count = max(
        8,
        int(follower_count * rand_float_range(rng, tier_config["received_rate"])),
    )
    made_interaction_count = max(
        6,
        int(friend_count * rand_float_range(rng, tier_config["made_rate"])),
    )
    neighbor_count = max(
        rand_range(rng, tier_config["neighbors"]),
        min(220, int((received_interaction_count + made_interaction_count) ** 0.5)),
    )
    mutual_neighbor_count = int(neighbor_count * rng.uniform(0.18, 0.48))

    received_comment_ratio = rng.uniform(0.45, 0.70)
    made_comment_ratio = rng.uniform(0.55, 0.82)
    received_comment_count = int(received_interaction_count * received_comment_ratio)
    made_comment_count = int(made_interaction_count * made_comment_ratio)

    interests = list(template["interests"])
    extra_terms = [
        "热点事件",
        "公共讨论",
        "社会影响",
        "信息整理",
        "理性表达",
        "数据观察",
    ]
    rng.shuffle(extra_terms)
    interests = dedupe_texts([*interests, *extra_terms[:2]])[:8]
    description = rng.choice(template["description_patterns"])

    rewritten = {
        "user_id": str(original.get("user_id") or user_id),
        "user_name": str(original.get("user_name") or f"user_{user_id}"),
        "user_followers": follower_count,
        "user_friends": friend_count,
        "user_interests": interests,
        "user_description": description,
        "graph_attributes": {
            "neighbor_count": neighbor_count,
            "mutual_neighbor_count": mutual_neighbor_count,
            "self_interaction_count": rng.choice([0, 0, 0, 1]),
            "received_interaction_count": received_interaction_count,
            "received_comment_count": received_comment_count,
            "received_repost_count": max(0, received_interaction_count - received_comment_count),
            "made_interaction_count": made_interaction_count,
            "made_comment_count": made_comment_count,
            "made_repost_count": max(0, made_interaction_count - made_comment_count),
            "isolated": False,
        },
    }
    manifest_item = {
        "user_id": str(user_id),
        "rewrite_reason": "empty_profile_low_signal",
        "assigned_category": template["category"],
        "assigned_influence_tier": tier,
        "original": {
            "user_followers": safe_int(original.get("user_followers")),
            "user_friends": safe_int(original.get("user_friends")),
            "user_interests": original.get("user_interests", []),
            "user_description": original.get("user_description", ""),
            "graph_attributes": original.get("graph_attributes", {}),
        },
        "rewritten": rewritten,
    }
    return rewritten, manifest_item


def build_reworked_dataset(
    raw_payload: dict[str, Any],
    *,
    rewrite_count: int,
    keep_empty_unrewritten: bool,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    normalized_records: dict[str, dict[str, Any]] = {}
    total_removed_interest_terms = 0
    for user_id, raw_record in raw_payload.items():
        if not isinstance(raw_record, dict):
            continue
        normalized, removed_count = normalize_profile_record(str(user_id), raw_record)
        normalized_records[str(user_id)] = normalized
        total_removed_interest_terms += removed_count

    rewrite_ids = select_rewrite_ids(
        normalized_records,
        rewrite_count=max(0, rewrite_count),
        seed=seed,
    )

    reworked_payload: dict[str, dict[str, Any]] = {}
    manifest_items: list[dict[str, Any]] = []
    rewrite_index = 0
    dropped_count = 0
    kept_original_count = 0
    for user_id, normalized in normalized_records.items():
        if user_id in rewrite_ids:
            rewritten, manifest_item = build_rewritten_profile(
                user_id,
                normalized,
                rewrite_index=rewrite_index,
                seed=seed,
            )
            reworked_payload[user_id] = rewritten
            manifest_items.append(manifest_item)
            rewrite_index += 1
            continue

        if keep_empty_unrewritten or has_profile_signal(normalized):
            reworked_payload[user_id] = normalized
            kept_original_count += 1
        else:
            dropped_count += 1

    category_distribution = Counter(
        item["assigned_category"] for item in manifest_items
    )
    tier_distribution = Counter(
        item["assigned_influence_tier"] for item in manifest_items
    )
    description_lengths = [
        len(str(item.get("user_description") or ""))
        for item in reworked_payload.values()
    ]
    interest_counts = [
        len(item.get("user_interests", []))
        for item in reworked_payload.values()
    ]

    summary = {
        "input_node_count": len(normalized_records),
        "output_node_count": len(reworked_payload),
        "kept_original_node_count": kept_original_count,
        "rewritten_node_count": len(manifest_items),
        "dropped_low_value_node_count": dropped_count,
        "empty_profile_input_count": sum(
            1 for item in normalized_records.values() if not has_profile_signal(item)
        ),
        "removed_product_noise_interest_terms": total_removed_interest_terms,
        "nonempty_description_ratio": round(
            sum(1 for length in description_lengths if length > 0) / max(len(description_lengths), 1),
            6,
        ),
        "avg_description_length": round(mean(description_lengths), 4) if description_lengths else 0.0,
        "avg_interest_count": round(mean(interest_counts), 4) if interest_counts else 0.0,
        "rewrite_category_distribution": dict(category_distribution),
        "rewrite_influence_tier_distribution": dict(tier_distribution),
        "seed": seed,
        "notes": [
            "raw dataset is not modified",
            "main output keeps the original profile schema",
            "empty low-value profiles are removed unless rewritten",
            "rewrite manifest is for experiment tracing and is not required by the runtime pipeline",
        ],
    }
    return reworked_payload, manifest_items, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a slimmer Weibo-style profile dataset for hot-event simulation."
    )
    parser.add_argument(
        "--input",
        default="data/derived/abc_reading_profile_compact.graph.anon",
        help="Input compact profile dataset.",
    )
    parser.add_argument(
        "--output",
        default="data/derived/weibo_profile_reworked.graph.anon",
        help="Output reworked profile dataset.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/derived/weibo_profile_reworked_summary.json",
        help="Output dataset summary JSON.",
    )
    parser.add_argument(
        "--manifest-output",
        default="data/derived/weibo_profile_rewrite_manifest.jsonl",
        help="Output rewrite tracing manifest JSONL.",
    )
    parser.add_argument(
        "--rewrite-count",
        type=int,
        default=360,
        help="Number of low-value empty profiles to rewrite.",
    )
    parser.add_argument(
        "--keep-empty-unrewritten",
        action="store_true",
        help="Keep empty profiles that are not selected for rewriting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260623,
        help="Deterministic seed for node selection and generated profile variation.",
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
    manifest_output_path = Path(args.manifest_output)
    indent = None if args.indent <= 0 else args.indent

    raw_payload = load_json_object(input_path)
    reworked_payload, manifest_items, summary = build_reworked_dataset(
        raw_payload,
        rewrite_count=args.rewrite_count,
        keep_empty_unrewritten=args.keep_empty_unrewritten,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(reworked_payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )

    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output_path.open("w", encoding="utf-8") as handle:
        for item in manifest_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "manifest_output_path": str(manifest_output_path),
            "input_size_bytes": input_path.stat().st_size if input_path.exists() else 0,
            "output_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        }
    )
    if summary["input_size_bytes"]:
        summary["size_reduction_ratio"] = round(
            1 - summary["output_size_bytes"] / summary["input_size_bytes"],
            6,
        )
    else:
        summary["size_reduction_ratio"] = 0.0

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
