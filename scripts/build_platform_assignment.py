
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


#: 三个模拟平台在数据集 ``platform`` 字段中的取值。
DEFAULT_PLATFORM_VALUES: tuple[str, ...] = (
    "facebook_simulated",
    "instagram_simulated",
    "twitter_simulated",
)


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def stable_hash_int(text: str, seed: int) -> int:
    """基于 blake2b 的稳定哈希，保证跨进程/跨平台可复现。"""

    digest = hashlib.blake2b(
        f"{seed}:{text}".encode("utf-8"), digest_size=8
    ).hexdigest()
    return int(digest, 16)


def assign_platform(
    user_id: str,
    *,
    platform_values: tuple[str, ...],
    seed: int,
) -> str:
    """把 ``user_id`` 确定性地映射到一个模拟平台。

    使用稳定哈希取模，同一 user_id + seed 永远得到同一平台；当平台数量
    不能整除哈希空间时，余数分布仍然是近似均匀的，因此各平台人数接近。
    """

    return platform_values[stable_hash_int(user_id, seed) % len(platform_values)]


def build_platform_dataset(
    *,
    profiles: dict[str, Any],
    platform_values: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """为每条画像注入 ``platform`` 字段，返回新数据集与平台计数。"""

    output: dict[str, dict[str, Any]] = {}
    counter: Counter[str] = Counter()
    for raw_user_id, record in profiles.items():
        if not isinstance(record, dict):
            raise ValueError(f"profile '{raw_user_id}' must be a JSON object.")
        user_id = str(record.get("user_id", raw_user_id))
        platform = assign_platform(
            user_id, platform_values=platform_values, seed=seed
        )
        counter[platform] += 1
        item = dict(record)
        item["user_id"] = user_id
        item["platform"] = platform
        output[str(raw_user_id)] = item
    return output, dict(counter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "为热点评测数据集中的每个数字人注入模拟平台字段（platform）。"
            "输出为可直接被 DataLoader 加载的新数据集。"
        )
    )
    parser.add_argument(
        "--input",
        default="data/derived/weibo_profile_reworked_with_neighbors.graph.anon",
        help="输入画像数据集（JSON 对象，键为 user_id）。",
    )
    parser.add_argument(
        "--output",
        default="data/derived/weibo_profile_reworked_with_neighbors_with_platform.graph.anon",
        help="输出带 platform 字段的数据集。",
    )
    parser.add_argument(
        "--summary-output",
        default="data/derived/weibo_profile_platform_assignment_summary.json",
        help="输出平台分配摘要 JSON。",
    )
    parser.add_argument(
        "--platforms",
        default=",".join(DEFAULT_PLATFORM_VALUES),
        help="逗号分隔的模拟平台取值，默认 facebook_simulated,instagram_simulated,twitter_simulated。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260624,
        help="确定性种子，改变它会得到不同的平台分配结果。",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=0,
        help="JSON 缩进；0 表示紧凑输出（推荐，数据集很大）。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output)

    platform_values = tuple(
        item.strip() for item in str(args.platforms).split(",") if item.strip()
    )
    if not platform_values:
        raise ValueError("--platforms 不能为空。")
    if len(set(platform_values)) != len(platform_values):
        raise ValueError("--platforms 中存在重复的平台取值。")

    profiles = load_json_object(input_path)
    output_payload, platform_counts = build_platform_dataset(
        profiles=profiles,
        platform_values=platform_values,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent <= 0 else args.indent
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=indent)

    total = len(output_payload) or 1
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "node_count": len(output_payload),
        "platform_values": list(platform_values),
        "platform_counts": platform_counts,
        "platform_ratios": {
            key: round(count / total, 6) for key, count in platform_counts.items()
        },
        "seed": args.seed,
        "notes": [
            "每个 user_id 通过稳定哈希确定性映射到一个模拟平台",
            "platform 字段取值形如 facebook_simulated / instagram_simulated / twitter_simulated",
            "同一输入 + 同一 seed 永远得到同一分配结果，可复现",
        ],
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
