"""生成"白库"低活跃度水军库。

白库是一组几乎无历史互动的模拟账号，供"水军参与"功能使用：它们只做点赞 /
转发（不附带文本），用来放大策略的参与人数，且不计入选人上限
（``max_selected_nodes``）。

账号档案结构与正式数字人一致（``UserProfile``），并打上 ``is_water_army=True`` 标记：

    {
      "wa_facebook_0001": {
        "user_id": "wa_facebook_0001",
        "user_name": "白号_facebook_0001",
        "platform": "facebook_simulated",
        "user_followers": 3,
        "user_friends": 5,
        "user_interests": [],
        "user_description": "白库低活跃水军账号",
        "is_water_army": true
      },
      ...
    }

用法::

    python scripts/gen_water_army.py                 # 默认 60 个，三平台均分
    python scripts/gen_water_army.py --count 120     # 指定数量
    python scripts/gen_water_army.py --product abc_reading
    python scripts/gen_water_army.py --out data/raw/abc_reading_water_army.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PLATFORMS = [
    "facebook_simulated",
    "instagram_simulated",
    "twitter_simulated",
]

PLATFORM_PREFIX = {
    "facebook_simulated": "wa_facebook",
    "instagram_simulated": "wa_instagram",
    "twitter_simulated": "wa_twitter",
}


def generate(
    count: int,
    platforms: list[str],
    seed: int = 20261029,
) -> dict[str, dict[str, object]]:
    rng = random.Random(seed)
    profiles: dict[str, dict[str, object]] = {}
    per_platform = max(1, count // len(platforms))
    remainder = count - per_platform * len(platforms)
    plan = []
    for platform in platforms:
        plan.append(per_platform)
    for i in range(remainder):
        plan[i % len(platforms)] += 1

    counters = {p: 0 for p in platforms}
    total = 0
    for platform in platforms:
        prefix = PLATFORM_PREFIX[platform]
        for _ in range(plan[platforms.index(platform)]):
            counters[platform] += 1
            total += 1
            user_id = f"{prefix}_{counters[platform]:04d}"
            profiles[user_id] = {
                "user_id": user_id,
                "user_name": f"白号_{platform.split('_')[0]}_{counters[platform]:04d}",
                "platform": platform,
                "user_followers": rng.randint(0, 12),
                "user_friends": rng.randint(0, 24),
                "user_interests": [],
                "user_description": "白库低活跃水军账号（仅用于点赞/转发放大参与度）",
                "is_water_army": True,
            }
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="生成白库低活跃度水军库")
    parser.add_argument("--count", type=int, default=60, help="水军数量（三平台均分）")
    parser.add_argument("--product", default="abc_reading", help="产品名，决定输出文件名")
    parser.add_argument(
        "--out",
        default=None,
        help="输出 JSON 路径，默认 data/raw/<product>_water_army.json",
    )
    parser.add_argument("--seed", type=int, default=20261029, help="随机种子")
    args = parser.parse_args()

    profiles = generate(args.count, PLATFORMS, seed=args.seed)

    if args.out:
        out_path = Path(args.out)
    else:
        repo_root = Path(__file__).resolve().parent.parent
        out_path = repo_root / "data" / "raw" / f"{args.product}_water_army.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {len(profiles)} 个白库水军账号 -> {out_path}")


if __name__ == "__main__":
    main()
