"""为时事政治领域补充数字人节点。

为什么需要
----------
现有 35,636 个节点来自微博娱乐兴趣数据，与事件库的时政定位严重错配
（实测"加息通胀/物价"类事件全库只有 16 个节点命中关键词，"人工智能治理"
也只有 668 个，占 1.87%）。这直接导致时政事件的召回池里几乎没有对口的人。

本脚本按 15 个 ``event_type`` 合成领域节点，合并进现有档案：
  - 兴趣标签取自 ``EVENT_TYPE_RULES`` 的该类关键词（与事件解析同源，命中率高）；
  - 简介复用该类的真实 ``opinion_variants``（观点型短文本，质量优于模板）；
  - 粉丝/图属性按分层模拟，保证能过稳定度与风险门槛。

生成后**必须重建语义索引**，否则新节点语义分为 0（BGE 下限 0.30 会把它们
全部挡在门外）：::

    python scripts/build_local_models.py --skip-graph

图索引不必重建：新节点没有真实邻居边，GraphSAGE 给不出有意义的潜力值，
保留旧索引让新节点 ``graph_potential`` 取 0 即可（只影响 0.35×potential 的
传播力增量，不影响主流程）。

用法::

    python scripts/build_political_domain_nodes.py                # 每类 200 个
    python scripts/build_political_domain_nodes.py --per-type 300
    python scripts/build_political_domain_nodes.py --dry-run      # 只看统计不写盘
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from influence_strategy.event_parser import EVENT_TYPE_RULES, RuleBasedEventParser  # noqa: E402

SOURCE_EVENTS = ROOT / "eval" / "hot_event_opinion_variants_200.json"
TARGET_PROFILE = (
    ROOT / "data" / "derived" / "weibo_profile_reworked_with_neighbors_with_platform.graph.anon"
)
BACKUP_PROFILE = TARGET_PROFILE.with_name(TARGET_PROFILE.name + ".bak")

PLATFORMS = ("facebook_simulated", "instagram_simulated", "twitter_simulated")

#: 领域中文标签，用于 user_name 与兜底简介。
DOMAIN_LABELS: dict[str, str] = {
    "military_conflict": "军事观察",
    "international_relations": "国际关系",
    "macro_economy": "宏观经济",
    "financial_market": "金融市场",
    "tech_ai": "科技前沿",
    "cyber_data_governance": "网络安全",
    "energy_resources": "能源产业",
    "climate_disaster": "气候环境",
    "public_health": "公共卫生",
    "politics_election": "时政选举",
    "social_unrest": "社会舆情",
    "industry_supply_chain": "产业供应链",
    "agriculture_food": "农业食品",
    "labor_demographics": "劳动人口",
    "social_livelihood": "社会民生",
}


def collect_variants_by_type() -> dict[str, list[str]]:
    """按 event_type 归集真实观点变体，作为新节点简介的素材。"""

    events = json.loads(SOURCE_EVENTS.read_text(encoding="utf-8"))
    parser = RuleBasedEventParser()
    grouped: dict[str, list[str]] = defaultdict(list)
    for hot in events:
        parsed = parser.parse(
            {
                "event_id": str(hot.get("event_id") or ""),
                "event_title": str(hot.get("event_title") or ""),
                "event_description": str(hot.get("event_summary") or ""),
                "target_goal": str(hot.get("target") or ""),
            }
        )
        for variant in hot.get("opinion_variants") or []:
            text = str(variant).strip()
            if text:
                grouped[parsed.event_type].append(text)
    return grouped


def build_nodes(*, per_type: int, seed: int) -> dict[str, dict[str, Any]]:
    grouped = collect_variants_by_type()
    rng = random.Random(seed)
    nodes: dict[str, dict[str, Any]] = {}

    for event_type, keywords in sorted(EVENT_TYPE_RULES.items()):
        label = DOMAIN_LABELS.get(event_type, event_type)
        pool = grouped.get(event_type) or []
        keyword_list = list(keywords)

        for index in range(per_type):
            # 兴趣标签：与事件解析词典同源，保证关键词命中率
            size = min(len(keyword_list), rng.randint(3, 5))
            interests = rng.sample(keyword_list, k=size)

            # 简介：优先复用真实观点变体，且优先挑含兴趣词的那些 ——
            # 让 BGE 编出的语义向量与标签一致，避免"标签写农业、简介讲战争"。
            matched = [text for text in pool if any(key in text for key in interests)]
            if matched:
                description = matched[index % len(matched)]
            elif pool:
                description = f"{label}观察者：{pool[index % len(pool)]}"
            else:
                description = f"{label}领域观察者，关注" + "、".join(interests[:3])

            # 粉丝分层：留出头部/腰部/长尾，保证打分有区分度
            tier = rng.random()
            if tier < 0.15:
                followers = rng.randint(50_000, 500_000)
            elif tier < 0.40:
                followers = rng.randint(5_000, 50_000)
            else:
                followers = rng.randint(200, 5_000)
            friends = rng.randint(100, 2_000)

            # 图属性：neighbor 必须 > 0（否则判孤立，风险 +0.20）
            neighbor = max(1, followers // 200)
            received = max(1, followers // 50)
            made = rng.randint(50, 500)

            user_id = f"pol_{event_type}_{index:04d}"
            nodes[user_id] = {
                "user_id": user_id,
                "user_name": f"{label}_{index:04d}",
                "user_followers": followers,
                "user_friends": friends,
                "user_interests": interests,
                "user_description": description,
                "graph_attributes": {
                    "neighbor_count": neighbor,
                    "engaged_by_neighbor_count": max(1, neighbor // 2),
                    "engaged_to_neighbor_count": max(1, neighbor // 2),
                    "mutual_neighbor_count": max(1, neighbor // 3),
                    # 自互动必须为 0，否则触发风险分量 self_interaction
                    "self_interaction_count": 0,
                    "self_comment_count": 0,
                    "self_repost_count": 0,
                    "received_interaction_count": received,
                    "received_comment_count": int(received * 0.6),
                    "received_repost_count": int(received * 0.4),
                    "made_interaction_count": made,
                    "made_comment_count": int(made * 0.7),
                    "made_repost_count": int(made * 0.3),
                    "isolated": False,
                },
                "neighbors": [],
                "platform": PLATFORMS[index % len(PLATFORMS)],
            }
    return nodes


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="补充时事政治领域数字人节点")
    arg_parser.add_argument("--per-type", type=int, default=200, help="每个 event_type 生成的节点数")
    arg_parser.add_argument("--seed", type=int, default=20261027, help="随机种子，保证可复现")
    arg_parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    args = arg_parser.parse_args()

    nodes = build_nodes(per_type=args.per_type, seed=args.seed)
    counts: dict[str, int] = defaultdict(int)
    for user_id in nodes:
        counts[user_id.rsplit("_", 1)[0]] += 1
    print("待生成节点 %d 个，按类目：" % len(nodes))
    for event_type, count in sorted(counts.items()):
        print("   %-28s %d" % (event_type, count))

    if args.dry_run:
        sample = nodes[sorted(nodes)[0]]
        print("\n样例节点：")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return

    payload = json.loads(TARGET_PROFILE.read_text(encoding="utf-8"))
    before = len(payload)
    if not BACKUP_PROFILE.exists():
        shutil.copy2(TARGET_PROFILE, BACKUP_PROFILE)
        print(f"\n已备份原档案 -> {BACKUP_PROFILE.name}")
    payload.update(nodes)
    TARGET_PROFILE.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"\n写入 {TARGET_PROFILE.name}: {before} -> {len(payload)} 个节点")
    print("\n下一步：重建语义索引（否则新节点语义分为 0）")
    print("    python scripts/build_local_models.py --skip-graph")


if __name__ == "__main__":
    main()
