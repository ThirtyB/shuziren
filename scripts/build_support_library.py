"""从原始真实数字人数据中自动抽取「特殊支持节点库」（白库 / 水军库）。

与 ``scripts/gen_water_army.py``（随机合成低活跃账号）不同，本脚本直接从
abc_reading 的真实画像 + 图特征里，按一套"低质量/沉默号"标准筛出**真实存在**
的节点，写入 ``data/raw/abc_reading_water_army.json``，字段结构与 ``UserProfile``
完全一致并打 ``is_water_army=True``。

白库定位（本脚本标准）：**粉丝极少甚至为 0、几乎无互动、资料少、信息少**的普通
账号。任何"高粉丝"或"高点赞/高互动"的账号都**不得**进入白库——它们应留给核心
发布，而不是当作只点赞/转发的水军。

下游（``Selector._build_water_army_nodes``）对这些节点的处理方式不变：仅在启用
水军参与时，按事件平台过滤、排除真实核心演员后随机抽取，只做点赞/转发。

注意：因为现在是真实节点，它们同时也存在于主画像池，故 ``selector.py`` 已同步
把白库 user_id 从核心选人候选中排除，避免同一人被既选为核心又被当作水军。

---------------------------------------------------------------------------
抽取标准：
  【硬过滤 —— 必须全部满足，目的：排除高粉丝/高点赞账号】
    1. 真实账号：不在 product_info 的 ``influencer_ids``（官方/KOL 不当水军）。
    2. 低粉丝：``user_followers <= FOLLOWER_MAX``（默认 100）—— 超过即视为有影响力，
       排除出白库。目标：非常少甚至没有粉丝。
    3. 低互动（排除高点赞/高互动）：``received_interaction_count <= RECEIVED_MAX`` 且
       ``made_interaction_count <= MADE_MAX``（默认均 100）—— 收到或发出过多互动的
       账号视为"热门/活跃"，不得作为水军。
    4. （可选硬门槛）``--desc-len-max`` / ``--interest-max`` 可进一步卡死资料长度/兴趣数，
       默认极大（不生效）。
  【软偏好 —— 排序优先序，目的：资料少、信息少】
    5. 放宽项（降低门槛）：允许孤立号（``NEIGHBOR_MIN=0``）、容忍一定自 loop
       （``SELF_LOOP_MAX=0.50``）。

不设数量上限，凡符合硬过滤标准的真实节点**一律入库**（仅按 ``junk_score`` 升序
排列，越像沉默号/壳号越靠前）：
    junk_score = followers + received + made
                 + DESC_LEN_WEIGHT * len(description)
                 + INTEREST_COUNT_WEIGHT * len(interests)
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from influence_strategy.data_loader import DataLoader  # noqa: E402

# ---- 可调整标准常量（阈值都集中在此，方便调参） ----
# 白库定位 = "低质量/沉默号"：粉丝极少甚至为 0、几乎无互动、资料少、信息少。
# 任何"高粉丝"或"高点赞/高互动"的账号都不得进入白库（留给核心发布）。
EXCLUDE_INFLUENCERS = True
# 粉丝上限：超过即视为有影响力，排除出白库（目标：非常少甚至没有粉丝）
FOLLOWER_MAX = 100
# 收到的互动数上限：超过即"高点赞/高互动"，排除
RECEIVED_INTERACTION_MAX = 100
# 发出的互动数上限：同上，排除活跃账号
MADE_INTERACTION_MAX = 100
# 放宽：允许极低互动 / 孤立号（白库本就信息少）
NEIGHBOR_MIN = 0
# 放宽自 loop 容忍度
SELF_LOOP_MAX = 0.50
# 信息稀疏度在排序中的权重（越大越偏好"资料少/信息少"的账号）
DESC_LEN_WEIGHT = 5
INTEREST_COUNT_WEIGHT = 10
# 不设数量上限：凡符合硬过滤标准的真实节点一律入库（仅按 junk_score 升序排列）。


def _self_loop_ratio(ga) -> float:
    total = (ga.received_interaction_count or 0) + (ga.made_interaction_count or 0)
    if not total:
        return 0.0
    return (ga.self_interaction_count or 0) / total


def main() -> None:
    parser = argparse.ArgumentParser(description="从真实数据抽取特殊支持节点库（沉默号/壳号）")
    parser.add_argument("--product", default="abc_reading")
    parser.add_argument("--out", default=None, help="输出 JSON，默认 data/raw/<product>_water_army.json")
    parser.add_argument("--follower-max", type=int, default=FOLLOWER_MAX)
    parser.add_argument("--received-max", type=int, default=RECEIVED_INTERACTION_MAX)
    parser.add_argument("--made-max", type=int, default=MADE_INTERACTION_MAX)
    parser.add_argument("--neighbor-min", type=int, default=NEIGHBOR_MIN)
    parser.add_argument("--self-loop-max", type=float, default=SELF_LOOP_MAX)
    parser.add_argument("--desc-len-max", type=int, default=9999, help="description 长度硬上限，默认不限制")
    parser.add_argument("--interest-max", type=int, default=9999, help="interests 数量硬上限，默认不限制")
    args = parser.parse_args()

    loader = DataLoader(REPO_ROOT, product_name=args.product)
    enriched = loader.load_enriched_profiles()
    product = loader.load_product_context()
    influencer_ids = set(product.influencer_ids or [])

    # platform -> list[(user_id, prof, ga, followers, received, made, junk_score)]
    per_platform: dict[str, list] = {}
    for user_id, prof in enriched.items():
        if EXCLUDE_INFLUENCERS and user_id in influencer_ids:
            continue
        ga = prof.graph_attributes
        if (ga.neighbor_count or 0) < args.neighbor_min:
            continue
        if _self_loop_ratio(ga) > args.self_loop_max:
            continue
        received = ga.received_interaction_count or 0
        made = ga.made_interaction_count or 0
        if received > args.received_max:
            continue
        if made > args.made_max:
            continue
        followers = prof.user_followers or 0
        if followers > args.follower_max:
            continue
        desc = (prof.user_description or "").strip()
        interests = prof.user_interests or []
        if len(desc) > args.desc_len_max:
            continue
        if len(interests) > args.interest_max:
            continue
        junk_score = (
            followers + received + made
            + DESC_LEN_WEIGHT * len(desc)
            + INTEREST_COUNT_WEIGHT * len(interests)
        )
        platform = prof.platform or "unknown"
        per_platform.setdefault(platform, []).append(
            (user_id, prof, ga, followers, received, made, junk_score)
        )

    selected: dict[str, dict] = {}
    for platform, items in per_platform.items():
        # 升序：junk_score 越小越像沉默号/壳号，越靠前；不设数量上限，全部入库
        pool = sorted(items, key=lambda x: x[6])
        for user_id, prof, _ga, _f, _r, _m, _j in pool:
            selected[user_id] = {
                "user_id": prof.user_id,
                "user_name": prof.user_name,
                "user_followers": prof.user_followers,
                "user_friends": prof.user_friends,
                "user_interests": prof.user_interests,
                "user_description": prof.user_description,
                "platform": prof.platform,
                "is_water_army": True,
            }

    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "data" / "raw" / f"{args.product}_water_army.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_platform: dict[str, int] = {}
    for v in selected.values():
        by_platform[v["platform"]] = by_platform.get(v["platform"], 0) + 1
    fol = [v["user_followers"] for v in selected.values()]
    empty_desc = sum(1 for v in selected.values() if not (v["user_description"] or "").strip())
    zero_int = sum(1 for v in selected.values() if not v["user_interests"])
    print(f"已写入 {len(selected)} 个真实特殊支持节点 -> {out_path}")
    print("各平台分布:", by_platform)
    if fol:
        print(f"粉丝范围: min={min(fol)} max={max(fol)} | 空描述数={empty_desc} | 空兴趣数={zero_int}")


if __name__ == "__main__":
    main()
