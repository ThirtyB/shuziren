"""离线实验：模拟"分层召回层"，看把 35,636 人缩到 200 / 300 后会不会漏选。

**本脚本不修改任何项目代码**，只新增文件；召回逻辑在这里独立实现，
不导入/不改 ``pipeline.py`` / ``scorer.py`` / ``selector.py`` / ``feature_builder.py``。

设计（配对实验，冻结上游，0 抖动，可归因）：
    同一个 event + 同一份 feature_result（35,636 个 NodeFeature，LLM 关闭→确定性）
        ├─ baseline : Scorer(全量)      → Selector → 选中 5 人
        ├─ recall200: 分层召回 200 人  → Scorer(200) → Selector → 选中 5 人
        └─ recall300: 分层召回 300 人  → Scorer(300) → Selector → 选中 5 人

召回规则复刻 ``feature_builder._candidate_pool_for_llm``（feature_builder.py:963-1035）
的 4 桶思路：topic / interaction / diffusion / baseline，并带
``generic_high_influence_cap`` 配额。

两种平台处理方式都跑，便于判断哪种合理：
    platform_aware=True   先按平台硬过滤，再在平台内分层召回 N 人（推荐）
    platform_aware=False  先在全量里召回 N 人，再交给 Selector 按平台过滤（朴素）

用法：
    python tests/recall_experiment.py
    python tests/recall_experiment.py --sizes 200 300 --events hot_event_001
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.eval_hot_events import (  # noqa: E402
    hot_event_to_pipeline_payload,
    load_hot_events,
)
from influence_strategy.event_parser import RuleBasedEventParser  # noqa: E402
from influence_strategy.feature_builder import FeatureBuilder  # noqa: E402
from influence_strategy.models import FeatureBuildResult  # noqa: E402
from influence_strategy.pipeline import StrategyPipeline  # noqa: E402
from influence_strategy.platform_profiles import normalize_platform_key  # noqa: E402
from influence_strategy.scorer import Scorer  # noqa: E402
from influence_strategy.selector import Selector  # noqa: E402

EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
OUT_ROOT = PROJECT_ROOT / "outputs" / "recall"

MAX_SELECTED_NODES = 5
CAMPAIGN_WINDOW_HOURS = 24
MAX_FREQUENCY_PER_DAY = 3
DISPATCH_PLATFORM = "facebook_simulated"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 召回层（本脚本独立实现，未改项目代码）
# --------------------------------------------------------------------------- #
def stratified_recall(
    node_features: list,
    *,
    size: int,
    platform_aware: bool,
    dispatch_platform: str | None,
) -> tuple[list, dict]:
    """分层召回：35,636 → size 人。

    返回 ``(recalled, stats)``。桶顺序与配额复刻
    ``feature_builder._candidate_pool_for_llm``(963-1035)。
    """
    pool = list(node_features)
    if platform_aware and dispatch_platform:
        target = normalize_platform_key(dispatch_platform)
        pool = [f for f in pool if normalize_platform_key(f.platform) == target]

    cap = max(2, size // 6)  # 与 add_bucket 的 generic_high_influence_cap 同款

    diversified: list = []
    used_ids: set[str] = set()

    def add_bucket(features: list) -> None:
        for feature in features:
            if len(diversified) >= size:
                return
            if feature.user_id in used_ids:
                continue
            is_generic_high_influence = (
                feature.keyword_hit_count == 0
                and feature.topic_match_score == 0.0
                and feature.influence_score >= 0.65
            )
            if is_generic_high_influence and len(
                [
                    f
                    for f in diversified
                    if f.keyword_hit_count == 0
                    and f.topic_match_score == 0.0
                    and f.influence_score >= 0.65
                ]
            ) >= cap:
                continue
            diversified.append(feature)
            used_ids.add(feature.user_id)

    topic_bucket = sorted(
        pool,
        key=lambda i: (
            i.keyword_hit_count,
            i.topic_match_score,
            i.stability_score,
            i.activity_score,
        ),
        reverse=True,
    )
    interaction_bucket = sorted(
        [i for i in pool if i.is_interaction_source or i.role_hint == "interaction_response"],
        key=lambda i: (
            i.activity_score,
            i.diffusion_score,
            i.stability_score,
            i.influence_score,
        ),
        reverse=True,
    )
    diffusion_bucket = sorted(
        pool,
        key=lambda i: (
            i.diffusion_score,
            i.activity_score,
            i.stability_score,
            i.influence_score,
        ),
        reverse=True,
    )
    baseline_bucket = list(pool)

    add_bucket(topic_bucket)
    add_bucket(interaction_bucket)
    add_bucket(diffusion_bucket)
    add_bucket(baseline_bucket)

    recalled = diversified[:size]
    stats = {
        "source_pool_size": len(pool),
        "recalled_size": len(recalled),
        "generic_high_influence_cap": cap,
    }
    return recalled, stats


# --------------------------------------------------------------------------- #
# 单事件实验
# --------------------------------------------------------------------------- #
def run_event(event_id: str, sizes: list[int]) -> dict:
    events = load_hot_events(EVENTS_PATH)
    hot_event = next(e for e in events if str(e.get("event_id")) == event_id)

    payload = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=MAX_SELECTED_NODES,
        risk_level=None,
        campaign_window_hours=CAMPAIGN_WINDOW_HOURS,
        max_frequency_per_day=MAX_FREQUENCY_PER_DAY,
        allowed_platforms=None,
    )
    # use_llm=False → ①②③ 全关，确定性，可归因
    pipeline = StrategyPipeline(product_name="abc_reading")
    event = pipeline.event_parser.parse(payload, workspace_root=PROJECT_ROOT, use_llm=False)
    feature_result = pipeline.feature_builder.build_from_loader(
        workspace_root=PROJECT_ROOT,
        event=event,
        profile_limit=None,
        use_llm=False,
    )

    scorer = Scorer()
    selector = Selector()

    def select_from(fr: FeatureBuildResult) -> tuple[dict, float]:
        t0 = time.monotonic()
        score_result = scorer.score(fr)
        selection = selector.select(
            score_result,
            workspace_root=PROJECT_ROOT,
            use_llm=False,
            dispatch_platform=DISPATCH_PLATFORM,
        )
        elapsed = time.monotonic() - t0
        return {"score_result": score_result, "selection": selection}, elapsed

    # ---- baseline：全量 ----
    base, base_sec = select_from(feature_result)
    base_sel = base["selection"].selected_nodes
    base_score = base["score_result"]

    # baseline 选中者在「全量打分排序」里的名次（决定 N 至少要多大）
    full_order = [n.user_id for n in base_score.node_scores]
    rank_of = {uid: full_order.index(uid) for uid in full_order}
    # 平台内名次（真正决定能否被捞到的口径）
    plat_order = [
        n.user_id
        for n in base_score.node_scores
        if normalize_platform_key(n.platform) == normalize_platform_key(DISPATCH_PLATFORM)
    ]
    plat_rank = {uid: plat_order.index(uid) for uid in plat_order}

    def snapshot(nodes) -> list[dict]:
        return [
            {
                "user_id": n.user_id,
                "selected_role": n.selected_role,
                "dispatch_stage": n.dispatch_stage,
                "risk_level": n.risk_level,
                "final_score": round(n.final_score, 6),
                "risk_score": round(n.risk_score, 6),
                "eligible": n.eligible,
                "full_rank": rank_of.get(n.user_id),
                "platform_rank": plat_rank.get(n.user_id),
            }
            for n in nodes
        ]

    result = {
        "event_id": event_id,
        "domain": hot_event.get("domain"),
        "event_title": hot_event.get("event_title"),
        "total_nodes": len(feature_result.node_features),
        "platform": DISPATCH_PLATFORM,
        "platform_node_count": len(plat_order),
        "baseline": {
            "selected": snapshot(base_sel),
            "selected_ids": [n.user_id for n in base_sel],
            "score_sec": round(base_sec, 3),
            "eligible_count": base_score.summary.eligible_count,
            "node_count": base_score.summary.node_count,
        },
        "variants": {},
    }

    # ---- 各档召回 ----
    for size in sizes:
        for aware in (True, False):
            recalled, rstats = stratified_recall(
                feature_result.node_features,
                size=size,
                platform_aware=aware,
                dispatch_platform=DISPATCH_PLATFORM,
            )
            recalled_ids = {f.user_id for f in recalled}
            sub = feature_result.model_copy(
                update={"node_features": recalled}
            )
            var, var_sec = select_from(sub)
            var_sel = var["selection"].selected_nodes
            var_ids = [n.user_id for n in var_sel]

            base_ids = set(result["baseline"]["selected_ids"])
            hit_in_pool = sorted(base_ids & recalled_ids)
            miss_in_pool = sorted(base_ids - recalled_ids)
            hit_in_final = sorted(base_ids & set(var_ids))

            key = f"recall_{size}_{'platform_aware' if aware else 'naive'}"
            result["variants"][key] = {
                "size": size,
                "platform_aware": aware,
                "recall_stats": rstats,
                # 池子里的平台分布（朴素模式下会被 Selector 再砍一刀）
                "recalled_platform_count": sum(
                    1
                    for f in recalled
                    if normalize_platform_key(f.platform)
                    == normalize_platform_key(DISPATCH_PLATFORM)
                ),
                "selected": snapshot(var_sel),
                "selected_ids": var_ids,
                "score_sec": round(var_sec, 3),
                # 核心：是否遗漏 baseline 选中者
                "baseline_ids_in_recall_pool": hit_in_pool,
                "baseline_ids_missed_by_recall": miss_in_pool,
                "recall_hit_rate": round(len(hit_in_pool) / max(len(base_ids), 1), 4),
                "baseline_ids_in_final": hit_in_final,
                "final_overlap_rate": round(len(hit_in_final) / max(len(base_ids), 1), 4),
                "selected_ids_identical": var_ids == result["baseline"]["selected_ids"],
                "eligible_count": var["score_result"].summary.eligible_count,
                "node_count": var["score_result"].summary.node_count,
            }
            _log(
                f"[recall] {event_id} {key}: "
                f"pool={rstats['recalled_size']} "
                f"hit={len(hit_in_pool)}/{len(base_ids)} "
                f"identical={var_ids == result['baseline']['selected_ids']}"
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 300])
    parser.add_argument(
        "--events",
        nargs="+",
        default=["hot_event_001", "hot_event_002", "hot_event_003"],
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT_ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for event_id in args.events:
        _log(f"[recall] >>> {event_id}")
        rec = run_event(event_id, args.sizes)
        results.append(rec)
        (raw_dir / f"{event_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (OUT_ROOT / "recall_raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 报告 ----------------
    lines: list[str] = []
    out = lines.append
    out("# 分层召回层离线实验报告")
    out("")
    out(
        f"事件数 {len(results)} ｜ 召回档位 {args.sizes} ｜ 平台 `{DISPATCH_PLATFORM}` ｜ "
        "use_llm=False（确定性配对，冻结上游）"
    )
    out("")
    out("> 召回逻辑在本脚本内独立实现，**未修改任何项目代码**。")
    out("")

    for rec in results:
        out(f"## 事件 `{rec['event_id']}`（domain={rec['domain']}）")
        out("")
        out(
            f"全量 **{rec['total_nodes']}** 人，其中 `{rec['platform']}` "
            f"**{rec['platform_node_count']}** 人"
        )
        out("")
        base = rec["baseline"]
        out("### 基线（全量）")
        out("")
        out("| # | user_id | 角色 | 阶段 | risk | final_score | 全量名次 | 平台内名次 |")
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for i, s in enumerate(base["selected"], 1):
            out(
                f"| {i} | `{s['user_id']}` | {s['selected_role']} | {s['dispatch_stage']} | "
                f"{s['risk_level']} | {s['final_score']} | {s['full_rank']} | "
                f"{s['platform_rank']} |"
            )
        out("")
        out("### 召回各档对比")
        out("")
        out(
            "| 变体 | 池子 | 池内本平台人数 | 捞到基线几人 | 召回命中率 | 最终选中重合 | "
            "选中序列全等 | 耗时(s) |"
        )
        out("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key, v in rec["variants"].items():
            out(
                f"| {key} | {v['recall_stats']['recalled_size']} | "
                f"{v['recalled_platform_count']} | "
                f"{len(v['baseline_ids_in_recall_pool'])}/{len(base['selected_ids'])} | "
                f"{v['recall_hit_rate']:.0%} | "
                f"{len(v['baseline_ids_in_final'])}/{len(base['selected_ids'])} | "
                f"{'✅' if v['selected_ids_identical'] else '❌'} | "
                f"{v['score_sec']} |"
            )
        out("")
        for key, v in rec["variants"].items():
            if v["baseline_ids_missed_by_recall"]:
                out(
                    f"- ⚠️ **{key} 漏掉**："
                    f"`{', '.join(v['baseline_ids_missed_by_recall'])}`"
                )
            if not v["selected_ids_identical"]:
                out(f"- `{key}` 最终选中序列：`{v['selected_ids']}`（基线 `{base['selected_ids']}`）")
        out("")

    # ---------------- 总表 ----------------
    out("## 汇总")
    out("")
    out("| 事件 | 变体 | 召回命中率 | 最终重合率 | 序列全等 |")
    out("| --- | --- | --- | --- | --- |")
    agg: dict[str, list[dict]] = {}
    for rec in results:
        for key, v in rec["variants"].items():
            agg.setdefault(key, []).append(v)
    for key, items in agg.items():
        hr = [i["recall_hit_rate"] for i in items]
        fr = [i["final_overlap_rate"] for i in items]
        ident = sum(1 for i in items if i["selected_ids_identical"])
        out(
            f"| 全部 {len(items)} 个 | {key} | "
            f"{statistics.mean(hr):.0%} | {statistics.mean(fr):.0%} | "
            f"{ident}/{len(items)} |"
        )
    out("")
    out("### 耗时")
    out("")
    out("| 变体 | 平均 Scorer+Selector 耗时(s) |")
    out("| --- | --- |")
    base_secs = [r["baseline"]["score_sec"] for r in results]
    out(f"| baseline（全量） | {statistics.mean(base_secs):.3f} |")
    for key, items in agg.items():
        out(f"| {key} | {statistics.mean([i['score_sec'] for i in items]):.3f} |")
    out("")

    report = "\n".join(lines)
    (OUT_ROOT / "recall_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
