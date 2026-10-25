"""旧版（全量 35,636）vs 新版（platform_aware 召回 200）的性能与结果对比。

不改项目代码，只调 pipeline 的 ``recall_size`` / ``recall_mode`` 开关。
use_llm=False（①②③④ 全关）→ 确定性，可直接归因。

用法：
    python tests/compare_recall_before_after.py
    python tests/compare_recall_before_after.py --events hot_event_001 hot_event_002
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

from influence_strategy.eval_hot_events import load_hot_events  # noqa: E402
from influence_strategy.pipeline import StrategyPipeline  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "recall_compare"
DISPATCH_PLATFORM = "facebook_simulated"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def snapshot(selection) -> list[dict]:
    return [
        {
            "user_id": n.user_id,
            "selected_role": n.selected_role,
            "dispatch_stage": n.dispatch_stage,
            "platform": n.platform,
            "risk_level": n.risk_level,
            "final_score": round(n.final_score, 6),
        }
        for n in selection.selected_nodes
    ]


def run_once(*, event_id: str, recall_size: int | None, repeat: int) -> dict:
    pipeline = StrategyPipeline(product_name="abc_reading")
    payload_path = None
    # 直接用 event_id 走 pipeline（event_input 支持 dict）
    from influence_strategy.eval_hot_events import hot_event_to_pipeline_payload
    events = load_hot_events(PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json")
    hot = next(e for e in events if str(e.get("event_id")) == event_id)
    payload = hot_event_to_pipeline_payload(
        hot,
        max_selected_nodes=5,
        risk_level=None,
        campaign_window_hours=24,
        max_frequency_per_day=3,
        allowed_platforms=None,
    )

    # 第一次当预热（DataLoader 有模块级缓存），只计第二次起
    best = None
    for i in range(repeat):
        t0 = time.monotonic()
        if recall_size is None:
            arts = pipeline.run_with_artifacts(
                workspace_root=PROJECT_ROOT,
                event_input=payload,
                profile_limit=None,
                use_llm=False,
                dispatch_platform=DISPATCH_PLATFORM,
            )
        else:
            arts = pipeline.run_with_artifacts(
                workspace_root=PROJECT_ROOT,
                event_input=payload,
                profile_limit=None,
                use_llm=False,
                dispatch_platform=DISPATCH_PLATFORM,
                recall_size=recall_size,
                recall_mode="platform_aware",
            )
        elapsed = time.monotonic() - t0
        if i == 0:
            continue  # 预热不计
        if best is None or elapsed < best[0]:
            best = (elapsed, arts)
    elapsed, arts = best
    return {
        "elapsed_sec": round(elapsed, 3),
        "selected": snapshot(arts.selection_result),
        "selected_ids": [n.user_id for n in arts.selection_result.selected_nodes],
        "node_count": arts.feature_result.summary.node_count,
        "recall_size": arts.feature_result.summary.recall_size,
        "recall_mode": arts.feature_result.summary.recall_mode,
        "eligible_count": arts.score_result.summary.eligible_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--events",
        nargs="+",
        default=[
            "hot_event_001", "hot_event_002", "hot_event_003",
            "hot_event_004", "hot_event_005", "hot_event_006",
            "hot_event_007", "hot_event_008", "hot_event_009",
            "hot_event_010",
        ],
    )
    ap.add_argument("--repeat", type=int, default=3, help="每档跑几次取最快（第1次预热）")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for event_id in args.events:
        _log(f"[cmp] >>> {event_id}")
        old = run_once(event_id=event_id, recall_size=None, repeat=args.repeat)
        new = run_once(event_id=event_id, recall_size=200, repeat=args.repeat)
        identical = old["selected"] == new["selected"]
        results.append({"event_id": event_id, "old": old, "new": new, "identical": identical})
        _log(
            f"[cmp] {event_id}: old={old['elapsed_sec']}s new={new['elapsed_sec']}s "
            f"identical={identical}"
        )

    (OUT_ROOT / "raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 报告 ----------------
    lines: list[str] = []
    out = lines.append
    out("# 旧版（全量）vs 新版（召回 200）· 性能与结果对比")
    out("")
    out(
        f"事件 {len(results)} 个 ｜ 平台 `{DISPATCH_PLATFORM}` ｜ use_llm=False ｜ "
        f"每档跑 {args.repeat} 次取最快（第 1 次预热不计）"
    )
    out("")
    out("| 事件 | 旧版耗时 | 新版耗时 | 加速比 | 旧版 node_count | 新版 node_count | 结果全等 |")
    out("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in results:
        sp = r["old"]["elapsed_sec"] / max(r["new"]["elapsed_sec"], 0.001)
        out(
            f"| `{r['event_id']}` | {r['old']['elapsed_sec']}s | {r['new']['elapsed_sec']}s | "
            f"{sp:.2f}x | {r['old']['node_count']} | {r['new']['node_count']} | "
            f"{'✅' if r['identical'] else '❌'} |"
        )
    out("")

    old_secs = [r["old"]["elapsed_sec"] for r in results]
    new_secs = [r["new"]["elapsed_sec"] for r in results]
    ident = sum(1 for r in results if r["identical"])
    out("## 汇总")
    out("")
    out("| 指标 | 旧版 | 新版 |")
    out("| --- | ---: | ---: |")
    out(f"| 平均端到端 | {statistics.mean(old_secs):.3f}s | {statistics.mean(new_secs):.3f}s |")
    out(f"| 中位端到端 | {statistics.median(old_secs):.3f}s | {statistics.median(new_secs):.3f}s |")
    out(f"| 平均加速比 | — | {statistics.mean(old_secs) / statistics.mean(new_secs):.2f}x |")
    out(f"| 结果全等 | — | {ident}/{len(results)} |")
    out(f"| node_count | 35,636 | 200 |")
    out("")

    out("## 差异明细（字段级）")
    out("")
    any_diff = False
    for r in results:
        if r["identical"]:
            continue
        any_diff = True
        out(f"### `{r['event_id']}`")
        out("")
        out("| # | 旧版 | 新版 | 差异字段 |")
        out("| --- | --- | --- | --- |")
        for i, (o, n) in enumerate(zip(r["old"]["selected"], r["new"]["selected"])):
            if o == n:
                continue
            diffs = [k for k in o if o[k] != n.get(k)]
            out(f"| {i} | `{o['user_id']}` | `{n['user_id']}` | {diffs} |")
        out("")
    if not any_diff:
        out(
            "无差异：**全部 "
            f"{len(results)} 个事件的 selected_ids / selected_role / dispatch_stage / "
            "platform / risk_level / final_score 逐项全等**。"
        )
    out("")

    report = "\n".join(lines)
    (OUT_ROOT / "compare_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
