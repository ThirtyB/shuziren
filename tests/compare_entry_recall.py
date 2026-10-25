"""真实入口（eval_hot_events 的评测入口）回归：召回 ON vs OFF。

不改项目代码，只调公开入口 ``run_hot_event_evaluation``（CLI 也是走它），
分别用 recall 开 / 关跑同一批事件，比较最终选人是否一致。
use_llm=False 保证确定性，可直接归因（② 线上恒失败，影响为 0）。

用法：
    python tests/compare_entry_recall.py
    python tests/compare_entry_recall.py --events hot_event_001 hot_event_002
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.eval_hot_events import run_hot_event_evaluation  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "entry_recall"
DISPATCH_PLATFORM = "facebook_simulated"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _selected_ids(payload: dict) -> list[str]:
    """从真实出口 JSON 里取「选取数字人id组」。"""
    return list(payload.get("选取数字人id组") or [])


def _strategy_ids(payload: dict) -> list[str]:
    """从五维调度策略里取目标对象 id（选人的第二出口）。"""
    strategy = payload.get("五维调度策略") or {}
    obj = strategy.get("目标对象") or {}
    ids = obj.get("数字人id组") or []
    return list(ids)


def _run_once(*, event_id: str, recall_on: bool) -> dict:
    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        trace_dir = Path(tmp) / "trace"
        t0 = time.monotonic()
        _path, payload = run_hot_event_evaluation(
            workspace_root=PROJECT_ROOT,
            input_path=PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json",
            output_dir=out_dir,
            event_id=event_id,
            profile_limit=None,
            use_llm=False,
            trace_dir=trace_dir,
            dispatch_platform=DISPATCH_PLATFORM,
            recall_size=200 if recall_on else 0,
            recall_mode="platform_aware" if recall_on else "off",
        )
        elapsed = time.monotonic() - t0

        # 读 trace 里的候选口径视图，验证展示字段
        view = {}
        trace_file = (
            trace_dir / event_id / "03_feature_builder_output.json"
        )
        if trace_file.exists():
            data = json.loads(trace_file.read_text(encoding="utf-8"))
            view = data.get("candidate_counts") or {}
            summary = data.get("summary") or {}

        return {
            "elapsed_sec": round(elapsed, 3),
            "selected_ids": _selected_ids(payload),
            "strategy_ids": _strategy_ids(payload),
            "payload_keys": sorted(payload.keys()),
            "view": view,
            "summary_node_count": summary.get("node_count"),
            "summary_total": summary.get("total_candidate_count"),
            "summary_recall_size": summary.get("recall_size"),
            "summary_recall_mode": summary.get("recall_mode"),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--events",
        nargs="+",
        default=[
            "hot_event_001", "hot_event_002", "hot_event_003",
            "hot_event_004", "hot_event_005",
        ],
    )
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for event_id in args.events:
        _log(f"[entry] >>> {event_id}")
        off = _run_once(event_id=event_id, recall_on=False)
        on = _run_once(event_id=event_id, recall_on=True)
        identical = (
            off["selected_ids"] == on["selected_ids"]
            and off["strategy_ids"] == on["strategy_ids"]
        )
        results.append(
            {"event_id": event_id, "off": off, "on": on, "identical": identical}
        )
        _log(
            f"[entry] {event_id}: off={off['elapsed_sec']}s on={on['elapsed_sec']}s "
            f"identical={identical}"
        )

    (OUT_ROOT / "raw.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 报告 ----------------
    lines: list[str] = []
    out = lines.append
    out("# 真实入口（eval_hot_events）回归：召回 ON vs OFF")
    out("")
    out(
        f"事件 {len(results)} 个 ｜ 平台 `{DISPATCH_PLATFORM}` ｜ use_llm=False ｜ "
        "入口 = ``run_hot_event_evaluation``（CLI run_eval.py 走同一条）"
    )
    out("")
    out("| 事件 | 关闭召回 | 开启召回 | 加速比 | 选人一致 | 调度目标一致 |")
    out("| --- | ---: | ---: | ---: | --- | --- |")
    for r in results:
        sp = r["off"]["elapsed_sec"] / max(r["on"]["elapsed_sec"], 0.001)
        sel_ok = r["off"]["selected_ids"] == r["on"]["selected_ids"]
        strat_ok = r["off"]["strategy_ids"] == r["on"]["strategy_ids"]
        out(
            f"| `{r['event_id']}` | {r['off']['elapsed_sec']}s | "
            f"{r['on']['elapsed_sec']}s | {sp:.2f}x | {'✅' if sel_ok else '❌'} | "
            f"{'✅' if strat_ok else '❌'} |"
        )
    out("")

    out("## 口径字段（trace 03_feature_builder_output.json）")
    out("")
    out("| 事件 | 状态 | node_count | total_candidate_count | recall_size | recall_mode | display_text |")
    out("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for r in results:
        for label in ("off", "on"):
            d = r[label]
            out(
                f"| `{r['event_id']}` | {'关闭' if label == 'off' else '开启'} | "
                f"{d['summary_node_count']} | {d['summary_total']} | "
                f"{d['summary_recall_size']} | {d['summary_recall_mode']} | "
                f"{d['view'].get('display_text', '-')} |"
            )
    out("")

    ident = sum(1 for r in results if r["identical"])
    off_secs = [r["off"]["elapsed_sec"] for r in results]
    on_secs = [r["on"]["elapsed_sec"] for r in results]
    out("## 汇总")
    out("")
    out(
        f"- 选人一致：**{ident}/{len(results)}**"
    )
    out(
        f"- 平均耗时：关闭 {statistics.mean(off_secs):.3f}s → 开启 "
        f"{statistics.mean(on_secs):.3f}s "
        f"（{statistics.mean(off_secs) / statistics.mean(on_secs):.2f}x）"
    )
    out("")

    out("## 差异明细")
    out("")
    any_diff = False
    for r in results:
        if r["identical"]:
            continue
        any_diff = True
        out(f"- `{r['event_id']}`:")
        out(f"  - 关闭：{r['off']['selected_ids']} / 调度 {r['off']['strategy_ids']}")
        out(f"  - 开启：{r['on']['selected_ids']} / 调度 {r['on']['strategy_ids']}")
    if not any_diff:
        out(
            "无差异：**全部 "
            f"{len(results)} 个事件在真实入口下，召回 ON 与 OFF 的 "
            "「选取数字人id组」与「五维调度策略.目标对象.数字人id组」完全一致**。"
        )
    out("")

    report = "\n".join(lines)
    (OUT_ROOT / "entry_regression.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
