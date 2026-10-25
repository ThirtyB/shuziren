"""测量 ④ 内容生成在端到端链路中的耗时占比。

分三个阶段计时，避免把"核心策略已就绪"和"文案已生成"混为一谈：

    core_sec     ①事件解析 + ②特征增强 + ③选人重排 + strategy_generator
                 —— 到 ``strategy_result`` 生成为止，核心决策（谁/什么角色/
                    什么平台/什么时间/什么频率/什么风险）此时已全部确定
    content_sec  ④ 内容生成（``build_eval_output`` 内的 LLM 文案调用）
    e2e_sec      上面两段 + 落盘

用法：
    python tests/measure_content_phase.py --reps 3
    python tests/measure_content_phase.py --reps 3 --event-id hot_event_001
    python tests/measure_content_phase.py --reps 3 --use-llm 0     # ④ 关闭对照

输出写 ``outputs/content_phase/measure_report.md`` 与 ``raw/*.json``。
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
    build_eval_output,
    get_content_llm_metrics,
    hot_event_to_pipeline_payload,
    load_hot_events,
    reset_content_llm_metrics,
)
from influence_strategy.feature_builder import get_feature_llm_metrics  # noqa: E402
from influence_strategy.llm_client import OpenAICompatibleLLMClient  # noqa: E402
from influence_strategy.pipeline import StrategyPipeline  # noqa: E402

EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
OUT_ROOT = PROJECT_ROOT / "outputs" / "content_phase"

MAX_SELECTED_NODES = 5
CAMPAIGN_WINDOW_HOURS = 24
MAX_FREQUENCY_PER_DAY = 3
DISPATCH_PLATFORM = "facebook_simulated"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_once(
    *,
    event_id: str,
    use_llm: bool,
    content_use_llm: bool,
    llm_client,
    run_index: int,
) -> dict:
    """跑一次并分段计时。

    ``use_llm`` 控制 ①②③（核心链路），``content_use_llm`` 单独控制 ④，
    这样才能测出"核心开 + ④ 关"这个组合，而不是把 4 个一起关掉。
    """
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
    pipeline = StrategyPipeline(product_name="abc_reading")

    # ---- 核心阶段：①②③ + strategy_generator ----
    core_started = time.monotonic()
    artifacts = pipeline.run_with_artifacts(
        workspace_root=PROJECT_ROOT,
        event_input=payload,
        profile_limit=None,
        use_llm=use_llm,
        llm_client=llm_client,
        target_user_id=None,
        dispatch_platform=DISPATCH_PLATFORM,
    )
    core_sec = time.monotonic() - core_started

    # ---- ④ 内容生成阶段 ----
    reset_content_llm_metrics()
    content_started = time.monotonic()
    out_payload, content_meta = build_eval_output(
        hot_event=hot_event,
        strategy_result=artifacts.strategy_result,
        workspace_root=PROJECT_ROOT,
        use_llm=content_use_llm,
        llm_client=llm_client,
    )
    content_sec = time.monotonic() - content_started
    content_metrics = get_content_llm_metrics()

    fd = out_payload["五维调度策略"]
    record = {
        "event_id": event_id,
        "run_index": run_index,
        "use_llm": use_llm,
        # 三段耗时
        "core_sec": round(core_sec, 3),
        "content_sec": round(content_sec, 3),
        "e2e_sec": round(core_sec + content_sec, 3),
        # ④ 自己的指标
        "content_metrics": content_metrics,
        "content_meta": {
            "used": content_meta.get("used"),
            "fallback_reason": content_meta.get("fallback_reason"),
            "provider": content_meta.get("provider"),
            "model": content_meta.get("model"),
        },
        # ② 的指标（对照）
        "feature_metrics": get_feature_llm_metrics(),
        # 核心字段（回归对照用）
        "selected_ids": fd["目标对象"]["选取数字人id组"],
        "roles": {item["数字人id"]: item["节点角色"] for item in fd["目标对象"]["数字人分工"]},
        "platforms": {
            item["数字人id"]: item["分发平台"] for item in fd["平台"]["数字人平台分发"]
        },
        "time_plan": {
            item["数字人id"]: item["时间阶段"] for item in fd["时间"]["数字人时间安排"]
        },
        "frequency_plan": {
            item["数字人id"]: item["发帖频率"] for item in fd["频率"]["数字人频率"]
        },
        "risk": [
            {
                "user_id": n.user_id,
                "risk_level": n.risk_level,
                "manual_review_required": n.manual_review_required,
                "risk_flags": list(n.risk_flags),
            }
            for n in artifacts.strategy_result.selected_nodes
        ],
        "content_used": content_meta.get("used"),
        "content_fallback_reason": content_meta.get("fallback_reason"),
        "post_count": len(fd["内容"]["数字人发帖内容"]),
        "post_content_preview": [
            item["发帖内容"][:60] for item in fd["内容"]["数字人发帖内容"]
        ],
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", default="hot_event_001")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--use-llm",
        type=int,
        default=1,
        help="①②③ 核心链路 LLM 开关（默认 1）",
    )
    parser.add_argument(
        "--content-use-llm",
        type=int,
        default=1,
        help="④ 内容生成 LLM 开关（默认 1）；设为 0 即【核心开 + ④ 走规则兜底】对照",
    )
    parser.add_argument("--tag", default="", help="输出文件前缀，用于区分不同对照")
    args = parser.parse_args()

    use_llm = bool(args.use_llm)
    content_use_llm = bool(args.content_use_llm)
    need_client = use_llm or content_use_llm
    client = (
        OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT) if need_client else None
    )
    _log(
        f"[measure] use_llm={use_llm} content_use_llm={content_use_llm} "
        f"event={args.event_id} reps={args.reps}"
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT_ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(1, args.reps + 1):
        _log(f"[measure] >>> run{i} 开始")
        rec = run_once(
            event_id=args.event_id,
            use_llm=use_llm,
            content_use_llm=content_use_llm,
            llm_client=client,
            run_index=i,
        )
        records.append(rec)
        prefix = f"{args.tag}_" if args.tag else ""
        (raw_dir / f"{prefix}{args.event_id}_run{i}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log(
            f"[measure] <<< run{i} 完成 core={rec['core_sec']}s "
            f"content={rec['content_sec']}s e2e={rec['e2e_sec']}s "
            f"content_used={rec['content_used']}"
        )

    # ---- 汇总 ----
    def _stats(values: list[float]) -> str:
        if not values:
            return "-"
        mean = statistics.mean(values)
        if len(values) == 1:
            return f"{mean:.2f}"
        return f"{mean:.2f} (min {min(values):.2f} / max {max(values):.2f})"

    core_vals = [r["core_sec"] for r in records]
    content_vals = [r["content_sec"] for r in records]
    e2e_vals = [r["e2e_sec"] for r in records]

    attempted = sum(r["content_metrics"]["attempted"] for r in records)
    success = sum(r["content_metrics"]["success"] for r in records)
    failure = sum(r["content_metrics"]["failure"] for r in records)

    lines = []
    out = lines.append
    out("# ④ 内容生成耗时测量报告")
    out("")
    out(
        f"事件：`{args.event_id}` ｜ 核心①②③ LLM `use_llm={use_llm}` ｜ "
        f"④ 内容 LLM `content_use_llm={content_use_llm}` ｜ 重复 {args.reps} 次"
    )
    out("")
    out("## 1. 三段耗时（秒）")
    out("")
    out("| 阶段 | 含义 | 平均 (min / max) |")
    out("| --- | --- | --- |")
    out(f"| **核心策略** | ①+②+③+strategy_generator，核心决策就绪 | **{_stats(core_vals)}** |")
    out(f"| **④ 文案生成** | `build_eval_output` 内的 LLM 文案 | **{_stats(content_vals)}** |")
    out(f"| **端到端** | 上面两段之和 | **{_stats(e2e_vals)}** |")
    out("")
    if e2e_vals and statistics.mean(e2e_vals) > 0:
        ratio = statistics.mean(content_vals) / statistics.mean(e2e_vals) * 100
        out(f"④ 占端到端比例：**{ratio:.1f}%**")
        out("")
    out("## 2. ④ 的 LLM 请求统计")
    out("")
    out("| 指标 | 合计 |")
    out("| --- | --- |")
    out(f"| 实际发起请求次数 `attempted` | {attempted} |")
    out(f"| 成功 | {success} |")
    out(f"| 失败 | {failure} |")
    if attempted:
        out(f"| 失败率 | {failure / attempted * 100:.1f}% |")
    out("")
    out("### 每次运行的 fallback 原因")
    out("")
    out("| run | used | fallback_reason | ④耗时(s) | prompt_chars |")
    out("| --- | --- | --- | --- | --- |")
    for r in records:
        lc = r["content_metrics"]["last_call"] or {}
        out(
            f"| {r['run_index']} | {r['content_used']} | "
            f"`{r['content_fallback_reason']}` | {r['content_sec']} | "
            f"{lc.get('prompt_chars', '-')} |"
        )
    out("")
    out("## 3. 核心字段是否稳定（跨 run）")
    out("")
    for field in ("selected_ids", "roles", "platforms", "time_plan", "frequency_plan"):
        vals = [json.dumps(r[field], ensure_ascii=False, sort_keys=True) for r in records]
        stable = len(set(vals)) == 1
        out(f"- `{field}`：**{'一致' if stable else '不一致'}**")
    out("")
    out("## 4. 文案是否生成")
    out("")
    out("| run | 文案条数 | 首条预览 |")
    out("| --- | --- | --- |")
    for r in records:
        preview = r["post_count"] and r["post_content_preview"][0] or "-"
        out(f"| {r['run_index']} | {r['post_count']} | {preview} |")
    out("")

    report = "\n".join(lines)
    prefix = f"{args.tag}_" if args.tag else ""
    (OUT_ROOT / f"{prefix}measure_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
