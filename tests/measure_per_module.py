"""
按模块拆分策略计算耗时：直接调用各模块，外部用 ``time.perf_counter`` 计时。

不修改项目代码；脚本在 Python 里按 web_app /api/runs/text 走的同一条路径逐个调用
各模块，测出每个阶段的墙钟耗时。这样测得的"分模块耗时"是各模块纯计算时间，
不含 HTTP 入口与 uvicorn 队列等开销（那些已在 tests/test_strategy_perf.py
中以"端到端墙钟"形式统计过）。

用法（项目根目录下）：

    # LLM 关闭（默认）
    .\\.venv\\Scripts\\python.exe -u tests\\measure_per_module.py

    # LLM 开启（教师要求的口径）
    $env:USE_LLM = "1"
    .\\.venv\\Scripts\\python.exe -u tests\\measure_per_module.py

阶段定义（与 _run_hot_event_evaluation 中 multi_user 路径对齐）：

    01 pipeline_payload    : hot_event_to_pipeline_payload
    02 event_parser        : StrategyPipeline.event_parser.parse
    03 feature_builder     : StrategyPipeline.feature_builder.build_from_loader
    04 scorer              : StrategyPipeline.scorer.score
    05 selector            : StrategyPipeline.selector.select
    06 strategy_generator  : StrategyPipeline.strategy_generator.generate
    07 build_eval_output   : build_eval_output（含 LLM 内容生成 + 五维装配 + 节点输出）
    08 write               : 写盘 + 写 trace
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.eval_hot_events import (  # noqa: E402
    build_eval_output,
    hot_event_to_pipeline_payload,
    load_hot_events,
    select_hot_event,
)
from influence_strategy.llm_client import OpenAICompatibleLLMClient  # noqa: E402
from influence_strategy.pipeline import StrategyPipeline  # noqa: E402


USE_LLM = os.environ.get("USE_LLM", "0").lower() in ("1", "true", "yes")
EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
MAX_SELECTED_NODES = 5
DISPATCH_PLATFORM = "facebook_simulated"
ALLOWED_PLATFORMS = [
    "facebook_simulated",
    "instagram_simulated",
    "twitter_simulated",
]
TEST_EVENT_ID = "hot_event_001"  # 与 test_strategy_perf.py 保持一致


# ---- 阶段定义（中文标签 + 占比宽度固定） ----
PHASE_LABELS = {
    "pipeline_payload":   "01 pipeline_payload    (事件→Pipeline 输入)",
    "event_parser":       "02 event_parser         (事件归一化)",
    "feature_builder":    "03 feature_builder      (特征构建)",
    "scorer":             "04 scorer               (打分)",
    "selector":           "05 selector             (选人)",
    "strategy_generator": "06 strategy_generator   (策略生成)",
    "build_eval_output":  "07 build_eval_output    (LLM内容+五维装配+节点输出)",
    "write":              "08 write                (结果落盘+trace)",
}


def _t(name: str):
    """阶段计时装饰器：with _t("xxx") as bucket: bucket["secs"] = ...; bucket["result"] = ...。"""
    bucket: dict = {"secs": 0.0, "result": None}

    class _C:
        def __enter__(self_inner):
            self_inner._t0 = time.perf_counter()
            return bucket

        def __exit__(self_inner, *exc):
            bucket["secs"] = time.perf_counter() - self_inner._t0
            return False

    return _C()


def _measure_once() -> dict[str, float]:
    """跑一条事件（hot_event_001），返回各阶段耗时（秒）。"""
    client = (
        OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT) if USE_LLM else None
    )

    events = load_hot_events(EVENTS_PATH)
    hot_event = select_hot_event(events, event_id=TEST_EVENT_ID)

    pipeline = StrategyPipeline(product_name="abc_reading")
    timings: dict[str, float] = {}

    with _t("x") as b:
        pipeline_payload = hot_event_to_pipeline_payload(
            hot_event,
            max_selected_nodes=MAX_SELECTED_NODES,
            risk_level=None,
            campaign_window_hours=24,
            max_frequency_per_day=3,
            allowed_platforms=ALLOWED_PLATFORMS,
        )
    timings["pipeline_payload"] = b["secs"]

    with _t("x") as b:
        parsed = pipeline.event_parser.parse(
            pipeline_payload,
            workspace_root=str(PROJECT_ROOT),
            use_llm=USE_LLM,
            llm_client=client,
        )
    timings["event_parser"] = b["secs"]

    with _t("x") as b:
        feature = pipeline.feature_builder.build_from_loader(
            workspace_root=str(PROJECT_ROOT),
            event=parsed,
            use_llm=USE_LLM,
            llm_client=client,
        )
    timings["feature_builder"] = b["secs"]

    with _t("x") as b:
        score = pipeline.scorer.score(feature)
    timings["scorer"] = b["secs"]

    with _t("x") as b:
        selection = pipeline.selector.select(
            score,
            workspace_root=str(PROJECT_ROOT),
            use_llm=USE_LLM,
            llm_client=client,
            target_user_id=None,
            dispatch_platform=DISPATCH_PLATFORM,
        )
    timings["selector"] = b["secs"]

    with _t("x") as b:
        strategy = pipeline.strategy_generator.generate(selection)
    timings["strategy_generator"] = b["secs"]

    # build_eval_output 是包含 LLM 内容生成 + 节点输出 + 五维装配的一整个步骤；
    # 一次端到端调用，内部已对每个数字人调一次 LLM（当 use_llm=True）。
    with _t("x") as b:
        output_payload, content_llm_meta = build_eval_output(
            hot_event=hot_event,
            strategy_result=strategy,
            workspace_root=str(PROJECT_ROOT),
            use_llm=USE_LLM,
            llm_client=client,
        )
    timings["build_eval_output"] = b["secs"]

    # 落盘（写结果 JSON）+ 写 trace 8 个文件
    with _t("x") as b:
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="measure_per_module_"))
        output_path = out_dir / f"{strategy.event.event_id}_strategy_output.json"
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    timings["write"] = b["secs"]

    return timings


def _print_table(timings: dict[str, float], event_id: str) -> None:
    total = sum(timings.values())
    print()
    print(f"按模块耗时（事件: {event_id}, use_llm={USE_LLM}）")
    print("=" * 88)
    print(f"  {'阶段':<48} {'耗时(秒)':>12} {'占比':>8}")
    print("-" * 88)
    for key in [
        "pipeline_payload", "event_parser", "feature_builder",
        "scorer", "selector", "strategy_generator",
        "build_eval_output", "write",
    ]:
        label = PHASE_LABELS[key]
        dur = timings.get(key, 0.0)
        pct = (dur / total * 100) if total > 0 else 0.0
        print(f"  {label:<48} {dur:>12.3f} {pct:>7.1f}%")
    print("-" * 88)
    print(f"  {'总计':<48} {total:>12.3f} {'100.0%':>8}")
    print()


if __name__ == "__main__":
    print(
        f"[measure_per_module] START pid={os.getpid()} use_llm={USE_LLM} "
        f"event={TEST_EVENT_ID}",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.perf_counter()
    timings = _measure_once()
    wall = time.perf_counter() - t0
    _print_table(timings, event_id=TEST_EVENT_ID)
    print(
        f"[measure_per_module] DONE  wall_total={wall:.3f}s  "
        f"sum_phases={sum(timings.values()):.3f}s",
        file=sys.stderr,
        flush=True,
    )
