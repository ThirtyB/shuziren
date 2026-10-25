"""最小消融实验：4 个 LLM 调用点的单点开关影响。

设计要点
--------
项目本身只有一个全局 ``use_llm`` 开关（``RunOptions.use_llm``），一次会关掉 4 个
调用点，做不到单点消融。本脚本**不修改任何项目代码**，而是绕过
``StrategyPipeline.run_with_artifacts``（它把同一个 ``use_llm`` 传给全部 4 级），
改为手工逐级调用 —— 因为 ``pipeline.py`` 的每一级都把 ``use_llm`` / ``llm_client``
作为独立参数接收，因此可以给每一级传不同的值。

四个调用点：
  ① ``RuleBasedEventParser.parse(use_llm=...)``          —— 事件解析
  ② ``FeatureBuilder.build_from_loader(use_llm=...)``    —— 特征增强 / 语义打分
  ③ ``Selector.select(use_llm=...)``                     —— 选人重排
  ④ ``build_eval_output(use_llm=...)``                   —— 内容文案生成
  （``Scorer.score`` 与 ``StrategyGenerator.generate`` 内部无 LLM，恒等。）

场景
----
  S0  全部开启（基准）
  S4  仅关闭 ④ 内容生成
  S3  仅关闭 ③ 选人重排
  S1  仅关闭 ① 事件解析
  S2  仅关闭 ② 特征增强
  S5  全部关闭（对照）

用法
----
    # 跑实验（结果落到 outputs/ablation/raw/）
    .\\.venv\\Scripts\\python.exe -u tests\\ablation_llm.py --scenarios S0 S4 --reps 3

    # 只做比对，不重跑
    .\\.venv\\Scripts\\python.exe -u tests\\ablation_llm.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.eval_hot_events import (  # noqa: E402
    build_eval_output,
    hot_event_to_pipeline_payload,
    load_hot_events,
)
from influence_strategy.llm_client import OpenAICompatibleLLMClient  # noqa: E402
from influence_strategy.pipeline import StrategyPipeline  # noqa: E402


EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
OUT_ROOT = PROJECT_ROOT / "outputs" / "ablation"
RAW_DIR = OUT_ROOT / "raw"

DEFAULT_EVENT_ID = "hot_event_001"
DISPATCH_PLATFORM = "facebook_simulated"
MAX_SELECTED_NODES = 5
CAMPAIGN_WINDOW_HOURS = 24
MAX_FREQUENCY_PER_DAY = 3

# True = 该级开启 LLM；False = 该级走规则兜底
SCENARIOS: dict[str, dict[str, bool]] = {
    "S0": {"parser": True, "feature": True, "selector": True, "content": True},
    "S1": {"parser": False, "feature": True, "selector": True, "content": True},
    "S2": {"parser": True, "feature": False, "selector": True, "content": True},
    "S3": {"parser": True, "feature": True, "selector": False, "content": True},
    "S4": {"parser": True, "feature": True, "selector": True, "content": False},
    "S5": {"parser": False, "feature": False, "selector": False, "content": False},
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 实验执行
# ---------------------------------------------------------------------------


def _run_once(
    *,
    scenario: str,
    switches: dict[str, bool],
    hot_event: dict[str, Any],
    client: Any,
    workspace_root: Path,
) -> dict[str, Any]:
    """按给定开关跑一条事件，返回可序列化的记录。"""
    payload_input = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=MAX_SELECTED_NODES,
        risk_level=None,
        campaign_window_hours=CAMPAIGN_WINDOW_HOURS,
        max_frequency_per_day=MAX_FREQUENCY_PER_DAY,
        allowed_platforms=None,
    )

    pipeline = StrategyPipeline(product_name="abc_reading")
    started = time.perf_counter()

    # ① 事件解析 —— 只有这一级决定 extracted_keywords / risk_level / 权重
    event = pipeline.event_parser.parse(
        payload_input,
        workspace_root=workspace_root,
        use_llm=switches["parser"],
        llm_client=client,
    )
    # ② 特征增强 —— 决定 llm_feature_score / event_semantic_relevance_score 等
    feature_result = pipeline.feature_builder.build_from_loader(
        workspace_root=workspace_root,
        event=event,
        profile_limit=None,
        use_llm=switches["feature"],
        llm_client=client,
    )
    # Scorer 内部无 LLM 调用
    score_result = pipeline.scorer.score(feature_result)
    # ③ 选人重排 —— 改 safe_pool 顺序 / 角色映射 / 首选角色
    selection_result = pipeline.selector.select(
        score_result,
        workspace_root=workspace_root,
        use_llm=switches["selector"],
        llm_client=client,
        target_user_id=None,
        dispatch_platform=DISPATCH_PLATFORM,
    )
    # StrategyGenerator 内部无 LLM 调用
    strategy_result = pipeline.strategy_generator.generate(selection_result)
    # ④ 内容生成
    output_payload, content_llm_meta = build_eval_output(
        hot_event=hot_event,
        strategy_result=strategy_result,
        workspace_root=workspace_root,
        use_llm=switches["content"],
        llm_client=client,
    )
    elapsed = time.perf_counter() - started

    five = output_payload["五维调度策略"]

    return {
        "scenario": scenario,
        "switches": switches,
        "event_id": str(event.event_id),
        "elapsed_sec": round(elapsed, 3),
        # ---- 开关生效性自检 ----
        "selfcheck": {
            "parser_name": event.parser_name,
            "llm_feature_used_count": sum(
                1 for node in feature_result.node_features if node.llm_feature_used
            ),
            "content_llm_used": bool(content_llm_meta.get("used")),
            "content_llm_fallback_reason": content_llm_meta.get("fallback_reason"),
            "content_llm_provider": content_llm_meta.get("provider"),
            "content_llm_model": content_llm_meta.get("model"),
        },
        # ---- 风险信息（最终 JSON 里没有，必须从内部对象取）----
        "risk": {
            "event_risk_level": event.constraints.risk_level,
            "strategy_risk_control": strategy_result.strategy.risk_control.model_dump(
                mode="json"
            ),
            "eligible_count": sum(1 for node in score_result.node_scores if node.eligible),
            "manual_review_count": sum(
                1 for node in score_result.node_scores if node.manual_review_required
            ),
            "risk_level_histogram": dict(
                Counter(node.risk_level for node in score_result.node_scores)
            ),
        },
        # ---- 选中节点明细（角色 / 平台 / 阶段 / 频率 / 风险）----
        # 注意用 strategy_result.selected_nodes（StrategyNodePlan）而不是
        # selection_result.selected_nodes（SelectedNode）：后者没有
        # frequency_per_day / timing_window，那两个字段是 StrategyGenerator 补上的。
        "selected_nodes": [
            {
                "user_id": node.user_id,
                "selected_role": node.selected_role,
                "platform": node.platform,
                "dispatch_stage": node.dispatch_stage,
                "frequency_per_day": node.frequency_per_day,
                "timing_window": node.timing_window,
                "final_score": round(node.final_score, 6),
                "risk_level": node.risk_level,
                "risk_flags": list(node.risk_flags),
                "manual_review_required": node.manual_review_required,
            }
            for node in strategy_result.selected_nodes
        ],
        # ---- 最终业务 JSON（selected ids / role / platform / time / frequency / content）----
        "five_dimension": {
            "selected_ids": five["目标对象"]["选取数字人id组"],
            "role_plan": five["目标对象"]["数字人分工"],
            "time_plan": five["时间"]["数字人时间安排"],
            "frequency_plan": five["频率"]["数字人频率"],
            "platform_mode": five["平台"]["平台模式"],
            "platform_plan": five["平台"]["数字人平台分发"],
            "content_items": five["内容"]["数字人发帖内容"],
            "actions": five["内容"]["动作清单"],
            "content_llm_diag": five["内容"].get("内容生成诊断"),
        },
    }


def run_experiment(
    *,
    scenarios: list[str],
    reps: int,
    event_id: str,
    workspace_root: Path,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    events = load_hot_events(EVENTS_PATH)
    hot_event = next(
        (e for e in events if str(e.get("event_id")) == event_id), None
    )
    if hot_event is None:
        raise SystemExit(f"事件不存在: {event_id}")

    client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
    if client is None:
        raise SystemExit("LLM 未配置（.env 缺少 key），无法做 S0 基准")

    log(f"[ablation] client = {client.describe()}")
    log(f"[ablation] event   = {event_id}, scenarios = {scenarios}, reps = {reps}")

    for scenario in scenarios:
        switches = SCENARIOS[scenario]
        for rep in range(1, reps + 1):
            path = RAW_DIR / f"{scenario}_{event_id}_run{rep}.json"
            if path.exists():
                log(f"[ablation] {path.name} 已存在，跳过")
                continue

            log(f"[ablation] >>> {scenario} run{rep} 开始 switches={switches}")
            try:
                record = _run_once(
                    scenario=scenario,
                    switches=switches,
                    hot_event=hot_event,
                    client=client,
                    workspace_root=workspace_root,
                )
                record["rep"] = rep
                path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                log(
                    f"[ablation] <<< {scenario} run{rep} 完成 "
                    f"{record['elapsed_sec']}s "
                    f"ids={record['five_dimension']['selected_ids']} "
                    f"content_llm_used={record['selfcheck']['content_llm_used']}"
                )
            except Exception as exc:  # noqa: BLE001 - 单次失败不终止整个实验
                log(f"[ablation] !!! {scenario} run{rep} 失败: {type(exc).__name__}: {exc}")
                path.write_text(
                    json.dumps(
                        {"scenario": scenario, "rep": rep, "error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )


# ---------------------------------------------------------------------------
# 配对（受控）实验：冻结上游，只切换一个开关
# ---------------------------------------------------------------------------
#
# 为什么必须做配对：
#   `temperature=0.4` 且无 seed 使得 **S0 自己 3 次之间就有 9/10 个维度抖动**
#   （见 compare_report.md 第 1 节）。在 S0 自身都不稳定的情况下，
#   拿"S0 的 3 次"去比"S4 的 3 次"，差异无法归因给 ④ —— 上游 ①②③ 的抖动
#   会把结论淹没。
#
# 配对做法：把上游产物当"冻结快照"复用，只改一个 use_llm：
#   ④ 配对：同一个 strategy_result → build_eval_output(use_llm=True/False)
#   ③ 配对：同一个 score_result    → selector.select(use_llm=True/False)
# 这样任何差异都 100% 只可能来自被切换的那一级。


class _RecordingClient:
    """代理真实 client，顺带记录每次 LLM 调用的 prompt/响应规模。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def describe(self) -> dict[str, str]:
        return self._inner.describe()

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> Any:
        response = self._inner.generate_json(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        if isinstance(response, dict):
            keys = list(response.keys())
            try:
                size = len(json.dumps(response, ensure_ascii=False))
            except (TypeError, ValueError):
                size = -1
        else:
            keys = [type(response).__name__]
            size = -1
        self.calls.append(
            {
                "system_len": len(system_prompt),
                "user_len": len(user_prompt),
                "response_keys": keys,
                "response_chars": size,
            }
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _deep_diff(a: Any, b: Any, path: str = "") -> list[str]:
    """递归比较，返回所有不一致的路径。"""
    diffs: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                diffs.append(f"{path}.{key}  <仅一侧存在>")
            else:
                diffs.extend(_deep_diff(a[key], b[key], f"{path}.{key}"))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}  <长度 {len(a)} → {len(b)}>")
            return diffs
        for index, (x, y) in enumerate(zip(a, b)):
            diffs.extend(_deep_diff(x, y, f"{path}[{index}]"))
        return diffs
    if a != b:
        diffs.append(path)
    return diffs


def _summarize_diffs(diffs: list[str], depth: int = 2) -> dict[str, int]:
    """把逐叶差异按前 N 层路径聚合计数。"""
    counter: Counter[str] = Counter()
    for item in diffs:
        head = item.split("  <")[0].strip(".")
        parts = head.split(".")
        counter[".".join(parts[:depth])] += 1
    return dict(counter.most_common())


def _node_signature(nodes: Any) -> list[tuple[Any, ...]]:
    """节点指纹。用 getattr 兜底：SelectedNode 没有 frequency_per_day，
    该字段由 StrategyGenerator 加到 StrategyNodePlan 上。"""
    return [
        (
            n.user_id,
            n.selected_role,
            getattr(n, "platform", ""),
            n.dispatch_stage,
            getattr(n, "frequency_per_day", None),
            round(n.final_score, 6),
            n.risk_level,
            tuple(n.risk_flags),
            n.manual_review_required,
        )
        for n in nodes
    ]


def run_paired(*, event_id: str, workspace_root: Path) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    events = load_hot_events(EVENTS_PATH)
    hot_event = next((e for e in events if str(e.get("event_id")) == event_id), None)
    if hot_event is None:
        raise SystemExit(f"事件不存在: {event_id}")

    real_client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
    if real_client is None:
        raise SystemExit("LLM 未配置，无法做配对实验")
    client = _RecordingClient(real_client)

    payload_input = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=MAX_SELECTED_NODES,
        risk_level=None,
        campaign_window_hours=CAMPAIGN_WINDOW_HOURS,
        max_frequency_per_day=MAX_FREQUENCY_PER_DAY,
        allowed_platforms=None,
    )
    pipeline = StrategyPipeline(product_name="abc_reading")

    log(f"[paired] ① 事件解析（LLM 开）…")
    event = pipeline.event_parser.parse(
        payload_input, workspace_root=workspace_root, use_llm=True, llm_client=client
    )
    log(f"[paired] ② 特征增强（LLM 开）…")
    feature_result = pipeline.feature_builder.build_from_loader(
        workspace_root=workspace_root,
        event=event,
        profile_limit=None,
        use_llm=True,
        llm_client=client,
    )
    llm_feature_used = sum(1 for n in feature_result.node_features if n.llm_feature_used)

    # 上游冻结：score_result 只算一次，③ 的两种开关都基于它
    score_result = pipeline.scorer.score(feature_result)

    log(f"[paired] ③ 选人重排 配对（同一 score_result）…")
    sel_on = pipeline.selector.select(
        score_result,
        workspace_root=workspace_root,
        use_llm=True,
        llm_client=client,
        target_user_id=None,
        dispatch_platform=DISPATCH_PLATFORM,
    )
    sel_off = pipeline.selector.select(
        score_result,
        workspace_root=workspace_root,
        use_llm=False,
        llm_client=client,
        target_user_id=None,
        dispatch_platform=DISPATCH_PLATFORM,
    )

    # ④ 配对：都基于同一个 strategy_result（取 ③ 开启时的产物）
    # 同时把 ③ 的两种选人结果各推进到 StrategyNodePlan，便于看"最终计划"层面的差异
    strat_on = pipeline.strategy_generator.generate(sel_on)
    strat_off = pipeline.strategy_generator.generate(sel_off)
    strategy_result = strat_on

    log(f"[paired] ④ 内容生成 配对（同一 strategy_result）…")
    payload_on, meta_on = build_eval_output(
        hot_event=hot_event,
        strategy_result=strategy_result,
        workspace_root=workspace_root,
        use_llm=True,
        llm_client=client,
    )
    payload_off, meta_off = build_eval_output(
        hot_event=hot_event,
        strategy_result=strategy_result,
        workspace_root=workspace_root,
        use_llm=False,
        llm_client=client,
    )
    # 反向顺序再跑一次，排除"第一次调用有副作用"的可能
    payload_off_2, _ = build_eval_output(
        hot_event=hot_event,
        strategy_result=strategy_result,
        workspace_root=workspace_root,
        use_llm=False,
        llm_client=client,
    )

    content_diffs = _deep_diff(payload_on, payload_off)
    content_diffs_reverse = _deep_diff(payload_off, payload_off_2)

    sel_diffs = _deep_diff(
        {"nodes": _node_signature(strat_on.selected_nodes)},
        {"nodes": _node_signature(strat_off.selected_nodes)},
    )
    sel_summary_diffs = _deep_diff(
        sel_on.summary.model_dump(mode="json"), sel_off.summary.model_dump(mode="json")
    )

    lines: list[str] = []
    out = lines.append
    out("# LLM 消融 · 配对（受控）实验报告")
    out("")
    out(f"事件：`{event_id}` ｜ 分发平台：`{DISPATCH_PLATFORM}`")
    out("")
    out("## 0. 上游 LLM 调用记录")
    out("")
    out("| 次序 | system字符 | user字符 | 响应顶层key | 响应字符数 |")
    out("| --- | --- | --- | --- | --- |")
    for i, call in enumerate(client.calls, start=1):
        out(
            f"| {i} | {call['system_len']} | {call['user_len']} | "
            f"{call['response_keys']} | {call['response_chars']} |"
        )
    out("")
    out(f"- `llm_feature_used` 节点数：**{llm_feature_used}**"
        f"（0 表示 ② 的增强结果一个都没落到节点上）")
    out("")

    out("## 1. ④ 内容生成 LLM —— 配对结果")
    out("")
    out("同一个 `strategy_result`，只切换 `build_eval_output(use_llm=...)`：")
    out("")
    out(f"- `content_llm_used`：开=**{bool(meta_on.get('used'))}** / 关=**{bool(meta_off.get('used'))}**")
    out(f"- `fallback_reason`：开=`{meta_on.get('fallback_reason')}` / 关=`{meta_off.get('fallback_reason')}`")
    out(f"- 差异路径总数：**{len(content_diffs)}**")
    out("")
    out("按路径前缀聚合：")
    out("")
    out("| 路径前缀 | 差异叶子数 |")
    out("| --- | --- |")
    for prefix, count in _summarize_diffs(content_diffs).items():
        out(f"| `{prefix}` | {count} |")
    out("")
    out("**逐条差异明细（最多列 40 条）：**")
    out("")
    out("```")
    for item in content_diffs[:40]:
        out(item)
    if len(content_diffs) > 40:
        out(f"... 另有 {len(content_diffs) - 40} 条")
    out("```")
    out("")
    out(f"反向顺序自检（两次 `use_llm=False` 互比）差异数：**{len(content_diffs_reverse)}** "
        f"（应为 0，证明 ④ 关闭时是确定性的）")
    out("")

    out("## 2. ③ 选人重排 LLM —— 配对结果")
    out("")
    out("同一个 `score_result`，只切换 `selector.select(use_llm=...)`：")
    out("")
    out(f"- 选中 id（开）：{sel_on.selected_nodes and [n.user_id for n in sel_on.selected_nodes]}")
    out(f"- 选中 id（关）：{[n.user_id for n in sel_off.selected_nodes]}")
    out(f"- id 集合是否相同：**{set(n.user_id for n in sel_on.selected_nodes) == set(n.user_id for n in sel_off.selected_nodes)}**")
    out(f"- id 顺序是否相同：**{[n.user_id for n in sel_on.selected_nodes] == [n.user_id for n in sel_off.selected_nodes]}**")
    out(f"- 节点签名差异数：**{len(sel_diffs)}**")
    out(f"- summary 差异数：**{len(sel_summary_diffs)}**")
    out("")
    out("| 数字人id | 角色(开) | 角色(关) | 阶段(开) | 阶段(关) |")
    out("| --- | --- | --- | --- | --- |")
    for on_node, off_node in zip(sel_on.selected_nodes, sel_off.selected_nodes):
        out(
            f"| {on_node.user_id} | {on_node.selected_role} | {off_node.selected_role} | "
            f"{on_node.dispatch_stage} | {off_node.dispatch_stage} |"
        )
    out("")
    out(f"- summary（开）：`{json.dumps(sel_on.summary.model_dump(mode='json'), ensure_ascii=False)}`")
    out(f"- summary（关）：`{json.dumps(sel_off.summary.model_dump(mode='json'), ensure_ascii=False)}`")
    out("")

    report = "\n".join(lines)
    (OUT_ROOT / "paired_report.md").write_text(report, encoding="utf-8")

    raw = {
        "event_id": event_id,
        "llm_calls": client.calls,
        "llm_feature_used_count": llm_feature_used,
        "content_pair": {
            "meta_on": meta_on,
            "meta_off": meta_off,
            "diff_count": len(content_diffs),
            "diff_prefixes": _summarize_diffs(content_diffs),
            "diff_detail": content_diffs,
            "reverse_self_check_diff_count": len(content_diffs_reverse),
        },
        "selector_pair": {
            "ids_on": [n.user_id for n in sel_on.selected_nodes],
            "ids_off": [n.user_id for n in sel_off.selected_nodes],
            "same_id_set": set(n.user_id for n in sel_on.selected_nodes)
            == set(n.user_id for n in sel_off.selected_nodes),
            "same_id_order": [n.user_id for n in sel_on.selected_nodes]
            == [n.user_id for n in sel_off.selected_nodes],
            "node_diff_count": len(sel_diffs),
            "summary_diff_count": len(sel_summary_diffs),
            "summary_on": sel_on.summary.model_dump(mode="json"),
            "summary_off": sel_off.summary.model_dump(mode="json"),
        },
    }
    (OUT_ROOT / "paired_raw.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report)
    print(f"\n[paired] 报告已写入 {OUT_ROOT / 'paired_report.md'}")


# ---------------------------------------------------------------------------
# 比对
# ---------------------------------------------------------------------------


def _role_map(record: dict[str, Any]) -> dict[str, str]:
    return {
        item["数字人id"]: item["节点角色"]
        for item in record["five_dimension"]["role_plan"]
    }


def _platform_map(record: dict[str, Any]) -> dict[str, str]:
    return {
        item["数字人id"]: item["分发平台"]
        for item in record["five_dimension"]["platform_plan"]
    }


def _texts(record: dict[str, Any]) -> list[str]:
    return [item["发帖内容"] for item in record["five_dimension"]["content_items"]]


def _dimension_states(a: dict[str, Any], b: dict[str, Any]) -> dict[str, bool]:
    """返回各维度是否“完全一致”。True = 一致。"""
    fa, fb = a["five_dimension"], b["five_dimension"]
    ids_a, ids_b = fa["selected_ids"], fb["selected_ids"]

    risk_a = {
        "event": a["risk"]["event_risk_level"],
        "strategy": a["risk"]["strategy_risk_control"],
        "eligible": a["risk"]["eligible_count"],
        "review": a["risk"]["manual_review_count"],
        "hist": a["risk"]["risk_level_histogram"],
        "nodes": [
            (n["user_id"], n["risk_level"], tuple(n["risk_flags"]), n["manual_review_required"])
            for n in a["selected_nodes"]
        ],
    }
    risk_b = {
        "event": b["risk"]["event_risk_level"],
        "strategy": b["risk"]["strategy_risk_control"],
        "eligible": b["risk"]["eligible_count"],
        "review": b["risk"]["manual_review_count"],
        "hist": b["risk"]["risk_level_histogram"],
        "nodes": [
            (n["user_id"], n["risk_level"], tuple(n["risk_flags"]), n["manual_review_required"])
            for n in b["selected_nodes"]
        ],
    }

    actions_a = [
        (x.get("动作编号"), x.get("动作类型"), x.get("执行平台"), x.get("目标帖子"), x.get("执行主体"))
        for x in fa["actions"]
    ]
    actions_b = [
        (x.get("动作编号"), x.get("动作类型"), x.get("执行平台"), x.get("目标帖子"), x.get("执行主体"))
        for x in fb["actions"]
    ]

    return {
        "选中id_集合": set(ids_a) == set(ids_b),
        "选中id_顺序": ids_a == ids_b,
        "角色role": _role_map(a) == _role_map(b),
        "平台platform": _platform_map(a) == _platform_map(b),
        "平台模式": fa["platform_mode"] == fb["platform_mode"],
        "时间time": fa["time_plan"] == fb["time_plan"],
        "频率frequency": fa["frequency_plan"] == fb["frequency_plan"],
        "风险risk": risk_a == risk_b,
        "动作清单actions": actions_a == actions_b,
        "发帖内容content": _texts(a) == _texts(b),
    }


def _load_records() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if not RAW_DIR.exists():
        return grouped
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "error" in data:
            continue
        grouped.setdefault(data["scenario"], []).append(data)
    for bucket in grouped.values():
        bucket.sort(key=lambda r: r["rep"])
    return grouped


def compare(baseline: str = "S0") -> None:
    grouped = _load_records()
    if not grouped:
        raise SystemExit(f"没有原始记录（{RAW_DIR}），请先运行实验")
    if baseline not in grouped:
        fallback = sorted(grouped)[0]
        print(f"[ablation] 警告：基准场景 {baseline} 尚无记录，改用 {fallback}")
        baseline = fallback

    base = grouped[baseline]
    dimensions = list(_dimension_states(base[0], base[0]).keys())

    lines: list[str] = []
    out = lines.append

    out("# LLM 消融实验比对报告")
    out("")
    out(f"基准场景：`{baseline}`（{len(base)} 次）")
    out("")
    out("## 0. 各场景开关生效性自检")
    out("")
    out("| 场景 | 次数 | 平均耗时(s) | parser_name | llm_feature_used | content_llm_used | fallback_reason |")
    out("| --- | --- | --- | --- | --- | --- | --- |")
    for scenario in sorted(grouped):
        bucket = grouped[scenario]
        avg = sum(r["elapsed_sec"] for r in bucket) / len(bucket)
        sc = bucket[0]["selfcheck"]
        out(
            f"| {scenario} | {len(bucket)} | {avg:.1f} | {sc['parser_name']} | "
            f"{sc['llm_feature_used_count']} | {sc['content_llm_used']} | "
            f"{sc['content_llm_fallback_reason']} |"
        )
    out("")

    # 同场景内部抖动基线：同一场景 3 次之间是否一致
    out("## 1. 同场景内部抖动（非确定性基线）")
    out("")
    out("| 场景 | " + " | ".join(dimensions) + " |")
    out("| --- | " + " | ".join(["---"] * len(dimensions)) + " |")
    for scenario in sorted(grouped):
        bucket = grouped[scenario]
        if len(bucket) < 2:
            out(f"| {scenario} | " + " | ".join(["-"] * len(dimensions)) + " |")
            continue
        cells = []
        for dim in dimensions:
            same = all(
                _dimension_states(bucket[0], bucket[i])[dim]
                for i in range(1, len(bucket))
            )
            cells.append("一致" if same else "**抖动**")
        out(f"| {scenario} | " + " | ".join(cells) + " |")
    out("")

    # 跨场景对比：场景 X 的每次 vs 基准的每次
    out("## 2. 各场景 vs 基准（逐 pair 全比，共 n×m 对）")
    out("")
    out("| 场景 | " + " | ".join(dimensions) + " |")
    out("| --- | " + " | ".join(["---"] * len(dimensions)) + " |")
    for scenario in sorted(grouped):
        if scenario == baseline:
            continue
        bucket = grouped[scenario]
        cells = []
        for dim in dimensions:
            same_pairs = 0
            total_pairs = 0
            for ra in base:
                for rb in bucket:
                    total_pairs += 1
                    if _dimension_states(ra, rb)[dim]:
                        same_pairs += 1
            if same_pairs == total_pairs:
                cells.append("全部一致")
            elif same_pairs == 0:
                cells.append("**全部不同**")
            else:
                cells.append(f"{same_pairs}/{total_pairs} 一致")
        out(f"| {scenario} | " + " | ".join(cells) + " |")
    out("")

    # 明细
    out("## 3. 明细对比")
    out("")
    for scenario in sorted(grouped):
        if scenario == baseline:
            continue
        a, b = base[0], grouped[scenario][0]
        out(f"### {baseline} run1  vs  {scenario} run1")
        out("")
        out(f"- 选中 id（基准）: {a['five_dimension']['selected_ids']}")
        out(f"- 选中 id（{scenario}）: {b['five_dimension']['selected_ids']}")
        out(f"- 角色（基准）  : {_role_map(a)}")
        out(f"- 角色（{scenario}）  : {_role_map(b)}")
        out(f"- 平台（基准）  : {_platform_map(a)}")
        out(f"- 平台（{scenario}）  : {_platform_map(b)}")
        out(f"- 风险（基准）  : event={a['risk']['event_risk_level']}, "
            f"eligible={a['risk']['eligible_count']}, "
            f"review={a['risk']['manual_review_count']}")
        out(f"- 风险（{scenario}）  : event={b['risk']['event_risk_level']}, "
            f"eligible={b['risk']['eligible_count']}, "
            f"review={b['risk']['manual_review_count']}")
        out(f"- 动作数（基准/{scenario}）: {len(a['five_dimension']['actions'])}"
            f" / {len(b['five_dimension']['actions'])}")
        out(f"- 内容生成诊断（基准/{scenario}）: "
            f"{a['five_dimension']['content_llm_diag']} / {b['five_dimension']['content_llm_diag']}")
        ta, tb = _texts(a), _texts(b)
        out(f"- 发帖内容首条（基准）: {ta[0][:80] if ta else ''}...")
        out(f"- 发帖内容首条（{scenario}）: {tb[0][:80] if tb else ''}...")
        out("")

    report = "\n".join(lines)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "compare_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[ablation] 报告已写入 {OUT_ROOT / 'compare_report.md'}")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 调用点最小消融实验")
    parser.add_argument("--scenarios", nargs="*", default=["S0", "S4"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID)
    parser.add_argument("--compare", action="store_true", help="只比对，不跑实验")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="配对受控实验：冻结上游，只切换一个开关（推荐，结果可归因）",
    )
    parser.add_argument("--baseline", default="S0")
    args = parser.parse_args()

    if args.paired:
        run_paired(event_id=args.event_id, workspace_root=PROJECT_ROOT)
        return

    if args.compare:
        compare(baseline=args.baseline)
        return

    unknown = [s for s in args.scenarios if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"未知场景: {unknown}，可选 {sorted(SCENARIOS)}")

    run_experiment(
        scenarios=args.scenarios,
        reps=args.reps,
        event_id=args.event_id,
        workspace_root=PROJECT_ROOT,
    )
    compare(baseline=args.baseline)


if __name__ == "__main__":
    main()
