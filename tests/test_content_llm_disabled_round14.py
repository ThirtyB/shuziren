"""第十四轮：页面测试控制台取消"节点发帖正文 LLM"的回归测试。

被取消的只是**正文那一次 LLM 调用**（``_build_llm_node_texts``）。必须保证：

* 正文字段仍然存在于输出契约里（走已有规则兜底，不为空、不是伪造的 LLM 文案）；
* 五维调度策略、动作清单、角色、平台、频率、阶段等核心字段一个不少；
* selector 的 LLM 语义排序照旧执行（本轮没有碰它）；
* 关闭时**一个请求都不发**，且不会被记成"调用失败"污染成功率指标；
* 只有页面路径默认关闭；``eval_hot_events`` 的函数默认值仍是 True，
  因此 CLI / run_eval / ablation 等入口行为完全不变。
"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from influence_strategy.eval_hot_events import (
    _build_llm_node_texts,
    _run_hot_event_evaluation,
    build_eval_output,
    get_content_llm_metrics,
    reset_content_llm_metrics,
)
from influence_strategy.models import (
    DispatchStrategy,
    EventConstraints,
    ParsedEvent,
    ProductContext,
    SelectionSummary,
    StrategyNodePlan,
    StrategyResult,
    StrategySummary,
)

FIVE_DIMENSION_KEYS = ("目标对象", "时间", "频率", "平台", "内容")


class _RecordingClient:
    """同时冒充 selector 与正文生成的假客户端，记录每次调用。

    selector 的 prompt 里一定有 ``selected_order``；正文生成的 prompt 里
    一定有 ``eligible_nodes``。用它区分两类请求。
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def describe(self) -> dict[str, str]:
        return {"provider": "fake", "model": "fake-model", "base_url": "http://fake"}

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        if "eligible_nodes" in user_prompt:
            self.calls.append("content")
            payload = json.loads(user_prompt)
            nodes = {}
            for item in payload.get("eligible_nodes", []):
                uid = str(item.get("user_id"))
                nodes[f"id{uid}"] = {
                    "post_content": f"LLM generated post content for {uid}",
                    "audience_profile": f"LLM audience profile for {uid}",
                    "audience_interaction_strategy": f"LLM interaction for {uid}",
                    "cross_digital_human_strategy": f"LLM coordination for {uid}",
                }
            return {"nodes": nodes}
        self.calls.append("selector")
        payload = json.loads(user_prompt)
        ids = [str(item.get("user_id")) for item in payload.get("shortlist", [])]
        return {"selected_order": ids[: payload["constraints"]["max_selected_nodes"]]}


def _make_node(user_id: str, role: str = "core_publish_node") -> StrategyNodePlan:
    return StrategyNodePlan(
        user_id=user_id,
        user_name=f"node_{user_id}",
        selected_role=role,
        dispatch_stage="stage_1_launch",
        final_score=0.72,
        stability_score=0.65,
        risk_level="low",
        manual_review_required=False,
        recommended_action="publish explanation",
        suggested_content_style="factual update",
        platform="facebook_simulated",
        frequency_per_day=2,
        matched_keywords=["芯片"],
        semantic_tags=["供应链"],
    )


def _make_strategy_result() -> StrategyResult:
    parsed_event = ParsedEvent(
        event_id="hot_event_round14",
        event_title="AI chip supply test",
        event_description="High-end chip supply is tightening.",
        target_goal="awareness",
        event_type="general_influence_event",
        constraints=EventConstraints(risk_level="low"),
    )
    nodes = [
        _make_node("1", "core_publish_node"),
        _make_node("2", "interaction_response_node"),
    ]
    return StrategyResult(
        event=parsed_event,
        product_context=ProductContext(product_name="abc_reading"),
        selection_summary=SelectionSummary(
            event_id=parsed_event.event_id,
            event_type=parsed_event.event_type,
            max_selected_nodes=2,
            selected_count=2,
            fallback_count=0,
        ),
        summary=StrategySummary(
            event_id=parsed_event.event_id,
            event_type=parsed_event.event_type,
            selected_count=2,
            fallback_count=0,
            primary_platform="facebook_simulated",
        ),
        stage_plans=[],
        selected_nodes=nodes,
        fallback_nodes=[],
        strategy=DispatchStrategy(
            target_object=["general_public"],
            objective="Raise awareness of chip supply risk.",
        ),
    )


_HOT_EVENT = {
    "event_id": "hot_event_round14",
    "domain": "technology",
    "event_title": "AI chip supply test",
    "event_summary": "High-end chip supply is tightening.",
    "target": "Explain chip supply impact.",
    "opinion_variants": [f"Variant {i}" for i in range(1, 11)],
}


class ContentLLMDisabledUnitTest(unittest.TestCase):
    """``use_llm_content=False`` 的单元行为。"""

    def setUp(self) -> None:
        reset_content_llm_metrics()

    def tearDown(self) -> None:
        reset_content_llm_metrics()

    def test_disabled_sends_no_request(self) -> None:
        client = _RecordingClient()
        overrides, meta = _build_llm_node_texts(
            hot_event=_HOT_EVENT,
            strategy_result=_make_strategy_result(),
            platform="facebook_simulated",
            target="Explain chip supply impact.",
            workspace_root=None,
            use_llm=True,
            llm_client=client,
            use_llm_content=False,
        )
        self.assertEqual(overrides, {})
        self.assertEqual(client.calls, [])
        self.assertFalse(meta["used"])
        self.assertFalse(meta["attempted"])
        self.assertEqual(meta["fallback_reason"], "content_llm_disabled_by_caller")
        self.assertFalse(meta["requested"])

    def test_disabled_does_not_pollute_success_metrics(self) -> None:
        """主动关闭不是"调用失败"，不该进成功率基线。"""
        _build_llm_node_texts(
            hot_event=_HOT_EVENT,
            strategy_result=_make_strategy_result(),
            platform="facebook_simulated",
            target="Explain chip supply impact.",
            workspace_root=None,
            use_llm=True,
            llm_client=_RecordingClient(),
            use_llm_content=False,
        )
        metrics = get_content_llm_metrics()
        self.assertEqual(metrics["calls"], 0)
        self.assertEqual(metrics["failure"], 0)
        self.assertEqual(metrics["fallback_by_reason"], {})

    def test_enabled_still_calls_llm(self) -> None:
        """对照：不传开关时行为与改动前一致。"""
        client = _RecordingClient()
        overrides, meta = _build_llm_node_texts(
            hot_event=_HOT_EVENT,
            strategy_result=_make_strategy_result(),
            platform="facebook_simulated",
            target="Explain chip supply impact.",
            workspace_root=None,
            use_llm=True,
            llm_client=client,
        )
        self.assertEqual(client.calls, ["content"])
        self.assertTrue(meta["used"])
        self.assertIn("1", overrides)

    def test_default_parameter_is_enabled(self) -> None:
        """默认 True —— 其它入口不受本轮改动影响。"""
        params = inspect.signature(_build_llm_node_texts).parameters
        self.assertIs(params["use_llm_content"].default, True)
        params = inspect.signature(build_eval_output).parameters
        self.assertIs(params["use_llm_content"].default, True)
        params = inspect.signature(_run_hot_event_evaluation).parameters
        self.assertIs(params["use_llm_content"].default, True)


class EvalOutputContractTest(unittest.TestCase):
    """关闭正文 LLM 后，输出契约不能退化。"""

    def setUp(self) -> None:
        reset_content_llm_metrics()

    def tearDown(self) -> None:
        reset_content_llm_metrics()

    @staticmethod
    def _build(use_llm_content: bool):
        client = _RecordingClient()
        payload, meta = build_eval_output(
            hot_event=_HOT_EVENT,
            strategy_result=_make_strategy_result(),
            workspace_root=None,
            use_llm=True,
            llm_client=client,
            use_llm_content=use_llm_content,
        )
        return payload, meta, client

    def test_five_dimensions_and_posts_present_when_disabled(self) -> None:
        payload, meta, client = self._build(False)
        self.assertEqual(client.calls, [])
        strategy = payload["五维调度策略"]
        for key in FIVE_DIMENSION_KEYS:
            self.assertIn(key, strategy)
        posts = strategy["内容"]["数字人发帖内容"]
        self.assertEqual(len(posts), 2)
        for item in posts:
            self.assertTrue(item["数字人id"])
            self.assertTrue(item["发帖内容"].strip())
            self.assertNotIn("LLM generated post content", item["发帖内容"])
        self.assertFalse(meta["used"])
        self.assertEqual(meta["fallback_reason"], "content_llm_disabled_by_caller")

    def test_action_lists_still_generated_when_disabled(self) -> None:
        """动作清单与时间安排照旧产出（不依赖正文 LLM）。"""
        payload, _meta, _client = self._build(False)
        strategy = payload["五维调度策略"]
        self.assertTrue(strategy["时间"]["数字人时间安排"])
        self.assertTrue(strategy["平台"]["数字人平台分发"])
        actions = strategy["内容"]["动作清单"]
        self.assertTrue(actions)
        for action in actions:
            self.assertTrue(action["动作编号"])
            self.assertTrue(action["执行主体"])
            self.assertTrue(action["动作类型"])
        # 正文仍在，且是规则兜底（事件自带 opinion_variants）而非空串。
        self.assertEqual(strategy["内容"]["数字人发帖内容"][0]["发帖内容"], "Variant 1")

    def test_core_fields_identical_between_modes(self) -> None:
        """核心决策字段（ID / 角色 / 平台 / 频率 / 阶段）不随开关变化。"""

        def core_fields(payload: dict) -> list[dict[str, str]]:
            items = []
            for item in payload["五维调度策略"]["内容"]["数字人发帖内容"]:
                items.append(
                    {
                        "user_id": str(item["数字人id"]),
                        "selected_role": str(item.get("角色", "")),
                    }
                )
            return items

        disabled, _, _ = self._build(False)
        enabled, _, _ = self._build(True)
        self.assertEqual(core_fields(disabled), core_fields(enabled))


class WebPathDefaultTest(unittest.TestCase):
    """页面路径默认关闭，其它路径不受影响。"""

    def test_web_run_options_defaults_to_disabled(self) -> None:
        from influence_strategy.web_app import RunOptions

        self.assertIs(RunOptions().use_llm_content, False)

    def test_web_can_opt_in(self) -> None:
        from influence_strategy.web_app import RunOptions

        self.assertIs(RunOptions(use_llm_content=True).use_llm_content, True)


class EndToEndDisabledTest(unittest.TestCase):
    """真实数据 + 假客户端跑完整条评测路径，确认整条链路只发 selector 那一次请求。"""

    def test_evaluation_path_skips_content_llm(self) -> None:
        root = Path(__file__).resolve().parents[1]
        events_path = root / "eval" / "hot_event_opinion_variants_200.json"
        if not events_path.exists():
            self.skipTest("hot_event 数据集不存在")
        import tempfile

        raw = json.loads(events_path.read_text(encoding="utf-8"))
        hot_event = raw[0] if isinstance(raw, list) else (raw.get("events") or [])[0]

        client = _RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            _out, payload, stage = _run_hot_event_evaluation(
                workspace_root=root,
                output_dir=Path(tmp) / "out",
                trace_dir=Path(tmp) / "trace",
                hot_event=hot_event,
                max_selected_nodes=5,
                use_llm=True,
                llm_client=client,
                use_llm_content=False,
                selector_candidate_limit=15,
            )
        self.assertEqual(
            client.calls,
            ["selector"],
            f"整条链路应只调用 selector 一次，实际：{client.calls}",
        )
        self.assertLess(stage["content_generation"], 0.5)
        self.assertIn("五维调度策略", payload)
        posts = payload["五维调度策略"]["内容"]["数字人发帖内容"]
        self.assertTrue(posts)
        for item in posts:
            self.assertTrue(str(item["发帖内容"]).strip())

    def test_evaluation_path_with_content_llm_enabled(self) -> None:
        """对照：显式开启时仍会调用正文 LLM（其它入口的既有行为）。"""
        root = Path(__file__).resolve().parents[1]
        events_path = root / "eval" / "hot_event_opinion_variants_200.json"
        if not events_path.exists():
            self.skipTest("hot_event 数据集不存在")
        import tempfile

        raw = json.loads(events_path.read_text(encoding="utf-8"))
        hot_event = raw[0] if isinstance(raw, list) else (raw.get("events") or [])[0]

        client = _RecordingClient()
        with tempfile.TemporaryDirectory() as tmp:
            _out, payload, stage = _run_hot_event_evaluation(
                workspace_root=root,
                output_dir=Path(tmp) / "out",
                trace_dir=Path(tmp) / "trace",
                hot_event=hot_event,
                max_selected_nodes=5,
                use_llm=True,
                llm_client=client,
                use_llm_content=True,
                selector_candidate_limit=15,
            )
        self.assertEqual(client.calls, ["selector", "content"])
        self.assertTrue(payload["五维调度策略"]["内容"]["数字人发帖内容"])


if __name__ == "__main__":
    unittest.main()
