"""④ 内容生成 LLM 的可观测性与"核心决策不受文案影响"的回归测试。

覆盖第二轮要求的检查项：

* ④ 的耗时 / 成功失败 / fallback 原因 / 实际请求次数被正确记录；
* **核心字段（选中 ID / 角色 / 平台 / 时间 / 频率 / 风险）不随 ④ 开关闭合变化**；
* 文案最终仍能生成；失败时有明确的降级状态（``fallback_reason``）；
* 同步接口 ``build_eval_output`` 的返回结构兼容（仍是 ``(payload, meta)``，
  ``payload`` 顶层 key 不变）。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.eval_hot_events import (
    _build_llm_node_texts,
    build_eval_output,
    get_content_llm_metrics,
    reset_content_llm_metrics,
)
from influence_strategy.llm_client import LLMClientError
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


class _OkContentClient:
    """返回合法文案。"""

    def __init__(self) -> None:
        self.calls = 0

    def describe(self) -> dict[str, str]:
        return {"provider": "fake", "model": "fake-model", "base_url": "http://fake"}

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls += 1
        payload = json.loads(user_prompt)
        # ``_llm_prompt`` 里候选节点的键名是 ``eligible_nodes``，
        # 响应要求 key 形如 ``id<user_id>``（见 constraints 第 5 条）。
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


class _RaisingContentClient(OSError):
    """占位：真正抛异常的 client 在下面按类型定义。"""


def _make_client_raising(exc: BaseException):
    class _Raiser:
        def __init__(self) -> None:
            self.calls = 0

        def describe(self) -> dict[str, str]:
            return {"provider": "fake", "model": "fake-model", "base_url": "http://fake"}

        def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
            self.calls += 1
            raise exc

    return _Raiser()


def _make_node(user_id: str, role: str = "core_publish_node") -> StrategyNodePlan:
    """``final_score`` / ``stability_score`` 取足够高的值以满足 ``_can_connect_llm``。"""
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
        event_id="hot_event_content_metrics",
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
    "event_id": "hot_event_content_metrics",
    "domain": "technology",
    "event_title": "AI chip supply test",
    "event_summary": "High-end chip supply is tightening.",
    "target": "Explain chip supply impact.",
    "opinion_variants": [f"Variant {i}" for i in range(1, 11)],
}


class ContentLLMMetricsTest(unittest.TestCase):
    """④ 的统计与降级路径。"""

    def setUp(self) -> None:
        reset_content_llm_metrics()

    def tearDown(self) -> None:
        reset_content_llm_metrics()

    def _build(self, client, use_llm: bool = True):
        return build_eval_output(
            hot_event=_HOT_EVENT,
            strategy_result=_make_strategy_result(),
            workspace_root=None,
            use_llm=use_llm,
            llm_client=client,
        )

    def test_records_success_with_attempted_and_elapsed(self) -> None:
        client = _OkContentClient()
        payload, meta = self._build(client)

        self.assertTrue(meta["used"], meta)
        self.assertIsNone(meta["fallback_reason"])
        self.assertEqual(client.calls, 1)

        m = get_content_llm_metrics()
        self.assertEqual(m["calls"], 1)
        self.assertEqual(m["success"], 1)
        self.assertEqual(m["failure"], 0)
        self.assertEqual(m["attempted"], 1)
        self.assertIsNone(m["last_call"]["fallback_reason"])
        self.assertTrue(m["last_call"]["attempted"])
        self.assertGreater(m["last_call"]["elapsed_sec"], -1.0)
        self.assertGreater(m["last_call"]["prompt_chars"], 0)

        # 文案确实被 LLM 覆盖
        posts = payload["五维调度策略"]["内容"]["数字人发帖内容"]
        self.assertEqual(len(posts), 2)
        self.assertTrue(posts[0]["发帖内容"].startswith("LLM generated post content"))

    def test_records_llm_disabled_without_request(self) -> None:
        payload, meta = self._build(None, use_llm=False)

        self.assertFalse(meta["used"])
        self.assertEqual(meta["fallback_reason"], "llm_disabled")

        m = get_content_llm_metrics()
        self.assertEqual(m["failure"], 1)
        self.assertEqual(m["attempted"], 0, "关闭时不应发起任何远端请求")
        self.assertEqual(m["fallback_by_reason"].get("llm_disabled"), 1)
        self.assertFalse(m["last_call"]["attempted"])

        # 文案仍能生成（走规则兜底）
        posts = payload["五维调度策略"]["内容"]["数字人发帖内容"]
        self.assertEqual(len(posts), 2)
        self.assertTrue(posts[0]["发帖内容"])

    def test_records_request_failed_with_error_type(self) -> None:
        client = _make_client_raising(OSError("simulated connection reset"))
        payload, meta = self._build(client)

        self.assertFalse(meta["used"])
        self.assertEqual(meta["fallback_reason"], "llm_request_failed:OSError")
        self.assertEqual(client.calls, 1)

        m = get_content_llm_metrics()
        self.assertEqual(m["failure"], 1)
        self.assertEqual(m["attempted"], 1)
        self.assertEqual(m["fallback_by_reason"].get("llm_request_failed:OSError"), 1)
        self.assertEqual(m["last_call"]["error_type"], "OSError")
        self.assertIn("simulated connection reset", m["last_call"]["error_message"])

        # 降级：文案仍存在（规则兜底），不为空
        posts = payload["五维调度策略"]["内容"]["数字人发帖内容"]
        self.assertEqual(len(posts), 2)
        self.assertTrue(all(item["发帖内容"] for item in posts))

    def test_records_llm_client_error(self) -> None:
        client = _make_client_raising(LLMClientError("simulated HTTP 500"))
        _payload, meta = self._build(client)

        self.assertEqual(meta["fallback_reason"], "llm_request_failed:LLMClientError")
        m = get_content_llm_metrics()
        self.assertEqual(m["last_call"]["error_type"], "LLMClientError")

    def test_records_value_error(self) -> None:
        client = _make_client_raising(ValueError("simulated bad payload"))
        _payload, meta = self._build(client)

        self.assertEqual(meta["fallback_reason"], "llm_request_failed:ValueError")
        m = get_content_llm_metrics()
        self.assertEqual(m["last_call"]["error_type"], "ValueError")

    def test_does_not_swallow_unexpected_exception(self) -> None:
        """``RuntimeError`` 不在 ``except`` 列表内，必须原样向上抛。"""

        class _Booster:
            def describe(self) -> dict[str, str]:
                return {"provider": "fake", "model": "fake", "base_url": "http://fake"}

            def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
                raise RuntimeError("unexpected boom")

        with self.assertRaises(RuntimeError):
            self._build(_Booster())


class CoreFieldsUnaffectedByContentLLMTest(unittest.TestCase):
    """**核心不变量**：④ 的开关只影响「内容」，不影响任何核心决策字段。

    这是第二轮"能否把 ④ 挪出实时链路"的前提条件。若这个测试失败，
    说明文案与核心字段存在隐藏耦合，异步化就不再安全。
    """

    def setUp(self) -> None:
        reset_content_llm_metrics()

    def tearDown(self) -> None:
        reset_content_llm_metrics()

    @staticmethod
    def _core_fields(payload: dict) -> dict:
        fd = payload["五维调度策略"]
        return {
            "selected_ids": fd["目标对象"]["选取数字人id组"],
            "roles": {
                item["数字人id"]: item["节点角色"]
                for item in fd["目标对象"]["数字人分工"]
            },
            "platforms": {
                item["数字人id"]: item["分发平台"]
                for item in fd["平台"]["数字人平台分发"]
            },
            "platform_mode": fd["平台"]["平台模式"],
            "time_plan": {
                item["数字人id"]: item["时间阶段"]
                for item in fd["时间"]["数字人时间安排"]
            },
            "frequency_plan": {
                item["数字人id"]: item["发帖频率"] for item in fd["频率"]["数字人频率"]
            },
            "actions": fd["内容"]["动作清单"],
        }

    def test_core_fields_identical_when_content_llm_toggled(self) -> None:
        strategy_result = _make_strategy_result()

        payload_on, meta_on = build_eval_output(
            hot_event=_HOT_EVENT,
            strategy_result=strategy_result,
            workspace_root=None,
            use_llm=True,
            llm_client=_OkContentClient(),
        )
        payload_off, meta_off = build_eval_output(
            hot_event=_HOT_EVENT,
            strategy_result=strategy_result,
            workspace_root=None,
            use_llm=False,
            llm_client=None,
        )

        self.assertTrue(meta_on["used"])
        self.assertFalse(meta_off["used"])

        core_on = self._core_fields(payload_on)
        core_off = self._core_fields(payload_off)

        for key in core_on:
            with self.subTest(field=key):
                self.assertEqual(core_on[key], core_off[key], f"核心字段 {key} 被 ④ 改变了")

    def test_only_content_differs(self) -> None:
        strategy_result = _make_strategy_result()
        payload_on, _ = build_eval_output(
            hot_event=_HOT_EVENT,
            strategy_result=strategy_result,
            workspace_root=None,
            use_llm=True,
            llm_client=_OkContentClient(),
        )
        payload_off, _ = build_eval_output(
            hot_event=_HOT_EVENT,
            strategy_result=strategy_result,
            workspace_root=None,
            use_llm=False,
            llm_client=None,
        )

        fd_on = payload_on["五维调度策略"]
        fd_off = payload_off["五维调度策略"]

        # 只有「内容.数字人发帖内容」与「内容.内容生成诊断」允许不同
        for dimension in ("目标对象", "时间", "频率", "平台"):
            with self.subTest(dimension=dimension):
                self.assertEqual(fd_on[dimension], fd_off[dimension])

        self.assertNotEqual(
            fd_on["内容"]["数字人发帖内容"], fd_off["内容"]["数字人发帖内容"]
        )
        self.assertEqual(len(fd_on["内容"]["动作清单"]), len(fd_off["内容"]["动作清单"]))

    def test_sync_interface_shape_is_compatible(self) -> None:
        """同步接口 ``build_eval_output`` 的返回结构必须保持不变。"""
        payload, meta = build_eval_output(
            hot_event=_HOT_EVENT,
            strategy_result=_make_strategy_result(),
            workspace_root=None,
            use_llm=True,
            llm_client=_OkContentClient(),
        )
        self.assertIsInstance(payload, dict)
        self.assertIsInstance(meta, dict)
        # 用集合比较：中文按 Unicode 码点排序，写死 sorted list 会脆弱。
        self.assertEqual(
            set(payload.keys()), {"事件名称", "五维调度策略", "输出格式版本"}
        )
        self.assertEqual(
            set(payload["五维调度策略"].keys()),
            {"内容", "目标对象", "平台", "时间", "频率"},
        )
        for key in ("requested", "configured", "attempted", "used", "fallback_reason"):
            self.assertIn(key, meta)


class BuildLLMNodeTextsDirectTest(unittest.TestCase):
    """直接测 ``_build_llm_node_texts`` 的降级分支。"""

    def setUp(self) -> None:
        reset_content_llm_metrics()

    def tearDown(self) -> None:
        reset_content_llm_metrics()

    def test_no_eligible_nodes_records_fallback(self) -> None:
        """全部节点都很低分时，不应发起请求，且 fallback 原因明确。"""
        parsed_event = ParsedEvent(
            event_id="evt_low",
            event_title="low score event",
            event_description="desc",
            target_goal="awareness",
            event_type="general_influence_event",
            constraints=EventConstraints(risk_level="low"),
        )
        low_node = StrategyNodePlan(
            user_id="1",
            user_name="low",
            selected_role="core_publish_node",
            dispatch_stage="stage_1_launch",
            final_score=0.05,  # < 0.30 → 不满足 _can_connect_llm
            stability_score=0.01,
            risk_level="low",
            manual_review_required=False,
        )
        strategy_result = StrategyResult(
            event=parsed_event,
            product_context=ProductContext(product_name="abc_reading"),
            selection_summary=SelectionSummary(
                event_id="evt_low",
                event_type="general_influence_event",
                max_selected_nodes=1,
                selected_count=1,
                fallback_count=0,
            ),
            summary=StrategySummary(
                event_id="evt_low",
                event_type="general_influence_event",
                selected_count=1,
                fallback_count=0,
                primary_platform="facebook_simulated",
            ),
            stage_plans=[],
            selected_nodes=[low_node],
            fallback_nodes=[],
            strategy=DispatchStrategy(target_object=["general_public"]),
        )
        overrides, meta = _build_llm_node_texts(
            hot_event=_HOT_EVENT,
            strategy_result=strategy_result,
            platform="facebook_simulated",
            target="t",
            workspace_root=None,
            use_llm=True,
            llm_client=_OkContentClient(),
        )
        self.assertEqual(overrides, {})
        self.assertEqual(meta["fallback_reason"], "no_eligible_nodes")
        m = get_content_llm_metrics()
        self.assertEqual(m["attempted"], 0)
        self.assertEqual(m["fallback_by_reason"].get("no_eligible_nodes"), 1)


if __name__ == "__main__":
    unittest.main()
