"""第十三轮 · selector「只输出排序 ID」模式的回归测试。

全部使用脚本化的假 LLM 客户端，不发起任何真实请求、不访问云端模型。

覆盖：
1. 模型排序与规则排序不同时，结果保留模型排序；
2. 池外 ID 被丢弃，重复 ID 不会重复选中；
3. 输出不完整时按既有逻辑补齐并标记 ``partial``；
4. 空数组 / 无法解析时标记 ``invalid``，**不伪报为成功重排**；
5. 角色、阶段、理由等下游字段完整，StrategyGenerator 能正常消费；
6. 安全过滤、人工复核与角色平衡照常执行。
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from influence_strategy.models import (
    EventDispatchPreferences,
    NodeScore,
    ParsedEvent,
    ProductContext,
    ScoreResult,
    ScoreSummary,
)
from influence_strategy.prompts import build_selector_id_order_user_prompt
from influence_strategy.selector import DEFAULT_SELECTOR_OUTPUT_MODE, Selector
from influence_strategy.strategy_generator import StrategyGenerator


def _node(user_id: str, role_hint: str = "amplification", final_score: float = 0.5, **over: Any) -> NodeScore:
    # NodeScore.eligible 默认 False，这里默认置 True 再让 over 覆盖，
    # 这样"构造不合格节点"时可以直接传 eligible=False。
    fields: dict[str, Any] = {
        "user_id": user_id,
        "user_name": f"user_{user_id}",
        "event_id": "evt_r13",
        "event_type": "parent_child_reading",
        "role_hint": role_hint,
        "final_score": final_score,
        "topic_match_score": final_score,
        "risk_level": "low",
        "eligible": True,
    }
    fields.update(over)
    return NodeScore(**fields)


def _event() -> ParsedEvent:
    return ParsedEvent(
        event_id="evt_r13",
        event_title="亲子阅读与英语启蒙",
        event_description="围绕亲子阅读与英语启蒙做一次传播活动",
        target_goal="engagement",
        event_type="parent_child_reading",
        target_audience=["parent_child"],
        target_roles=["core_publish_node", "interaction_response_node", "amplification_node"],
        dispatch_preferences=EventDispatchPreferences(candidate_pool_size=60, rerank_top_k=30),
    )


def _score_result(nodes: list[NodeScore]) -> ScoreResult:
    return ScoreResult(
        event=_event(),
        product_context=ProductContext(product_name="abc_reading"),
        summary=ScoreSummary(
            event_id="evt_r13",
            event_type="parent_child_reading",
            node_count=len(nodes),
            eligible_count=len(nodes),
            manual_review_count=0,
            high_priority_count=0,
            avg_final_score=0.0,
            avg_risk_score=0.0,
        ),
        node_scores=nodes,
    )


class _ScriptedLLM:
    def __init__(self, response: Any, recorder: list[str] | None = None) -> None:
        self.response = response
        self.recorder = recorder

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: Any) -> Any:
        if self.recorder is not None:
            self.recorder.append(user_prompt)
        return self.response


class _Pool:
    """40 个合格节点（分数递减）+ 20 个不合格节点，模拟生产候选池。"""

    def __init__(self) -> None:
        self.nodes = [_node(f"{1000 + i}", final_score=0.9 - i * 0.001) for i in range(40)]
        self.nodes += [
            _node(f"{2000 + i}", final_score=0.5, eligible=False) for i in range(20)
        ]
        # 两个人工复核节点：分数高但必须被排除出主选。
        self.nodes += [
            _node(f"{3000 + i}", final_score=0.95, manual_review_required=True) for i in range(2)
        ]


class ModelOrderAppliedTest(unittest.TestCase):
    """A3.1 / A3.3：模型排序优先，重复与池外 ID 不会污染结果。"""

    def setUp(self) -> None:
        self.selector = Selector()
        self.pool = _Pool()

    def _select(self, response: Any, recorder: list[str] | None = None) -> Any:
        return self.selector.select(
            _score_result(self.pool.nodes),
            use_llm=True,
            llm_client=_ScriptedLLM(response, recorder),
            candidate_limit=15,
        )

    def test_model_order_beats_rule_order(self) -> None:
        """模型给出的顺序（倒序）必须真正在结果里生效。"""
        wanted = [f"{1000 + i}" for i in range(9, -1, -1)]
        result = self._select({"selected_order": wanted})
        self.assertEqual([n.user_id for n in result.selected_nodes], wanted)

    def test_out_of_pool_ids_are_dropped(self) -> None:
        wanted = ["999999"] + [f"{1000 + i}" for i in range(9)]
        result = self._select({"selected_order": wanted})
        self.assertNotIn("999999", [n.user_id for n in result.selected_nodes])
        self.assertEqual(result.summary.selected_count, 10)

    def test_duplicate_ids_do_not_duplicate_selection(self) -> None:
        duplicated = [f"{1000}" for _ in range(4)] + [f"{1000 + i}" for i in range(1, 10)]
        result = self._select({"selected_order": duplicated})
        ids = [n.user_id for n in result.selected_nodes]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result.summary.selected_count, 10)

    def test_rank_marker_records_model_order(self) -> None:
        """模型只给顺序不给文字理由时，结果里要留下可审计的位次标记。"""
        wanted = [f"{1000 + i}" for i in range(9, -1, -1)]
        result = self._select({"selected_order": wanted})
        by_id = {n.user_id: n.selection_reasons for n in result.selected_nodes}
        self.assertIn("llm_reorder=applied", by_id[wanted[0]])
        self.assertIn("llm_order_rank=1", by_id[wanted[0]])
        self.assertIn("llm_order_rank=10", by_id[wanted[-1]])

    def test_id_only_mode_is_the_default_prompt(self) -> None:
        """默认走"只输出排序 ID"的提示词。"""
        recorder: list[str] = []
        self._select({"selected_order": [f"{1000 + i}" for i in range(10)]}, recorder)
        self.assertEqual(DEFAULT_SELECTOR_OUTPUT_MODE, "id_order")
        self.assertEqual(len(recorder), 1)
        payload = json.loads(recorder[0])
        self.assertEqual(list(payload["required_json_schema"]), ["selected_order"])
        self.assertEqual(len(payload["shortlist"]), 15)


class FallbackAndInvalidTest(unittest.TestCase):
    """A3.4 / A3.5：补齐与失效都必须可观测。"""

    def setUp(self) -> None:
        self.selector = Selector()
        self.pool = _Pool()

    def _select(self, response: Any) -> Any:
        return self.selector.select(
            _score_result(self.pool.nodes),
            use_llm=True,
            llm_client=_ScriptedLLM(response),
            candidate_limit=15,
        )

    def test_partial_order_is_filled_and_marked(self) -> None:
        partial = [f"{1000 + i}" for i in range(4, -1, -1)]
        result = self._select({"selected_order": partial})
        self.assertEqual(result.summary.selected_count, 10)
        # 前 5 位仍是模型给出的顺序
        self.assertEqual([n.user_id for n in result.selected_nodes[:5]], partial)
        for node in result.selected_nodes:
            self.assertIn("llm_reorder=partial(5/10)", node.selection_reasons)

    def test_empty_array_is_marked_invalid(self) -> None:
        result = self._select({"selected_order": []})
        self.assertEqual(result.summary.selected_count, 10)
        for node in result.selected_nodes:
            self.assertIn("llm_reorder=invalid", node.selection_reasons)
        # 失效时不允许出现"模型排序位次"这类伪成功标记
        self.assertFalse(
            any(r.startswith("llm_order_rank=") for n in result.selected_nodes for r in n.selection_reasons)
        )

    def test_missing_selected_order_key_is_marked_invalid(self) -> None:
        result = self._select({"fallback_order": [], "global_notes": []})
        self.assertEqual(result.summary.selected_count, 10)
        for node in result.selected_nodes:
            self.assertIn("llm_reorder=invalid", node.selection_reasons)

    def test_all_ids_out_of_pool_is_marked_invalid(self) -> None:
        result = self._select({"selected_order": ["111111", "222222"]})
        self.assertEqual(result.summary.selected_count, 10)
        for node in result.selected_nodes:
            self.assertIn("llm_reorder=invalid", node.selection_reasons)

    def test_invalid_order_falls_back_to_rule_score_order(self) -> None:
        invalid = self._select({"selected_order": []})
        rule_order = [n.user_id for n in invalid.selected_nodes]
        # 规则顺序：final_score 降序（候选池本身按分数递减构造）
        self.assertEqual(rule_order, [f"{1000 + i}" for i in range(10)])


class DownstreamContractTest(unittest.TestCase):
    """A3.6 / A3.7：角色、阶段与策略生成所需字段完整，安全过滤照常。"""

    def setUp(self) -> None:
        self.selector = Selector()
        self.pool = _Pool()

    def _select(self, response: Any) -> Any:
        return self.selector.select(
            _score_result(self.pool.nodes),
            use_llm=True,
            llm_client=_ScriptedLLM(response),
            candidate_limit=15,
        )

    def test_roles_and_stages_are_complete(self) -> None:
        result = self._select({"selected_order": [f"{1000 + i}" for i in range(9, -1, -1)]})
        allowed_roles = {
            "core_publish_node",
            "interaction_response_node",
            "amplification_node",
            "support_node",
        }
        for node in result.selected_nodes:
            self.assertIn(node.selected_role, allowed_roles)
            self.assertTrue(node.dispatch_stage.startswith("stage_"))
            self.assertTrue(node.selection_reasons)
            self.assertIn(node.dispatch_priority, {"p1", "p2", "p3"})
        self.assertIn("core_publish_node", [n.selected_role for n in result.selected_nodes])

    def test_security_filters_still_apply(self) -> None:
        result = self._select({"selected_order": [f"{1000 + i}" for i in range(9, -1, -1)]})
        selected_ids = {n.user_id for n in result.selected_nodes}
        # 不合格节点与人工复核节点都不能进主选
        self.assertFalse(selected_ids & {f"{2000 + i}" for i in range(20)})
        self.assertFalse(selected_ids & {f"{3000 + i}" for i in range(2)})
        self.assertTrue(all(n.eligible and not n.manual_review_required for n in result.selected_nodes))

    def test_strategy_generator_consumes_result(self) -> None:
        """下游策略生成必须能完整消费"只给排序"的选人结果。"""
        result = self._select({"selected_order": [f"{1000 + i}" for i in range(9, -1, -1)]})
        strategy = StrategyGenerator().generate(result)
        self.assertEqual(strategy.summary.selected_count, 10)
        self.assertEqual(strategy.summary.status, "ok")
        self.assertEqual(len(strategy.selected_nodes), 10)
        self.assertTrue(strategy.stage_plans)
        for node in strategy.selected_nodes:
            self.assertTrue(node.selected_role)
            self.assertTrue(node.dispatch_stage)
            self.assertTrue(node.recommended_action)
            self.assertTrue(node.rationale)


class IdOrderPromptTest(unittest.TestCase):
    """只输出排序 ID 的提示词本身：保留全部决策证据，只削减输出字段。"""

    def _shortlist(self, n: int = 15) -> list[dict[str, Any]]:
        return [
            {
                "user_id": f"{1000 + i}",
                "user_name": f"user_{1000 + i}",
                "role_hint": "amplification",
                "final_score": 0.9 - i * 0.001,
                "llm_feature_score": 0.0,
                "topic_match_score": 0.9,
                "risk_level": "low",
                "semantic_tags": ["亲子阅读"],
                "matched_keywords": ["阅读"],
                "selection_reasons": ["high_influence"],
            }
            for i in range(n)
        ]

    def test_payload_keeps_event_and_candidate_evidence(self) -> None:
        payload = json.loads(
            build_selector_id_order_user_prompt(event=_event(), max_selected=10, shortlist=self._shortlist())
        )
        for key in ("event_title", "target_goal", "event_type", "risk_level", "target_audience", "semantic_tags", "negative_constraints"):
            self.assertIn(key, payload["event"])
        self.assertEqual(len(payload["shortlist"]), 15)
        for item in payload["shortlist"]:
            for key in ("user_id", "role_hint", "final_score", "topic_match_score", "risk_level", "semantic_tags"):
                self.assertIn(key, item)

    def test_output_schema_is_id_array_only(self) -> None:
        payload = json.loads(
            build_selector_id_order_user_prompt(event=_event(), max_selected=10, shortlist=self._shortlist())
        )
        self.assertEqual(list(payload["required_json_schema"]), ["selected_order"])
        joined = " ".join(payload["requirements"])
        self.assertNotIn("recommended_role 只能是", joined)
        self.assertNotIn("reasoning 每个节点必须写", joined)

    def test_prompt_is_compact(self) -> None:
        text = build_selector_id_order_user_prompt(event=_event(), max_selected=10, shortlist=self._shortlist())
        self.assertNotIn("\n", text)


if __name__ == "__main__":
    unittest.main()
