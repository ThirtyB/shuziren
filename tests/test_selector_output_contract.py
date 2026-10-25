"""第八轮 · selector 输出契约与输出侧压缩的回归测试。

全部使用脚本化的假 LLM 客户端，不发起任何真实请求、不访问云端模型。
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
from influence_strategy.prompts import build_selector_user_prompt
from influence_strategy.selector import Selector


def _node(user_id: str, role_hint: str, final_score: float) -> NodeScore:
    # NodeScore.eligible 默认是 False（见 models.py），必须显式置 True，
    # 否则节点不会进入 safe_pool，select() 会退化成空池。
    return NodeScore(
        user_id=user_id,
        user_name=f"user_{user_id}",
        event_id="evt_output",
        event_type="parent_child_reading",
        role_hint=role_hint,
        final_score=final_score,
        topic_match_score=final_score,
        risk_level="low",
        eligible=True,
    )


def _event() -> ParsedEvent:
    return ParsedEvent(
        event_id="evt_output",
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
            event_id="evt_output",
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


def _shortlist(n: int = 15) -> list[dict[str, Any]]:
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
            "selection_reasons": ["topic_match"],
        }
        for i in range(n)
    ]


def _payload() -> dict[str, Any]:
    """把提示词反解回 dict。第三轮起提示词就是紧凑 JSON，可以直接解析。"""
    return json.loads(build_selector_user_prompt(event=_event(), max_selected=10, shortlist=_shortlist()))


class _ScriptedLLM:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: Any) -> dict[str, Any]:
        return self.response


class OutputContractPreservedTest(unittest.TestCase):
    """输出契约必须保留：字段还在，只是内容被要求缩短。"""

    def test_all_three_top_level_fields_still_in_schema(self) -> None:
        schema = _payload()["required_json_schema"]
        for key in ("selected_order", "fallback_order", "global_notes"):
            self.assertIn(key, schema, f"输出契约字段 {key} 被删除了")

    def test_fallback_order_schema_is_empty_list(self) -> None:
        self.assertEqual(_payload()["required_json_schema"]["fallback_order"], [])

    def test_global_notes_schema_is_empty_list(self) -> None:
        self.assertEqual(_payload()["required_json_schema"]["global_notes"], [])

    def test_selected_order_item_keeps_three_fields(self) -> None:
        item = _payload()["required_json_schema"]["selected_order"][0]
        for key in ("user_id", "recommended_role", "reasoning"):
            self.assertIn(key, item, f"selected_order 项缺少 {key}")

    def test_reasoning_example_is_single_short_phrase(self) -> None:
        reasoning = _payload()["required_json_schema"]["selected_order"][0]["reasoning"]
        self.assertEqual(len(reasoning), 1)
        self.assertLessEqual(len(reasoning[0]), 8)

    def test_role_constraint_requirement_still_present(self) -> None:
        joined = " ".join(_payload()["requirements"])
        for role in ("core_publish_node", "interaction_response_node", "amplification_node", "support_node"):
            self.assertIn(role, joined)

    def test_role_coverage_requirement_still_present(self) -> None:
        joined = " ".join(_payload()["requirements"])
        self.assertIn("核心首发", joined)
        self.assertIn("互动承接", joined)

    def test_shortlist_input_is_untouched(self) -> None:
        """本轮只动输出，输入侧（候选人数/字段）必须逐字不变。"""
        payload = _payload()
        self.assertEqual(len(payload["shortlist"]), 15)
        for item in payload["shortlist"]:
            self.assertEqual(
                set(item),
                {
                    "user_id", "user_name", "role_hint", "final_score",
                    "llm_feature_score", "topic_match_score", "risk_level",
                    "semantic_tags", "matched_keywords", "selection_reasons",
                },
            )

    def test_max_selected_unchanged(self) -> None:
        self.assertEqual(_payload()["constraints"]["max_selected_nodes"], 10)


class OutputInstructionTest(unittest.TestCase):
    """提示词里确实写了输出克制要求。"""

    def test_reasoning_brevity_requirement_present(self) -> None:
        joined = " ".join(_payload()["requirements"])
        self.assertIn("reasoning", joined)
        self.assertIn("1 条", joined)

    def test_reasoning_must_not_be_empty(self) -> None:
        """只写"少写"会让 qwen3:8b 直接退化成纯 ID 数组，必须写明"不能为空"。"""
        joined = " ".join(_payload()["requirements"])
        self.assertIn("必须写", joined)
        self.assertIn("不能为空数组", joined)

    def test_empty_array_requirement_present(self) -> None:
        joined = " ".join(_payload()["requirements"])
        self.assertIn("fallback_order", joined)
        self.assertIn("global_notes", joined)
        self.assertIn("空数组", joined)

    def test_prompt_is_still_compact_json(self) -> None:
        """第三轮的紧凑序列化不能被回退。"""
        text = build_selector_user_prompt(event=_event(), max_selected=10, shortlist=_shortlist())
        self.assertNotIn("\n", text)
        self.assertNotIn(": ", text.replace('", "', ""))


class RobustnessTest(unittest.TestCase):
    """模型不遵守克制要求时，生产路径必须照常工作。"""

    def setUp(self) -> None:
        self.selector = Selector()

    def _select(self, response: dict[str, Any]) -> Any:
        # 40 个合格节点 + 20 个不合格节点。
        # 不合格节点进 fallback_pool，这样备选名单能像生产环境一样填满 20 人
        # （初选 15 - 主选 10 只剩 5 个余量，不够 20）。
        nodes = [_node(f"{1000 + i}", "amplification", 0.9 - i * 0.001) for i in range(40)]
        nodes += [
            _node(f"{2000 + i}", "amplification", 0.5).model_copy(update={"eligible": False})
            for i in range(20)
        ]
        return self.selector.select(
            _score_result(nodes),
            use_llm=True,
            llm_client=_ScriptedLLM(response),
            candidate_limit=15,
        )

    def test_verbose_response_still_works(self) -> None:
        """模型照旧写满 fallback/notes/长 reasoning 时不应出错。"""
        result = self._select(
            {
                "selected_order": [
                    {
                        "user_id": f"{1000 + i}",
                        "recommended_role": "core_publish_node" if i == 0 else "amplification_node",
                        "reasoning": ["这条理由故意写得很长很长很长", "第二条", "第三条"],
                    }
                    for i in range(10)
                ],
                "fallback_order": [f"{1000 + i}" for i in range(10, 20)],
                "global_notes": ["很长的全局说明" * 5],
            }
        )
        self.assertEqual(result.summary.selected_count, 10)
        self.assertEqual(result.summary.fallback_count, 20)
        self.assertIn("core_publish_node", [n.selected_role for n in result.selected_nodes])

    def test_minimal_response_still_works(self) -> None:
        """模型完全遵守克制要求（空 fallback/notes、单条短 reasoning）时也要正常。"""
        result = self._select(
            {
                "selected_order": [
                    {
                        "user_id": f"{1000 + i}",
                        "recommended_role": "amplification_node",
                        "reasoning": ["主题贴合"],
                    }
                    for i in range(10)
                ],
                "fallback_order": [],
                "global_notes": [],
            }
        )
        self.assertEqual(result.summary.selected_count, 10)
        # 关键：fallback 名单仍由候选池构建，不受 fallback_order 变空影响
        self.assertEqual(result.summary.fallback_count, 20)
        self.assertEqual(len(result.fallback_nodes), 20)

    def test_fallback_nodes_do_not_depend_on_llm_fallback_order(self) -> None:
        """fallback_order 变空前后，真实 fallback 名单必须一致。"""
        base = {
            "selected_order": [
                {"user_id": f"{1000 + i}", "recommended_role": "amplification_node", "reasoning": ["贴合"]}
                for i in range(10)
            ],
            "global_notes": [],
        }
        with_notes = self._select({**base, "fallback_order": [f"{1000 + i}" for i in range(10, 20)]})
        without = self._select({**base, "fallback_order": []})
        self.assertEqual(
            [n.user_id for n in with_notes.fallback_nodes],
            [n.user_id for n in without.fallback_nodes],
        )

    def test_reasoning_still_propagates_to_llm_selector_tag(self) -> None:
        """下游依赖的 llm_selector= 前缀理由必须继续生效。"""
        result = self._select(
            {
                "selected_order": [
                    {"user_id": f"{1000 + i}", "recommended_role": "amplification_node", "reasoning": ["主题贴合"]}
                    for i in range(10)
                ],
                "fallback_order": [],
                "global_notes": [],
            }
        )
        for node in result.selected_nodes:
            self.assertIn("llm_selector=主题贴合", node.selection_reasons)

    def test_recommended_role_still_drives_selected_role(self) -> None:
        result = self._select(
            {
                "selected_order": [
                    {"user_id": "1000", "recommended_role": "core_publish_node", "reasoning": ["首发"]},
                    {"user_id": "1001", "recommended_role": "interaction_response_node", "reasoning": ["互动"]},
                    {"user_id": "1002", "recommended_role": "support_node", "reasoning": ["支持"]},
                ]
                + [
                    {"user_id": f"{1000 + i}", "recommended_role": "amplification_node", "reasoning": ["扩散"]}
                    for i in range(3, 10)
                ],
                "fallback_order": [],
                "global_notes": [],
            }
        )
        roles = {n.user_id: n.selected_role for n in result.selected_nodes}
        self.assertEqual(roles["1000"], "core_publish_node")
        self.assertEqual(roles["1001"], "interaction_response_node")
        self.assertEqual(roles["1002"], "support_node")

    def test_bare_id_selected_order_still_applies_llm_order(self) -> None:
        """模型把 selected_order 退化成纯 ID 数组时：顺序仍生效，且不报错。

        第八轮前半段的实测现象：提示词只要求"少写"时，qwen3:8b 直接返回
        ``"selected_order": ["96961", ...]``，若整批丢弃则 LLM 重排静默失效
        （名单与规则基线逐位相同，6s 调用白花）。这里锁定"接纳裸 ID"的行为。
        """
        # 初选上限 15 → 候选池是 1000..1014；倒序给出，便于验证顺序确实被采纳。
        wanted = [f"{1000 + i}" for i in range(9, -1, -1)]
        result = self._select({"selected_order": wanted, "fallback_order": [], "global_notes": []})
        self.assertEqual(result.summary.selected_count, 10)
        self.assertEqual([n.user_id for n in result.selected_nodes], wanted)
        # 裸 ID 不携带理由，因此不应出现 llm_selector= 前缀。
        self.assertFalse(
            any(r.startswith("llm_selector=") for n in result.selected_nodes for r in n.selection_reasons)
        )
        self.assertEqual(result.summary.fallback_count, 20)

    def test_bare_id_outside_candidate_pool_is_dropped(self) -> None:
        """裸 ID 同样要过候选集校验，不能把池外人放进名单。"""
        result = self._select(
            {
                "selected_order": ["999999"] + [f"{1000 + i}" for i in range(9)],
                "fallback_order": [],
                "global_notes": [],
            }
        )
        self.assertNotIn("999999", [n.user_id for n in result.selected_nodes])
        self.assertEqual(result.summary.selected_count, 10)

    def test_missing_reasoning_does_not_break(self) -> None:
        result = self._select(
            {
                "selected_order": [
                    {"user_id": f"{1000 + i}", "recommended_role": "amplification_node"}
                    for i in range(10)
                ],
                "fallback_order": [],
                "global_notes": [],
            }
        )
        self.assertEqual(result.summary.selected_count, 10)
        self.assertFalse(
            any(r.startswith("llm_selector=") for n in result.selected_nodes for r in n.selection_reasons)
        )


if __name__ == "__main__":
    unittest.main()
