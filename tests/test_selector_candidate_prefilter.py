"""第七轮 · 规则初选 + 大模型小范围重排 的回归测试。

全部使用脚本化的假 LLM 客户端，不发起任何真实请求、不访问云端模型。
"""

from __future__ import annotations

import unittest
from typing import Any

from influence_strategy import selector as selector_module
from influence_strategy.models import (
    EventDispatchPreferences,
    NodeScore,
    ParsedEvent,
    ProductContext,
    ScoreResult,
    ScoreSummary,
)
from influence_strategy.selector import (
    DEFAULT_SELECTOR_CANDIDATE_LIMIT,
    Selector,
    prefilter_selector_candidates,
)

ALL_HINTS = ("core_broadcast", "interaction_response", "amplification", "general_support")


def _node(
    user_id: str,
    role_hint: str,
    final_score: float,
    *,
    eligible: bool = True,
    manual_review_required: bool = False,
) -> NodeScore:
    return NodeScore(
        user_id=user_id,
        user_name=f"user_{user_id}",
        event_id="evt_prefilter",
        event_type="parent_child_reading",
        role_hint=role_hint,
        final_score=final_score,
        topic_match_score=final_score,
        eligible=eligible,
        manual_review_required=manual_review_required,
        risk_level="low",
    )


def _score_result(
    nodes: list[NodeScore],
    *,
    target_roles: list[str] | None = None,
    max_selected_nodes: int = 10,
    rerank_top_k: int = 30,
) -> ScoreResult:
    # 必须与生产一致：规则事件解析在 max_selected_nodes=10 时产出的是
    # candidate_pool_size=60 / rerank_top_k=30（见 _default_dispatch_preferences）。
    # 若直接用 EventDispatchPreferences 的模型默认值（rerank_top_k=12），
    # 会掩盖"初选上限 15 与 rerank_top_k 是两个独立截断"这一事实。
    event = ParsedEvent(
        event_id="evt_prefilter",
        event_title="亲子阅读与英语启蒙",
        event_description="围绕亲子阅读与英语启蒙做一次传播活动",
        target_goal="engagement",
        event_type="parent_child_reading",
        target_audience=["parent_child"],
        target_roles=target_roles or ["core_publish_node", "interaction_response_node", "amplification_node"],
        dispatch_preferences=EventDispatchPreferences(
            # 生产规则解析在 max_selected_nodes=10 时产出的是 60/30；
            # EventDispatchPreferences 的模型默认值是 24/12，直接用默认值
            # 会掩盖"初选上限与 rerank_top_k 是两个独立截断"这一事实。
            candidate_pool_size=60,
            rerank_top_k=rerank_top_k,
        ),
    )
    if max_selected_nodes != 10:
        event = event.model_copy(
            update={"constraints": event.constraints.model_copy(update={"max_selected_nodes": max_selected_nodes})}
        )
    return ScoreResult(
        event=event,
        product_context=ProductContext(product_name="abc_reading"),
        summary=ScoreSummary(
            event_id=event.event_id,
            event_type=event.event_type,
            node_count=len(nodes),
            eligible_count=sum(1 for n in nodes if n.eligible),
            manual_review_count=sum(1 for n in nodes if n.manual_review_required),
            high_priority_count=0,
            avg_final_score=0.0,
            avg_risk_score=0.0,
        ),
        node_scores=nodes,
    )


class _ScriptedLLM:
    """按脚本返回 JSON 的假客户端；raise_error 时模拟调用失败。"""

    def __init__(self, response: dict[str, Any] | None = None, raise_error: bool = False) -> None:
        self.response = response or {}
        self.raise_error = raise_error
        self.user_prompts: list[str] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str, **_: Any) -> dict[str, Any]:
        if self.raise_error:
            # 必须是 selector 真正捕获的异常类型之一
            # （LLMClientError / OSError / ValueError / TypeError），
            # 否则测的就不是"调用失败走兜底"这条路径。
            raise OSError("模拟 LLM 调用失败")
        self.user_prompts.append(user_prompt)
        return self.response


def _build_pool(total: int) -> list[NodeScore]:
    """同质候选池：全是 amplification，分数递减。用于测池大小的截断行为。"""
    return [
        _node(f"{1000 + i}", "amplification", 0.90 - i * 0.001)
        for i in range(total)
    ]


def _build_amplification_dominant_pool(
    *,
    amp_count: int = 35,
    low_role_counts: dict[str, int] | None = None,
) -> list[NodeScore]:
    """构造"amplification 霸榜"的候选池。

    amplification 不仅数量占绝对多数，而且**分数最高**；其他角色分数明显更低。
    这样"纯按分数取前 N"会退化成清一色 amplification，从而可以验证
    角色配额是否真的落实了，而不是只写在 prompt 文字里。
    """
    low_role_counts = low_role_counts or {
        "core_broadcast": 2,
        "interaction_response": 2,
        "general_support": 1,
    }
    nodes: list[NodeScore] = []
    score = 0.90
    for i in range(amp_count):
        nodes.append(_node(f"amp_{i}", "amplification", score))
        score -= 0.001
    low_base = 0.50
    offset = 0
    for hint, count in low_role_counts.items():
        for i in range(count):
            nodes.append(_node(f"{hint}_{i}", hint, low_base - offset * 0.001))
            offset += 1
    return nodes


class PrefilterUnitTest(unittest.TestCase):
    """规则初选函数本身：不足 / 恰好 / 超过 15。"""

    def test_pool_smaller_than_limit_returns_all(self) -> None:
        pool = _build_pool(total=8)
        self.assertEqual(len(prefilter_selector_candidates(pool, limit=15)), 8)

    def test_pool_exactly_limit_returns_all(self) -> None:
        pool = _build_pool(total=15)
        self.assertEqual(len(prefilter_selector_candidates(pool, limit=15)), 15)

    def test_pool_larger_than_limit_is_truncated(self) -> None:
        pool = _build_pool(total=40)
        result = prefilter_selector_candidates(pool, limit=15)
        self.assertEqual(len(result), 15)

    def test_zero_limit_disables_prefilter(self) -> None:
        """limit<=0 表示关闭初选，必须原样返回全池。"""
        pool = _build_pool(total=40)
        self.assertEqual(len(prefilter_selector_candidates(pool, limit=0)), 40)
        self.assertEqual(len(prefilter_selector_candidates(pool, limit=-5)), 40)

    def test_empty_pool_returns_empty(self) -> None:
        self.assertEqual(prefilter_selector_candidates([], limit=15), [])

    def test_result_keeps_score_descending_order(self) -> None:
        pool = _build_pool(total=40)
        result = prefilter_selector_candidates(pool, limit=15)
        scores = [node.final_score for node in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_scoring_nodes_are_kept(self) -> None:
        """初选必须保住分数最高的那批人，不能随机丢弃高分节点。"""
        pool = _build_pool(total=40)
        result = prefilter_selector_candidates(pool, limit=15)
        self.assertEqual(
            [node.user_id for node in result],
            [node.user_id for node in pool[:15]],
        )

    def test_role_diversity_is_enforced_not_just_prompt_text(self) -> None:
        """分数最高的 15 人若全是 amplification，初选必须补进其他角色。

        这是对"角色多样性"的实质检查：如果实现退化成"直接按分数取前 N"，
        这个用例会失败。
        """
        pool = _build_amplification_dominant_pool(amp_count=35)
        # 先确认"纯按分数取前 15"确实会退化成单一角色，否则用例没有意义。
        top15_by_score = sorted(pool, key=lambda n: (-n.final_score, n.user_id))[:15]
        self.assertEqual({n.role_hint for n in top15_by_score}, {"amplification"})

        result = prefilter_selector_candidates(pool, limit=15)
        hints = {node.role_hint for node in result}
        self.assertIn("core_broadcast", hints)
        self.assertIn("interaction_response", hints)
        self.assertIn("general_support", hints)
        self.assertGreaterEqual(len(hints), 3)

    def test_prefilter_is_deterministic(self) -> None:
        pool = _build_amplification_dominant_pool(
            amp_count=32,
            low_role_counts={"core_broadcast": 4, "interaction_response": 3},
        )
        first = [n.user_id for n in prefilter_selector_candidates(pool, limit=15)]
        second = [n.user_id for n in prefilter_selector_candidates(pool, limit=15)]
        self.assertEqual(first, second)

    def test_preferred_role_hints_are_respected(self) -> None:
        """传入偏好的角色顺序时，优先保证这批角色进入初选池。"""
        pool = _build_amplification_dominant_pool(
            amp_count=34,
            low_role_counts={"core_broadcast": 3, "interaction_response": 3},
        )
        result = prefilter_selector_candidates(
            pool,
            limit=15,
            preferred_role_hints=("core_broadcast", "interaction_response"),
        )
        hints = [node.role_hint for node in result]
        self.assertIn("core_broadcast", hints)
        self.assertIn("interaction_response", hints)


class SelectorPrefilterIntegrationTest(unittest.TestCase):
    """selector 集成：初选必须发生在 prompt 构建之前。"""

    def setUp(self) -> None:
        self.selector = Selector()
        # 第十三轮：生产默认是"只输出排序 ID"的提示词（``id_order`` 模式），
        # 旧的完整提示词仍然保留可切回。这里两个构建器都挂上记录器，
        # 这样无论当前默认哪种模式，"初选先于 prompt 构建"这个断言都成立。
        self._real_builders = {
            "build_selector_user_prompt": selector_module.build_selector_user_prompt,
            "build_selector_id_order_user_prompt": selector_module.build_selector_id_order_user_prompt,
        }
        self.shortlists: list[list[dict[str, Any]]] = []

        def make_recorder(name: str, real):
            def recorder(**kwargs):
                self.shortlists.append(kwargs["shortlist"])
                return real(**kwargs)

            return recorder

        self._recorders = {
            name: make_recorder(name, real) for name, real in self._real_builders.items()
        }
        for name, recorder in self._recorders.items():
            setattr(selector_module, name, recorder)

    def tearDown(self) -> None:
        for name, real in self._real_builders.items():
            setattr(selector_module, name, real)

    def test_shortlist_in_prompt_is_at_most_default_limit(self) -> None:
        """真正生效的是候选列表本身被收敛，而不是提示词字符串被截断。"""
        nodes = _build_pool(total=40)
        self.selector.select(
            _score_result(nodes),
            use_llm=True,
            llm_client=_ScriptedLLM({"selected_order": [], "fallback_order": [], "global_notes": []}),
        )
        self.assertEqual(len(self.shortlists), 1)
        self.assertEqual(len(self.shortlists[0]), DEFAULT_SELECTOR_CANDIDATE_LIMIT)
        self.assertEqual(DEFAULT_SELECTOR_CANDIDATE_LIMIT, 15)

    def test_prefilter_and_rerank_top_k_are_independent_caps(self) -> None:
        """初选上限与 rerank_top_k 是两个独立截断，生效的是较小者。

        生产规则解析产出 rerank_top_k=30 > 15，因此默认 15 人初选能完整生效；
        若 rerank_top_k 被调小到 12，则会再截一刀到 12（本轮不改 rerank_top_k，
        这里只把两者的关系固化下来）。
        """
        nodes = _build_pool(40)
        self.selector.select(
            _score_result(nodes, rerank_top_k=30),
            use_llm=True,
            llm_client=_ScriptedLLM({"selected_order": []}),
        )
        self.assertEqual(len(self.shortlists[-1]), 15)

        self.selector.select(
            _score_result(nodes, rerank_top_k=12),
            use_llm=True,
            llm_client=_ScriptedLLM({"selected_order": []}),
        )
        self.assertEqual(len(self.shortlists[-1]), 12)

    def test_custom_limit_is_honoured(self) -> None:
        nodes = _build_pool(total=40)
        self.selector.select(
            _score_result(nodes),
            use_llm=True,
            llm_client=_ScriptedLLM({"selected_order": []}),
            candidate_limit=6,
        )
        self.assertEqual(len(self.shortlists[0]), 6)

    def test_small_pool_uses_actual_count_without_error(self) -> None:
        """候选不足 15 人时用实际人数，不报错。"""
        nodes = _build_pool(total=8)
        result = self.selector.select(
            _score_result(nodes),
            use_llm=True,
            llm_client=_ScriptedLLM({"selected_order": []}),
        )
        self.assertEqual(len(self.shortlists[0]), 8)
        self.assertEqual(result.summary.selected_count, 8)

    def test_disabled_llm_keeps_original_unprefiltered_pool(self) -> None:
        """use_llm=False → 走原有纯规则路径，候选池不做任何收敛。"""
        nodes = _build_pool(total=40)
        result = self.selector.select(_score_result(nodes), use_llm=False)
        self.assertEqual(self.shortlists, [])
        # 纯规则路径仍按分数取满 max_selected_nodes=10
        self.assertEqual(result.summary.selected_count, 10)
        self.assertEqual(
            [n.user_id for n in result.selected_nodes],
            [n.user_id for n in sorted(nodes, key=lambda n: (-n.final_score, n.user_id))[:10]],
        )

    def test_llm_disabled_ignores_candidate_limit(self) -> None:
        nodes = _build_pool(total=40)
        result = self.selector.select(
            _score_result(nodes), use_llm=False, candidate_limit=5
        )
        self.assertEqual(result.summary.selected_count, 10)

    def test_role_diversity_survives_into_shortlist(self) -> None:
        nodes = _build_amplification_dominant_pool(amp_count=35)
        self.selector.select(
            _score_result(nodes),
            use_llm=True,
            llm_client=_ScriptedLLM({"selected_order": []}),
        )
        hints = {item["role_hint"] for item in self.shortlists[0]}
        self.assertIn("core_broadcast", hints)
        self.assertIn("interaction_response", hints)
        self.assertGreaterEqual(len(hints), 3)


class LlmIdValidationTest(unittest.TestCase):
    """LLM 返回的 user_id 必须来自本次初选候选集。"""

    def _run(self, pool_total: int, response: dict[str, Any], limit: int | None = None):
        nodes = _build_pool(total=pool_total)
        self.selector.select(
            _score_result(nodes),
            use_llm=True,
            llm_client=_ScriptedLLM(response),
            candidate_limit=limit,
        )
        return nodes

    def setUp(self) -> None:
        self.selector = Selector()

    def test_out_of_pool_ids_are_ignored(self) -> None:
        """返回集外 ID 与"只返回集内 ID"必须得到完全相同的结果。"""
        nodes = _build_pool(total=40)
        shortlist_ids = {
            n.user_id
            for n in sorted(nodes, key=lambda n: (-n.final_score, n.user_id))[:15]
        }
        outside_ids = [n.user_id for n in nodes if n.user_id not in shortlist_ids][:3]
        valid_ids = list(shortlist_ids)[:6]

        clean_response = {
            "selected_order": [
                {"user_id": uid, "recommended_role": "amplification_node", "reasoning": ["贴合"]}
                for uid in valid_ids
            ],
            "fallback_order": [],
            "global_notes": [],
        }
        polluted_response = {
            "selected_order": [
                {"user_id": "ghost_1", "recommended_role": "core_publish_node", "reasoning": ["幻觉"]},
                {"user_id": outside_ids[0], "recommended_role": "core_publish_node", "reasoning": ["集外"]},
                *clean_response["selected_order"],
                {"user_id": "ghost_2", "recommended_role": "support_node", "reasoning": ["幻觉"]},
            ],
            "fallback_order": ["ghost_3", outside_ids[1]],
            "global_notes": [],
        }

        clean = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM(clean_response)
        )
        polluted = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM(polluted_response)
        )

        self.assertEqual(
            [n.user_id for n in clean.selected_nodes],
            [n.user_id for n in polluted.selected_nodes],
        )
        # 集外 ID 绝不能出现在最终结果里
        selected = {n.user_id for n in polluted.selected_nodes}
        self.assertFalse(selected & set(outside_ids))
        self.assertFalse(selected & {"ghost_1", "ghost_2"})

    def test_all_invalid_ids_fall_back_to_rule_path(self) -> None:
        """全部 ID 无效 → 沿用原有"返回 None → 走纯规则"的兜底。"""
        nodes = _build_pool(total=40)
        all_ghost = {
            "selected_order": [
                {"user_id": "ghost_1", "recommended_role": "core_publish_node", "reasoning": []},
                {"user_id": "ghost_2", "recommended_role": "amplification_node", "reasoning": []},
            ],
            "fallback_order": [],
            "global_notes": [],
        }
        errored = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM(raise_error=True)
        )
        ghosted = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM(all_ghost)
        )
        self.assertEqual(
            [n.user_id for n in errored.selected_nodes],
            [n.user_id for n in ghosted.selected_nodes],
        )
        # 两者都等价于：在 15 人初选池上做纯规则选择
        expected = [
            n.user_id
            for n in sorted(nodes, key=lambda n: (-n.final_score, n.user_id))[:10]
        ]
        self.assertEqual([n.user_id for n in ghosted.selected_nodes], expected)

    def test_llm_error_falls_back_without_raising(self) -> None:
        nodes = _build_pool(total=40)
        result = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM(raise_error=True)
        )
        self.assertEqual(result.summary.selected_count, 10)
        self.assertFalse(any("llm_selector=" in r for n in result.selected_nodes for r in n.selection_reasons))

    def test_plan_dict_filters_invalid_ids(self) -> None:
        """直接检查 _llm_selection_plan 的输出 dict。"""
        nodes = _build_pool(total=40)
        score = _score_result(nodes)
        plan = self.selector._llm_selection_plan(
            score_result=score,
            safe_pool=prefilter_selector_candidates(nodes, limit=15),
            max_selected=10,
            workspace_root=None,
            use_llm=True,
            llm_client=_ScriptedLLM(
                {
                    "selected_order": [
                        {"user_id": "ghost_1", "recommended_role": "core_publish_node", "reasoning": ["幻觉"]},
                        {"user_id": "1000", "recommended_role": "amplification_node", "reasoning": ["可用"]},
                    ],
                    "fallback_order": ["ghost_2", "1001"],
                    "global_notes": ["note"],
                }
            ),
        )
        self.assertIsNotNone(plan)
        self.assertEqual(list(plan["order_map"]), ["1000"])
        self.assertEqual(plan["fallback_order"], ["1001"])
        self.assertEqual(plan["global_notes"], ["note"])

    def test_recommended_role_and_reasoning_still_propagate(self) -> None:
        """下游依赖的 recommended_role / reasoning 字段必须继续生效。"""
        nodes = _build_pool(total=40)
        valid = [f"{1000 + i}" for i in range(15)]
        result = self.selector.select(
            _score_result(nodes),
            use_llm=True,
            llm_client=_ScriptedLLM(
                {
                    "selected_order": [
                        {
                            "user_id": uid,
                            "recommended_role": "core_publish_node" if i == 0 else "amplification_node",
                            "reasoning": ["首发", "贴合"],
                        }
                        for i, uid in enumerate(valid[:3])
                    ],
                    "fallback_order": [],
                    "global_notes": [],
                }
            ),
        )
        top = result.selected_nodes[0]
        self.assertEqual(top.selected_role, "core_publish_node")
        self.assertIn("llm_selector=首发", top.selection_reasons)
        self.assertIn("llm_selector=贴合", top.selection_reasons)


class CompatibilityTest(unittest.TestCase):
    """保持既有输出契约：人数、fallback、角色再平衡。"""

    def setUp(self) -> None:
        self.selector = Selector()

    def test_final_selected_count_is_unchanged(self) -> None:
        nodes = _build_pool(total=40)
        result = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM({"selected_order": []})
        )
        self.assertEqual(result.summary.selected_count, 10)
        self.assertEqual(result.summary.max_selected_nodes, 10)

    def test_fallback_nodes_still_produced(self) -> None:
        nodes = _build_pool(total=40)
        result = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM({"selected_order": []})
        )
        self.assertGreater(result.summary.fallback_count, 0)
        self.assertEqual(len(result.fallback_nodes), result.summary.fallback_count)
        # 主选与备选不能重复
        self.assertFalse(
            {n.user_id for n in result.selected_nodes}
            & {n.user_id for n in result.fallback_nodes}
        )

    def test_role_rebalance_still_applies(self) -> None:
        """最终名单仍要有核心发布节点且角色种类 >= min(3, n)。"""
        nodes = _build_amplification_dominant_pool(
            amp_count=36,
            low_role_counts={"core_broadcast": 2, "interaction_response": 2},
        )
        result = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM({"selected_order": []})
        )
        roles = [n.selected_role for n in result.selected_nodes]
        self.assertIn("core_publish_node", roles)
        self.assertGreaterEqual(len(set(roles)), 3)

    def test_empty_eligible_pool_does_not_raise(self) -> None:
        """初选结果为空 → 走既有安全路径，不产生无效结果。"""
        nodes = [
            _node(f"{1000 + i}", "amplification", 0.9, eligible=False)
            for i in range(20)
        ]
        result = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM({"selected_order": []})
        )
        self.assertEqual(result.summary.selected_count, 0)
        self.assertEqual(result.selected_nodes, [])
        # ineligible 节点仍可作为 fallback 名单透出
        self.assertGreaterEqual(result.summary.fallback_count, 0)

    def test_manual_review_nodes_excluded_from_shortlist(self) -> None:
        """需人工复核的节点不进初选池（沿用原 safe_pool 定义）。"""
        nodes = [
            _node(f"{1000 + i}", "amplification", 0.9 - i * 0.001)
            for i in range(20)
        ]
        nodes[0] = nodes[0].model_copy(update={"manual_review_required": True})
        result = self.selector.select(
            _score_result(nodes), use_llm=True, llm_client=_ScriptedLLM({"selected_order": []})
        )
        self.assertNotIn("1000", [n.user_id for n in result.selected_nodes])


if __name__ == "__main__":
    unittest.main()
