from __future__ import annotations

import json
import unittest
from collections import Counter

from pydantic import ValidationError

from influence_strategy.models import (
    NodeScore,
    ParsedEvent,
    ProductContext,
    ScoreResult,
    ScoreSummary,
)
from influence_strategy.platform_dispatcher import (
    PlatformDispatcher,
    PlatformSpec,
    PlatformTimePlan,
    PlatformTimeStage,
    UnsupportedPlatformError,
    to_json_dict,
)
from influence_strategy.platform_profiles import get_platform_profile


#: 数据集中可用的三个模拟平台 key（小写、无后缀）。
PLATFORM_KEYS = ("facebook", "instagram", "twitter")


def _dataset_platform(platform_key: str) -> str:
    return f"{platform_key}_simulated"


def _make_event(**overrides) -> ParsedEvent:
    payload = {
        "event_id": "event_test",
        "event_title": "亲子阅读推广活动",
        "event_description": "围绕亲子阅读和英语启蒙做一次传播活动。",
        "target_goal": "engagement",
        "event_type": "parent_child_reading",
        "target_audience": ["parent_child", "reading_interest"],
        "extracted_keywords": ["亲子阅读", "英语启蒙"],
        "constraints": {
            "risk_level": "low",
            "max_selected_nodes": 2,
            "campaign_window_hours": 48,
        },
        "negative_constraints": ["避免节点内容完全同质化"],
    }
    payload.update(overrides)
    return ParsedEvent(**payload)


def _make_node(user_id: str, *, platform: str = "facebook_simulated", **overrides) -> NodeScore:
    base = {
        "user_id": user_id,
        "user_name": f"user_{user_id}",
        "event_id": "event_test",
        "event_type": "parent_child_reading",
        "role_hint": "core_broadcast",
        "platform": platform,
        "final_score": 0.50,
        "topic_match_score": 0.40,
        "influence_score": 0.50,
        "diffusion_score": 0.30,
        "stability_score": 0.60,
        "activity_score": 0.30,
        "eligible": True,
        "manual_review_required": False,
        "risk_level": "low",
        "risk_flags": [],
        "selection_reasons": ["final_score=0.500"],
    }
    base.update(overrides)
    return NodeScore(**base)


def _pool(**counts: int) -> list[NodeScore]:
    """按平台构建节点池，例如 ``_pool(facebook=3, twitter=5)``。"""

    nodes: list[NodeScore] = []
    for key, count in counts.items():
        for index in range(1, count + 1):
            nodes.append(
                _make_node(f"{key}_{index}", platform=_dataset_platform(key))
            )
    return nodes


def _make_score_result(nodes: list[NodeScore], event: ParsedEvent | None = None) -> ScoreResult:
    parsed_event = event or _make_event()
    return ScoreResult(
        event=parsed_event,
        product_context=ProductContext(product_name="abc_reading"),
        summary=ScoreSummary(
            event_id=parsed_event.event_id,
            event_type=parsed_event.event_type,
            node_count=len(nodes),
            eligible_count=sum(1 for node in nodes if node.eligible),
            manual_review_count=sum(1 for node in nodes if node.manual_review_required),
            high_priority_count=0,
            avg_final_score=round(
                sum(node.final_score for node in nodes) / max(len(nodes), 1), 6
            ),
            avg_risk_score=0.0,
        ),
        node_scores=nodes,
    )


def _specs(**counts: int) -> list[PlatformSpec]:
    return [
        PlatformSpec(platform_key=key, requested_count=count)
        for key, count in counts.items()
    ]


class PlatformDispatcherTest(unittest.TestCase):
    def test_dispatch_across_three_platforms(self) -> None:
        nodes = _pool(facebook=3, twitter=5, instagram=2)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3, twitter=5, instagram=2),
        )

        self.assertEqual(result.dispatch_mode, "cross_platform")
        self.assertEqual(
            [platform.platform_key for platform in result.platforms],
            ["facebook", "twitter", "instagram"],
        )
        self.assertEqual(result.total_requested_count, 10)
        self.assertEqual(result.total_selected_count, 10)

    def test_selected_count_matches_requested_count(self) -> None:
        nodes = _pool(facebook=8, twitter=8, instagram=8)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3, twitter=5, instagram=2),
        )
        for platform, expected in zip(result.platforms, (3, 5, 2)):
            self.assertEqual(platform.requested_count, expected)
            self.assertEqual(platform.selected_count, expected)
            self.assertEqual(platform.shortfall_count, 0)

    def test_platform_only_selects_its_own_members(self) -> None:
        """每个平台只能从归属自己的数字人中选人。"""

        nodes = _pool(facebook=2, twitter=3, instagram=1)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2, twitter=3, instagram=1),
        )
        for platform in result.platforms:
            for assignment in platform.selected_nodes:
                self.assertEqual(
                    assignment.platform,
                    _dataset_platform(platform.platform_key),
                )

    def test_shortfall_recorded_when_pool_exhausted(self) -> None:
        nodes = _pool(facebook=2)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=5),
        )
        facebook = result.platforms[0]
        self.assertEqual(facebook.requested_count, 5)
        self.assertEqual(facebook.selected_count, 2)
        self.assertEqual(facebook.shortfall_count, 3)
        self.assertTrue(
            any("候选不足" in warning for warning in facebook.warnings),
            facebook.warnings,
        )
        # no fabricated nodes
        self.assertEqual(
            {node.user_id for node in facebook.selected_nodes},
            {"facebook_1", "facebook_2"},
        )

    def test_no_duplicate_user_within_platform(self) -> None:
        nodes = _pool(facebook=10, twitter=10, instagram=10)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=4, twitter=5, instagram=3),
        )
        for platform in result.platforms:
            user_ids = [node.user_id for node in platform.selected_nodes]
            self.assertEqual(len(user_ids), len(set(user_ids)), platform.platform_key)

    def test_each_user_belongs_to_exactly_one_platform(self) -> None:
        """每个数字人只归属唯一模拟平台，不再出现跨平台重复选人。"""

        nodes = _pool(facebook=3, twitter=3, instagram=3)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3, twitter=3, instagram=3),
        )
        self.assertEqual(result.unique_user_count, result.total_selected_count)
        self.assertEqual(result.overlapping_user_ids, [])

    def test_content_plans_differ_across_platforms(self) -> None:
        nodes = _pool(facebook=8, twitter=8, instagram=8)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3, twitter=5, instagram=2),
        )
        section_signatures = [
            tuple(section.section_key for section in platform.content_plan.sections)
            for platform in result.platforms
        ]
        self.assertEqual(len(set(section_signatures)), 3, section_signatures)

        mode_signatures = [
            tuple(platform.content_plan.content_modes) for platform in result.platforms
        ]
        self.assertEqual(len(set(mode_signatures)), 3, mode_signatures)

        # event context must be rendered into the guidance
        facebook_guidance = " ".join(
            section.guidance for section in result.platforms[0].content_plan.sections
        )
        self.assertIn("亲子阅读推广活动", facebook_guidance)

    def test_time_plans_differ_across_platforms(self) -> None:
        nodes = _pool(facebook=2, twitter=2, instagram=2)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2, twitter=2, instagram=2),
        )
        stage_signatures = [
            tuple(stage.stage_key for stage in platform.time_plan.stages)
            for platform in result.platforms
        ]
        self.assertEqual(len(set(stage_signatures)), 3, stage_signatures)

    def test_final_score_is_not_mutated(self) -> None:
        nodes = (
            [
                _make_node(f"facebook_{index}", platform="facebook_simulated", final_score=0.10 * index)
                for index in range(1, 4)
            ]
            + [
                _make_node(f"twitter_{index}", platform="twitter_simulated", final_score=0.10 * index)
                for index in range(1, 6)
            ]
            + [
                _make_node(f"instagram_{index}", platform="instagram_simulated", final_score=0.10 * index)
                for index in range(1, 3)
            ]
        )
        score_result = _make_score_result(nodes)
        original_scores = {node.user_id: node.final_score for node in nodes}
        snapshot = score_result.model_dump()

        result = PlatformDispatcher().dispatch(
            score_result=score_result,
            platform_specs=_specs(facebook=3, twitter=5, instagram=2),
        )

        for platform in result.platforms:
            for assignment in platform.selected_nodes:
                self.assertEqual(
                    assignment.final_score, original_scores[assignment.user_id]
                )
        # shared ScoreResult was not modified
        self.assertEqual(score_result.model_dump(), snapshot)

    def test_platform_fit_scores_are_unit_range(self) -> None:
        nodes = _pool(facebook=8, twitter=8, instagram=8)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3, twitter=5, instagram=2),
        )
        for platform in result.platforms:
            for assignment in platform.selected_nodes:
                self.assertGreaterEqual(assignment.platform_fit_score, 0.0)
                self.assertLessEqual(assignment.platform_fit_score, 1.0)
                self.assertGreaterEqual(assignment.platform_dispatch_score, 0.0)
                self.assertLessEqual(assignment.platform_dispatch_score, 1.0)

    def test_platform_fit_is_not_a_copy_of_final_score(self) -> None:
        """Platform fit must react to platform-specific weights.

        每个平台只评估本平台成员，因此在各自平台上放置一对特征互补的节点，
        验证 Facebook 偏好稳定性节点、Twitter 偏好扩散型节点。
        """

        nodes = [
            _make_node(
                "fb_stable",
                platform="facebook_simulated",
                final_score=0.5,
                topic_match_score=0.9,
                stability_score=0.9,
                diffusion_score=0.0,
                activity_score=0.0,
                influence_score=0.0,
                role_hint="general_support",
            ),
            _make_node(
                "fb_diffuse",
                platform="facebook_simulated",
                final_score=0.5,
                topic_match_score=0.0,
                stability_score=0.0,
                diffusion_score=0.9,
                activity_score=0.9,
                influence_score=0.5,
                role_hint="amplification",
            ),
            _make_node(
                "tw_stable",
                platform="twitter_simulated",
                final_score=0.5,
                topic_match_score=0.9,
                stability_score=0.9,
                diffusion_score=0.0,
                activity_score=0.0,
                influence_score=0.0,
                role_hint="general_support",
            ),
            _make_node(
                "tw_diffuse",
                platform="twitter_simulated",
                final_score=0.5,
                topic_match_score=0.0,
                stability_score=0.0,
                diffusion_score=0.9,
                activity_score=0.9,
                influence_score=0.5,
                role_hint="amplification",
            ),
        ]
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2, twitter=2),
        )
        facebook_fit = {
            node.user_id: node.platform_fit_score for node in result.platforms[0].selected_nodes
        }
        twitter_fit = {
            node.user_id: node.platform_fit_score for node in result.platforms[1].selected_nodes
        }
        # Facebook favours the stability/topic node, Twitter favours diffusion/activity.
        self.assertGreater(facebook_fit["fb_stable"], facebook_fit["fb_diffuse"])
        self.assertGreater(twitter_fit["tw_diffuse"], twitter_fit["tw_stable"])
        # even though both share the same base final_score
        self.assertEqual(
            {node.user_id: node.final_score for node in result.platforms[0].selected_nodes},
            {"fb_stable": 0.5, "fb_diffuse": 0.5},
        )

    def test_safe_nodes_are_ranked_before_unsafe_nodes(self) -> None:
        nodes = [
            _make_node(
                "risky",
                final_score=0.99,
                eligible=False,
                manual_review_required=True,
                risk_level="high",
                risk_flags=["topic_mismatch"],
            ),
            _make_node("safe", final_score=0.10),
        ]
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=1),
        )
        selected = result.platforms[0].selected_nodes
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].user_id, "safe")
        self.assertEqual(selected[0].safety_tier, "safe")

    def test_risk_information_is_preserved_and_not_marked_safe(self) -> None:
        nodes = [
            _make_node("safe1"),
            _make_node(
                "risky",
                eligible=False,
                manual_review_required=True,
                risk_level="high",
                risk_flags=["topic_mismatch", "profile_sparse"],
                selection_reasons=["risk_flags=topic_mismatch"],
            ),
        ]
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2),
        )
        assignments = {node.user_id: node for node in result.platforms[0].selected_nodes}
        risky = assignments["risky"]

        self.assertFalse(risky.eligible)
        self.assertTrue(risky.manual_review_required)
        self.assertEqual(risky.risk_level, "high")
        self.assertEqual(risky.risk_flags, ["topic_mismatch", "profile_sparse"])
        self.assertEqual(risky.safety_tier, "not_eligible")
        self.assertNotEqual(risky.safety_tier, "safe")

        joined_warnings = " ".join(result.platforms[0].warnings)
        self.assertIn("未通过资格筛选", joined_warnings)
        self.assertIn("人工复核", joined_warnings)
        self.assertIn("high", joined_warnings)

    def test_platform_roles_assigned_from_profile(self) -> None:
        nodes = _pool(facebook=3, twitter=3, instagram=3)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3, twitter=3, instagram=3),
        )
        for platform in result.platforms:
            profile = get_platform_profile(platform.platform_key)
            assigned = [node.platform_role for node in platform.selected_nodes]
            for role in assigned:
                self.assertIn(role, profile.platform_roles)
            # 3 nodes / 3 roles -> each role used once
            self.assertEqual(set(assigned), set(profile.platform_roles))

    def test_fit_basis_is_recorded(self) -> None:
        nodes = [_make_node("1")]
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=1),
        )
        assignment = result.platforms[0].selected_nodes[0]
        self.assertTrue(assignment.fit_basis)
        self.assertTrue(
            any("platform_fit_score=" in item for item in assignment.fit_basis),
            assignment.fit_basis,
        )
        self.assertTrue(
            any("platform=" in reason for reason in assignment.selection_reasons)
        )

    def test_policy_disclaimer_present_on_every_platform(self) -> None:
        nodes = _pool(facebook=1, twitter=1, instagram=1)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=1, twitter=1, instagram=1),
        )
        for platform in result.platforms:
            self.assertIn("真实账号", platform.policy_disclaimer)
            self.assertTrue(platform.fit_limitations)
        self.assertIn("真实账号", result.policy_disclaimer)

    def test_serializable_to_json(self) -> None:
        nodes = _pool(facebook=2, twitter=3, instagram=1)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2, twitter=3, instagram=1),
        )
        payload = to_json_dict(result)
        dumped = json.dumps(payload, ensure_ascii=False)
        self.assertIn("cross_platform", dumped)
        self.assertEqual(
            json.loads(dumped)["dispatch_mode"], "cross_platform"
        )

    def test_requested_count_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PlatformSpec(platform_key="facebook", requested_count=0)
        with self.assertRaises(ValueError):
            PlatformSpec(platform_key="facebook", requested_count=-2)

    def test_duplicate_platform_specs_rejected(self) -> None:
        nodes = [_make_node("1"), _make_node("2")]
        with self.assertRaises(ValueError) as ctx:
            PlatformDispatcher().dispatch(
                score_result=_make_score_result(nodes),
                platform_specs=[
                    PlatformSpec(platform_key="facebook", requested_count=1),
                    PlatformSpec(platform_key="facebook", requested_count=1),
                ],
            )
        self.assertIn("重复", str(ctx.exception))

    def test_unsupported_platform_rejected(self) -> None:
        nodes = [_make_node("1")]
        with self.assertRaises(UnsupportedPlatformError) as ctx:
            PlatformDispatcher().dispatch(
                score_result=_make_score_result(nodes),
                platform_specs=[PlatformSpec(platform_key="myspace", requested_count=1)],
            )
        message = str(ctx.exception)
        self.assertIn("myspace", message)
        for key in PLATFORM_KEYS:
            self.assertIn(key, message)

    def test_empty_platform_specs_rejected(self) -> None:
        nodes = [_make_node("1")]
        with self.assertRaises(ValueError):
            PlatformDispatcher().dispatch(
                score_result=_make_score_result(nodes),
                platform_specs=[],
            )

    def test_missing_profile_fields_do_not_crash(self) -> None:
        minimal_node = NodeScore(
            user_id="lonely",
            user_name="",
            event_id="event_test",
            event_type="parent_child_reading",
            role_hint="",
            platform="facebook_simulated",
        )
        event = _make_event(
            constraints={
                "risk_level": "low",
                "max_selected_nodes": 1,
                "campaign_window_hours": 1,
            },
            extracted_keywords=[],
            target_audience=[],
        )
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result([minimal_node], event=event),
            platform_specs=_specs(facebook=1, twitter=1, instagram=1),
        )
        self.assertEqual(result.unique_user_count, 1)
        # 只有 facebook 有归属该平台的候选，其余平台缺口如实记录。
        self.assertEqual(result.platforms[0].selected_count, 1)
        self.assertEqual(result.platforms[1].selected_count, 0)
        self.assertEqual(result.platforms[2].selected_count, 0)
        for platform in result.platforms:
            self.assertTrue(platform.time_plan.stages)

    def test_empty_candidate_pool_reports_shortfall(self) -> None:
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result([]),
            platform_specs=_specs(facebook=3, twitter=2),
        )
        for platform in result.platforms:
            self.assertEqual(platform.selected_count, 0)
            self.assertEqual(platform.shortfall_count, platform.requested_count)
            self.assertTrue(
                any("候选不足" in warning for warning in platform.warnings)
            )
        self.assertEqual(result.unique_user_count, 0)
        self.assertEqual(result.overlapping_user_ids, [])

    def test_campaign_window_override(self) -> None:
        nodes = [_make_node("1")]
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=1),
            campaign_window_hours=10,
        )
        self.assertEqual(result.campaign_window_hours, 10)
        self.assertEqual(result.platforms[0].time_plan.campaign_window_hours, 10)


class PlatformSafetyDisclosureTest(unittest.TestCase):
    """风险/资格披露：人数缺口与合格缺口必须分开、可机读、口径一致。"""

    def test_safety_tier_counts_expose_qualified_gap(self) -> None:
        nodes = [
            _make_node("safe1"),
            _make_node("unsafe1", eligible=False, risk_flags=["topic_mismatch"]),
            _make_node("unsafe2", eligible=False, risk_flags=["topic_mismatch"]),
        ]
        facebook = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3),
        ).platforms[0]

        # 人数选满了，但合格的人不够：两个口径必须分开
        self.assertEqual(facebook.requested_count, 3)
        self.assertEqual(facebook.selected_count, 3)
        self.assertEqual(facebook.shortfall_count, 0)
        self.assertEqual(facebook.safe_shortfall_count, 2)
        self.assertEqual(
            facebook.selected_safety_tiers.model_dump(),
            {"safe": 1, "review_required": 0, "not_eligible": 2},
        )
        # 计数必须与逐节点 safety_tier 一致，不能出现两套口径
        counter = Counter(item.safety_tier for item in facebook.selected_nodes)
        self.assertEqual(counter["safe"], facebook.selected_safety_tiers.safe)
        self.assertEqual(
            counter["not_eligible"], facebook.selected_safety_tiers.not_eligible
        )
        # 不合格节点必须保持 eligible=false
        for item in facebook.selected_nodes:
            if item.safety_tier == "not_eligible":
                self.assertFalse(item.eligible)

    def test_review_required_nodes_form_their_own_tier(self) -> None:
        nodes = [
            _make_node("safe1"),
            _make_node("review1", manual_review_required=True, risk_level="medium"),
        ]
        facebook = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2),
        ).platforms[0]

        self.assertEqual(
            facebook.selected_safety_tiers.model_dump(),
            {"safe": 1, "review_required": 1, "not_eligible": 0},
        )
        review_node = next(
            item for item in facebook.selected_nodes if item.user_id == "review1"
        )
        # 需人工复核的节点仍然是 eligible，但不能被当作 safe
        self.assertTrue(review_node.eligible)
        self.assertTrue(review_node.manual_review_required)
        self.assertEqual(review_node.safety_tier, "review_required")
        self.assertEqual(facebook.shortfall_count, 0)
        self.assertEqual(facebook.safe_shortfall_count, 1)

    def test_qualified_gap_warning_is_separate_from_headcount_gap(self) -> None:
        nodes = [_make_node("safe1"), _make_node("unsafe1", eligible=False)]
        facebook = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=3),
        ).platforms[0]
        joined = " ".join(facebook.warnings)
        self.assertIn("合格节点不足", joined)
        self.assertIn("合格缺口 2 个", joined)
        self.assertIn("候选不足", joined)

    def test_no_qualified_gap_when_pool_is_fully_qualified(self) -> None:
        nodes = _pool(facebook=3, twitter=3)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2, twitter=1),
        )
        for platform in result.platforms:
            self.assertEqual(platform.safe_shortfall_count, 0)
            self.assertEqual(platform.selected_safety_tiers.not_eligible, 0)
        self.assertEqual(result.total_safe_shortfall_count, 0)
        self.assertFalse(
            any("合格节点不足" in warning for warning in result.warnings)
        )

    def test_multi_platform_aggregates_safety_tiers_consistently(self) -> None:
        nodes = [
            # facebook：1 个合格 + 1 个不合格
            _make_node("fb_safe"),
            _make_node("fb_unsafe", eligible=False),
            # twitter：1 个合格 + 1 个待复核 + 1 个不合格
            _make_node("tw_safe", platform="twitter_simulated"),
            _make_node(
                "tw_review",
                platform="twitter_simulated",
                manual_review_required=True,
            ),
            _make_node(
                "tw_unsafe",
                platform="twitter_simulated",
                eligible=False,
            ),
        ]
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=2, twitter=3),
        )

        self.assertEqual(
            result.total_selected_safety_tiers.model_dump(),
            {"safe": 2, "review_required": 1, "not_eligible": 2},
        )
        # facebook 2/2 合格 1 → 合格缺口 1；twitter 3/3 合格 1 → 合格缺口 2
        self.assertEqual(result.total_safe_shortfall_count, 3)
        for field in ("safe", "review_required", "not_eligible"):
            self.assertEqual(
                getattr(result.total_selected_safety_tiers, field),
                sum(
                    getattr(platform.selected_safety_tiers, field)
                    for platform in result.platforms
                ),
                field,
            )
        self.assertTrue(
            any("跨平台合格节点缺口" in warning for warning in result.warnings)
        )

    def test_empty_pool_safety_tiers_are_zero(self) -> None:
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result([]),
            platform_specs=_specs(facebook=2),
        )
        platform = result.platforms[0]
        self.assertEqual(
            platform.selected_safety_tiers.model_dump(),
            {"safe": 0, "review_required": 0, "not_eligible": 0},
        )
        self.assertEqual(platform.shortfall_count, platform.requested_count)
        self.assertEqual(platform.safe_shortfall_count, platform.requested_count)


class PlatformFitDisclosureTest(unittest.TestCase):
    """平台适配分：输入来源、未声明输入与免责声明必须可追溯。"""

    def test_undeclared_role_hint_is_recorded(self) -> None:
        node = _make_node("1", role_hint="custom_bridge_role")
        assignment = PlatformDispatcher().dispatch(
            score_result=_make_score_result([node]),
            platform_specs=_specs(facebook=1),
        ).platforms[0].selected_nodes[0]

        self.assertTrue(assignment.undeclared_inputs)
        self.assertIn("custom_bridge_role", assignment.undeclared_inputs[0])
        self.assertTrue(
            any(
                "未在平台 role_affinity 中声明" in item
                for item in assignment.fit_basis
            ),
            assignment.fit_basis,
        )

    def test_empty_role_hint_is_recorded_as_undeclared(self) -> None:
        node = NodeScore(
            user_id="lonely",
            user_name="",
            event_id="event_test",
            event_type="parent_child_reading",
            role_hint="",
            platform="facebook_simulated",
        )
        assignment = PlatformDispatcher().dispatch(
            score_result=_make_score_result([node]),
            platform_specs=_specs(facebook=1),
        ).platforms[0].selected_nodes[0]

        self.assertEqual(len(assignment.undeclared_inputs), 1)
        self.assertIn("unknown", assignment.undeclared_inputs[0])

    def test_declared_role_hint_is_not_flagged(self) -> None:
        node = _make_node("1", role_hint="core_broadcast")
        assignment = PlatformDispatcher().dispatch(
            score_result=_make_score_result([node]),
            platform_specs=_specs(facebook=1),
        ).platforms[0].selected_nodes[0]

        self.assertEqual(assignment.undeclared_inputs, [])
        self.assertFalse(
            any(
                "未在平台 role_affinity 中声明" in item
                for item in assignment.fit_basis
            )
        )

    def test_fit_basis_records_every_weighted_signal(self) -> None:
        node = _make_node("1")
        assignment = PlatformDispatcher().dispatch(
            score_result=_make_score_result([node]),
            platform_specs=_specs(facebook=1),
        ).platforms[0].selected_nodes[0]

        joined = " ".join(assignment.fit_basis)
        for field in (
            "topic_match_score",
            "stability_score",
            "influence_score",
            "diffusion_score",
            "activity_score",
        ):
            self.assertIn(f"{field}=", joined)
        self.assertIn("platform_fit_score=", joined)
        # 原始 final_score 必须逐字复制，不能被平台分覆盖
        self.assertEqual(assignment.final_score, node.final_score)

    def test_raw_signal_values_are_copied_verbatim(self) -> None:
        node = _make_node(
            "1",
            platform="twitter_simulated",
            final_score=0.123456,
            topic_match_score=0.234567,
            influence_score=0.345678,
            diffusion_score=0.456789,
            stability_score=0.567891,
            activity_score=0.678912,
        )
        assignment = PlatformDispatcher().dispatch(
            score_result=_make_score_result([node]),
            platform_specs=_specs(twitter=1),
        ).platforms[0].selected_nodes[0]

        self.assertEqual(assignment.final_score, node.final_score)
        self.assertEqual(assignment.topic_match_score, node.topic_match_score)
        self.assertEqual(assignment.influence_score, node.influence_score)
        self.assertEqual(assignment.diffusion_score, node.diffusion_score)
        self.assertEqual(assignment.stability_score, node.stability_score)
        self.assertEqual(assignment.activity_score, node.activity_score)

    def test_top_level_disclaimers_cover_all_platforms(self) -> None:
        nodes = _pool(facebook=1, twitter=1, instagram=1)
        result = PlatformDispatcher().dispatch(
            score_result=_make_score_result(nodes),
            platform_specs=_specs(facebook=1, twitter=1, instagram=1),
        )
        self.assertTrue(result.policy_disclaimer)
        for platform in result.platforms:
            self.assertIn(platform.policy_disclaimer, result.policy_disclaimer)
            for note in platform.fit_limitations:
                self.assertIn(note, result.fit_limitations)
        self.assertIn("真实账号", result.policy_disclaimer)
        self.assertIn("策略与内容草案", result.policy_disclaimer)

    def test_new_fields_are_json_serializable(self) -> None:
        nodes = [_make_node("safe1"), _make_node("unsafe1", eligible=False)]
        payload = to_json_dict(
            PlatformDispatcher().dispatch(
                score_result=_make_score_result(nodes),
                platform_specs=_specs(facebook=2),
            )
        )
        platform = payload["platforms"][0]
        self.assertEqual(
            platform["selected_safety_tiers"],
            {"safe": 1, "review_required": 0, "not_eligible": 1},
        )
        self.assertEqual(platform["safe_shortfall_count"], 1)
        self.assertEqual(
            payload["total_selected_safety_tiers"],
            {"safe": 1, "review_required": 0, "not_eligible": 1},
        )
        self.assertEqual(payload["total_safe_shortfall_count"], 1)
        self.assertIn("undeclared_inputs", platform["selected_nodes"][0])
        self.assertIn("platform", platform["selected_nodes"][0])

    def test_non_eligible_is_only_used_after_qualified_layers_are_exhausted(self) -> None:
        """不合格节点只能出现在「合格层已用尽」的补位位置，不能被优先选中。"""

        nodes = [
            _make_node("safe_high", final_score=0.10),
            _make_node("safe_low", final_score=0.05),
            _make_node("review1", final_score=0.90, manual_review_required=True),
            _make_node(
                "unsafe_hot",
                final_score=0.99,
                eligible=False,
                risk_level="high",
                risk_flags=["topic_mismatch"],
            ),
        ]
        tier_order = {"safe": 0, "review_required": 1, "not_eligible": 2}

        for requested in (1, 2, 3, 4):
            with self.subTest(requested=requested):
                facebook = PlatformDispatcher().dispatch(
                    score_result=_make_score_result(nodes),
                    platform_specs=_specs(facebook=requested),
                ).platforms[0]
                tiers = facebook.selected_safety_tiers

                # 选中顺序必须是 safe -> review_required -> not_eligible
                order = [
                    tier_order[item.safety_tier] for item in facebook.selected_nodes
                ]
                self.assertEqual(order, sorted(order), facebook.selected_nodes)

                # 不合格节点数量不可能超过合格层的缺口
                self.assertLessEqual(
                    tiers.not_eligible,
                    max(0, requested - tiers.safe - tiers.review_required),
                )
                if tiers.not_eligible > 0:
                    # 即使最高分的是不合格节点，也只有补位时才会用到它
                    self.assertGreater(facebook.safe_shortfall_count, 0)
                    self.assertNotIn(
                        "unsafe_hot",
                        [
                            item.user_id
                            for item in facebook.selected_nodes[: tiers.safe]
                        ],
                    )


class PlatformTimePlanBoundaryTest(unittest.TestCase):
    """活动窗口是硬上限：任何阶段都必须落在 [0, campaign_window_hours] 内。

    回归背景：``_build_time_stage`` 原先用 ``end = max(end, start + 1)``
    保证每阶段至少 1 小时，却没有把结果截回窗口，导致 ``campaign_window_hours=1``
    时产出 ``T+1h - T+2h``（超出窗口）。
    """

    WINDOWS = (1, 2, 3, 4, 5, 6, 12, 24)
    PLATFORM_KEYS = ("facebook", "twitter", "instagram")

    def _dispatch(self, window_hours: int):
        return PlatformDispatcher().dispatch(
            score_result=_make_score_result([_make_node("1")]),
            platform_specs=_specs(**{key: 1 for key in self.PLATFORM_KEYS}),
            campaign_window_hours=window_hours,
        )

    def test_every_stage_stays_inside_the_campaign_window(self) -> None:
        for window in self.WINDOWS:
            with self.subTest(window=window):
                result = self._dispatch(window)
                self.assertEqual(result.campaign_window_hours, window)
                for platform in result.platforms:
                    plan = platform.time_plan
                    self.assertEqual(plan.campaign_window_hours, window)
                    self.assertTrue(plan.stages, platform.platform_key)
                    for stage in plan.stages:
                        ctx = f"window={window} {platform.platform_key} {stage.stage_key}"
                        self.assertGreaterEqual(stage.start_hour, 0, ctx)
                        # 不允许负时长
                        self.assertGreaterEqual(
                            stage.end_hour, stage.start_hour, ctx
                        )
                        # 不允许超出活动窗口
                        self.assertLessEqual(stage.end_hour, window, ctx)

    def test_time_plan_model_rejects_out_of_window_stage(self) -> None:
        with self.assertRaises(ValidationError):
            PlatformTimePlan(
                platform_key="facebook",
                campaign_window_hours=1,
                stages=[
                    PlatformTimeStage(
                        stage_key="bad",
                        label="bad",
                        window="T+1h - T+2h",
                        start_hour=1,
                        end_hour=2,
                        focus="focus",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
