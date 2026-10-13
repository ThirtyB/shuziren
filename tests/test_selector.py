from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.data_loader import DataLoader
from influence_strategy.event_parser import RuleBasedEventParser
from influence_strategy.feature_builder import FeatureBuilder
from influence_strategy.models import SelectedNode
from influence_strategy.scorer import Scorer
from influence_strategy.selector import Selector


class FakeSelectorLLMClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        payload = json.loads(user_prompt)
        shortlisted = payload["shortlist"]
        selected_order = []
        preferred_roles = [
            "interaction_response_node",
            "core_publish_node",
            "amplification_node",
        ]
        for index, item in enumerate(shortlisted[: payload["constraints"]["max_selected_nodes"]]):
            selected_order.append(
                {
                    "user_id": item["user_id"],
                    "recommended_role": preferred_roles[index % len(preferred_roles)],
                    "reasoning": [f"picked_{item['user_id']}"],
                }
            )
        return {
            "selected_order": selected_order,
            "fallback_order": [item["user_id"] for item in shortlisted[payload["constraints"]["max_selected_nodes"] :]],
            "global_notes": ["selector rerank used"],
        }


def _build_targeted_fixture():
    """Build a real score_result using a temp workspace with 4 profiles.

    Returns a tuple ``(score_result, candidate_user_ids)`` where
    ``candidate_user_ids`` are the user ids that the scorer produced.
    """
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_dir = root / "data" / "raw"
        derived_dir = root / "data" / "derived"
        raw_dir.mkdir(parents=True)
        derived_dir.mkdir(parents=True)

        product_info = {
            "product_name": "abc_reading",
            "domain": "linguistics",
            "ads": "reading context",
            "influencer_ids": ["1"],
        }
        profiles = {
            "1": {
                "user_id": 1,
                "user_name": "user_1",
                "user_followers": 1200,
                "user_friends": 80,
                "user_interests": ["亲子阅读", "英语启蒙"],
                "user_description": "专注亲子阅读和英语启蒙",
                "platform": "facebook_simulated",
            },
            "2": {
                "user_id": 2,
                "user_name": "user_2",
                "user_followers": 350,
                "user_friends": 40,
                "user_interests": ["转发传播", "阅读活动"],
                "user_description": "适合扩散",
                "platform": "facebook_simulated",
            },
            "3": {
                "user_id": 3,
                "user_name": "user_3",
                "user_followers": 180,
                "user_friends": 25,
                "user_interests": ["亲子问答", "教育交流"],
                "user_description": "擅长评论互动与答疑",
                "platform": "instagram_simulated",
            },
            "4": {
                "user_id": 4,
                "user_name": "user_4",
                "user_followers": 15,
                "user_friends": 3,
                "user_interests": [],
                "user_description": "",
            },
        }
        interactions = {
            "1": [{"interact_id": 2, "interact_type": "comment"}],
            "2": [{"interact_id": 1, "interact_type": "reposts"}],
            "3": [{"interact_id": 1, "interact_type": "comment"}],
        }
        enriched_profiles = {
            "1": {
                **profiles["1"],
                "graph_attributes": {
                    "neighbor_count": 3,
                    "mutual_neighbor_count": 1,
                    "received_interaction_count": 8,
                    "received_comment_count": 6,
                    "received_repost_count": 2,
                    "made_interaction_count": 2,
                    "made_repost_count": 2,
                    "isolated": False,
                },
                "neighbors": [],
            },
            "2": {
                **profiles["2"],
                "graph_attributes": {
                    "neighbor_count": 4,
                    "mutual_neighbor_count": 1,
                    "received_interaction_count": 2,
                    "received_comment_count": 1,
                    "received_repost_count": 1,
                    "made_interaction_count": 8,
                    "made_comment_count": 2,
                    "made_repost_count": 6,
                    "isolated": False,
                },
                "neighbors": [],
            },
            "3": {
                **profiles["3"],
                "graph_attributes": {
                    "neighbor_count": 2,
                    "mutual_neighbor_count": 1,
                    "received_interaction_count": 4,
                    "received_comment_count": 4,
                    "made_interaction_count": 3,
                    "made_comment_count": 3,
                    "isolated": False,
                },
                "neighbors": [],
            },
            "4": {
                **profiles["4"],
                "graph_attributes": {
                    "neighbor_count": 0,
                    "mutual_neighbor_count": 0,
                    "received_interaction_count": 1,
                    "made_interaction_count": 1,
                    "self_interaction_count": 1,
                    "isolated": True,
                },
                "neighbors": [],
            },
        }

        (raw_dir / "abc_reading_product_info.json").write_text(
            json.dumps(product_info, ensure_ascii=False),
            encoding="utf-8",
        )
        (raw_dir / "abc_reading_profile.graph.anon").write_text(
            json.dumps(profiles, ensure_ascii=False),
            encoding="utf-8",
        )
        (raw_dir / "abc_reading_interaction.graph.anon").write_text(
            json.dumps(interactions, ensure_ascii=False),
            encoding="utf-8",
        )
        (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
            json.dumps(enriched_profiles, ensure_ascii=False),
            encoding="utf-8",
        )

        loader = DataLoader(root)
        event = RuleBasedEventParser().parse(
            {
                "event_description": "围绕亲子阅读和英语启蒙做一次传播活动，提升讨论度并控制风险。",
                "constraints": {"max_selected_nodes": 3},
            }
        )
        feature_result = FeatureBuilder().build_features(
            product_context=loader.load_product_context(),
            profiles=loader.load_profiles(),
            event=event,
            enriched_profiles=loader.load_enriched_profiles(),
            source_user_ids=set(loader.load_interactions().keys()),
        )
        score_result = Scorer().score(feature_result)
    return score_result


class SelectorTest(unittest.TestCase):
    def test_selects_primary_and_fallback_nodes(self) -> None:
        score_result = _build_targeted_fixture()
        selection_result = Selector().select(score_result)

        self.assertEqual(selection_result.summary.selected_count, 3)
        self.assertLessEqual(
            selection_result.summary.selected_count,
            score_result.event.constraints.max_selected_nodes,
        )
        selected_roles = {node.selected_role for node in selection_result.selected_nodes}
        self.assertIn("core_publish_node", selected_roles)
        self.assertIn("amplification_node", selected_roles)
        self.assertGreaterEqual(selection_result.summary.fallback_count, 1)

    def test_automatic_mode_keeps_selection_mode_default(self) -> None:
        """Regression: when target_user_id is None, the summary must stay automatic."""
        score_result = _build_targeted_fixture()
        selection_result = Selector().select(score_result)
        self.assertEqual(selection_result.summary.selection_mode, "automatic")
        self.assertIsNone(selection_result.summary.target_user_id)
        self.assertEqual(selection_result.summary.target_warning, [])
        self.assertNotEqual(len(selection_result.selected_nodes), 0)

    def test_to_frame_supports_primary_and_fallback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            (raw_dir / "abc_reading_product_info.json").write_text(
                json.dumps({"product_name": "abc_reading"}),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_profile.graph.anon").write_text(
                json.dumps({"1": {"user_id": 1, "user_name": "user_1"}}),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_interaction.graph.anon").write_text(
                json.dumps({}),
                encoding="utf-8",
            )
            (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {
                            "user_id": 1,
                            "user_name": "user_1",
                            "graph_attributes": {},
                            "neighbors": [],
                        }
                    }
                ),
                encoding="utf-8",
            )

            loader = DataLoader(root)
            event = RuleBasedEventParser().parse("普通传播活动")
            feature_result = FeatureBuilder().build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles=loader.load_enriched_profiles(),
                source_user_ids=set(),
            )
            score_result = Scorer().score(feature_result)
            selection_result = Selector().select(score_result)

            primary_frame = Selector().to_frame(selection_result)
            fallback_frame = Selector().to_frame(selection_result, bucket="fallback")

            self.assertIn("selected_role", primary_frame.columns)
            self.assertIn("selection_bucket", primary_frame.columns)
            self.assertIn("selection_bucket", fallback_frame.columns)

    def test_selector_can_apply_llm_rerank_plan(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            product_info = {"product_name": "abc_reading", "influencer_ids": []}
            profiles = {
                "1": {
                    "user_id": 1,
                    "user_name": "user_1",
                    "user_followers": 600,
                    "user_friends": 40,
                    "user_interests": ["阅读", "传播"],
                    "user_description": "适合首发",
                },
                "2": {
                    "user_id": 2,
                    "user_name": "user_2",
                    "user_followers": 220,
                    "user_friends": 60,
                    "user_interests": ["问答", "评论"],
                    "user_description": "擅长互动答疑",
                },
                "3": {
                    "user_id": 3,
                    "user_name": "user_3",
                    "user_followers": 260,
                    "user_friends": 90,
                    "user_interests": ["扩散", "转述"],
                    "user_description": "适合扩散",
                },
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {
                        "neighbor_count": 2,
                        "received_interaction_count": 5,
                        "made_interaction_count": 2,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "2": {
                    **profiles["2"],
                    "graph_attributes": {
                        "neighbor_count": 2,
                        "received_interaction_count": 4,
                        "made_interaction_count": 4,
                        "made_comment_count": 4,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "3": {
                    **profiles["3"],
                    "graph_attributes": {
                        "neighbor_count": 4,
                        "received_interaction_count": 3,
                        "made_interaction_count": 7,
                        "made_repost_count": 5,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
            }

            (raw_dir / "abc_reading_product_info.json").write_text(
                json.dumps(product_info), encoding="utf-8"
            )
            (raw_dir / "abc_reading_profile.graph.anon").write_text(
                json.dumps(profiles, ensure_ascii=False), encoding="utf-8"
            )
            (raw_dir / "abc_reading_interaction.graph.anon").write_text(
                json.dumps({}), encoding="utf-8"
            )
            (
                derived_dir / "abc_reading_profile_with_neighbors.graph.anon"
            ).write_text(
                json.dumps(enriched_profiles, ensure_ascii=False), encoding="utf-8"
            )

            loader = DataLoader(root)
            event = RuleBasedEventParser().parse(
                {
                    "event_description": "围绕阅读讨论做一次传播活动，重点做好互动承接。",
                    "constraints": {"max_selected_nodes": 2},
                }
            )
            feature_result = FeatureBuilder().build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles=loader.load_enriched_profiles(),
                source_user_ids=set(),
            )
            score_result = Scorer().score(feature_result)
            selection_result = Selector().select(
                score_result,
                use_llm=True,
                llm_client=FakeSelectorLLMClient(),
            )

            self.assertEqual(
                selection_result.selected_nodes[0].selected_role, "interaction_response_node"
            )
            self.assertTrue(
                any(
                    reason.startswith("llm_selector=")
                    for reason in selection_result.selected_nodes[0].selection_reasons
                )
            )

    def test_selector_rebalances_homogeneous_selected_roles(self) -> None:
        homogeneous_nodes = [
            SelectedNode(
                user_id=str(index),
                user_name=f"user_{index}",
                event_id="event_test",
                event_type="public_affairs",
                role_hint="amplification",
                final_score=0.8,
                eligible=True,
                priority_tier="high",
                selection_rank=index,
                selected_role="amplification_node",
                dispatch_stage="stage_3_amplify",
                selection_bucket="primary",
                selection_reasons=[
                    "selected_role=amplification_node",
                    "dispatch_stage=stage_3_amplify",
                    "selection_bucket=primary",
                ],
            )
            for index in range(1, 5)
        ]

        rebalanced_nodes = Selector()._rebalance_selected_roles(homogeneous_nodes)

        selected_roles = [node.selected_role for node in rebalanced_nodes]
        self.assertEqual(
            selected_roles,
            [
                "core_publish_node",
                "interaction_response_node",
                "amplification_node",
                "support_node",
            ],
        )
        self.assertTrue(
            any(
                reason.startswith("role_rebalanced=")
                for node in rebalanced_nodes
                for reason in node.selection_reasons
            )
        )


class TargetedSelectorTest(unittest.TestCase):
    """Tests covering single-user targeted selection (Selector._select_targeted)."""

    def test_targeted_mode_picks_only_target_user(self) -> None:
        score_result = _build_targeted_fixture()
        target_id = "2"
        selection_result = Selector().select(score_result, target_user_id=target_id)

        self.assertEqual(len(selection_result.selected_nodes), 1)
        self.assertEqual(selection_result.selected_nodes[0].user_id, target_id)
        self.assertEqual(
            selection_result.selected_nodes[0].selected_role, "core_publish_node"
        )
        self.assertEqual(
            selection_result.selected_nodes[0].dispatch_stage, "stage_1_launch"
        )
        self.assertEqual(selection_result.selected_nodes[0].selection_rank, 1)
        self.assertEqual(
            selection_result.selected_nodes[0].selection_bucket, "primary"
        )
        self.assertEqual(selection_result.fallback_nodes, [])

        summary = selection_result.summary
        self.assertEqual(summary.selection_mode, "targeted")
        self.assertEqual(summary.target_user_id, target_id)
        self.assertEqual(summary.target_role, "core_publish_node")
        self.assertEqual(
            summary.selected_role_distribution, {"core_publish_node": 1}
        )

    def test_targeted_mode_does_not_mutate_scorer_output(self) -> None:
        score_result = _build_targeted_fixture()
        target_id = "1"
        target_score = next(
            node for node in score_result.node_scores if node.user_id == target_id
        )
        original_final = target_score.final_score
        original_topic = target_score.topic_match_score
        original_eligible = target_score.eligible
        original_manual_review = target_score.manual_review_required
        original_risk_level = target_score.risk_level

        selection_result = Selector().select(score_result, target_user_id=target_id)

        self.assertEqual(
            selection_result.selected_nodes[0].final_score, original_final
        )
        self.assertEqual(
            selection_result.selected_nodes[0].topic_match_score, original_topic
        )
        self.assertEqual(
            selection_result.selected_nodes[0].eligible, original_eligible
        )
        self.assertEqual(
            selection_result.selected_nodes[0].manual_review_required,
            original_manual_review,
        )
        self.assertEqual(
            selection_result.selected_nodes[0].risk_level, original_risk_level
        )

        other_ids = {
            node.user_id
            for node in score_result.node_scores
            if node.user_id != target_id
        }
        self.assertEqual(selection_result.selected_nodes[0].user_id, target_id)
        self.assertFalse(
            any(n.user_id in other_ids for n in selection_result.selected_nodes)
        )

    def test_targeted_mode_emits_warning_for_low_topic_match(self) -> None:
        score_result = _build_targeted_fixture()
        candidates = list(score_result.node_scores)
        self.assertGreater(len(candidates), 0, "fixture must contain candidates")

        topic_scores = [node.topic_match_score for node in candidates]
        median_topic = sorted(topic_scores)[len(topic_scores) // 2]
        weakest = min(candidates, key=lambda node: node.topic_match_score)
        if weakest.topic_match_score >= median_topic:
            self.skipTest(
                "scorer did not produce a below-median topic_match candidate; "
                f"weakest={weakest.topic_match_score:.3f} median={median_topic:.3f}"
            )

        selection_result = Selector().select(
            score_result, target_user_id=weakest.user_id
        )
        self.assertEqual(len(selection_result.selected_nodes), 1)
        self.assertEqual(
            selection_result.selected_nodes[0].user_id, weakest.user_id
        )
        joined_warnings = "; ".join(selection_result.summary.target_warning)
        self.assertTrue(
            any(
                phrase in joined_warnings
                for phrase in ("话题匹配度", "综合评分", "资格", "复核", "风险")
            ),
            f"expected at least one warning, got {selection_result.summary.target_warning}",
        )

    def test_targeted_mode_emits_warning_for_high_risk(self) -> None:
        score_result = _build_targeted_fixture()
        high_risk = [
            node
            for node in score_result.node_scores
            if node.risk_level == "high"
        ]
        if not high_risk:
            self.skipTest(
                "scorer did not flag any candidate as high risk in this fixture"
            )

        target = high_risk[0]
        selection_result = Selector().select(
            score_result, target_user_id=target.user_id
        )
        self.assertEqual(len(selection_result.selected_nodes), 1)
        self.assertEqual(
            selection_result.selected_nodes[0].user_id, target.user_id
        )
        self.assertEqual(
            selection_result.selected_nodes[0].risk_level, "high"
        )
        joined_warnings = "; ".join(selection_result.summary.target_warning)
        self.assertTrue(
            any(phrase in joined_warnings for phrase in ("high", "风险")),
            f"expected high-risk warning, got {selection_result.summary.target_warning}",
        )

    def test_targeted_mode_raises_for_unknown_user(self) -> None:
        score_result = _build_targeted_fixture()
        with self.assertRaises(ValueError) as ctx:
            Selector().select(score_result, target_user_id="not-a-real-user")
        self.assertIn("not-a-real-user", str(ctx.exception))

    def test_targeted_mode_reports_target_platform(self) -> None:
        """单人分发结果必须能看到被选中的人属于哪个平台。"""

        score_result = _build_targeted_fixture()
        selection_result = Selector().select(score_result, target_user_id="2")

        summary = selection_result.summary
        self.assertEqual(summary.target_platform, "facebook")
        self.assertEqual(
            selection_result.selected_nodes[0].platform, "facebook_simulated"
        )
        self.assertIn(
            "platform=facebook",
            selection_result.selected_nodes[0].selection_reasons,
        )


class PlatformScopedSelectionTest(unittest.TestCase):
    """多人分发：指定平台后只能从该平台已有的数字人中选人。"""

    def test_dispatch_platform_restricts_candidates(self) -> None:
        score_result = _build_targeted_fixture()
        selection_result = Selector().select(
            score_result, dispatch_platform="instagram"
        )

        self.assertEqual(selection_result.summary.dispatch_platform, "instagram")
        self.assertTrue(selection_result.selected_nodes)
        for node in (
            selection_result.selected_nodes + selection_result.fallback_nodes
        ):
            self.assertEqual(node.platform, "instagram_simulated")

    def test_dispatch_platform_with_no_members_selects_nothing(self) -> None:
        score_result = _build_targeted_fixture()
        selection_result = Selector().select(
            score_result, dispatch_platform="twitter"
        )

        self.assertEqual(selection_result.summary.dispatch_platform, "twitter")
        self.assertEqual(selection_result.selected_nodes, [])
        self.assertEqual(selection_result.summary.selected_count, 0)

    def test_dispatch_platform_accepts_dataset_suffix(self) -> None:
        score_result = _build_targeted_fixture()
        selection_result = Selector().select(
            score_result, dispatch_platform="instagram_simulated"
        )
        self.assertEqual(selection_result.summary.dispatch_platform, "instagram")

    def test_no_dispatch_platform_keeps_full_pool(self) -> None:
        score_result = _build_targeted_fixture()
        selection_result = Selector().select(score_result)
        self.assertIsNone(selection_result.summary.dispatch_platform)
        platforms = {
            node.platform for node in selection_result.selected_nodes
        }
        self.assertGreater(len(platforms), 1, platforms)


if __name__ == "__main__":
    unittest.main()