from __future__ import annotations

import unittest

from influence_strategy.feature_builder import FeatureBuilder
from influence_strategy.models import (
    EnrichedUserProfile,
    EventConstraints,
    GraphAttributes,
    ParsedEvent,
    ProductContext,
    UserProfile,
)
from influence_strategy.scorer import Scorer
from influence_strategy.selector import Selector
from influence_strategy.strategy_generator import StrategyGenerator


class StrategyGeneratorTest(unittest.TestCase):
    def test_generate_builds_structured_strategy(self) -> None:
        product_context = ProductContext(
            product_name="abc_reading",
            domain="reading",
            influencer_ids=["1"],
        )
        profiles = {
            "1": UserProfile(
                user_id="1",
                user_name="core_parent_reader",
                user_followers=1200,
                user_friends=80,
                user_interests=["parent_child_reading", "english_learning"],
                user_description="focus on parent child reading and english learning discussions",
            ),
            "2": UserProfile(
                user_id="2",
                user_name="reading_amplifier",
                user_followers=120,
                user_friends=90,
                user_interests=["parent_child_reading", "content_diffusion"],
                user_description="good at diffusion for reading related activities",
            ),
            "3": UserProfile(
                user_id="3",
                user_name="interactive_support",
                user_followers=90,
                user_friends=40,
                user_interests=["english_learning", "parent_child_support"],
                user_description="good at qa and interactive response for parent users",
            ),
            "4": UserProfile(
                user_id="4",
                user_name="weak_node",
                user_followers=12,
                user_friends=3,
                user_interests=[],
                user_description="",
            ),
        }
        enriched_profiles = {
            "1": EnrichedUserProfile(
                **profiles["1"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=3,
                    mutual_neighbor_count=1,
                    received_interaction_count=8,
                    received_comment_count=6,
                    received_repost_count=2,
                    made_interaction_count=2,
                    made_repost_count=2,
                    isolated=False,
                ),
            ),
            "2": EnrichedUserProfile(
                **profiles["2"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=4,
                    mutual_neighbor_count=1,
                    received_interaction_count=2,
                    received_comment_count=1,
                    received_repost_count=1,
                    made_interaction_count=8,
                    made_comment_count=2,
                    made_repost_count=6,
                    isolated=False,
                ),
            ),
            "3": EnrichedUserProfile(
                **profiles["3"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=2,
                    mutual_neighbor_count=1,
                    received_interaction_count=4,
                    received_comment_count=4,
                    made_interaction_count=5,
                    made_comment_count=5,
                    isolated=False,
                ),
            ),
            "4": EnrichedUserProfile(
                **profiles["4"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=0,
                    mutual_neighbor_count=0,
                    received_interaction_count=1,
                    made_interaction_count=1,
                    self_interaction_count=1,
                    isolated=True,
                ),
            ),
        }
        event = ParsedEvent(
            event_id="event_001",
            product_name="abc_reading",
            event_title="Parent child reading campaign",
            event_description="Promote parent child reading and english learning engagement with controlled risk.",
            target_goal="engagement",
            event_type="parent_child_reading",
            target_audience=["parent_child", "english_learning"],
            extracted_keywords=["parent_child_reading", "english_learning", "engagement"],
            constraints=EventConstraints(
                risk_level="medium",
                max_selected_nodes=3,
                max_frequency_per_day=3,
                campaign_window_hours=24,
            ),
        )

        feature_result = FeatureBuilder().build_features(
            product_context=product_context,
            profiles=profiles,
            event=event,
            enriched_profiles=enriched_profiles,
            source_user_ids={"1", "2", "3"},
        )
        score_result = Scorer().score(feature_result)
        selection_result = Selector().select(score_result)
        strategy_result = StrategyGenerator().generate(selection_result)

        self.assertEqual(strategy_result.summary.selected_count, 3)
        self.assertEqual(strategy_result.strategy.platform_plan.primary_platform, "facebook_simulated")
        self.assertTrue(strategy_result.stage_plans)

        stage_names = {item.stage_name for item in strategy_result.stage_plans}
        self.assertIn("stage_1_launch", stage_names)

        primary_roles = {item.selected_role for item in strategy_result.selected_nodes}
        self.assertIn("core_publish_node", primary_roles)
        self.assertTrue(primary_roles)

    def test_to_frame_and_stage_frame_export_columns(self) -> None:
        product_context = ProductContext(product_name="abc_reading")
        profiles = {
            "1": UserProfile(
                user_id="1",
                user_name="user_1",
                user_followers=100,
                user_interests=["reading"],
                user_description="reading sharing",
            ),
        }
        enriched_profiles = {
            "1": EnrichedUserProfile(
                **profiles["1"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=1,
                    received_interaction_count=3,
                    made_interaction_count=1,
                    isolated=False,
                ),
            ),
        }
        event = ParsedEvent(
            event_id="event_002",
            product_name="abc_reading",
            event_title="Reading activity",
            event_description="General reading event.",
            target_goal="awareness",
            event_type="general_influence_event",
            target_audience=["general_public"],
            extracted_keywords=["reading"],
            constraints=EventConstraints(max_selected_nodes=1),
        )

        feature_result = FeatureBuilder().build_features(
            product_context=product_context,
            profiles=profiles,
            event=event,
            enriched_profiles=enriched_profiles,
            source_user_ids={"1"},
        )
        score_result = Scorer().score(feature_result)
        selection_result = Selector().select(score_result)
        strategy_result = StrategyGenerator().generate(selection_result)

        primary_frame = StrategyGenerator().to_frame(strategy_result)
        stage_frame = StrategyGenerator().stage_frame(strategy_result)

        self.assertIn("recommended_action", primary_frame.columns)
        self.assertIn("timing_window", primary_frame.columns)
        self.assertIn("content_focus", stage_frame.columns)
        self.assertIn("success_signal", stage_frame.columns)


if __name__ == "__main__":
    unittest.main()
