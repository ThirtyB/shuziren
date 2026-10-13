from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.data_loader import DataLoader
from influence_strategy.event_parser import RuleBasedEventParser
from influence_strategy.feature_builder import FeatureBuilder
from influence_strategy.scorer import Scorer


class ScorerTest(unittest.TestCase):
    def test_scores_and_ranks_nodes(self) -> None:
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
                },
                "2": {
                    "user_id": 2,
                    "user_name": "user_2",
                    "user_followers": 300,
                    "user_friends": 40,
                    "user_interests": ["热点", "转发"],
                    "user_description": "适合扩散",
                },
                "3": {
                    "user_id": 3,
                    "user_name": "user_3",
                    "user_followers": 50,
                    "user_friends": 5,
                    "user_interests": [],
                    "user_description": "",
                },
            }
            interactions = {
                "1": [{"interact_id": 2, "interact_type": "comment"}],
                "2": [{"interact_id": 1, "interact_type": "reposts"}],
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {
                        "neighbor_count": 2,
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
                        "neighbor_count": 3,
                        "mutual_neighbor_count": 1,
                        "received_interaction_count": 2,
                        "received_comment_count": 1,
                        "received_repost_count": 1,
                        "made_interaction_count": 7,
                        "made_comment_count": 2,
                        "made_repost_count": 5,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "3": {
                    **profiles["3"],
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
                "围绕亲子阅读和英语启蒙做一次传播活动，提升讨论度并控制风险。"
            )
            feature_result = FeatureBuilder().build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles=loader.load_enriched_profiles(),
                source_user_ids=set(loader.load_interactions().keys()),
            )
            score_result = Scorer().score(feature_result)

            self.assertEqual(score_result.summary.node_count, 3)
            self.assertEqual(score_result.node_scores[0].user_id, "1")
            self.assertTrue(score_result.node_scores[0].eligible)
            self.assertGreater(score_result.node_scores[0].final_score, score_result.node_scores[1].final_score)
            self.assertEqual(score_result.node_scores[-1].risk_level, "high")
            self.assertIn("isolated_node", score_result.node_scores[-1].risk_flags)

    def test_to_frame_contains_score_columns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)

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

            loader = DataLoader(root)
            event = RuleBasedEventParser().parse("普通传播活动")
            feature_result = FeatureBuilder().build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles={},
                source_user_ids=set(),
            )
            score_result = Scorer().score(feature_result)
            frame = Scorer().to_frame(score_result)

            self.assertIn("final_score", frame.columns)
            self.assertIn("risk_score", frame.columns)
            self.assertIn("eligible", frame.columns)

    def test_high_influence_irrelevant_node_is_filtered(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            product_info = {"product_name": "abc_reading", "influencer_ids": ["2"]}
            profiles = {
                "1": {
                    "user_id": 1,
                    "user_name": "shipping_node",
                    "user_followers": 200,
                    "user_friends": 40,
                    "user_interests": ["shipping", "energy", "logistics"],
                    "user_description": "shipping and logistics updates",
                },
                "2": {
                    "user_id": 2,
                    "user_name": "lifestyle_star",
                    "user_followers": 500000,
                    "user_friends": 600,
                    "user_interests": ["travel", "food", "photography"],
                    "user_description": "travel and lifestyle account",
                },
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {
                        "neighbor_count": 2,
                        "received_interaction_count": 5,
                        "made_interaction_count": 3,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "2": {
                    **profiles["2"],
                    "graph_attributes": {
                        "neighbor_count": 12,
                        "received_interaction_count": 30,
                        "made_interaction_count": 10,
                        "isolated": False,
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
            (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
                json.dumps(enriched_profiles, ensure_ascii=False),
                encoding="utf-8",
            )

            event = RuleBasedEventParser().parse(
                {
                    "event_title": "Shipping disruption",
                    "event_description": "Shipping disruption affects energy logistics.",
                    "target_goal": "awareness",
                    "target_audience": ["general_public"],
                }
            )
            feature_result = FeatureBuilder().build_features(
                product_context=DataLoader(root).load_product_context(),
                profiles=DataLoader(root).load_profiles(),
                event=event,
                enriched_profiles=DataLoader(root).load_enriched_profiles(),
                source_user_ids=set(),
            )
            score_result = Scorer().score(feature_result)

            shipping_node = next(node for node in score_result.node_scores if node.user_id == "1")
            lifestyle_node = next(node for node in score_result.node_scores if node.user_id == "2")

            self.assertTrue(shipping_node.eligible)
            self.assertFalse(lifestyle_node.eligible)
            self.assertIn("generic_high_influence_mismatch", lifestyle_node.risk_flags)


if __name__ == "__main__":
    unittest.main()
