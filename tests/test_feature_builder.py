from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.data_loader import DataLoader
from influence_strategy.event_parser import RuleBasedEventParser
from influence_strategy.feature_builder import FeatureBuilder
from influence_strategy.models import UserProfile


class FakeFeatureLLMClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        payload = json.loads(user_prompt)
        nodes = {}
        for card in payload["candidate_cards"]:
            user_id = card["user_id"]
            is_target = str(user_id) == "2"
            nodes[f"id{user_id}"] = {
                "semantic_relevance_score": 0.95 if is_target else 0.35,
                "audience_fit_score": 0.90 if is_target else 0.30,
                "role_fit_score": 0.88 if is_target else 0.32,
                "narrative_fit_score": 0.86 if is_target else 0.28,
                "risk_conflict_score": 0.05,
                "novelty_score": 0.70 if is_target else 0.20,
                "semantic_tags": [f"fit_{user_id}"],
                "reasoning": [f"reason_{user_id}"],
            }
        return {"nodes": nodes}


class FeatureBuilderTest(unittest.TestCase):
    def test_builds_ranked_node_features(self) -> None:
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
                    "user_followers": 1000,
                    "user_friends": 50,
                    "user_interests": ["亲子阅读", "英语启蒙"],
                    "user_description": "专注亲子阅读和英语启蒙",
                },
                "2": {
                    "user_id": 2,
                    "user_name": "user_2",
                    "user_followers": 200,
                    "user_friends": 80,
                    "user_interests": ["互动", "问答"],
                    "user_description": "擅长评论互动",
                },
                "3": {
                    "user_id": 3,
                    "user_name": "user_3",
                    "user_followers": 10,
                    "user_friends": 5,
                    "user_interests": [],
                    "user_description": "",
                },
            }
            interactions = {
                "1": [
                    {
                        "text_raw": "raw",
                        "text_comment": "comment",
                        "interact_type": "comment",
                        "interact_id": 2,
                    }
                ],
                "2": [
                    {
                        "text_raw": "raw",
                        "text_comment": "repost",
                        "interact_type": "reposts",
                        "interact_id": 1,
                    }
                ],
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {
                        "neighbor_count": 1,
                        "mutual_neighbor_count": 1,
                        "received_interaction_count": 5,
                        "received_comment_count": 4,
                        "received_repost_count": 1,
                        "made_interaction_count": 1,
                        "made_comment_count": 0,
                        "made_repost_count": 1,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "2": {
                    **profiles["2"],
                    "graph_attributes": {
                        "neighbor_count": 1,
                        "mutual_neighbor_count": 1,
                        "received_interaction_count": 1,
                        "received_comment_count": 1,
                        "received_repost_count": 0,
                        "made_interaction_count": 4,
                        "made_comment_count": 3,
                        "made_repost_count": 1,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "3": {
                    **profiles["3"],
                    "graph_attributes": {
                        "neighbor_count": 0,
                        "mutual_neighbor_count": 0,
                        "received_interaction_count": 0,
                        "made_interaction_count": 0,
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
            event = RuleBasedEventParser().parse("围绕亲子阅读和英语启蒙做一次传播活动，提升讨论度。")
            builder = FeatureBuilder()
            result = builder.build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles=loader.load_enriched_profiles(),
                source_user_ids=set(loader.load_interactions().keys()),
            )

            self.assertEqual(result.summary.node_count, 3)
            self.assertEqual(result.node_features[0].user_id, "1")
            self.assertGreater(result.node_features[0].topic_match_score, 0.0)
            self.assertEqual(result.node_features[0].role_hint, "core_broadcast")
            self.assertTrue(all(0.0 <= item.feature_ready_score <= 1.0 for item in result.node_features))

    def test_to_frame_exposes_feature_columns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)

            (raw_dir / "abc_reading_product_info.json").write_text(
                json.dumps({"product_name": "abc_reading", "influencer_ids": []}),
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
            builder = FeatureBuilder()
            event = RuleBasedEventParser().parse("普通传播活动")
            result = builder.build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles={},
                source_user_ids=set(),
            )
            frame = builder.to_frame(result)

            self.assertIn("feature_ready_score", frame.columns)
            self.assertIn("topic_match_score", frame.columns)
            self.assertEqual(len(frame), 1)

    def test_llm_feature_enrichment_updates_candidate_scores(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            product_info = {
                "product_name": "abc_reading",
                "domain": "reading",
                "influencer_ids": [],
            }
            profiles = {
                "1": {
                    "user_id": 1,
                    "user_name": "node_1",
                    "user_followers": 300,
                    "user_friends": 50,
                    "user_interests": ["泛话题"],
                    "user_description": "general topic account",
                },
                "2": {
                    "user_id": 2,
                    "user_name": "node_2",
                    "user_followers": 180,
                    "user_friends": 60,
                    "user_interests": ["亲子阅读", "英语启蒙"],
                    "user_description": "focused on parent child reading",
                },
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {
                        "neighbor_count": 3,
                        "received_interaction_count": 5,
                        "made_interaction_count": 4,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "2": {
                    **profiles["2"],
                    "graph_attributes": {
                        "neighbor_count": 2,
                        "received_interaction_count": 3,
                        "made_interaction_count": 2,
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
                "围绕亲子阅读和英语启蒙做一次传播活动，提升讨论度。",
                use_llm=True,
                llm_client=None,
            )
            builder = FeatureBuilder()
            result = builder.build_features(
                product_context=DataLoader(root).load_product_context(),
                profiles=DataLoader(root).load_profiles(),
                event=event,
                enriched_profiles=DataLoader(root).load_enriched_profiles(),
                source_user_ids=set(),
                use_llm=True,
                llm_client=FakeFeatureLLMClient(),
            )

            top_feature = result.node_features[0]
            self.assertEqual(top_feature.user_id, "2")
            self.assertTrue(top_feature.llm_feature_used)
            self.assertGreater(top_feature.llm_feature_score, 0.5)
            self.assertIn("fit_2", top_feature.semantic_tags)

    def test_llm_feature_enrichment_accepts_list_style_response(self) -> None:
        class ListStyleFeatureLLMClient:
            def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                items = []
                for card in payload["candidate_cards"]:
                    items.append(
                        {
                            "user_id": card["user_id"],
                            "semantic_relevance_score": 0.80 if str(card["user_id"]) == "1" else 0.25,
                            "audience_fit_score": 0.78,
                            "role_fit_score": 0.74,
                            "narrative_fit_score": 0.70,
                            "risk_conflict_score": 0.05,
                            "novelty_score": 0.30,
                            "semantic_tags": "semantics,event_fit",
                            "reasoning": "reranked by list response",
                        }
                    )
                return {"data": items}

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
                    "user_name": "node_1",
                    "user_followers": 200,
                    "user_friends": 30,
                    "user_interests": ["shipping", "energy"],
                    "user_description": "shipping and energy account",
                },
                "2": {
                    "user_id": 2,
                    "user_name": "node_2",
                    "user_followers": 800,
                    "user_friends": 70,
                    "user_interests": ["travel", "food"],
                    "user_description": "travel lifestyle account",
                },
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {
                        "neighbor_count": 2,
                        "received_interaction_count": 4,
                        "made_interaction_count": 3,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "2": {
                    **profiles["2"],
                    "graph_attributes": {
                        "neighbor_count": 5,
                        "received_interaction_count": 8,
                        "made_interaction_count": 6,
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
            result = FeatureBuilder().build_features(
                product_context=DataLoader(root).load_product_context(),
                profiles=DataLoader(root).load_profiles(),
                event=event,
                enriched_profiles=DataLoader(root).load_enriched_profiles(),
                source_user_ids=set(),
                use_llm=True,
                llm_client=ListStyleFeatureLLMClient(),
            )

            self.assertEqual(result.node_features[0].user_id, "1")
            self.assertTrue(result.node_features[0].llm_feature_used)
            self.assertIn("semantics", result.node_features[0].semantic_tags)

    def test_generic_semantic_tags_do_not_create_false_keyword_hits(self) -> None:
        event = RuleBasedEventParser().parse(
            {
                "event_title": "Shipping disruption",
                "event_description": "Military shipping disruption affects energy logistics.",
                "target_goal": "awareness",
                "target_audience": ["general_public"],
            }
        )
        builder = FeatureBuilder()
        lifestyle_profile = UserProfile(
            user_id="1",
            user_name="lifestyle_node",
            user_followers=1000,
            user_friends=100,
            user_interests=["travel", "food", "photography"],
            user_description="daily lifestyle notes",
        )
        matches = builder._match_keywords(  # noqa: SLF001 - unit test for regression
            profile=lifestyle_profile,
            normalized_keywords=builder._prepare_keywords(builder._event_match_keywords(event)),
        )

        self.assertNotIn("general_public", matches)
        self.assertNotIn("military", matches)


if __name__ == "__main__":
    unittest.main()
