from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.data_loader import DataLoader


class DataLoaderTest(unittest.TestCase):
    def test_loads_standardized_dataset_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            product_info = {
                "product_name": "abc_reading",
                "domain": "linguistics",
                "ads": "sample context",
                "influencer_ids": ["1", "2"],
            }
            profiles = {
                "1": {
                    "user_id": 1,
                    "user_name": "user_1",
                    "user_followers": 100,
                    "user_friends": 20,
                    "user_interests": ["reading"],
                    "user_description": "profile one",
                }
            }
            interactions = {
                "1": [
                    {
                        "text_raw": "raw",
                        "text_comment": "comment",
                        "interact_type": "comment",
                        "interact_id": 2,
                    }
                ]
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {"neighbor_count": 1, "isolated": False},
                    "neighbors": [
                        {
                            "neighbor_id": "2",
                            "relation": "engaged_by",
                            "received_comment_count": 1,
                            "total_interaction_count": 1,
                        }
                    ],
                }
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
            product_context = loader.load_product_context()
            profile_map = loader.load_profiles()
            interaction_map = loader.load_interactions()
            enriched_map = loader.load_enriched_profiles()
            summary = loader.build_summary()

            self.assertEqual(product_context.product_name, "abc_reading")
            self.assertEqual(profile_map["1"].user_followers, 100)
            self.assertEqual(interaction_map["1"][0].source_user_id, "1")
            self.assertEqual(interaction_map["1"][0].target_user_id, "2")
            self.assertEqual(enriched_map["1"].graph_attributes.neighbor_count, 1)
            self.assertEqual(summary.interaction_record_count, 1)

    def test_load_dataset_bundle_respects_limits(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)

            (raw_dir / "abc_reading_product_info.json").write_text(
                json.dumps({"product_name": "abc_reading"}),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_profile.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {"user_id": 1, "user_name": "user_1"},
                        "2": {"user_id": 2, "user_name": "user_2"},
                    }
                ),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_interaction.graph.anon").write_text(
                json.dumps(
                    {
                        "1": [
                            {"interact_id": 2, "interact_type": "comment"},
                            {"interact_id": 3, "interact_type": "reposts"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loader = DataLoader(root)
            bundle = loader.load_dataset_bundle(
                include_enriched_profiles=False,
                profile_limit=1,
                interaction_source_limit=1,
                interaction_record_limit=1,
            )

            self.assertEqual(len(bundle.profiles), 1)
            self.assertEqual(len(bundle.interactions["1"]), 1)
            self.assertIsNone(bundle.enriched_profiles)

    def test_reworked_weibo_profile_has_loader_priority(self) -> None:
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
                json.dumps(
                    {
                        "1": {
                            "user_id": 1,
                            "user_name": "raw_user",
                            "user_followers": 10,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (derived_dir / "abc_reading_profile_compact.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {
                            "user_id": 1,
                            "user_name": "compact_user",
                            "user_followers": 20,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (derived_dir / "weibo_profile_reworked.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {
                            "user_id": 1,
                            "user_name": "reworked_user",
                            "user_followers": 300,
                            "graph_attributes": {"neighbor_count": 6, "isolated": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (derived_dir / "weibo_profile_reworked_with_neighbors.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {
                            "user_id": 1,
                            "user_name": "reworked_neighbor_user",
                            "user_followers": 600,
                            "graph_attributes": {"neighbor_count": 1, "isolated": False},
                            "neighbors": [
                                {
                                    "neighbor_id": "2",
                                    "relation": "engaged_by",
                                    "received_comment_count": 1,
                                    "received_repost_count": 0,
                                    "made_comment_count": 0,
                                    "made_repost_count": 0,
                                    "received_interaction_count": 1,
                                    "made_interaction_count": 0,
                                    "total_interaction_count": 1,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            loader = DataLoader(root)
            profiles = loader.load_profiles()
            enriched_profiles = loader.load_enriched_profiles()

            self.assertEqual(profiles["1"].user_name, "reworked_neighbor_user")
            self.assertEqual(profiles["1"].user_followers, 600)
            self.assertEqual(enriched_profiles["1"].graph_attributes.neighbor_count, 1)
            self.assertEqual(enriched_profiles["1"].neighbors[0].neighbor_id, "2")

    def test_platform_enriched_profile_has_highest_priority(self) -> None:
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
            (derived_dir / "weibo_profile_reworked_with_neighbors.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {
                            "user_id": 1,
                            "user_name": "no_platform_user",
                            "user_followers": 600,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (derived_dir / "weibo_profile_reworked_with_neighbors_with_platform.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {
                            "user_id": 1,
                            "user_name": "platform_user",
                            "user_followers": 900,
                            "platform": "twitter_simulated",
                        }
                    }
                ),
                encoding="utf-8",
            )

            loader = DataLoader(root)
            self.assertEqual(
                loader.platform_enriched_profile_path,
                derived_dir
                / "weibo_profile_reworked_with_neighbors_with_platform.graph.anon",
            )
            profiles = loader.load_profiles()
            self.assertEqual(profiles["1"].user_name, "platform_user")
            self.assertEqual(profiles["1"].platform, "twitter_simulated")

    def test_summarize_profile_includes_platform(self) -> None:
        loader = DataLoader(Path("."))
        from influence_strategy.models import UserProfile

        profile = UserProfile(
            user_id="1",
            user_name="user_1",
            platform="instagram_simulated",
        )
        summary = loader.summarize_profile_for_semantics(profile)
        self.assertIn("platform=instagram_simulated", summary)


if __name__ == "__main__":
    unittest.main()
