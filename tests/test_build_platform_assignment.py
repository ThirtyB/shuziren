from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_platform_assignment.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_platform_assignment", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


assignment = _load_script_module()

PLATFORM_VALUES = (
    "facebook_simulated",
    "instagram_simulated",
    "twitter_simulated",
)


class BuildPlatformAssignmentTest(unittest.TestCase):
    def test_assign_platform_is_deterministic(self) -> None:
        first = assignment.assign_platform(
            "user_42", platform_values=PLATFORM_VALUES, seed=7
        )
        second = assignment.assign_platform(
            "user_42", platform_values=PLATFORM_VALUES, seed=7
        )
        self.assertEqual(first, second)
        self.assertIn(first, PLATFORM_VALUES)

    def test_seed_changes_assignment(self) -> None:
        assignments_a = {
            str(index): assignment.assign_platform(
                str(index), platform_values=PLATFORM_VALUES, seed=1
            )
            for index in range(200)
        }
        assignments_b = {
            str(index): assignment.assign_platform(
                str(index), platform_values=PLATFORM_VALUES, seed=2
            )
            for index in range(200)
        }
        self.assertNotEqual(assignments_a, assignments_b)

    def test_distribution_is_roughly_even(self) -> None:
        counter = Counter(
            assignment.assign_platform(
                f"user_{index}", platform_values=PLATFORM_VALUES, seed=20260624
            )
            for index in range(9000)
        )
        for platform in PLATFORM_VALUES:
            ratio = counter[platform] / 9000
            self.assertGreater(ratio, 0.28, counter)
            self.assertLess(ratio, 0.39, counter)

    def test_build_platform_dataset_injects_platform_field(self) -> None:
        profiles = {
            "1": {"user_id": 1, "user_name": "user_1", "user_followers": 10},
            "2": {"user_id": 2, "user_name": "user_2", "user_followers": 20},
        }
        output, counts = assignment.build_platform_dataset(
            profiles=profiles,
            platform_values=PLATFORM_VALUES,
            seed=20260624,
        )
        self.assertEqual(set(output), {"1", "2"})
        for key, record in output.items():
            self.assertIn("platform", record)
            self.assertIn(record["platform"], PLATFORM_VALUES)
            # 原有字段必须保留
            self.assertEqual(record["user_name"], profiles[key]["user_name"])
            self.assertEqual(record["user_followers"], profiles[key]["user_followers"])
        self.assertEqual(sum(counts.values()), 2)

    def test_build_platform_dataset_preserves_user_id_type_consistency(self) -> None:
        profiles = {"1": {"user_id": 1, "user_name": "user_1"}}
        output, _ = assignment.build_platform_dataset(
            profiles=profiles,
            platform_values=PLATFORM_VALUES,
            seed=1,
        )
        self.assertEqual(output["1"]["user_id"], "1")

    def test_build_platform_dataset_rejects_non_object_record(self) -> None:
        with self.assertRaises(ValueError):
            assignment.build_platform_dataset(
                profiles={"1": ["not", "an", "object"]},  # type: ignore[dict-item]
                platform_values=PLATFORM_VALUES,
                seed=1,
            )

    def test_default_platform_values_match_registry(self) -> None:
        self.assertEqual(
            set(assignment.DEFAULT_PLATFORM_VALUES), set(PLATFORM_VALUES)
        )


if __name__ == "__main__":
    unittest.main()
