from __future__ import annotations

import unittest

from pydantic import ValidationError

from influence_strategy.platform_profiles import (
    DATASET_PLATFORM_VALUES,
    SIMULATED_PLATFORM_KEYS,
    PlatformProfile,
    UnsupportedPlatformError,
    dataset_platform_value,
    get_platform_profile,
    list_supported_platforms,
    normalize_platform_key,
)


EXPECTED_PLATFORMS = {"facebook", "instagram", "twitter"}


class PlatformProfileRegistryTest(unittest.TestCase):
    def test_three_platforms_present_with_complete_fields(self) -> None:
        for key in sorted(EXPECTED_PLATFORMS):
            profile = get_platform_profile(key)
            self.assertEqual(profile.platform_key, key)
            self.assertTrue(profile.display_name)
            self.assertTrue(profile.platform_roles)
            self.assertTrue(profile.content_template.content_modes)
            self.assertTrue(profile.content_template.sections)
            self.assertTrue(profile.content_template.guardrails)
            self.assertTrue(profile.time_blueprint)
            self.assertTrue(profile.signal_weights)
            self.assertTrue(profile.dispatch_score_weights)
            self.assertTrue(profile.policy_disclaimer)

    def test_platform_keys_are_lowercase(self) -> None:
        for key in list_supported_platforms():
            self.assertEqual(key, key.lower().strip())

    def test_list_supported_platforms(self) -> None:
        self.assertEqual(set(list_supported_platforms()), EXPECTED_PLATFORMS)

    def test_get_platform_profile_normalizes_key(self) -> None:
        self.assertEqual(
            get_platform_profile("  FaceBook  ").platform_key, "facebook"
        )
        self.assertEqual(get_platform_profile("TWITTER").platform_key, "twitter")

    def test_get_platform_profile_accepts_dataset_simulated_suffix(self) -> None:
        for key in sorted(EXPECTED_PLATFORMS):
            self.assertEqual(
                get_platform_profile(dataset_platform_value(key)).platform_key,
                key,
            )

    def test_simulated_platform_constants_are_consistent(self) -> None:
        self.assertEqual(set(SIMULATED_PLATFORM_KEYS), EXPECTED_PLATFORMS)
        self.assertEqual(
            set(DATASET_PLATFORM_VALUES),
            {f"{key}_simulated" for key in EXPECTED_PLATFORMS},
        )
        self.assertEqual(normalize_platform_key("facebook_simulated"), "facebook")
        self.assertEqual(normalize_platform_key(" Twitter "), "twitter")
        self.assertEqual(normalize_platform_key(None), "")

    def test_unknown_platform_raises_clear_error(self) -> None:
        with self.assertRaises(UnsupportedPlatformError) as ctx:
            get_platform_profile("myspace")
        message = str(ctx.exception)
        self.assertIn("myspace", message)
        for key in EXPECTED_PLATFORMS:
            self.assertIn(key, message)

    def test_empty_platform_key_raises(self) -> None:
        with self.assertRaises(UnsupportedPlatformError):
            get_platform_profile("   ")

    def test_platform_roles_are_unique_and_non_empty(self) -> None:
        for key in list_supported_platforms():
            roles = get_platform_profile(key).platform_roles
            self.assertGreater(len(roles), 0)
            self.assertEqual(len(roles), len(set(roles)))

    def test_signal_weights_sum_to_one(self) -> None:
        for key in list_supported_platforms():
            profile = get_platform_profile(key)
            total = sum(profile.signal_weights.values())
            self.assertAlmostEqual(total, 1.0, places=3)
            dispatch_total = sum(profile.dispatch_score_weights.values())
            self.assertAlmostEqual(dispatch_total, 1.0, places=3)

    def test_role_affinity_within_unit_range(self) -> None:
        for key in list_supported_platforms():
            for affinity in get_platform_profile(key).role_affinity.values():
                self.assertGreaterEqual(affinity, 0.0)
                self.assertLessEqual(affinity, 1.0)

    def test_content_templates_differ_across_platforms(self) -> None:
        section_keys = {
            key: tuple(
                section.section_key
                for section in get_platform_profile(key).content_template.sections
            )
            for key in list_supported_platforms()
        }
        self.assertEqual(len(set(section_keys.values())), 3, section_keys)

        content_modes = {
            key: tuple(get_platform_profile(key).content_template.content_modes)
            for key in list_supported_platforms()
        }
        self.assertEqual(len(set(content_modes.values())), 3, content_modes)

    def test_time_blueprints_differ_across_platforms(self) -> None:
        stage_keys = {
            key: tuple(stage.stage_key for stage in get_platform_profile(key).time_blueprint)
            for key in list_supported_platforms()
        }
        self.assertEqual(len(set(stage_keys.values())), 3, stage_keys)

    def test_disclaimer_states_simulated_platform(self) -> None:
        for key in list_supported_platforms():
            disclaimer = get_platform_profile(key).policy_disclaimer
            self.assertIn("模拟平台", disclaimer)
            self.assertIn("真实账号", disclaimer)

    def test_expected_role_vocabulary(self) -> None:
        self.assertEqual(
            set(get_platform_profile("facebook").platform_roles),
            {"community_discussion", "story_explainer", "community_moderator"},
        )
        self.assertEqual(
            set(get_platform_profile("twitter").platform_roles),
            {"thread_writer", "retweet_amplifier", "reply_engager"},
        )
        self.assertEqual(
            set(get_platform_profile("instagram").platform_roles),
            {"visual_storyteller", "reels_creator", "story_interactor"},
        )

    def test_twitter_guardrail_forbids_fabrication(self) -> None:
        guardrails = " ".join(
            get_platform_profile("twitter").content_template.guardrails
        )
        self.assertIn("不得", guardrails)
        self.assertIn("捏造", guardrails)


class PlatformProfileValidationTest(unittest.TestCase):
    def _base_payload(self) -> dict:
        return {
            "platform_key": "demo",
            "display_name": "Demo",
            "platform_roles": ["role_a"],
            "content_template": {
                "content_modes": ["mode"],
                "sections": [
                    {"section_key": "s1", "label": "label", "guidance": "guide"}
                ],
                "guardrails": ["guardrail"],
            },
            "time_blueprint": [
                {
                    "stage_key": "stage",
                    "label": "stage",
                    "ratio_start": 0.0,
                    "ratio_end": 1.0,
                    "focus": "focus",
                }
            ],
            "role_affinity": {"core_broadcast": 0.5},
            "signal_weights": {"topic_match_score": 1.0},
            "dispatch_score_weights": {"final_score": 0.5, "platform_fit_score": 0.5},
        }

    def test_uppercase_platform_key_is_rejected(self) -> None:
        payload = self._base_payload()
        payload["platform_key"] = "Demo"
        with self.assertRaises(ValidationError):
            PlatformProfile.model_validate(payload)

    def test_signal_weights_must_sum_to_one(self) -> None:
        payload = self._base_payload()
        payload["signal_weights"] = {"topic_match_score": 0.4}
        with self.assertRaises(ValidationError):
            PlatformProfile.model_validate(payload)

    def test_empty_roles_rejected(self) -> None:
        payload = self._base_payload()
        payload["platform_roles"] = []
        with self.assertRaises(ValidationError):
            PlatformProfile.model_validate(payload)

    def test_invalid_role_affinity_rejected(self) -> None:
        payload = self._base_payload()
        payload["role_affinity"] = {"core_broadcast": 1.5}
        with self.assertRaises(ValidationError):
            PlatformProfile.model_validate(payload)

    def test_time_stage_window_must_increase(self) -> None:
        payload = self._base_payload()
        payload["time_blueprint"] = [
            {
                "stage_key": "stage",
                "label": "stage",
                "ratio_start": 0.8,
                "ratio_end": 0.2,
                "focus": "focus",
            }
        ]
        with self.assertRaises(ValidationError):
            PlatformProfile.model_validate(payload)

    def test_valid_payload_is_accepted(self) -> None:
        profile = PlatformProfile.model_validate(self._base_payload())
        self.assertEqual(profile.platform_key, "demo")


if __name__ == "__main__":
    unittest.main()
