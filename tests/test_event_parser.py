from __future__ import annotations

import unittest

from influence_strategy.event_parser import RuleBasedEventParser


class FakeEventParserLLMClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        return {
            "event_type": "public_opinion_response",
            "target_goal": "response",
            "target_audience": ["general_public", "education_practitioner"],
            "extracted_keywords": ["澄清", "质疑", "回应"],
            "risk_level": "high",
            "semantic_tags": ["争议回应", "公共讨论"],
            "narrative_frames": ["事实澄清", "问答回应"],
            "target_roles": [
                "core_publish_node",
                "interaction_response_node",
                "support_node",
            ],
            "negative_constraints": ["避免情绪化表达"],
            "dispatch_preferences": {
                "candidate_pool_size": 18,
                "rerank_top_k": 9,
            },
            "reasoning": ["llm parsed event"],
        }


class EventParserTest(unittest.TestCase):
    def test_parses_natural_language_event(self) -> None:
        parser = RuleBasedEventParser()
        parsed = parser.parse(
            "希望围绕亲子阅读和英语启蒙做一次传播活动，重点提升讨论度，并控制刷屏争议。"
        )

        self.assertEqual(parsed.product_name, "abc_reading")
        self.assertEqual(parsed.event_type, "english_learning_engagement")
        self.assertEqual(parsed.target_goal, "engagement")
        self.assertIn("parent_child", parsed.target_audience)
        self.assertIn("english_learning", parsed.target_audience)
        self.assertEqual(parsed.constraints.risk_level, "medium")

    def test_respects_explicit_payload_fields(self) -> None:
        parser = RuleBasedEventParser()
        parsed = parser.parse(
            {
                "event_id": "evt_manual_001",
                "product_name": "abc_reading",
                "event_title": "争议回应",
                "event_description": "针对近期投诉和质疑做一次集中回应与澄清。",
                "target_goal": "response",
                "target_audience": ["general_public"],
                "constraints": {"max_selected_nodes": 6, "risk_level": "high"},
            }
        )

        self.assertEqual(parsed.event_id, "evt_manual_001")
        self.assertEqual(parsed.target_goal, "response")
        self.assertEqual(parsed.event_type, "public_opinion_response")
        self.assertEqual(parsed.constraints.max_selected_nodes, 6)
        self.assertEqual(parsed.constraints.risk_level, "high")
        self.assertIn("general_public", parsed.target_audience)

    def test_llm_parser_enriches_structured_event_fields(self) -> None:
        parser = RuleBasedEventParser()
        parsed = parser.parse(
            "针对近期争议做一次集中回应与澄清，稳定讨论节奏。",
            use_llm=True,
            llm_client=FakeEventParserLLMClient(),
        )

        self.assertEqual(parsed.parser_name, "rule_llm_v1")
        self.assertEqual(parsed.target_goal, "response")
        self.assertEqual(parsed.constraints.risk_level, "high")
        self.assertIn("争议回应", parsed.semantic_tags)
        self.assertIn("事实澄清", parsed.narrative_frames)
        self.assertIn("support_node", parsed.target_roles)
        self.assertTrue(parsed.llm_metadata["used"])

    def test_extracts_generic_keywords_for_non_reading_hot_event(self) -> None:
        parser = RuleBasedEventParser()
        parsed = parser.parse(
            {
                "event_id": "hot_event_generic_001",
                "event_title": "Iran shipping disruption",
                "event_description": "Hormuz shipping disruption affects global energy and logistics.",
                "target_goal": "awareness",
                "target_audience": ["general_public"],
            }
        )

        lowered_keywords = [item.lower() for item in parsed.extracted_keywords]
        self.assertIn("iran", lowered_keywords)
        self.assertIn("shipping", lowered_keywords)
        self.assertIn("general_public", parsed.semantic_tags)


if __name__ == "__main__":
    unittest.main()
