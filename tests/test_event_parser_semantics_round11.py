"""第十一轮 · 规则事件解析的语义修复回归测试。

覆盖三块：
1. 校园 / 环保 / 隐私 主题词进入 ``semantic_tags``，校园受众有上下文依据；
2. 预防语境 vs 已发生语境的区分（风险等级、事件类型、负面约束）；
3. 六个指定案例 + 一个"不含新主题词"的普通事件，确认原规则未被意外改变。

全部为纯规则解析，不发起任何 LLM 请求。
"""

from __future__ import annotations

import unittest

from influence_strategy.event_parser import RuleBasedEventParser


class TopicRecognitionTest(unittest.TestCase):
    """校园 / 环保 / 隐私 主题词必须进入结构化字段。"""

    def setUp(self) -> None:
        self.parser = RuleBasedEventParser()

    def test_campus_environment_event_has_campus_and_environment_tags(self) -> None:
        parsed = self.parser.parse("围绕校园环保宣传开展一次活动，提升学生参与度。")

        self.assertEqual(parsed.event_type, "activity_announcement")
        self.assertIn("campus_student", parsed.target_audience)
        for tag in ("校园", "学生", "环保"):
            with self.subTest(tag=tag):
                self.assertIn(tag, parsed.semantic_tags)

    def test_campus_audience_needs_context_not_just_campus_word(self) -> None:
        """只有教职工线索、没有学生线索时，不应把受众推定为学生。"""
        practitioner_only = self.parser.parse("邀请老师参加校园教研活动，分享课堂教学经验。")
        self.assertIn("education_practitioner", practitioner_only.target_audience)
        self.assertNotIn("campus_student", practitioner_only.target_audience)

        with_students = self.parser.parse("校园环保宣传活动，面向学生社团招募志愿者。")
        self.assertIn("campus_student", with_students.target_audience)

    def test_plain_event_does_not_gain_new_topics(self) -> None:
        """不含校园 / 环保 / 隐私 / 争议的事件：原有行为不变。"""
        parsed = self.parser.parse("品牌新品直播专场，邀请用户参与互动讨论。")
        self.assertEqual(parsed.event_type, "activity_announcement")
        self.assertEqual(parsed.target_audience, ["general_public"])
        self.assertEqual(parsed.constraints.risk_level, "low")
        self.assertEqual(parsed.negative_constraints, ["避免节点内容完全同质化"])
        for tag in ("校园", "学生", "环保", "隐私", "个人信息", "争议"):
            with self.subTest(tag=tag):
                self.assertNotIn(tag, parsed.semantic_tags)

    def test_parent_child_event_is_unchanged(self) -> None:
        parsed = self.parser.parse("亲子英语阅读活动，提升家长参与度。")
        self.assertEqual(parsed.event_type, "parent_child_reading")
        self.assertIn("parent_child", parsed.target_audience)
        self.assertNotIn("campus_student", parsed.target_audience)
        self.assertNotIn("环保", parsed.semantic_tags)


class PreventionVersusOccurredTest(unittest.TestCase):
    """预防语境与已发生语境必须分开处理。"""

    def setUp(self) -> None:
        self.parser = RuleBasedEventParser()

    def test_prevention_is_not_public_opinion_response(self) -> None:
        """「避免引发争议」是预防措施，不是已经发生的舆情事件。"""
        parsed = self.parser.parse("开展校园交流活动，避免引发争议。")
        self.assertEqual(parsed.event_type, "activity_announcement")
        self.assertEqual(parsed.constraints.risk_level, "low")

    def test_prevention_still_gets_risk_constraint(self) -> None:
        """写了"避免"不代表没有风险：仍要给出明确的负面约束。"""
        constraints = self.parser.parse("开展校园交流活动，避免引发争议。").negative_constraints
        self.assertIn("避免引发争议或误解，表述保持中立克制", constraints)

    def test_occurred_controversy_is_still_public_opinion_response(self) -> None:
        parsed = self.parser.parse("该话题已经引发大量争议，需要尽快回应澄清。")
        self.assertEqual(parsed.event_type, "public_opinion_response")
        self.assertEqual(parsed.constraints.risk_level, "medium")

    def test_occurred_cue_beats_prevention_in_same_clause(self) -> None:
        """同一短句里先有"避免"又有"已出现"，应按已发生处理。"""
        parsed = self.parser.parse("为避免已出现的争议进一步发酵，需要先回应。")
        self.assertEqual(parsed.constraints.risk_level, "medium")

    def test_clause_split_prevents_cross_sentence_leak(self) -> None:
        """上一句的"避免"不能压掉下一句已发生的风险。"""
        self.assertEqual(
            self.parser.parse("先处理已有投诉，再避免后续争议。").constraints.risk_level,
            "high",
        )

    def test_negation_forms_do_not_raise_risk(self) -> None:
        for text in (
            "本次传播不希望引起争议",
            "放大正面声量并避免引发争议",
            "本次投放要防止刷屏，保持稳定节奏",
            "投放时不愿造成刷屏",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.parser.parse(text).constraints.risk_level, "low")

    def test_topic_keywords_still_visible_when_preventive(self) -> None:
        """预防语境不删除关键词本身，"争议"仍要能在标签里查到。"""
        parsed = self.parser.parse("开展校园交流活动，避免引发争议。")
        self.assertIn("争议", parsed.semantic_tags)


class PrivacyContextTest(unittest.TestCase):
    """隐私：区分"要保护"与"已经泄露"。"""

    def setUp(self) -> None:
        self.parser = RuleBasedEventParser()

    def test_protection_intent_is_preventive(self) -> None:
        parsed = self.parser.parse("校园交流活动中不得泄露参与者个人信息。")

        self.assertIn("个人信息", parsed.semantic_tags)
        self.assertEqual(parsed.constraints.risk_level, "low")
        self.assertIn("避免收集、展示或传播参与者的个人信息", parsed.negative_constraints)
        self.assertNotIn(
            "不得二次传播当事人个人信息与未经核实的细节", parsed.negative_constraints
        )
        self.assertNotEqual(parsed.event_type, "public_opinion_response")

    def test_occurred_leak_is_escalated(self) -> None:
        parsed = self.parser.parse("学校已经发生学生个人信息泄露事件，需要开展回应。")

        self.assertEqual(parsed.constraints.risk_level, "medium")
        self.assertIn("不得二次传播当事人个人信息与未经核实的细节", parsed.negative_constraints)
        self.assertIn("campus_student", parsed.target_audience)
        self.assertEqual(parsed.event_type, "public_opinion_response")
        self.assertEqual(parsed.target_goal, "response")

    def test_privacy_word_alone_does_not_raise_risk(self) -> None:
        """只说"注意保护隐私"，不能推断已经泄露。"""
        parsed = self.parser.parse("开展校园交流活动，注意参与者隐私保护。")
        self.assertEqual(parsed.constraints.risk_level, "low")
        self.assertIn("隐私", parsed.semantic_tags)
        self.assertNotIn(
            "不得二次传播当事人个人信息与未经核实的细节", parsed.negative_constraints
        )

    def test_no_privacy_mention_means_no_privacy_constraint(self) -> None:
        constraints = self.parser.parse("围绕校园环保宣传开展一次活动。").negative_constraints
        self.assertFalse(any("个人信息" in item for item in constraints))

    def test_context_is_recorded_in_reasoning(self) -> None:
        """下游可从 reasoning 读到隐私语境标记。"""
        preventive = self.parser.parse("校园交流活动中不得泄露参与者个人信息。")
        self.assertIn("privacy_context=preventive", preventive.reasoning)

        occurred = self.parser.parse("学校已经发生学生个人信息泄露事件，需要开展回应。")
        self.assertIn("privacy_context=occurred", occurred.reasoning)


if __name__ == "__main__":
    unittest.main()
