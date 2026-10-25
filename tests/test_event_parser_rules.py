"""第六轮 · 规则事件解析的修复回归测试。

覆盖三块：
1. 风险关键词的预防语境（第六轮补录 不希望 / 不想 / 不愿）；
2. semantic_tags / extracted_keywords 的从句片段噪声清理；
3. 事件类型对否定语境的行为（**诊断性**测试，记录当前行为，见类注释）。

全部为纯规则解析，不发起任何 LLM 请求。
"""

from __future__ import annotations

import re
import unittest

from influence_strategy.event_parser import RuleBasedEventParser

_CJK_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")

# 与生产实现保持一致：无分词器时，纯中文标签的合理上限。
_MAX_CJK_TERM = 4


def _cjk_len(text: str) -> int:
    return len(text) if _CJK_ONLY.match(text) else 0


class RiskPreventionTest(unittest.TestCase):
    """风险等级：预防语境不应抬升，真实风险不能被预防词遮蔽。"""

    def setUp(self) -> None:
        self.parser = RuleBasedEventParser()

    def _risk(self, text: str) -> str:
        return self.parser.parse(text).constraints.risk_level

    # --- 第六轮补录的三个预防词 ---

    def test_bu_xi_wang_does_not_raise_risk(self) -> None:
        self.assertEqual(self._risk("本次传播不希望引起争议"), "low")

    def test_bu_xiang_does_not_raise_risk(self) -> None:
        self.assertEqual(self._risk("服务升级后不想收到投诉"), "low")

    def test_bu_yuan_does_not_raise_risk(self) -> None:
        self.assertEqual(self._risk("投放时不愿造成刷屏"), "low")

    # --- 真实风险必须保留 ---

    def test_real_controversy_is_still_medium(self) -> None:
        """后半句有预防表达，但前半句是已发生的事实，不能整句压低。"""
        self.assertEqual(
            self._risk("近期该话题已经引发大量争议，需要避免进一步发酵"),
            "medium",
        )

    def test_existing_complaint_before_prevention_is_high(self) -> None:
        """不能因后半句的预防表达忽略前半句已发生的风险。"""
        self.assertEqual(self._risk("先处理已有投诉，再避免后续争议"), "high")

    # --- 既有预防规则不得回归 ---

    def test_existing_prevention_rules_still_work(self) -> None:
        for text in (
            "放大正面声量并避免引发争议",
            "本次投放要防止刷屏，保持稳定节奏",
            "服务升级后杜绝投诉，提升体验",
        ):
            with self.subTest(text=text):
                self.assertEqual(self._risk(text), "low")

    def test_control_is_not_treated_as_prevention(self) -> None:
        """「控制」可能描述已发生、正在处理，不算预防——既有测试依赖此行为。"""
        self.assertEqual(
            self._risk(
                "希望围绕亲子阅读和英语启蒙做一次传播活动，重点提升讨论度，并控制刷屏争议。"
            ),
            "medium",
        )


class SemanticTagsQualityTest(unittest.TestCase):
    """semantic_tags / extracted_keywords 的从句片段噪声清理。"""

    def setUp(self) -> None:
        self.parser = RuleBasedEventParser()

    def _parsed(self, text: str):
        return self.parser.parse(text)

    # --- 四个案例的期望结果 ---

    def test_parent_child_english_case(self) -> None:
        parsed = self._parsed("围绕亲子阅读与英语启蒙做一次传播活动，提升家长群体的讨论互动")
        self.assertEqual(
            parsed.semantic_tags,
            [
                "parent_child", "reading_interest", "english_learning",
                "活动", "英语", "启蒙", "亲子", "家长", "阅读",
            ],
        )
        self.assertIn("parent_child", parsed.target_audience)

    def test_colloquial_parent_child_case(self) -> None:
        parsed = self._parsed("想让更多爸爸妈妈陪孩子一起读英文绘本，并鼓励大家在评论区交流心得")
        self.assertEqual(
            parsed.semantic_tags,
            [
                "parent_child", "reading_interest", "english_learning",
                "英文", "绘本", "评论", "交流", "爸爸妈妈", "爸妈",
            ],
        )
        self.assertIn("parent_child", parsed.target_audience)

    def test_education_kol_case(self) -> None:
        parsed = self._parsed(
            "近期有教育类KOL就素质教育话题发表了不同看法，"
            "希望邀请数位泛知识创作者理性发声，放大正面声量并避免引发争议"
        )
        self.assertEqual(
            parsed.semantic_tags,
            ["education_practitioner", "争议", "教育", "KOL"],
        )
        self.assertIn("education_practitioner", parsed.target_audience)

    def test_unrelated_event_keeps_domain_terms(self) -> None:
        """与亲子/教育无关的新事件：保留领域词，丢掉从句。"""
        parsed = self._parsed("新款纯电SUV上市，重点展示续航与智能座舱，面向首次购车的年轻用户")
        self.assertEqual(
            parsed.semantic_tags,
            ["general_public", "新款纯电", "SUV", "上市"],
        )

    # --- 明确证明被清掉的噪声 ---

    def test_clause_fragments_are_removed(self) -> None:
        removed = {
            "想让更多爸爸妈妈陪孩子一起读英文绘本，并鼓励大家在评论区交流心得": [
                "并鼓励大家",
                "并鼓励大家在评论区交流心得",
                "想让更多爸爸妈妈陪孩子一起读英文",
            ],
            "近期有教育类KOL就素质教育话题发表了不同看法，"
            "希望邀请数位泛知识创作者理性发声，放大正面声量并避免引发争议": [
                "近期有教育类",
                "就素质教育话题发表了不同看法",
                "希望邀请数位泛知识创作者理性发声",
                "放大正面声量并避免引发争议",
            ],
            "新款纯电SUV上市，重点展示续航与智能座舱，面向首次购车的年轻用户": [
                "重点展示续航与智能座舱",
                "面向首次购车的年轻用户",
                "面向",
            ],
        }
        for text, fragments in removed.items():
            parsed = self._parsed(text)
            for fragment in fragments:
                with self.subTest(text=text, fragment=fragment):
                    self.assertNotIn(fragment, parsed.extracted_keywords)
                    self.assertNotIn(fragment, parsed.semantic_tags)

    def test_no_overlong_cjk_tag_remains(self) -> None:
        """兜底断言：任何纯中文标签都不应超过 4 个汉字。"""
        for text in (
            "围绕亲子阅读与英语启蒙做一次传播活动，提升家长群体的讨论互动",
            "想让更多爸爸妈妈陪孩子一起读英文绘本，并鼓励大家在评论区交流心得",
            "近期有教育类KOL就素质教育话题发表了不同看法，"
            "希望邀请数位泛知识创作者理性发声，放大正面声量并避免引发争议",
            "新款纯电SUV上市，重点展示续航与智能座舱，面向首次购车的年轻用户",
        ):
            parsed = self._parsed(text)
            overlong = [t for t in parsed.semantic_tags if _cjk_len(t) > _MAX_CJK_TERM]
            with self.subTest(text=text):
                self.assertEqual(overlong, [])

    def test_ascii_domain_terms_are_preserved(self) -> None:
        """英文/数字领域词不受中文过滤影响。"""
        self.assertIn("KOL", self._parsed("教育类KOL就素质教育话题发表了不同看法").extracted_keywords)
        self.assertIn("SUV", self._parsed("新款纯电SUV上市，面向年轻用户").extracted_keywords)

    def test_english_audience_identifiers_still_present(self) -> None:
        """英文受众标识符来自 target_audience，本轮刻意保留，不被清理。"""
        parsed = self._parsed("围绕亲子阅读与英语启蒙做一次传播活动")
        self.assertEqual(
            parsed.semantic_tags[:3],
            ["parent_child", "reading_interest", "english_learning"],
        )

    def test_generic_english_keywords_still_extracted(self) -> None:
        """既有测试依赖：非中文事件的英文关键词仍要抓得到。"""
        parsed = self._parsed(
            {
                "event_id": "hot_event_generic_001",
                "event_title": "Iran shipping disruption",
                "event_description": "Hormuz shipping disruption affects global energy and logistics.",
                "target_goal": "awareness",
                "target_audience": ["general_public"],
            }
        )
        lowered = [item.lower() for item in parsed.extracted_keywords]
        self.assertIn("iran", lowered)
        self.assertIn("shipping", lowered)


class EventTypeNegationTest(unittest.TestCase):
    """事件类型对否定语境的行为。

    第六轮这里记录的是**已知缺陷**：风险等级能正确处理预防语境，但事件类型
    还没有接入同一套判断，「避免引发争议」会被误判成 ``public_opinion_response``。

    第十一轮已修复：``_infer_event_type`` 对舆情响应类关键词先过
    ``_keyword_prevented``，预防语境下的风险词不再参与事件类型打分。
    下面断言的是**修复后的期望行为**（原作者注释已说明修复后应改成期望值）。
    """

    def setUp(self) -> None:
        self.parser = RuleBasedEventParser()

    def _type(self, text: str) -> str:
        return self.parser.parse(text).event_type

    def test_prevention_is_no_longer_classified_as_opinion_response(self) -> None:
        """预防措施不是舆情回应事件。"""
        self.assertEqual(self._type("本次传播不希望引起争议"), "general_influence_event")
        self.assertEqual(self._type("放大正面声量并避免引发争议"), "general_influence_event")

    def test_risk_level_and_event_type_now_agree(self) -> None:
        """同一个输入：风险等级与事件类型不再分裂。"""
        parsed = self.parser.parse("本次传播不希望引起争议")
        self.assertEqual(parsed.constraints.risk_level, "low")
        self.assertNotEqual(parsed.event_type, "public_opinion_response")

    def test_real_complaint_still_classified_as_opinion_response(self) -> None:
        """真实风险不应被误伤：这个用例当前分类是正确的。"""
        self.assertEqual(self._type("先处理已有投诉，再避免后续争议"), "public_opinion_response")
        self.assertEqual(self._type("该话题已经引发大量争议，需要尽快回应澄清"), "public_opinion_response")

    def test_prevention_only_case_has_no_real_risk(self) -> None:
        """纯预防句式不应被判成高风险。"""
        self.assertEqual(
            self.parser.parse("本次投放要防止刷屏，保持稳定节奏").constraints.risk_level,
            "low",
        )


if __name__ == "__main__":
    unittest.main()
