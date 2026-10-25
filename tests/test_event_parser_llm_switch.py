"""事件解析 LLM 独立开关 + 规则解析修复的回归测试。

覆盖两类改动：
1. ``pipeline.run(use_llm_event_parser=...)`` 让事件解析阶段的 LLM 调用
   可以独立于 selector 控制；
2. 规则解析器的两处修复：亲子口语说法识别、风险关键词的否定/预防语境。

所有测试都用假的 LLM 客户端，不发起任何真实 HTTP 请求。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from influence_strategy.event_parser import RuleBasedEventParser
from influence_strategy.pipeline import StrategyPipeline

EVENT_PARSER_MARK = "事件解析助手"
SELECTOR_MARK = "候选组合选择助手"


class RecordingLLMClient:
    """按 system prompt 区分被调用的阶段，并记录每一次调用。

    事件解析与 selector 共用同一个 client 实例，所以只能靠提示词区分阶段。
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if EVENT_PARSER_MARK in system_prompt:
            self.calls.append("event_parser")
            return {
                "event_type": "parent_child_reading",
                "target_goal": "engagement",
                "target_audience": ["parent_child"],
                "extracted_keywords": ["亲子阅读", "英语启蒙"],
                "risk_level": "low",
                "semantic_tags": ["亲子阅读", "英语启蒙"],
                "narrative_frames": ["议题讨论"],
                "target_roles": ["core_publish_node"],
                "negative_constraints": ["避免同质化"],
                "dispatch_preferences": {
                    "candidate_pool_size": 12,
                    "rerank_top_k": 6,
                },
                "reasoning": ["llm parsed event"],
            }

        if SELECTOR_MARK in system_prompt:
            self.calls.append("selector")
            # 从提示词里回读候选名单，回显前 3 个，走通重排逻辑。
            try:
                payload = json.loads(user_prompt)
                shortlist = payload.get("shortlist") or []
            except (ValueError, TypeError):
                shortlist = []
            return {
                "selected_order": [
                    {
                        "user_id": str(item.get("user_id")),
                        "recommended_role": "core_publish_node",
                        "reasoning": ["适合首发"],
                    }
                    for item in shortlist[:3]
                    if item.get("user_id")
                ],
                "fallback_order": [],
                "global_notes": ["角色覆盖"],
            }

        self.calls.append("unknown")
        return {}

    @property
    def event_parser_calls(self) -> int:
        return self.calls.count("event_parser")

    @property
    def selector_calls(self) -> int:
        return self.calls.count("selector")


def write_minimal_workspace(root: Path) -> None:
    """写出能让整条流水线离线跑通的最小数据目录。"""
    raw_dir = root / "data" / "raw"
    derived_dir = root / "data" / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    product_info = {
        "product_name": "abc_reading",
        "domain": "reading",
        "ads": "reading context",
        "influencer_ids": ["1"],
    }
    # 兴趣标签同时给中英文，保证三种措辞的新事件都能命中至少一个候选人
    # ——否则 safe_pool 为空，selector 会走"不调用 LLM"的早退分支，
    # 测不到"selector 仍然被调用"这一点。
    profiles = {
        "1": {
            "user_id": 1,
            "user_name": "core_parenting_observer",
            "user_followers": 2400,
            "user_friends": 80,
            "user_interests": [
                "parent_child", "reading", "english_learning",
                "亲子", "阅读", "英语", "启蒙", "绘本",
            ],
            "user_description": "亲子阅读与英语启蒙话题的核心观察者，擅长家长群体讨论互动。",
            "platform": "facebook_simulated",
        },
        "2": {
            "user_id": 2,
            "user_name": "amplifier_reading",
            "user_followers": 650,
            "user_friends": 90,
            "user_interests": [
                "reading", "content_diffusion",
                "阅读", "传播", "讨论", "扩散",
            ],
            "user_description": "适合做阅读类话题的二次扩散与传播放大。",
            "platform": "twitter_simulated",
        },
        "3": {
            "user_id": 3,
            "user_name": "responder_qa",
            "user_followers": 380,
            "user_friends": 50,
            "user_interests": [
                "education_exchange", "industry_qa",
                "教育", "知识", "素质", "评论", "交流",
            ],
            "user_description": "知识类创作者，擅长教育话题评论互动与观点回应。",
            "platform": "instagram_simulated",
        },
    }
    interactions = {
        "1": [{"interact_id": 2, "interact_type": "comment"}],
        "2": [{"interact_id": 1, "interact_type": "reposts"}],
        "3": [{"interact_id": 1, "interact_type": "comment"}],
    }
    enriched = {
        key: {
            **profile,
            "graph_attributes": {
                "neighbor_count": 3,
                "mutual_neighbor_count": 1,
                "received_interaction_count": 4,
                "received_comment_count": 3,
                "received_repost_count": 1,
                "made_interaction_count": 2,
                "made_repost_count": 1,
                "isolated": False,
            },
            "neighbors": [],
        }
        for key, profile in profiles.items()
    }

    (raw_dir / "abc_reading_product_info.json").write_text(
        json.dumps(product_info, ensure_ascii=False), encoding="utf-8"
    )
    (raw_dir / "abc_reading_profile.graph.anon").write_text(
        json.dumps(profiles, ensure_ascii=False), encoding="utf-8"
    )
    (raw_dir / "abc_reading_interaction.graph.anon").write_text(
        json.dumps(interactions, ensure_ascii=False), encoding="utf-8"
    )
    (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
        json.dumps(enriched, ensure_ascii=False), encoding="utf-8"
    )


class EventParserLLMSwitchTest(unittest.TestCase):
    """三种开关组合下，两个阶段各自的 LLM 调用情况。"""

    def _run(self, *, use_llm: bool, use_llm_event_parser: bool | None) -> tuple[RecordingLLMClient, Any]:
        client = RecordingLLMClient()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_minimal_workspace(root)
            pipeline = StrategyPipeline(product_name="abc_reading")
            artifacts = pipeline.run_with_artifacts(
                workspace_root=root,
                event_input="围绕亲子阅读与英语启蒙做一次传播活动，提升讨论度",
                use_llm=use_llm,
                llm_client=client,
                use_llm_event_parser=use_llm_event_parser,
            )
        return client, artifacts

    def test_parser_off_selector_on(self) -> None:
        """use_llm_event_parser=False + use_llm=True：事件解析走规则，selector 仍可调用 LLM。"""
        client, artifacts = self._run(use_llm=True, use_llm_event_parser=False)

        self.assertEqual(client.event_parser_calls, 0, "事件解析不应调用 LLM")
        self.assertEqual(client.selector_calls, 1, "selector 应仍调用 LLM")
        self.assertEqual(artifacts.event.parser_name, "rule_based_v1")
        self.assertFalse(artifacts.event.llm_metadata.get("used"))

    def test_parser_on_selector_off(self) -> None:
        """use_llm_event_parser=True + use_llm=False：事件解析仍可用 LLM，selector 不用。"""
        client, artifacts = self._run(use_llm=False, use_llm_event_parser=True)

        self.assertEqual(client.event_parser_calls, 1, "事件解析应调用 LLM")
        self.assertEqual(client.selector_calls, 0, "selector 不应调用 LLM")
        self.assertEqual(artifacts.event.parser_name, "rule_llm_v1")
        self.assertTrue(artifacts.event.llm_metadata.get("used"))

    def test_both_off(self) -> None:
        client, artifacts = self._run(use_llm=False, use_llm_event_parser=False)

        self.assertEqual(client.event_parser_calls, 0)
        self.assertEqual(client.selector_calls, 0)
        self.assertEqual(artifacts.event.parser_name, "rule_based_v1")

    def test_both_on(self) -> None:
        client, artifacts = self._run(use_llm=True, use_llm_event_parser=True)

        self.assertEqual(client.event_parser_calls, 1)
        self.assertEqual(client.selector_calls, 1)
        self.assertEqual(artifacts.event.parser_name, "rule_llm_v1")

    def test_default_uses_production_default_parser_off(self) -> None:
        """第十轮：不显式传 use_llm_event_parser 时，走生产默认（事件解析不开 LLM）。

        旧语义是"跟随 use_llm"，第十轮为了省掉这次约 12 秒的调用改成默认关闭；
        selector 的 LLM 不受影响，仍然必须调用。显式传 True 仍可恢复（见
        ``test_both_on`` / ``test_parser_on_selector_off``）。
        """
        client_on, _ = self._run(use_llm=True, use_llm_event_parser=None)
        self.assertEqual(client_on.event_parser_calls, 0, "默认不应调用事件解析 LLM")
        self.assertEqual(client_on.selector_calls, 1, "selector 的 LLM 必须保留")

        client_off, _ = self._run(use_llm=False, use_llm_event_parser=None)
        self.assertEqual(client_off.event_parser_calls, 0)
        self.assertEqual(client_off.selector_calls, 0)

    def test_selector_llm_reasoning_present_when_parser_off(self) -> None:
        """关掉事件解析 LLM 后，selector 的 LLM 理由仍应出现在结果里。"""
        _, artifacts = self._run(use_llm=True, use_llm_event_parser=False)
        reasons = [
            reason
            for node in artifacts.selection_result.selected_nodes
            for reason in node.selection_reasons
        ]
        self.assertTrue(
            any(reason.startswith("llm_selector=") for reason in reasons),
            f"selector 的 LLM 理由应保留，实际为 {reasons}",
        )


class NewEventInputTest(unittest.TestCase):
    """三种不同表达的新事件，在关闭事件解析 LLM 时仍能跑通全流程。"""

    EVENTS = [
        "围绕亲子阅读与英语启蒙做一次传播活动，提升家长群体的讨论互动",
        "想让更多爸爸妈妈陪孩子一起读英文绘本，并鼓励大家在评论区交流心得",
        "近期有教育类KOL就素质教育话题发表了不同看法，希望邀请数位泛知识创作者理性发声，放大正面声量并避免引发争议",
    ]

    def test_three_new_events_complete_pipeline(self) -> None:
        for text in self.EVENTS:
            with self.subTest(event=text):
                client = RecordingLLMClient()
                with TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    write_minimal_workspace(root)
                    pipeline = StrategyPipeline(product_name="abc_reading")
                    artifacts = pipeline.run_with_artifacts(
                        workspace_root=root,
                        event_input=text,
                        use_llm=True,
                        llm_client=client,
                        use_llm_event_parser=False,
                    )

                # 事件对象完整
                self.assertTrue(artifacts.event.event_id)
                self.assertTrue(artifacts.event.event_type)
                self.assertTrue(artifacts.event.target_goal)
                self.assertTrue(artifacts.event.target_audience)
                self.assertTrue(artifacts.event.semantic_tags)
                self.assertTrue(artifacts.event.narrative_frames)
                self.assertTrue(artifacts.event.target_roles)
                # 后续阶段产出完整
                self.assertGreater(len(artifacts.feature_result.node_features), 0)
                self.assertGreater(len(artifacts.score_result.node_scores), 0)
                self.assertIsNotNone(artifacts.strategy_result)
                self.assertTrue(artifacts.strategy_result.summary.status)
                # selector 的 LLM 仍然被调用
                self.assertEqual(client.selector_calls, 1)


class RuleParserFixTest(unittest.TestCase):
    """规则解析器的两处修复。"""

    def setUp(self) -> None:
        self.parser = RuleBasedEventParser()

    def _parse(self, text: str):
        return self.parser.parse(text)

    # --- 亲子受众口语化表达 ---

    def test_parent_child_colloquial_expressions(self) -> None:
        for text in (
            "想让更多爸爸妈妈陪孩子一起读英文绘本",
            "爸妈带孩子做英语启蒙，希望提升互动",
            "面向宝妈宝爸的亲子阅读活动",
        ):
            with self.subEvent(text):
                self.assertIn("parent_child", self._parse(text).target_audience)

    def test_bare_child_does_not_over_classify(self) -> None:
        """只出现"孩子"不应判为亲子受众，避免过度分类。"""
        parsed = self._parse("孩子所在学校的老师反映课堂情况")
        self.assertNotIn("parent_child", parsed.target_audience)
        self.assertIn("education_practitioner", parsed.target_audience)

    # --- 风险关键词的否定/预防语境 ---

    def test_prevention_expressions_do_not_raise_risk(self) -> None:
        for text in (
            "放大正面声量并避免引发争议",
            "本次投放要防止刷屏，保持稳定节奏",
            "服务升级后杜绝投诉，提升体验",
        ):
            with self.subEvent(text):
                self.assertEqual(self._parse(text).constraints.risk_level, "low")

    def test_real_risk_still_detected(self) -> None:
        """否定规则不能把真实风险也一起抹掉。"""
        self.assertEqual(
            self._parse("近期该话题已经引发大量争议，需要尽快回应").constraints.risk_level,
            "medium",
        )

    def test_mixed_real_risk_and_prevention_stays_high(self) -> None:
        """既有真实风险、又有预防表述时，不应被预防词整句遮蔽。"""
        self.assertEqual(
            self._parse("先处理已有投诉，再避免后续争议").constraints.risk_level,
            "high",
        )

    def subEvent(self, text: str):  # noqa: N802 - 与 unittest 命名风格保持一致
        return self.subTest(text=text)


if __name__ == "__main__":
    unittest.main()
