from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.image_event_loader import (
    HotEventMatcher,
    ImageEventRecognizer,
    ImageRecognitionResult,
    load_image_events,
)


class FakeVisionClient:
    provider = "dashscope"
    model = "qwen3.7-plus"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def generate_json_with_image(self, *, system_prompt: str, user_prompt: str, image_path: Path) -> dict:
        return {
            "event_title": "高端AI芯片供应与出口限制牵动产业链",
            "event_summary": "高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
            "domain": "technology",
            "keywords": ["AI芯片", "出口管制", "替代方案", "本土化供应"],
            "target": "说明高端AI芯片供应限制对模型训练、算力成本和产业链竞争的影响。",
            "opinion_variants": ["芯片供应紧张推动企业评估替代方案。"],
            "confidence": 0.91,
        }


class ImageEventLoaderTest(unittest.TestCase):
    def test_recognizer_normalizes_vision_client_response(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "4.png"
            image_path.write_bytes(b"image")

            result = ImageEventRecognizer().recognize(
                image_path=image_path,
                workspace_root=tmpdir,
                use_llm=True,
                vision_client=FakeVisionClient(),
            )

            self.assertEqual(result.event_title, "高端AI芯片供应与出口限制牵动产业链")
            self.assertEqual(result.domain, "technology")
            self.assertEqual(result.method, "vision_llm")
            self.assertEqual(result.confidence, 0.91)
            self.assertEqual(result.source_image, str(image_path))
            self.assertFalse(result.fallback_used)

    def test_matcher_reuses_existing_hot_event_above_threshold(self) -> None:
        recognition = ImageRecognitionResult(
            source_image="eval/image/4.png",
            method="vision_llm",
            event_title="高端AI芯片供应与出口限制牵动产业链",
            event_summary="高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
            domain="technology",
            keywords=["AI芯片", "出口管制", "替代方案", "本土化供应"],
            target="说明芯片供应限制影响。",
            opinion_variants=["芯片供应紧张。"],
            confidence=0.91,
        )
        reference_events = [
            {
                "event_id": "hot_event_005",
                "domain": "technology",
                "event_title": "高端AI芯片供应和出口限制引发产业链担忧",
                "event_summary": "高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
                "target": "说明高端AI芯片供应限制对模型训练、算力成本、产业竞争和本土替代的影响。",
                "opinion_variants": ["高端AI芯片越来越像战略资源。"],
            }
        ]

        event, metadata = HotEventMatcher(match_threshold=0.72).match(recognition, reference_events)

        self.assertEqual(event["event_id"], "hot_event_005")
        self.assertEqual(event["source_type"], "image")
        self.assertEqual(event["source_image"], "eval/image/4.png")
        self.assertEqual(event["image_recognition"]["matched_event_id"], "hot_event_005")
        self.assertGreaterEqual(metadata["match_score"], 0.72)

    def test_matcher_generates_image_event_below_threshold(self) -> None:
        recognition = ImageRecognitionResult(
            source_image="eval/image/new.png",
            method="fallback",
            event_title="完全不同的新事件",
            event_summary="这是一条和参考数据集没有关系的新事件。",
            domain="general",
            keywords=["新事件"],
            target="传播目标",
            opinion_variants=["新观点"],
            confidence=0.25,
            fallback_used=True,
            warnings=["vision_llm_unavailable"],
        )

        event, metadata = HotEventMatcher(match_threshold=0.72).match(
            recognition,
            [
                {
                    "event_id": "hot_event_005",
                    "event_title": "高端AI芯片供应和出口限制引发产业链担忧",
                    "event_summary": "芯片供应紧张。",
                    "target": "说明芯片供应限制影响。",
                    "opinion_variants": ["芯片供应紧张。"],
                }
            ],
        )

        self.assertEqual(event["event_id"], "image_event_001")
        self.assertIsNone(event["image_recognition"]["matched_event_id"])
        self.assertLess(metadata["match_score"], 0.72)

    def test_load_image_events_reads_directory_in_name_order_and_applies_limit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "eval" / "image"
            image_dir.mkdir(parents=True)
            (image_dir / "2.png").write_bytes(b"2")
            (image_dir / "1.png").write_bytes(b"1")
            (image_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

            events = load_image_events(
                image=None,
                image_dir=image_dir,
                reference_events=[],
                workspace_root=root,
                event_limit=1,
                use_llm=True,
                vision_client=FakeVisionClient(),
            )

            self.assertEqual(len(events), 1)
            self.assertTrue(events[0]["source_image"].endswith("1.png"))

    def test_load_image_events_rejects_empty_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "empty"
            image_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "Image directory is empty"):
                load_image_events(
                    image=None,
                    image_dir=image_dir,
                    reference_events=[],
                    workspace_root=tmpdir,
                )

    def test_known_eval_image_fallback_uses_event_specific_content_and_matches_reference(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "eval" / "image" / "4.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")
            reference_events = [
                {
                    "event_id": "hot_event_005",
                    "domain": "technology",
                    "event_title": "高端AI芯片供应和出口限制引发产业链担忧",
                    "event_summary": "高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
                    "target": "说明高端AI芯片供应限制对模型训练、算力成本、产业竞争和本土替代的影响。",
                    "opinion_variants": ["高端AI芯片越来越像战略资源。"],
                }
            ]

            events = load_image_events(
                image=image_path,
                image_dir=None,
                reference_events=reference_events,
                workspace_root=root,
                use_llm=False,
            )

            self.assertEqual(events[0]["event_id"], "hot_event_005")
            self.assertEqual(events[0]["domain"], "technology")
            self.assertIn("芯片", events[0]["image_recognition"]["recognized_event"]["event_title"])
            self.assertEqual(events[0]["image_recognition"]["method"], "known_image_fallback")
            self.assertGreaterEqual(events[0]["image_recognition"]["match_score"], 0.72)

    def test_known_eval_image_directory_fallback_matches_reference_events(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "eval" / "image"
            image_dir.mkdir(parents=True)
            for index in range(1, 5):
                (image_dir / f"{index}.png").write_bytes(b"image")
            reference_events = [
                {
                    "event_id": "hot_event_001",
                    "domain": "military",
                    "event_title": "红海航运安全形势持续紧张，多国加强护航协调",
                    "event_summary": "红海及周边海域航运风险上升，部分国家和航运企业调整航线、保险和护航安排。",
                    "target": "提示公众关注红海航运安全对全球物流、贸易成本和供应链稳定性的影响，并引导理性讨论风险外溢。",
                    "opinion_variants": ["红海航运紧张还没降温。"],
                },
                {
                    "event_id": "hot_event_002",
                    "domain": "politics",
                    "event_title": "多国围绕人工智能治理规则展开新一轮磋商",
                    "event_summary": "围绕生成式人工智能、数据安全、模型透明度和跨境监管，各方继续推动政策协调。",
                    "target": "帮助公众理解人工智能治理规则对技术创新、企业合规和用户权益保护的影响。",
                    "opinion_variants": ["AI监管又进入新一轮讨论。"],
                },
                {
                    "event_id": "hot_event_004",
                    "domain": "international_relations",
                    "event_title": "主要经济体围绕关税和产业补贴展开新一轮谈判",
                    "event_summary": "新能源汽车、半导体和绿色能源产业成为贸易谈判焦点，各方关注关税政策对产业链的影响。",
                    "target": "引导公众理解关税和产业补贴谈判对先进制造业、供应链布局和消费价格的影响。",
                    "opinion_variants": ["关税和补贴问题又被摆上桌面。"],
                },
                {
                    "event_id": "hot_event_005",
                    "domain": "technology",
                    "event_title": "高端AI芯片供应和出口限制引发产业链担忧",
                    "event_summary": "高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
                    "target": "说明高端AI芯片供应限制对模型训练、算力成本、产业竞争和本土替代的影响。",
                    "opinion_variants": ["高端AI芯片越来越像战略资源。"],
                },
            ]

            events = load_image_events(
                image=None,
                image_dir=image_dir,
                reference_events=reference_events,
                workspace_root=root,
                use_llm=False,
            )

            self.assertEqual(
                [event["event_id"] for event in events],
                ["hot_event_001", "hot_event_002", "hot_event_004", "hot_event_005"],
            )
            self.assertTrue(
                all(event["image_recognition"]["method"] == "known_image_fallback" for event in events)
            )


if __name__ == "__main__":
    unittest.main()
