from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from influence_strategy.image_event_loader import (
    HotEventMatcher,
    ImageEventRecognizer,
    ImageRecognitionResult,
    load_image_events,
)
from influence_strategy.ocr_recognizer import (
    OcrBlock,
    OcrOutcome,
    build_event_fields,
    compute_confidence,
    infer_domain,
    is_noise,
)

#: 真实评测图只在显式开启时跑 —— OCR 单张在 CPU 上要 3~7 秒，不应拖慢日常测试。
RUN_OCR_INTEGRATION = os.environ.get("OCR_INTEGRATION") == "1"


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


def _outcome(texts: list[str], score: float = 0.99) -> OcrOutcome:
    """构造一个已过滤噪声音的 OCR 结果，用于跳过真实引擎。"""

    blocks = [OcrBlock(text=text, score=score) for text in texts]
    return OcrOutcome(
        blocks=blocks,
        kept_texts=list(texts),
        dropped_texts=[],
        mean_score=score,
    )


class OcrTextRulesTest(unittest.TestCase):
    """OCR 文字 → 事件字段 的纯规则逻辑。"""

    def test_noise_filter_drops_station_logos_timecodes_and_single_chars(self) -> None:
        for text in ["NEWSLIVE", "LIVENEWS", "国际新闻", "最新动态", "12:30", "20:30", "1", "——", "//"]:
            self.assertTrue(is_noise(text), f"应判为噪声: {text}")
        for text in ["红海", "苏伊士运河", "护航动态", "AI", "数据安全与隐私保护"]:
            self.assertFalse(is_noise(text), f"不应判为噪声: {text}")

    def test_build_event_fields_uses_longest_sentence_as_title(self) -> None:
        texts = [
            "红海",
            "苏伊士运河",
            "护航动态",
            "红海及周边海域航运风险上升",
            "多国与航运企业同步调整航线、保险与护航安排",
        ]

        fields = build_event_fields(texts)

        self.assertEqual(fields["event_title"], "多国与航运企业同步调整航线、保险与护航安排")
        self.assertIn("红海", fields["keywords"])
        self.assertIn("苏伊士运河", fields["keywords"])
        self.assertNotIn("NEWSLIVE", fields["keywords"])

    def test_summary_skips_fragments_ending_with_comma(self) -> None:
        # 注意每句都要 >= LONG_BLOCK_MIN_CHARS(12)，否则会被判为短实体而进入关键词。
        texts = [
            "改线避险：部分船舶绕行好望角，",
            "航程与燃油成本因此明显增加。",
            "保险收紧：战争险与附加费波动，",
            "承保审核流程也随之趋于谨慎。",
        ]

        fields = build_event_fields(texts)

        # 以逗号结尾的碎片不应进入摘要，否则会出现"…绕行好望角，承保审核…"的断裂。
        self.assertNotIn("绕行好望角，承", fields["event_summary"])
        self.assertIn("航程与燃油成本因此明显增加", fields["event_summary"])
        self.assertIn("承保审核流程也随之趋于谨慎", fields["event_summary"])

    def test_infer_domain_matches_keyword_sets(self) -> None:
        self.assertEqual(infer_domain(["红海", "护航", "海域", "安全形势"]), "military")
        self.assertEqual(infer_domain(["人工智能", "模型透明度", "算力"]), "technology")
        self.assertEqual(infer_domain(["完全没有线索的文本"]), "general")

    def test_confidence_discounts_text_poor_images(self) -> None:
        rich = _outcome(
            [
                "红海及周边海域航运风险上升",
                "多家航运企业发布航行安全预警",
                "部分船公司暂停红海航线挂靠",
            ]
        )
        poor = _outcome(["红海"])

        self.assertGreater(
            compute_confidence(rich, {"event_summary": "红海及周边海域航运风险上升。" * 3}),
            compute_confidence(poor, {"event_summary": "红海"}),
        )


class ImageEventLoaderTest(unittest.TestCase):
    def test_recognizer_uses_ocr_and_builds_structured_event(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "news.png"
            image_path.write_bytes(b"image")

            with patch(
                "influence_strategy.ocr_recognizer.run_ocr",
                return_value=_outcome(
                    [
                        "红海",
                        "苏伊士运河",
                        "国际新闻",
                        "红海及周边海域航运风险上升",
                        "多国与航运企业同步调整航线、保险与护航安排",
                    ]
                ),
            ):
                result = ImageEventRecognizer().recognize(
                    image_path=image_path,
                    workspace_root=tmpdir,
                )

            self.assertEqual(result.method, "ocr_local")
            self.assertEqual(result.domain, "military")
            self.assertFalse(result.fallback_used)
            self.assertEqual(result.warnings, [])
            self.assertIn("红海", result.keywords)
            self.assertIn("护航", " ".join(result.keywords) + result.event_title)

    def test_recognizer_returns_unrecognized_when_ocr_yields_no_text(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "blank.png"
            image_path.write_bytes(b"image")

            with patch(
                "influence_strategy.ocr_recognizer.run_ocr",
                return_value=OcrOutcome(
                    blocks=[],
                    kept_texts=[],
                    dropped_texts=[],
                    mean_score=0.0,
                ),
            ):
                result = ImageEventRecognizer().recognize(
                    image_path=image_path,
                    workspace_root=tmpdir,
                )

            # 关键：不再按文件名编造事件内容，而是如实报告"没读出文字"。
            self.assertEqual(result.method, "unrecognized")
            self.assertTrue(result.fallback_used)
            self.assertLessEqual(result.confidence, 0.1)
            self.assertTrue(any("ocr_" in warning for warning in result.warnings))

    def test_recognizer_falls_back_to_vision_client_when_ocr_disabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "4.png"
            image_path.write_bytes(b"image")

            result = ImageEventRecognizer().recognize(
                image_path=image_path,
                workspace_root=tmpdir,
                use_ocr=False,
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
            method="ocr_local",
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
            method="unrecognized",
            event_title="完全不同的新事件",
            event_summary="这是一条和参考数据集没有关系的新事件。",
            domain="general",
            keywords=["新事件"],
            target="传播目标",
            opinion_variants=["新观点"],
            confidence=0.25,
            fallback_used=True,
            warnings=["ocr_insufficient_text:0"],
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
                use_ocr=False,
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


@unittest.skipUnless(RUN_OCR_INTEGRATION, "设置 OCR_INTEGRATION=1 才跑真实图片 OCR")
class OcrIntegrationTest(unittest.TestCase):
    """真实图片 OCR 集成测试（慢，默认跳过）。"""

    def test_real_news_screenshot_is_recognized_as_local_ocr(self) -> None:
        image_path = Path(__file__).resolve().parent.parent / "eval" / "image" / "1.png"
        if not image_path.exists():
            self.skipTest(f"评测图片不存在: {image_path}")

        result = ImageEventRecognizer().recognize(
            image_path=image_path,
            workspace_root=image_path.parent,
        )

        self.assertEqual(result.method, "ocr_local")
        self.assertEqual(result.fallback_used, False)
        self.assertGreater(len(result.keywords), 0)
        self.assertTrue(any(token in result.event_title + result.event_summary for token in ("红海", "护航", "航运")))


if __name__ == "__main__":
    unittest.main()
