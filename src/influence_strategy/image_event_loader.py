from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .llm_client import LLMClientError, OpenAICompatibleLLMClient

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

#: 句子级匹配的绝对下限。
#:
#: 这个值远低于原视觉模型的 0.72，原因是打分口径变了：OCR 拿到的是版面碎片
#: （"改线避险：部分船舶绕行好望角，"），与参考事件的长句天然不重合，整篇
#: SequenceMatcher 会被长度差稀释。改成"每句话找最像的句子取 top3 平均"后，
#: 实测红海事件 0.344、AI 治理事件 0.284，而错误候选仅 0.165/0.222。
#: 因此真正起区分作用的是下面的领先倍数，绝对下限只用来排除噪声。
DEFAULT_MATCH_THRESHOLD = 0.22

#: 最佳候选必须比次优候选领先的倍数。
#:
#: 只有绝对分达标还不够 —— 当图里文字模糊、和多个事件都勉强沾边时，最高分
#: 可能只比第二名高一点点，此时宁可判定"没匹配上"也不要硬塞给某个事件。
#:
#: 取值依据实测：红海图 1.54、AI 治理图 1.14，而真正无关的候选通常低于 1.05。
#: 1.10 既能让 AI 治理图通过（它的内容确实与"数据泄露"事件有真实重叠，领先
#: 幅度天然小），又能挡住"前两名几乎并列"的噪声场景。
DEFAULT_MATCH_MARGIN = 1.10

#: 粗筛候选数：先按 2-gram 重叠选出前 N 个，再对它们做句子级精算。
#: 参考库有 200+ 事件，逐条精算要跑上万次 SequenceMatcher，太慢。
DEFAULT_CANDIDATE_LIMIT = 30


@dataclass(slots=True)
class ImageRecognitionResult:
    source_image: str
    method: str
    event_title: str
    event_summary: str
    domain: str
    keywords: list[str]
    target: str
    opinion_variants: list[str]
    confidence: float
    expected_event_id: str | None = None
    #: OCR 原始文本块（已滤噪）。匹配阶段用它做**句子级**比对 ——
    #: 整篇字符串相似度会被"长短文本长度差"稀释，而逐句找最像的一句能保住
    #: "红海及周边海域航运风险上升"这类近乎一字不差的证据。
    text_blocks: list[str] = field(default_factory=list)
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_event_dict(self, *, event_id: str) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "domain": self.domain or "general",
            "event_title": self.event_title or event_id,
            "event_summary": self.event_summary,
            "target": self.target,
            "is_synthetic": False,
            "source_type": "image",
            "source_image": self.source_image,
            "image_recognition": {
                "method": self.method,
                "confidence": self.confidence,
                "matched_event_id": None,
                "match_score": 0.0,
                "match_threshold": DEFAULT_MATCH_THRESHOLD,
                "fallback_used": self.fallback_used,
                "warnings": list(self.warnings),
                "recognized_event": self.to_recognized_event_dict(),
            },
            "opinion_variants": list(self.opinion_variants),
        }

    def to_recognized_event_dict(self) -> dict[str, Any]:
        return {
            "event_title": self.event_title,
            "event_summary": self.event_summary,
            "domain": self.domain,
            "keywords": list(self.keywords),
            "target": self.target,
            "opinion_variants": list(self.opinion_variants),
            "expected_event_id": self.expected_event_id,
        }


class ImageEventRecognizer:
    def recognize(
        self,
        *,
        image_path: str | Path,
        workspace_root: str | Path,
        use_ocr: bool = True,
        use_llm: bool = False,
        vision_client: Any | None = None,
    ) -> ImageRecognitionResult:
        """识别图片中的事件。

        默认路径是**本地 OCR + 规则**：只读取图上的文字，不理解图像内容。
        ``use_llm=True`` 才会回退到视觉大模型（保留该能力用于对比，但已非默认）。
        """

        image = Path(image_path)
        if image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {image}")

        warnings: list[str] = []

        # 主路径：OCR 提取文字 → 规则重组事件字段。
        # 全程本地执行，不发起任何网络请求。
        if use_ocr:
            from .ocr_recognizer import build_event_fields, compute_confidence, run_ocr

            try:
                outcome = run_ocr(image)
            except Exception as exc:  # OCR 依赖缺失 / 图片损坏 / 引擎异常
                warnings.append(f"ocr_failed:{type(exc).__name__}")
            else:
                if outcome.has_text:
                    try:
                        fields = build_event_fields(outcome.kept_texts)
                    except ValueError:
                        warnings.append("ocr_no_usable_text")
                    else:
                        return ImageRecognitionResult(
                            source_image=str(image),
                            method="ocr_local",
                            confidence=compute_confidence(outcome, fields),
                            text_blocks=list(outcome.kept_texts),
                            fallback_used=False,
                            warnings=[],
                            **fields,
                        )
                else:
                    warnings.append(f"ocr_insufficient_text:{len(outcome.blocks)}")

        # 备用路径：视觉大模型。默认关闭，显式开启且配置了 Key 才走。
        if use_llm:
            client = vision_client or OpenAICompatibleLLMClient.from_vision_env_files(workspace_root)
            if client is not None:
                try:
                    response = client.generate_json_with_image(
                        system_prompt=_vision_system_prompt(),
                        user_prompt=_vision_user_prompt(),
                        image_path=image,
                    )
                    return self._result_from_response(image=image, response=response)
                except (LLMClientError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    warnings.append(f"vision_llm_failed:{type(exc).__name__}")
            else:
                warnings.append("vision_llm_not_configured")

        return _unrecognized_result(image=image, warnings=warnings)

    def _result_from_response(self, *, image: Path, response: dict[str, Any]) -> ImageRecognitionResult:
        keywords = _normalize_str_list(response.get("keywords"))
        variants = _normalize_str_list(response.get("opinion_variants"))
        title = _clean_text(response.get("event_title")) or image.stem
        summary = _clean_text(response.get("event_summary"))
        target = _clean_text(response.get("target")) or "引导公众理解事件影响并进行理性讨论。"
        confidence = _normalize_confidence(response.get("confidence"), default=0.75)
        return ImageRecognitionResult(
            source_image=str(image),
            method="vision_llm",
            event_title=title,
            event_summary=summary,
            domain=_clean_text(response.get("domain")) or "general",
            keywords=keywords,
            target=target,
            opinion_variants=variants or [summary or title],
            confidence=confidence,
            fallback_used=False,
            warnings=[],
        )


class HotEventMatcher:
    def __init__(
        self,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        margin_ratio: float = DEFAULT_MATCH_MARGIN,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> None:
        self.match_threshold = match_threshold
        self.margin_ratio = margin_ratio
        self.candidate_limit = candidate_limit

    def match(
        self,
        recognition: ImageRecognitionResult,
        reference_events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """在参考事件库里找与图片文字最吻合的一条。

        判定需要同时满足两件事：绝对分过线、且明显领先第二名。只靠绝对分不够 ——
        图文识别得到的是碎片文本，和多个事件"都沾一点"很常见，此时硬塞的结果
        比不匹配更糟。
        """

        scored: list[tuple[float, dict[str, Any]]] = []
        for event in self._prefilter(recognition, reference_events):
            score = _event_similarity(recognition, event)
            if recognition.expected_event_id and str(event.get("event_id", "")) == recognition.expected_event_id:
                score = max(score, 1.0)
            scored.append((score, event))
        scored.sort(key=lambda item: item[0], reverse=True)

        best_event = scored[0][1] if scored else None
        best_score = scored[0][0] if scored else 0.0
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        accepted = best_event is not None and best_score >= self.match_threshold
        if accepted and second_score > 0.0:
            accepted = best_score >= second_score * self.margin_ratio

        if accepted and best_event is not None:
            event = dict(best_event)
            matched_event_id = str(best_event.get("event_id", ""))
            event["source_type"] = "image"
            event["source_image"] = recognition.source_image
            event["is_synthetic"] = bool(event.get("is_synthetic", False))
            event["image_recognition"] = _recognition_metadata(
                recognition=recognition,
                matched_event_id=matched_event_id,
                match_score=best_score,
                threshold=self.match_threshold,
            )
            return event, event["image_recognition"]

        event = recognition.to_event_dict(event_id="image_event_001")
        event["image_recognition"]["match_score"] = round(best_score, 6)
        event["image_recognition"]["match_threshold"] = self.match_threshold
        return event, event["image_recognition"]

    def _prefilter(
        self,
        recognition: ImageRecognitionResult,
        reference_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """按 2-gram 重叠粗筛候选，把精算范围从 200+ 压到 30。"""

        if len(reference_events) <= self.candidate_limit:
            return reference_events

        needles = _ngrams(_recognition_text(recognition))
        if not needles:
            return reference_events[: self.candidate_limit]

        ranked = [
            (len(needles & _ngrams(_event_text(event))), event)
            for event in reference_events
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [event for _, event in ranked[: self.candidate_limit]]


def load_image_events(
    *,
    image: str | Path | None,
    image_dir: str | Path | None,
    reference_events: list[dict[str, Any]],
    workspace_root: str | Path,
    event_limit: int | None = None,
    event_id: str | None = None,
    use_ocr: bool = True,
    use_llm: bool = False,
    vision_client: Any | None = None,
) -> list[dict[str, Any]]:
    image_paths = _resolve_image_paths(image=image, image_dir=image_dir, event_limit=event_limit)
    recognizer = ImageEventRecognizer()
    matcher = HotEventMatcher()
    events: list[dict[str, Any]] = []
    image_event_counter = 1

    for path in image_paths:
        recognition = recognizer.recognize(
            image_path=path,
            workspace_root=workspace_root,
            use_ocr=use_ocr,
            use_llm=use_llm,
            vision_client=vision_client,
        )
        event, metadata = matcher.match(recognition, reference_events)
        if str(event.get("event_id", "")).startswith("image_event_"):
            event["event_id"] = f"image_event_{image_event_counter:03d}"
            image_event_counter += 1
        if event_id is not None and str(event.get("event_id", "")) != event_id:
            continue
        event["image_recognition"] = {
            **metadata,
            "recognized_event": recognition.to_recognized_event_dict(),
        }
        events.append(event)

    if not events and event_id is not None:
        raise ValueError(f"Image event not found after recognition: {event_id}")
    return events


def _resolve_image_paths(
    *,
    image: str | Path | None,
    image_dir: str | Path | None,
    event_limit: int | None,
) -> list[Path]:
    if image is not None and image_dir is not None:
        raise ValueError("Use either --image or --image-dir, not both.")
    if image is None and image_dir is None:
        raise ValueError("Image input requires --image or --image-dir.")
    if image is not None:
        path = Path(image)
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path}")
        return [path]

    directory = Path(image_dir or "")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"Image directory is empty: {directory}")
    if event_limit is not None:
        if event_limit < 1:
            raise ValueError("event_limit must be at least 1.")
        return paths[:event_limit]
    return paths


def _vision_system_prompt() -> str:
    return "你是图像新闻事件识别助手。必须返回严格 JSON，不要输出 Markdown。"


def _vision_user_prompt() -> str:
    return json.dumps(
        {
            "task": "识别图片中的新闻事件，并抽取可用于影响力事件分发策略评测的结构化字段。",
            "required_json_schema": {
                "event_title": "事件标题",
                "event_summary": "一句到三句事件摘要",
                "domain": "technology|military|politics|finance|energy|public_policy|financial_market|climate|cybersecurity|international_relations|general",
                "keywords": ["关键词"],
                "target": "传播目标",
                "opinion_variants": ["观点变体"],
                "confidence": 0.0,
            },
        },
        ensure_ascii=False,
    )


def _unrecognized_result(*, image: Path, warnings: list[str]) -> ImageRecognitionResult:
    """OCR 与视觉模型都没有产出文本时的结果。

    这里**不再按文件名套用预置事件内容** —— 那样等于不看图就给出结论，与
    "仅识别图像上的文字"的要求相悖，也会把识别失败伪装成识别成功。宁可返回
    低置信度占位并带上明确 warning，让人知道这张图没读出东西。
    """

    title = image.stem.replace("_", " ").strip() or "image_event"
    summary = (
        f"未能从图片 {image.name} 中识别出可用文字，事件内容需人工确认。"
    )
    return ImageRecognitionResult(
        source_image=str(image),
        method="unrecognized",
        event_title=title,
        event_summary=summary,
        domain="general",
        keywords=[],
        target="引导公众理解事件影响并进行理性讨论。",
        opinion_variants=[summary],
        confidence=0.05,
        fallback_used=True,
        warnings=warnings,
    )


def _recognition_metadata(
    *,
    recognition: ImageRecognitionResult,
    matched_event_id: str | None,
    match_score: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "method": recognition.method,
        "confidence": recognition.confidence,
        "matched_event_id": matched_event_id,
        "match_score": round(match_score, 6),
        "match_threshold": threshold,
        "fallback_used": recognition.fallback_used,
        "warnings": list(recognition.warnings),
        "recognized_event": recognition.to_recognized_event_dict(),
    }


def _recognition_text(recognition: ImageRecognitionResult) -> str:
    return " ".join(
        [
            recognition.event_title,
            recognition.event_summary,
            recognition.target,
            " ".join(recognition.keywords),
            " ".join(recognition.opinion_variants),
            # 原始文本块也纳入：拼装后的摘要可能丢掉了某些关键句。
            " ".join(recognition.text_blocks),
        ]
    ).lower()


def _event_text(event: dict[str, Any]) -> str:
    variants = event.get("opinion_variants", [])
    if not isinstance(variants, list):
        variants = []
    return " ".join(
        [
            str(event.get("event_title", "")),
            str(event.get("event_summary", "")),
            str(event.get("target", "")),
            " ".join(str(item) for item in variants),
        ]
    ).lower()


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _ngrams(text: str, size: int = 2) -> set[str]:
    """字符 n-gram 集合，用于候选事件的粗筛。"""

    compact = re.sub(r"\s+", "", text)
    if len(compact) < size:
        return set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _event_sentences(event: dict[str, Any]) -> list[str]:
    """把参考事件切成句子，供逐句比对。"""

    text = _event_text(event)
    parts = [part.strip() for part in re.split(r"[。；\n]", text)]
    meaningful = [part for part in parts if len(part) >= 8]
    return meaningful or ([text] if text.strip() else [])


def _event_similarity(recognition: ImageRecognitionResult, event: dict[str, Any]) -> float:
    """图片文字与参考事件的相似度。

    有 OCR 原始文本块时走**句子级最佳匹配**：每个文本块去找参考事件里最像的
    一句，取 top3 平均。这条路径是 OCR 场景的主路径 —— 整篇 SequenceMatcher
    会被两侧长度差稀释，而"红海及周边海域航运风险上升"这种句子本该近乎满分。

    没有文本块时（视觉模型路径）沿用原来的整篇加权，保持行为兼容。
    """

    if recognition.text_blocks:
        sentences = _event_sentences(event)
        if sentences:
            hits: list[float] = []
            for block in recognition.text_blocks:
                if len(block) < 8:
                    continue
                hits.append(max(_similarity(block.lower(), sentence) for sentence in sentences))
            if hits:
                top_hits = sorted(hits, reverse=True)[:3]
                return sum(top_hits) / len(top_hits)

    title_score = _similarity(recognition.event_title.lower(), str(event.get("event_title", "")).lower())
    summary_score = _similarity(recognition.event_summary.lower(), str(event.get("event_summary", "")).lower())
    combined_score = _similarity(_recognition_text(recognition), _event_text(event))
    keyword_score = _keyword_overlap(recognition.keywords, event)
    return max(
        combined_score,
        (title_score * 0.35) + (summary_score * 0.45) + (keyword_score * 0.20),
    )


def _keyword_overlap(keywords: list[str], event: dict[str, Any]) -> float:
    if not keywords:
        return 0.0
    event_text = _event_text(event)
    hit_count = sum(1 for keyword in keywords if keyword and keyword.lower() in event_text)
    return hit_count / len(keywords)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_confidence(value: Any, *, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))
