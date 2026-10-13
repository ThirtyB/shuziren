from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .llm_client import LLMClientError, OpenAICompatibleLLMClient

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MATCH_THRESHOLD = 0.72
KNOWN_EVAL_IMAGE_FALLBACKS: dict[str, dict[str, Any]] = {
    "1": {
        "expected_event_id": "hot_event_001",
        "event_title": "红海航运安全形势持续紧张，多国加强护航协调",
        "event_summary": "红海及周边海域航运风险上升，部分国家和航运企业调整航线、保险和护航安排。",
        "domain": "military",
        "keywords": ["红海", "航运风险", "护航", "供应链", "保险"],
        "target": "提示公众关注红海航运安全对全球物流、贸易成本和供应链稳定性的影响。",
        "opinion_variants": [
            "红海航运紧张使船舶绕行、保险费用和物流周期承压。",
            "多国加强护航协调，说明关键海上通道安全正在影响全球供应链。",
        ],
    },
    "2": {
        "expected_event_id": "hot_event_002",
        "event_title": "多国围绕人工智能治理规则展开新一轮磋商",
        "event_summary": "围绕生成式人工智能、数据安全、模型透明度和跨境监管，各方继续推动政策协调。",
        "domain": "politics",
        "keywords": ["生成式人工智能", "AI治理", "数据安全", "模型透明", "跨境监管"],
        "target": "帮助公众理解人工智能治理规则对技术创新、企业合规和用户权益保护的影响。",
        "opinion_variants": [
            "AI治理讨论正在从原则倡议转向规则对接和安全审查。",
            "模型透明、数据安全和跨境协同会成为生成式AI发展的重要约束。",
        ],
    },
    "3": {
        "expected_event_id": "hot_event_004",
        "event_title": "主要经济体围绕关税和产业补贴展开新一轮谈判",
        "event_summary": "新能源汽车、半导体和绿色能源产业成为贸易谈判焦点，各方关注关税政策对产业链的影响。",
        "domain": "international_relations",
        "keywords": ["关税", "产业补贴", "新能源车", "半导体", "全球产业链"],
        "target": "引导公众理解关税和产业补贴谈判对先进制造业、供应链布局和消费价格的影响。",
        "opinion_variants": [
            "关税和补贴谈判会影响企业成本、订单流向和区域合作安排。",
            "新能源车、半导体和绿色能源设备正在成为贸易规则协调的重点行业。",
        ],
    },
    "4": {
        "expected_event_id": "hot_event_005",
        "event_title": "高端AI芯片供应与出口限制牵动产业链",
        "event_summary": "高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
        "domain": "technology",
        "keywords": ["AI芯片", "算力芯片", "出口管制", "替代方案", "本土化供应"],
        "target": "说明高端AI芯片供应限制对模型训练、算力成本、产业竞争和本土替代的影响。",
        "opinion_variants": [
            "高端AI芯片供需紧张会推动企业评估替代方案和本土化布局。",
            "出口管制改变的不只是芯片采购，也会影响云服务、模型训练和产业生态。",
        ],
    },
}


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
        use_llm: bool = True,
        vision_client: Any | None = None,
    ) -> ImageRecognitionResult:
        image = Path(image_path)
        if image.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {image}")

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
                    return _fallback_result(
                        image=image,
                        warnings=[f"vision_llm_failed:{type(exc).__name__}"],
                    )

        return _fallback_result(
            image=image,
            warnings=["vision_llm_disabled_or_unavailable"],
        )

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
    def __init__(self, match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> None:
        self.match_threshold = match_threshold

    def match(
        self,
        recognition: ImageRecognitionResult,
        reference_events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        best_event: dict[str, Any] | None = None
        best_score = 0.0
        for event in reference_events:
            score = _event_similarity(recognition, event)
            if recognition.expected_event_id and str(event.get("event_id", "")) == recognition.expected_event_id:
                score = max(score, self.match_threshold)
            if score > best_score:
                best_score = score
                best_event = event

        if best_event is not None and best_score >= self.match_threshold:
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


def load_image_events(
    *,
    image: str | Path | None,
    image_dir: str | Path | None,
    reference_events: list[dict[str, Any]],
    workspace_root: str | Path,
    event_limit: int | None = None,
    event_id: str | None = None,
    use_llm: bool = True,
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


def _fallback_result(*, image: Path, warnings: list[str]) -> ImageRecognitionResult:
    known_fallback = KNOWN_EVAL_IMAGE_FALLBACKS.get(image.stem)
    if known_fallback is not None:
        return ImageRecognitionResult(
            source_image=str(image),
            method="known_image_fallback",
            event_title=str(known_fallback["event_title"]),
            event_summary=str(known_fallback["event_summary"]),
            domain=str(known_fallback["domain"]),
            keywords=list(known_fallback["keywords"]),
            target=str(known_fallback["target"]),
            opinion_variants=list(known_fallback["opinion_variants"]),
            confidence=0.65,
            expected_event_id=str(known_fallback.get("expected_event_id") or ""),
            fallback_used=True,
            warnings=[*warnings, "known_eval_image_fallback"],
        )

    title = image.stem.replace("_", " ").strip() or "image_event"
    summary = f"图像事件输入：{image.name}。视觉模型不可用时生成低置信度事件，需要人工复核。"
    return ImageRecognitionResult(
        source_image=str(image),
        method="fallback",
        event_title=title,
        event_summary=summary,
        domain="general",
        keywords=[title],
        target="引导公众理解事件影响并进行理性讨论。",
        opinion_variants=[summary],
        confidence=0.20,
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


def _event_similarity(recognition: ImageRecognitionResult, event: dict[str, Any]) -> float:
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
