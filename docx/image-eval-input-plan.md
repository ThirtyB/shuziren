# Image Eval Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `run_eval.py` to accept image inputs from `eval/image`, recognize the event in each image, normalize it into the existing hot_event shape, and run the current strategy pipeline unchanged.

**Architecture:** Add a focused image input layer before the existing eval pipeline. The layer uses a vision-capable OpenAI-compatible client when enabled, falls back to deterministic low-confidence extraction when unavailable, matches recognized events against the existing hot_event dataset, and writes image recognition trace files beside the existing pipeline trace.

**Tech Stack:** Python 3.11, stdlib `urllib`, `base64`, `mimetypes`, `difflib`, `json`, `pathlib`, existing `unittest` tests, existing OpenAI-compatible chat completions API.

## Global Constraints

- Keep existing text eval commands unchanged: `python run_eval.py` and `python run_eval.py --input eval/hot_event_opinion_variants.json`.
- Support image file types `.png`, `.jpg`, `.jpeg`, and `.webp`.
- Support both `--image eval/image/4.png` and `--image-dir eval/image`.
- In image mode, treat `--input` as the reference hot_event dataset for matching.
- Use `match_threshold = 0.72`.
- Prefer vision LLM when available and `use_llm=True`.
- If `VISION_*` variables are absent, reuse existing LLM/DashScope configuration.
- Do not pass images into `event_parser`, `feature_builder`, `scorer`, `selector`, or `strategy_generator`.
- Write image trace files under `tests/pipeline_step_outputs/<event_id>/`.
- Do not require offline OCR for the first implementation.
- Preserve current final output path pattern: `eval/output/<event_id>_strategy_output.json`.

---

## File Structure

- Modify `src/influence_strategy/llm_client.py`: add vision-aware JSON generation and `VISION_*` env resolution.
- Create `src/influence_strategy/image_event_loader.py`: image recognition, fallback event extraction, hot_event matching, and image input loading.
- Modify `src/influence_strategy/eval_hot_events.py`: add event-list evaluation entry point and write image recognition trace files when present.
- Modify `run_eval.py`: add `--image` and `--image-dir` CLI options and route image mode through `image_event_loader`.
- Modify `.env.example`: document optional `VISION_*` and `OCR_ENABLED`.
- Create `tests/test_image_event_loader.py`: unit tests for recognition normalization, matching, image listing, and fallback behavior.
- Modify `tests/test_eval_hot_events.py`: cover image metadata trace behavior and event-list evaluation.
- Create `tests/test_run_eval_cli.py`: CLI argument/routing tests using fakes.

---

### Task 1: Add Vision JSON Support To LLM Client

**Files:**
- Modify: `src/influence_strategy/llm_client.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: existing `OpenAICompatibleLLMClient._post_chat_completions(payload: dict[str, Any]) -> dict[str, Any]`
- Produces:
  - `OpenAICompatibleLLMClient.from_vision_env_files(workspace_root: str | Path) -> OpenAICompatibleLLMClient | None`
  - `OpenAICompatibleLLMClient.generate_json_with_image(system_prompt: str, user_prompt: str, image_path: str | Path, mime_type: str | None = None) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests for vision env fallback and image payload**

Add these tests to `tests/test_llm_client.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import base64
import unittest

from influence_strategy.llm_client import OpenAICompatibleLLMClient


class VisionLLMClientTest(unittest.TestCase):
    def test_from_vision_env_files_prefers_vision_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=text-key",
                        "LLM_BASE_URL=https://text.example/v1",
                        "LLM_MODEL=text-model",
                        "VISION_LLM_API_KEY=vision-key",
                        "VISION_LLM_BASE_URL=https://vision.example/v1",
                        "VISION_LLM_MODEL=qwen3.7-plus",
                    ]
                ),
                encoding="utf-8",
            )

            client = OpenAICompatibleLLMClient.from_vision_env_files(root)

            self.assertIsNotNone(client)
            assert client is not None
            self.assertEqual(client.api_key, "vision-key")
            self.assertEqual(client.base_url, "https://vision.example/v1")
            self.assertEqual(client.model, "qwen3.7-plus")
            self.assertEqual(client.provider, "dashscope")

    def test_from_vision_env_files_falls_back_to_text_llm_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "DASHSCOPE_API_KEY=dashscope-key",
                        "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "MODEL_NAME=qwen3.7-plus",
                    ]
                ),
                encoding="utf-8",
            )

            client = OpenAICompatibleLLMClient.from_vision_env_files(root)

            self.assertIsNotNone(client)
            assert client is not None
            self.assertEqual(client.api_key, "dashscope-key")
            self.assertEqual(client.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual(client.model, "qwen3.7-plus")

    def test_generate_json_with_image_sends_openai_compatible_image_content(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            image_path.write_bytes(b"fake-png-bytes")
            captured_payloads: list[dict] = []

            class CapturingClient(OpenAICompatibleLLMClient):
                def _post_chat_completions(self, payload: dict) -> dict:
                    captured_payloads.append(payload)
                    return {"choices": [{"message": {"content": '{"ok": true}'}}]}

            client = CapturingClient(
                api_key="key",
                base_url="https://example.test/v1",
                model="vision-model",
            )

            result = client.generate_json_with_image(
                system_prompt="system",
                user_prompt="describe",
                image_path=image_path,
            )

            self.assertEqual(result, {"ok": True})
            user_content = captured_payloads[0]["messages"][1]["content"]
            self.assertEqual(user_content[0]["type"], "text")
            self.assertEqual(user_content[0]["text"], "describe")
            self.assertEqual(user_content[1]["type"], "image_url")
            expected_b64 = base64.b64encode(b"fake-png-bytes").decode("ascii")
            self.assertEqual(
                user_content[1]["image_url"]["url"],
                f"data:image/png;base64,{expected_b64}",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm_client.py::VisionLLMClientTest -v`

Expected: FAIL because `from_vision_env_files` and `generate_json_with_image` do not exist.

- [ ] **Step 3: Implement vision env resolution and image payload**

In `src/influence_strategy/llm_client.py`, add imports:

```python
import base64
import mimetypes
```

Add this classmethod inside `OpenAICompatibleLLMClient` after `from_env_files`:

```python
    @classmethod
    def from_vision_env_files(cls, workspace_root: str | Path) -> "OpenAICompatibleLLMClient | None":
        env_values: dict[str, str] = {}
        root = Path(workspace_root)
        for path in (
            root / ".env",
            root / "src" / "influence_strategy" / ".env",
        ):
            env_values.update(_read_env_file(path))

        api_key = _first_non_empty(
            env_values.get("VISION_LLM_API_KEY"),
            os.environ.get("VISION_LLM_API_KEY"),
            env_values.get("LLM_API_KEY"),
            os.environ.get("LLM_API_KEY"),
            env_values.get("DASHSCOPE_API_KEY"),
            os.environ.get("DASHSCOPE_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        )
        base_url = _first_non_empty(
            env_values.get("VISION_LLM_BASE_URL"),
            os.environ.get("VISION_LLM_BASE_URL"),
            env_values.get("LLM_BASE_URL"),
            os.environ.get("LLM_BASE_URL"),
            env_values.get("DASHSCOPE_BASE_URL"),
            os.environ.get("DASHSCOPE_BASE_URL"),
            env_values.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            env_values.get("OPENAI_API_BASE"),
            os.environ.get("OPENAI_API_BASE"),
            "https://api.openai.com/v1",
        )
        model = _first_non_empty(
            env_values.get("VISION_LLM_MODEL"),
            os.environ.get("VISION_LLM_MODEL"),
            env_values.get("LLM_MODEL"),
            os.environ.get("LLM_MODEL"),
            env_values.get("MODEL_NAME"),
            os.environ.get("MODEL_NAME"),
            env_values.get("DASHSCOPE_MODEL"),
            os.environ.get("DASHSCOPE_MODEL"),
            env_values.get("OPENAI_MODEL"),
            os.environ.get("OPENAI_MODEL"),
            "gpt-4o-mini",
        )
        provider = _first_non_empty(
            env_values.get("VISION_LLM_PROVIDER"),
            os.environ.get("VISION_LLM_PROVIDER"),
            env_values.get("LLM_PROVIDER"),
            os.environ.get("LLM_PROVIDER"),
        )

        if not api_key:
            return None

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider or _infer_provider(base_url=base_url, model=model),
        )
```

Add this method inside `OpenAICompatibleLLMClient` after `generate_json`:

```python
    def generate_json_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        image = Path(image_path)
        resolved_mime_type = mime_type or mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        image_b64 = base64.b64encode(image.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{resolved_mime_type};base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            return self._post_chat_completions(payload)
        except LLMClientError:
            payload.pop("response_format", None)
            return self._post_chat_completions(payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_client.py::VisionLLMClientTest -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/influence_strategy/llm_client.py tests/test_llm_client.py
git commit -m "feat: add vision JSON support to LLM client"
```

---

### Task 2: Create Image Event Loader, Recognizer, Matcher, And Unit Tests

**Files:**
- Create: `src/influence_strategy/image_event_loader.py`
- Create: `tests/test_image_event_loader.py`

**Interfaces:**
- Consumes:
  - `OpenAICompatibleLLMClient.from_vision_env_files(workspace_root)`
  - `client.generate_json_with_image(system_prompt, user_prompt, image_path)`
- Produces:
  - `ImageRecognitionResult`
  - `ImageEventRecognizer.recognize(image_path: str | Path, workspace_root: str | Path, use_llm: bool = True, vision_client: Any | None = None) -> ImageRecognitionResult`
  - `HotEventMatcher.match(recognition: ImageRecognitionResult, reference_events: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]`
  - `load_image_events(image: str | Path | None, image_dir: str | Path | None, reference_events: list[dict[str, Any]], workspace_root: str | Path, event_limit: int | None = None, event_id: str | None = None, use_llm: bool = True, vision_client: Any | None = None) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing tests for recognition, matching, fallback, and image listing**

Create `tests/test_image_event_loader.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_event_loader.py -v`

Expected: FAIL because `image_event_loader.py` does not exist.

- [ ] **Step 3: Implement image event loader**

Create `src/influence_strategy/image_event_loader.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .llm_client import LLMClientError, OpenAICompatibleLLMClient

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MATCH_THRESHOLD = 0.72


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
            score = _similarity(_recognition_text(recognition), _event_text(event))
            if score > best_score:
                best_score = score
                best_event = event

        if best_event is not None and best_score >= self.match_threshold:
            event = dict(best_event)
            event["source_type"] = "image"
            event["source_image"] = recognition.source_image
            event["is_synthetic"] = bool(event.get("is_synthetic", False))
            event["image_recognition"] = _recognition_metadata(
                recognition=recognition,
                matched_event_id=str(best_event.get("event_id", "")),
                match_score=best_score,
                threshold=self.match_threshold,
            )
            return event, event["image_recognition"]

        event = recognition.to_event_dict(event_id="image_event_001")
        event["image_recognition"]["match_score"] = best_score
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
            assigned_id = f"image_event_{image_event_counter:03d}"
            event["event_id"] = assigned_id
            image_event_counter += 1
        if event_id is not None and str(event.get("event_id", "")) != event_id:
            continue
        event["image_recognition"] = metadata | {
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
```

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest tests/test_image_event_loader.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/influence_strategy/image_event_loader.py tests/test_image_event_loader.py
git commit -m "feat: normalize image inputs into hot events"
```

---

### Task 3: Add Event-List Evaluation And Image Trace Files

**Files:**
- Modify: `src/influence_strategy/eval_hot_events.py`
- Modify: `tests/test_eval_hot_events.py`

**Interfaces:**
- Consumes: hot_event dicts produced by `load_image_events()`
- Produces:
  - `run_hot_event_dict_evaluations(workspace_root: str | Path, events: list[dict[str, Any]], output_dir: str | Path | None = None, event_id: str | None = None, event_limit: int | None = None, profile_limit: int | None = None, max_selected_nodes: int = 5, risk_level: str | None = None, campaign_window_hours: int = 24, max_frequency_per_day: int = 3, allowed_platforms: list[str] | None = None, use_llm: bool = True, llm_client: Any | None = None, trace_dir: str | Path | None = None) -> list[tuple[Path, dict[str, Any]]]`
  - extra trace files `00_image_input.json` and `00_image_recognition.json` when `hot_event["source_type"] == "image"`

- [ ] **Step 1: Write failing tests for event-list evaluation and image trace**

Add this import in `tests/test_eval_hot_events.py`:

```python
    run_hot_event_dict_evaluations,
```

Add this test method to `EvalHotEventsTest`:

```python
    def test_run_hot_event_dict_evaluations_writes_image_trace_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            trace_dir = root / "tests" / "pipeline_step_outputs"
            image_path = eval_dir / "image" / "4.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")

            events = [
                {
                    "event_id": "hot_event_005",
                    "domain": "technology",
                    "event_title": "AI chip supply image event",
                    "event_summary": "High-end chip supply is tightening.",
                    "target": "Explain the impact of chip supply changes.",
                    "source_type": "image",
                    "source_image": str(image_path),
                    "image_recognition": {
                        "method": "vision_llm",
                        "confidence": 0.91,
                        "matched_event_id": "hot_event_005",
                        "match_score": 0.87,
                        "match_threshold": 0.72,
                        "fallback_used": False,
                        "warnings": [],
                        "recognized_event": {"event_title": "AI chip supply image event"},
                    },
                    "opinion_variants": ["AI chip supply variant"],
                }
            ]

            results = run_hot_event_dict_evaluations(
                workspace_root=root,
                events=events,
                output_dir=eval_dir / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )

            self.assertEqual(len(results), 1)
            event_trace_dir = trace_dir / "hot_event_005"
            image_input = json.loads((event_trace_dir / "00_image_input.json").read_text(encoding="utf-8"))
            recognition = json.loads((event_trace_dir / "00_image_recognition.json").read_text(encoding="utf-8"))
            self.assertEqual(image_input["source_image"], str(image_path))
            self.assertEqual(recognition["matched_event_id"], "hot_event_005")
            self.assertEqual(recognition["match_threshold"], 0.72)
            self.assertTrue((event_trace_dir / "00_hot_event_input.json").exists())
            self.assertTrue((event_trace_dir / "07_final_output.json").exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_hot_events.py::EvalHotEventsTest::test_run_hot_event_dict_evaluations_writes_image_trace_outputs -v`

Expected: FAIL because `run_hot_event_dict_evaluations` does not exist.

- [ ] **Step 3: Add event-list evaluation function**

In `src/influence_strategy/eval_hot_events.py`, add this function after `run_hot_event_evaluations`:

```python
def run_hot_event_dict_evaluations(
    *,
    workspace_root: str | Path,
    events: list[dict[str, Any]],
    output_dir: str | Path | None = None,
    event_id: str | None = None,
    event_limit: int | None = None,
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    selected_events = select_hot_events(
        events,
        event_id=event_id,
        event_limit=event_limit,
    )

    results: list[tuple[Path, dict[str, Any]]] = []
    for hot_event in selected_events:
        output_path, output_payload = _run_hot_event_evaluation(
            workspace_root=workspace_root,
            output_dir=output_dir,
            hot_event=hot_event,
            profile_limit=profile_limit,
            max_selected_nodes=max_selected_nodes,
            risk_level=risk_level,
            campaign_window_hours=campaign_window_hours,
            max_frequency_per_day=max_frequency_per_day,
            allowed_platforms=allowed_platforms,
            use_llm=use_llm,
            llm_client=llm_client,
            trace_dir=trace_dir,
        )
        results.append((output_path, output_payload))
    return results
```

- [ ] **Step 4: Write image trace files**

In `_write_pipeline_trace`, add this block before writing `00_hot_event_input.json`:

```python
    if hot_event.get("source_type") == "image":
        image_input = {
            "event_id": hot_event.get("event_id"),
            "source_type": hot_event.get("source_type"),
            "source_image": hot_event.get("source_image"),
        }
        _write_json(
            event_dir / "00_image_input.json",
            image_input,
        )
        _write_json(
            event_dir / "00_image_recognition.json",
            hot_event.get("image_recognition", {}),
        )
```

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_eval_hot_events.py::EvalHotEventsTest::test_run_hot_event_dict_evaluations_writes_image_trace_outputs -v`

Expected: PASS.

- [ ] **Step 6: Run existing eval tests**

Run: `python -m pytest tests/test_eval_hot_events.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/influence_strategy/eval_hot_events.py tests/test_eval_hot_events.py
git commit -m "feat: evaluate normalized image hot events"
```

---

### Task 4: Add Image CLI Routing To run_eval.py

**Files:**
- Modify: `run_eval.py`
- Create: `tests/test_run_eval_cli.py`

**Interfaces:**
- Consumes:
  - `load_hot_events(input_path)`
  - `load_image_events(image: str | Path | None, image_dir: str | Path | None, reference_events: list[dict[str, Any]], workspace_root: str | Path, event_limit: int | None = None, event_id: str | None = None, use_llm: bool = True, vision_client: Any | None = None) -> list[dict[str, Any]]`
  - `run_hot_event_dict_evaluations(workspace_root: str | Path, events: list[dict[str, Any]], output_dir: str | Path | None = None, event_id: str | None = None, event_limit: int | None = None, profile_limit: int | None = None, max_selected_nodes: int = 5, risk_level: str | None = None, campaign_window_hours: int = 24, max_frequency_per_day: int = 3, allowed_platforms: list[str] | None = None, use_llm: bool = True, llm_client: Any | None = None, trace_dir: str | Path | None = None) -> list[tuple[Path, dict[str, Any]]]`
- Produces:
  - CLI args `--image` and `--image-dir`
  - existing console summary shape with image-mode outputs

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_run_eval_cli.py`:

```python
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_eval


class RunEvalCliTest(unittest.TestCase):
    def test_arg_parser_accepts_image_inputs(self) -> None:
        parser = run_eval.build_arg_parser()
        args = parser.parse_args(["--image", "eval/image/4.png", "--event-limit", "1"])

        self.assertEqual(args.image, Path("eval/image/4.png"))
        self.assertIsNone(args.image_dir)
        self.assertEqual(args.event_limit, 1)

    def test_main_routes_image_mode_through_image_loader_and_dict_evaluation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "eval" / "image" / "4.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            input_path = root / "eval" / "hot_event_opinion_variants.json"
            input_path.write_text("[]", encoding="utf-8")

            with patch.object(
                run_eval,
                "load_hot_events",
                return_value=[{"event_id": "hot_event_005"}],
            ) as load_hot_events_mock, patch.object(
                run_eval,
                "load_image_events",
                return_value=[
                    {
                        "event_id": "hot_event_005",
                        "event_title": "Image event",
                        "source_type": "image",
                        "source_image": str(image),
                    }
                ],
            ) as load_image_events_mock, patch.object(
                run_eval,
                "run_hot_event_dict_evaluations",
                return_value=[(root / "eval" / "output" / "hot_event_005_strategy_output.json", {"事件名称": "Image event"})],
            ) as run_dict_mock, patch(
                "sys.argv",
                [
                    "run_eval.py",
                    "--workspace-root",
                    str(root),
                    "--input",
                    str(input_path),
                    "--image",
                    str(image),
                    "--disable-llm",
                ],
            ):
                exit_code = run_eval.main()

            self.assertEqual(exit_code, 0)
            load_hot_events_mock.assert_called_once_with(input_path.resolve())
            load_image_events_mock.assert_called_once()
            run_dict_mock.assert_called_once()
            self.assertFalse(run_dict_mock.call_args.kwargs["use_llm"])
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run: `python -m pytest tests/test_run_eval_cli.py -v`

Expected: FAIL because `--image`, `--image-dir`, and imported image functions are not wired.

- [ ] **Step 3: Import image routing functions**

In `run_eval.py`, change the imports to:

```python
from influence_strategy.eval_hot_events import (
    load_hot_events,
    run_hot_event_dict_evaluations,
    run_hot_event_evaluations,
)
from influence_strategy.image_event_loader import load_image_events
```

- [ ] **Step 4: Add CLI args**

Inside `build_arg_parser()`, after `--input`, add:

```python
    parser.add_argument(
        "--image",
        type=Path,
        help="单张图像输入；图像会先识别为 hot_event，再进入现有评测流水线。",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="图像目录输入；默认读取 .png/.jpg/.jpeg/.webp 并按文件名排序。",
    )
```

- [ ] **Step 5: Route image mode in main()**

In `main()`, replace the existing direct call to `run_hot_event_evaluations` with this image-aware branch:

```python
    if args.image is not None or args.image_dir is not None:
        reference_events = load_hot_events(args.input.resolve())
        image_events = load_image_events(
            image=args.image.resolve() if args.image is not None else None,
            image_dir=args.image_dir.resolve() if args.image_dir is not None else None,
            reference_events=reference_events,
            workspace_root=args.workspace_root.resolve(),
            event_limit=args.event_limit,
            event_id=args.event_id,
            use_llm=not args.disable_llm,
        )
        results = run_hot_event_dict_evaluations(
            workspace_root=args.workspace_root.resolve(),
            events=image_events,
            output_dir=args.output_dir.resolve(),
            event_id=None,
            event_limit=None,
            profile_limit=args.profile_limit,
            max_selected_nodes=args.max_selected_nodes,
            risk_level=args.risk_level,
            campaign_window_hours=args.campaign_window_hours,
            max_frequency_per_day=args.max_frequency_per_day,
            allowed_platforms=allowed_platforms,
            use_llm=not args.disable_llm,
            trace_dir=args.trace_dir.resolve(),
        )
    else:
        results = run_hot_event_evaluations(
            workspace_root=args.workspace_root.resolve(),
            input_path=args.input.resolve(),
            output_dir=args.output_dir.resolve(),
            event_id=args.event_id,
            event_limit=args.event_limit,
            profile_limit=args.profile_limit,
            max_selected_nodes=args.max_selected_nodes,
            risk_level=args.risk_level,
            campaign_window_hours=args.campaign_window_hours,
            max_frequency_per_day=args.max_frequency_per_day,
            allowed_platforms=allowed_platforms,
            use_llm=not args.disable_llm,
            trace_dir=args.trace_dir.resolve(),
        )
```

- [ ] **Step 6: Run CLI tests**

Run: `python -m pytest tests/test_run_eval_cli.py -v`

Expected: PASS.

- [ ] **Step 7: Run eval tests**

Run: `python -m pytest tests/test_eval_hot_events.py tests/test_image_event_loader.py tests/test_run_eval_cli.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add run_eval.py tests/test_run_eval_cli.py
git commit -m "feat: add image inputs to eval CLI"
```

---

### Task 5: Update Configuration Example And Acceptance Documentation

**Files:**
- Modify: `.env.example`
- Modify: `docx/image-eval-input-design.md`
- Test: no code test; verify docs mention exact image commands and env vars

**Interfaces:**
- Consumes: CLI and env behavior from Tasks 1-4
- Produces: documented `VISION_*`, `OCR_ENABLED`, and example image commands

- [ ] **Step 1: Update `.env.example`**

Append this block:

```env

# Optional vision model configuration for image eval inputs.
# If these are empty, image eval falls back to the normal LLM/DashScope settings.
VISION_LLM_API_KEY=
VISION_LLM_BASE_URL=
VISION_LLM_MODEL=
VISION_LLM_PROVIDER=

# OCR is a future fallback path. The first implementation does not require local OCR.
OCR_ENABLED=true
```

- [ ] **Step 2: Update design spec with actual implementation filenames**

In `docx/image-eval-input-design.md`, ensure the module and plan filenames are listed:

```markdown
## Implementation Artifacts

- Design spec: `docx/image-eval-input-design.md`
- Implementation plan: `docx/image-eval-input-plan.md`
- Image loader module: `src/influence_strategy/image_event_loader.py`
- CLI entry: `run_eval.py --image eval/image/4.png`
- Batch CLI entry: `run_eval.py --image-dir eval/image --event-limit 4`
```

- [ ] **Step 3: Verify docs contain the expected strings**

Run:

```powershell
Select-String -Path .env.example -Pattern "VISION_LLM_MODEL"
Select-String -Path docx\image-eval-input-design.md -Pattern "image-eval-input-plan.md"
Select-String -Path docx\image-eval-input-design.md -Pattern "--image-dir eval/image"
```

Expected: all three commands print at least one matching line.

- [ ] **Step 4: Commit**

```bash
git add .env.example docx/image-eval-input-design.md
git commit -m "docs: document image eval configuration"
```

---

### Task 6: Run Full Verification And Manual Smoke Commands

**Files:**
- Modify only files needed to fix failures found by verification
- Test: full test suite and smoke commands

**Interfaces:**
- Consumes: all prior tasks
- Produces: verified image eval behavior and final commit if fixes are needed

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest -v`

Expected: PASS.

- [ ] **Step 2: Run text eval smoke test with LLM disabled**

Run: `python run_eval.py --disable-llm --event-limit 1`

Expected: command exits `0` and prints JSON containing:

```json
"event_count": 1
```

- [ ] **Step 3: Run image eval fallback smoke test with LLM disabled**

Run: `python run_eval.py --image eval/image/4.png --disable-llm --profile-limit 20`

Expected: command exits `0`, writes one JSON output under `eval/output/`, and writes:

```text
tests/pipeline_step_outputs/image_event_001/00_image_input.json
tests/pipeline_step_outputs/image_event_001/00_image_recognition.json
```

- [ ] **Step 4: Run image directory fallback smoke test with LLM disabled**

Run: `python run_eval.py --image-dir eval/image --event-limit 2 --disable-llm --profile-limit 20`

Expected: command exits `0` and prints JSON containing:

```json
"event_count": 2
```

- [ ] **Step 5: Run vision smoke test when credentials are available**

Run only if `.env` has a valid API key for DashScope or `VISION_LLM_API_KEY`:

```powershell
python run_eval.py --image eval/image/4.png --profile-limit 20
```

Expected: `tests/pipeline_step_outputs/hot_event_005/00_image_recognition.json` exists and includes:

```json
"matched_event_id": "hot_event_005"
```

- [ ] **Step 6: Inspect git status**

Run: `git status --short`

Expected: only intentional files are modified. Existing untracked `.deps/` and `eval/image/` may remain untracked and must not be committed unless the user asks.

- [ ] **Step 7: Commit verification fixes if any were required**

If Step 1-5 required code or docs fixes, commit only those changed tracked files:

```bash
git add src/influence_strategy tests run_eval.py .env.example docx
git commit -m "fix: complete image eval verification"
```

If no fixes were required, do not create an empty commit.
