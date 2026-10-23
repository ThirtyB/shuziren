from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from rapidfuzz import fuzz

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import EventConstraints, EventDispatchPreferences, ParsedEvent
from .platform_profiles import DATASET_PLATFORM_VALUES
from .prompts import build_event_parser_system_prompt, build_event_parser_user_prompt

_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
_GENERIC_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+#./-]{1,31}|[\u4e00-\u9fff]{2,16}")

GENERIC_STOPWORDS = {
    "event",
    "general_public",
    "hot",
    "topic",
    "target",
    "goal",
    "event_id",
    "product_name",
    "传播",
    "信息",
    "事件",
    "热点",
    "话题",
    "内容",
    "平台",
    "受众",
    "目标",
    "阶段",
    "策略",
    "讨论",
    "发布",
    "回应",
    "影响",
}

EVENT_TYPE_RULES: dict[str, tuple[str, ...]] = {
    "public_opinion_response": ("争议", "投诉", "质疑", "回应", "澄清", "危机", "辟谣", "负面"),
    "activity_announcement": ("活动", "直播", "专场", "福利", "报名", "上线", "发布", "通知"),
    "english_learning_engagement": ("英语", "英文", "启蒙", "分级阅读", "单词", "口语"),
    "parent_child_reading": ("亲子", "家长", "阅读", "绘本", "家庭教育", "陪读"),
    "reading_habit_campaign": ("阅读计划", "阅读打卡", "书单", "读书", "阅读习惯", "阅读推广"),
}

GOAL_RULES: dict[str, tuple[str, ...]] = {
    "engagement": ("讨论", "互动", "评论", "参与", "交流"),
    "awareness": ("传播", "曝光", "扩散", "覆盖", "触达"),
    "response": ("回应", "澄清", "解释", "辟谣", "稳定"),
    "conversion": ("报名", "转化", "参与报名", "下载", "注册"),
}

AUDIENCE_RULES: dict[str, tuple[str, ...]] = {
    "parent_child": ("亲子", "家长", "育儿", "家庭教育"),
    "reading_interest": ("阅读", "绘本", "书单", "读书"),
    "english_learning": ("英语", "英文", "启蒙", "口语", "单词"),
    "education_practitioner": ("老师", "教育", "学校", "课堂"),
}

HIGH_RISK_KEYWORDS = ("危机", "投诉", "举报", "造假", "翻车", "维权", "道歉")
MEDIUM_RISK_KEYWORDS = ("争议", "质疑", "刷屏", "负面", "误解", "冲突")


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = _NON_TEXT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class RuleBasedEventParser:
    def __init__(self, default_product_name: str = "abc_reading") -> None:
        self.default_product_name = default_product_name

    def parse(
        self,
        payload: str | dict[str, Any],
        *,
        workspace_root: str | Path | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
    ) -> ParsedEvent:
        raw_event = self._normalize_payload(payload)
        parsed_event = self._parse_rule_based(raw_event)
        if not use_llm:
            return parsed_event

        client = llm_client
        if client is None and workspace_root is not None:
            client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
        if client is None:
            return parsed_event

        enriched_event = self._apply_llm_parse(
            raw_event=raw_event,
            parsed_event=parsed_event,
            llm_client=client,
        )
        return enriched_event or parsed_event

    def _parse_rule_based(self, raw_event: dict[str, Any]) -> ParsedEvent:
        merged_text = " ".join(
            [
                raw_event["event_title"],
                raw_event["event_description"],
                raw_event["target_goal"],
                " ".join(raw_event["target_audience"]),
            ]
        )
        normalized_text = _normalize_text(merged_text)

        event_type, event_reasons = self._infer_event_type(normalized_text)
        target_goal, goal_reasons = self._infer_target_goal(normalized_text, raw_event["target_goal"])
        target_audience, audience_reasons = self._infer_target_audience(normalized_text, raw_event["target_audience"])
        extracted_keywords = self._extract_keywords(raw_event, normalized_text, raw_event["target_audience"])
        constraints = self._normalize_constraints(raw_event.get("constraints", {}), normalized_text)

        reasoning = _dedupe_preserve_order(
            event_reasons + goal_reasons + audience_reasons + [f"risk_level={constraints.risk_level}"]
        )
        return ParsedEvent(
            event_id=raw_event["event_id"],
            product_name=raw_event["product_name"],
            event_title=raw_event["event_title"],
            event_description=raw_event["event_description"],
            target_goal=target_goal,
            event_type=event_type,
            target_audience=target_audience,
            extracted_keywords=extracted_keywords,
            constraints=constraints,
            semantic_tags=_dedupe_preserve_order(target_audience + extracted_keywords[:6]),
            narrative_frames=self._default_narrative_frames(event_type, target_goal),
            sensitive_entities=[],
            target_roles=self._default_target_roles(target_goal, constraints.risk_level),
            negative_constraints=self._default_negative_constraints(
                risk_level=constraints.risk_level,
                event_type=event_type,
            ),
            dispatch_preferences=self._default_dispatch_preferences(
                max_selected_nodes=constraints.max_selected_nodes,
                risk_level=constraints.risk_level,
            ),
            reasoning=reasoning,
            llm_metadata={
                "requested": False,
                "used": False,
            },
        )

    def _normalize_payload(self, payload: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, str):
            title = payload.strip()[:24] or "unnamed_event"
            return {
                "event_id": self._make_event_id(),
                "product_name": self.default_product_name,
                "event_title": title,
                "event_description": payload.strip(),
                "target_goal": "",
                "target_audience": [],
                "constraints": {},
            }

        if not isinstance(payload, dict):
            raise TypeError("Event payload must be a string or a dictionary.")

        event_description = str(payload.get("event_description", "")).strip()
        event_title = str(payload.get("event_title") or event_description[:24] or "unnamed_event").strip()
        target_goal = str(payload.get("target_goal", "")).strip()
        target_audience = [str(item).strip() for item in payload.get("target_audience", []) if str(item).strip()]
        return {
            "event_id": str(payload.get("event_id") or self._make_event_id()),
            "product_name": str(payload.get("product_name") or self.default_product_name),
            "event_title": event_title,
            "event_description": event_description,
            "target_goal": target_goal,
            "target_audience": target_audience,
            "constraints": payload.get("constraints", {}),
        }

    def _infer_event_type(self, normalized_text: str) -> tuple[str, list[str]]:
        best_type = "general_influence_event"
        best_score = 0.0
        reasons: list[str] = []

        for event_type, keywords in EVENT_TYPE_RULES.items():
            score = self._keyword_score(normalized_text, keywords)
            if score > best_score:
                best_type = event_type
                best_score = score
                reasons = [f"event_type={event_type}", f"matched_keywords={','.join(self._matched_keywords(normalized_text, keywords))}"]

        if best_score == 0.0:
            reasons = ["event_type=general_influence_event", "matched_keywords=none"]
        return best_type, reasons

    def _infer_target_goal(self, normalized_text: str, explicit_goal: str) -> tuple[str, list[str]]:
        if explicit_goal:
            return explicit_goal, [f"target_goal=explicit:{explicit_goal}"]

        best_goal = "awareness"
        best_score = 0.0
        reasons: list[str] = []
        for goal, keywords in GOAL_RULES.items():
            score = self._keyword_score(normalized_text, keywords)
            if score > best_score:
                best_goal = goal
                best_score = score
                reasons = [f"target_goal=inferred:{goal}"]
        if best_score == 0.0:
            reasons = ["target_goal=inferred:awareness"]
        return best_goal, reasons

    def _infer_target_audience(
        self,
        normalized_text: str,
        explicit_audience: list[str],
    ) -> tuple[list[str], list[str]]:
        inferred = list(explicit_audience)
        reasons = [f"target_audience=explicit:{','.join(explicit_audience)}"] if explicit_audience else []
        for audience_tag, keywords in AUDIENCE_RULES.items():
            if self._keyword_score(normalized_text, keywords) > 0.0:
                inferred.append(audience_tag)
        if not inferred:
            inferred.append("general_public")
            reasons.append("target_audience=inferred:general_public")
        else:
            reasons.append(f"target_audience=inferred:{','.join(_dedupe_preserve_order(inferred))}")
        return _dedupe_preserve_order(inferred), reasons

    def _normalize_constraints(self, raw_constraints: dict[str, Any], normalized_text: str) -> EventConstraints:
        constraints = dict(raw_constraints)
        if "risk_level" not in constraints:
            constraints["risk_level"] = self._infer_risk_level(normalized_text)
        if "allowed_platforms" not in constraints:
            constraints["allowed_platforms"] = list(DATASET_PLATFORM_VALUES)
        return EventConstraints.model_validate(constraints)

    def _infer_risk_level(self, normalized_text: str) -> str:
        if any(keyword in normalized_text for keyword in HIGH_RISK_KEYWORDS):
            return "high"
        if any(keyword in normalized_text for keyword in MEDIUM_RISK_KEYWORDS):
            return "medium"
        return "low"

    def _extract_keywords(
        self,
        raw_event: dict[str, Any],
        normalized_text: str,
        explicit_audience: list[str],
    ) -> list[str]:
        keywords: list[str] = []
        for rule_set in (EVENT_TYPE_RULES, GOAL_RULES, AUDIENCE_RULES):
            for candidate_keywords in rule_set.values():
                keywords.extend(self._matched_keywords(normalized_text, candidate_keywords))
        keywords.extend(
            self._extract_generic_keywords(
                raw_event.get("event_title", ""),
                raw_event.get("event_description", ""),
                raw_event.get("target_goal", ""),
            )
        )
        keywords.extend(explicit_audience)
        return _dedupe_preserve_order(keywords)[:12]

    def _extract_generic_keywords(self, *texts: str) -> list[str]:
        keywords: list[str] = []
        for text in texts:
            for match in _GENERIC_TERM_RE.finditer(str(text or "")):
                token = match.group(0).strip()
                normalized = _normalize_text(token)
                if not self._valid_generic_keyword(token=token, normalized=normalized):
                    continue
                keywords.append(token)
        return _dedupe_preserve_order(keywords)

    def _valid_generic_keyword(self, *, token: str, normalized: str) -> bool:
        if not token or not normalized:
            return False
        if normalized in GENERIC_STOPWORDS:
            return False
        if normalized.isdigit():
            return False
        if len(token) <= 1:
            return False
        if len(token) > 16 and any(char.isascii() for char in token):
            return False
        return True

    def _matched_keywords(self, normalized_text: str, keywords: tuple[str, ...]) -> list[str]:
        return [keyword for keyword in keywords if self._fuzzy_contains(normalized_text, keyword)]

    def _keyword_score(self, normalized_text: str, keywords: tuple[str, ...]) -> float:
        if not normalized_text:
            return 0.0
        matched = self._matched_keywords(normalized_text, keywords)
        if not matched:
            return 0.0
        return len(matched) / len(keywords)

    def _fuzzy_contains(self, normalized_text: str, keyword: str) -> bool:
        if keyword in normalized_text:
            return True
        tokens = normalized_text.split()
        if not tokens:
            return False
        return any(fuzz.partial_ratio(keyword, token) >= 90 for token in tokens)

    def _make_event_id(self) -> str:
        return f"event_{uuid4().hex[:10]}"

    def _default_narrative_frames(self, event_type: str, target_goal: str) -> list[str]:
        frames: list[str] = []
        if target_goal == "response":
            frames.extend(["事实澄清", "问答回应"])
        elif target_goal == "engagement":
            frames.extend(["议题讨论", "经验分享"])
        else:
            frames.extend(["信息触达", "摘要扩散"])

        if event_type == "public_opinion_response":
            frames.append("风险收敛")
        elif event_type == "activity_announcement":
            frames.append("活动提醒")
        return _dedupe_preserve_order(frames)

    def _default_target_roles(self, target_goal: str, risk_level: str) -> list[str]:
        roles = ["core_publish_node", "interaction_response_node", "amplification_node"]
        if risk_level == "high":
            roles = ["core_publish_node", "interaction_response_node", "support_node"]
        elif target_goal == "response":
            roles = ["core_publish_node", "interaction_response_node", "support_node"]
        elif target_goal == "engagement":
            roles = ["core_publish_node", "interaction_response_node", "amplification_node"]
        return roles

    def _default_negative_constraints(self, *, risk_level: str, event_type: str) -> list[str]:
        constraints = ["避免节点内容完全同质化"]
        if risk_level in {"medium", "high"}:
            constraints.append("避免情绪化和对立化表达")
        if risk_level == "high" or event_type == "public_opinion_response":
            constraints.append("避免未经证实的推测性扩写")
        return constraints

    def _default_dispatch_preferences(
        self,
        *,
        max_selected_nodes: int,
        risk_level: str,
    ) -> EventDispatchPreferences:
        candidate_pool_size = max(12, min(60, max_selected_nodes * 6))
        rerank_top_k = max(6, min(candidate_pool_size, max_selected_nodes * 3))
        semantic_weight = 0.42 if risk_level == "high" else 0.35
        diversity_weight = 0.16 if risk_level == "high" else 0.22
        risk_weight = 0.26 if risk_level == "high" else 0.18
        return EventDispatchPreferences(
            candidate_pool_size=candidate_pool_size,
            rerank_top_k=rerank_top_k,
            semantic_weight=semantic_weight,
            diversity_weight=diversity_weight,
            risk_weight=risk_weight,
        )

    def _apply_llm_parse(
        self,
        *,
        raw_event: dict[str, Any],
        parsed_event: ParsedEvent,
        llm_client: Any,
    ) -> ParsedEvent | None:
        fallback_parse = {
            "event_type": parsed_event.event_type,
            "target_goal": parsed_event.target_goal,
            "target_audience": parsed_event.target_audience,
            "extracted_keywords": parsed_event.extracted_keywords,
            "risk_level": parsed_event.constraints.risk_level,
            "semantic_tags": parsed_event.semantic_tags,
            "narrative_frames": parsed_event.narrative_frames,
            "target_roles": parsed_event.target_roles,
            "negative_constraints": parsed_event.negative_constraints,
            "dispatch_preferences": parsed_event.dispatch_preferences.model_dump(mode="json"),
            "reasoning": parsed_event.reasoning,
        }
        try:
            response = llm_client.generate_json(
                system_prompt=build_event_parser_system_prompt(),
                user_prompt=build_event_parser_user_prompt(
                    raw_event=raw_event,
                    fallback_parse=fallback_parse,
                ),
            )
        except (LLMClientError, OSError, ValueError, TypeError):
            return None

        try:
            event_type = str(response.get("event_type") or parsed_event.event_type).strip() or parsed_event.event_type
            target_goal = str(response.get("target_goal") or parsed_event.target_goal).strip() or parsed_event.target_goal
            target_audience = self._normalize_str_list(response.get("target_audience")) or parsed_event.target_audience
            extracted_keywords = self._merge_keyword_lists(
                self._normalize_str_list(response.get("extracted_keywords")),
                parsed_event.extracted_keywords,
            )
            risk_level = str(response.get("risk_level") or parsed_event.constraints.risk_level).strip().lower()
            if risk_level not in {"low", "medium", "high"}:
                risk_level = parsed_event.constraints.risk_level
            semantic_tags = self._merge_keyword_lists(
                self._normalize_str_list(response.get("semantic_tags")),
                parsed_event.semantic_tags,
            )
            narrative_frames = self._normalize_str_list(response.get("narrative_frames")) or parsed_event.narrative_frames
            sensitive_entities = self._normalize_str_list(response.get("sensitive_entities"))
            target_roles = self._normalize_roles(response.get("target_roles")) or parsed_event.target_roles
            negative_constraints = self._normalize_str_list(response.get("negative_constraints")) or parsed_event.negative_constraints

            raw_preferences = response.get("dispatch_preferences")
            if isinstance(raw_preferences, dict):
                merged_preferences = {
                    **parsed_event.dispatch_preferences.model_dump(mode="json"),
                    **raw_preferences,
                }
                dispatch_preferences = EventDispatchPreferences.model_validate(merged_preferences)
            else:
                dispatch_preferences = parsed_event.dispatch_preferences

            llm_reasoning = self._normalize_str_list(response.get("reasoning"))
            return parsed_event.model_copy(
                update={
                    "target_goal": target_goal,
                    "event_type": event_type,
                    "target_audience": _dedupe_preserve_order(target_audience),
                    "extracted_keywords": _dedupe_preserve_order(extracted_keywords),
                    "constraints": parsed_event.constraints.model_copy(update={"risk_level": risk_level}),
                    "semantic_tags": _dedupe_preserve_order(semantic_tags),
                    "narrative_frames": _dedupe_preserve_order(narrative_frames),
                    "sensitive_entities": _dedupe_preserve_order(sensitive_entities),
                    "target_roles": _dedupe_preserve_order(target_roles),
                    "negative_constraints": _dedupe_preserve_order(negative_constraints),
                    "dispatch_preferences": dispatch_preferences,
                    "parser_name": "rule_llm_v1",
                    "reasoning": _dedupe_preserve_order(parsed_event.reasoning + llm_reasoning + ["llm_event_parse_used"]),
                    "llm_metadata": {
                        "requested": True,
                        "used": True,
                        "provider": getattr(llm_client, "provider", None),
                        "model": getattr(llm_client, "model", None),
                    },
                }
            )
        except Exception:
            return None

    def _normalize_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return _dedupe_preserve_order([str(item).strip() for item in value if str(item).strip()])

    def _merge_keyword_lists(self, primary: list[str], fallback: list[str]) -> list[str]:
        """把 LLM 解析结果与规则解析结果**合并**（LLM 的在前，规则的补在后）。

        这里不能用 ``or`` 整体替换。规则解析会产出「讨论」「传播」「热点领域」这类
        能与 profile 兴趣标签（如「公共讨论」「热点事件」）对上的基础词；而 LLM
        倾向给出「货币政策传导滞后」「微观体感与宏观政策落差」这类新闻分析式长标签。
        一旦整体替换，``keyword_hit_count`` 与 ``topic_match_score`` 会全部归零，
        ``Scorer._semantic_gate`` 对每个节点都判否，整轮退化成 ``no_eligible_nodes``。

        大模型只负责事件解析之后这一点尤其关键：选人 / 打分都不再询问大模型，事件
        标签是唯一进入匹配环节的 LLM 产物，把基础词换掉就等于整轮白跑。
        """

        return _dedupe_preserve_order([*primary, *fallback])

    def _normalize_roles(self, value: Any) -> list[str]:
        allowed = {"core_publish_node", "interaction_response_node", "amplification_node", "support_node"}
        return [item for item in self._normalize_str_list(value) if item in allowed]
