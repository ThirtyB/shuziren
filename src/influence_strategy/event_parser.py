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

# --------------------------------------------------------------------------- #
# 通用正则抓取片段的过滤（第六轮）
#
# ``_GENERIC_TERM_RE`` 的中文分支是 ``[\u4e00-\u9fff]{2,16}``，它取的是
# **最长连续中文串**。中文没有空格，所以这个"串"经常是整条从句，例如
# 「就素质教育话题发表了不同看法」「并鼓励大家在评论区交流心得」。
# 下面这组配置只作用于**通用正则抓到的纯中文片段**，不动三套固定词表
# 命中的关键词（那部分本身就是业务词，直接放行）。
# --------------------------------------------------------------------------- #

# 从句标记：连词 / 指代 / 叙述性动词 / 泛指 / 助词。
# 出现其中任何一个，说明抓到的是"句子"而不是"词"。
_CLAUSE_MARKERS: tuple[str, ...] = (
    # 连词、副词
    "并", "而", "且", "也", "还", "又", "更", "就", "才", "却",
    # 指代、限定
    "该", "其", "此", "本次", "近期", "当前", "目前",
    # 叙述性动词
    "希望", "想让", "想要", "鼓励", "邀请", "发表", "表示", "认为",
    "引发", "引起", "产生", "放大", "展示", "面向", "围绕", "针对",
    "提升", "提高", "促进", "加强", "保持", "控制", "进行", "通过",
    # 泛指代词
    "大家", "我们", "他们",
    # 助词：几乎只出现在完整句子里
    "了", "的", "地", "得",
)

# 无分词器时，超过这个长度的连续中文串几乎必然是短语或从句而不是词。
# 注意这只是**兜底**条件，前面两层（词表覆盖、从句标记）已经先过滤过一轮。
_MAX_GENERIC_CJK_TERM = 4

_CJK_ONLY_RE = re.compile(r"^[\u4e00-\u9fff]+$")

EVENT_TYPE_RULES: dict[str, tuple[str, ...]] = {
    # 第十一轮：补录"泄露"。个人信息泄露事件同样需要舆情式回应，
    # 但"杜绝/不得泄露"属于预防语境，由 ``_keyword_prevented`` 排除。
    "public_opinion_response": (
        "争议", "投诉", "质疑", "回应", "澄清", "危机", "辟谣", "负面", "泄露",
    ),
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
    # 亲子受众：除"亲子/家长/育儿/家庭教育"这类书面词，也收录口语说法
    # （爸爸妈妈、爸妈、陪孩子、带孩子、宝妈、宝爸）。
    # 刻意**不**收录裸词"孩子"——"孩子"也会出现在与家长无关的语境里
    # （如"孩子所在学校的老师"偏教育从业者），只因出现"孩子"就判亲子
    # 会过度分类。
    "parent_child": (
        "亲子",
        "家长",
        "育儿",
        "家庭教育",
        "爸爸妈妈",
        "爸妈",
        "陪孩子",
        "带孩子",
        "宝妈",
        "宝爸",
    ),
    "reading_interest": ("阅读", "绘本", "书单", "读书"),
    "english_learning": ("英语", "英文", "启蒙", "口语", "单词"),
    "education_practitioner": ("老师", "教育", "学校", "课堂"),
    # 第十一轮：校园受众。只收录明确指向"在校学生群体"的词
    # （校园 / 学生 / 高校 / 大学 / 社团 / 班级 / 同学），
    # 刻意不收录"校园活动""教职工"这类指向组织或教职工的说法——
    # 出现"校园"并不等于受众一定是学生，判断见 ``_infer_target_audience``。
    "campus_student": ("校园", "学生", "高校", "大学", "社团", "班级", "同学"),
}

# 第十一轮：主题词表（校园 / 环保 / 隐私）。
# 复用与 EVENT_TYPE_RULES 完全相同的结构（dict[str, tuple[str, ...]]），
# 命中后走同一套 ``_matched_keywords`` → ``extracted_keywords`` →
# ``semantic_tags`` 流程，不新建平行字段。刻意不收录单字，避免误报。
TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "campus": ("校园", "学校", "高校", "大学", "学生", "社团", "班级", "校园活动"),
    "environment": ("环保", "环境保护", "节能减排", "垃圾分类", "低碳", "绿色校园", "生态保护"),
    "privacy": ("隐私", "个人信息", "隐私保护", "信息泄露", "未经授权", "敏感信息"),
}

# 隐私语境里，这几个词一旦**真实出现**（不在预防语境里）就说明泄露已发生，
# 而不只是"要注意保护"。
PRIVACY_LEAK_TERMS: tuple[str, ...] = ("泄露", "信息泄露", "未经授权")

# 受众判断的上下文依据：出现教职工专属词、且没有任何学生专属词时，
# "校园/学校"指的是单位而不是学生群体，此时不推定 campus_student。
_PRACTITIONER_ONLY_CUES: tuple[str, ...] = ("老师", "教师", "教职工", "教研", "校长")
# 注意：这里刻意**不含**"校园/学校"——它们是地点/单位词，本身不足以说明
# 受众是学生；真正的学生群体信号是"学生/班级/社团/同学"。
_STUDENT_SPECIFIC_CUES: tuple[str, ...] = ("学生", "班级", "社团", "同学")

HIGH_RISK_KEYWORDS = ("危机", "投诉", "举报", "造假", "翻车", "维权", "道歉")
# 第十一轮：补录"泄露"。它同样有"已发生"与"杜绝发生"两种语境，
# 由预防语境判断区分（见 ``_keyword_prevented``）。
MEDIUM_RISK_KEYWORDS = ("争议", "质疑", "刷屏", "负面", "误解", "冲突", "泄露")


def _collect_rule_keywords() -> frozenset[str]:
    """汇总四套固定词表里的全部关键词，用于判断"片段是否已含业务词"。"""
    terms: set[str] = set()
    for rule_set in (EVENT_TYPE_RULES, GOAL_RULES, AUDIENCE_RULES, TOPIC_RULES):
        for keywords in rule_set.values():
            terms.update(keywords)
    return frozenset(terms)


_RULE_KEYWORD_SET: frozenset[str] = _collect_rule_keywords()

# 预防性/否定线索词。当它们出现在风险关键词**前方**的局部窗口内时，
# 说明说话人是在**规避**该风险，而不是陈述风险已经发生，不应抬升风险等级。
# 只收录语义明确的"预防/禁止"词，不收录"控制/减少/降低"这类同样可能
# 描述"已发生、正在处理"的词，避免把真实风险误判成低风险。
_PREVENTION_CUES: tuple[str, ...] = (
    "避免",
    "防止",
    "杜绝",
    "规避",
    "防范",
    "以免",
    "不引发",
    "不要",
    "不应",
    # 第六轮补录：这三个同样是明确的"主观上希望不发生"，语义与"避免"
    # 等价。它们不描述已发生的事实，收录不会造成真实风险被压低。
    "不希望",
    "不想",
    "不愿",
    # 第十一轮补录：禁止类表达（"不得泄露个人信息"），语义与"避免"等价，
    # 不收录会让「不得泄露」被当成已经发生的泄露。
    "不得",
    "禁止",
    "严禁",
    "不可",
    "不应泄露",
)
# 第十一轮：已发生线索。当**同一个短句里**同时出现预防词和这些词时，
# 说明描述的是已经存在的风险（"为避免已出现的争议进一步发酵"），
# 不能因为句首有"避免"就整句压成低风险。已发生线索优先级更高。
_OCCURRED_CUES: tuple[str, ...] = (
    "已经",
    "已有",
    "已发生",
    "已出现",
    "出现",
    "发生",
    "正在",
    "持续",
    "发酵",
    "遭到",
    "被曝",
)
# 第十一轮：短句分段边界。预防/否定线索只在**同一个短句内**对其后的风险词
# 生效——过去是固定 8 字窗口（``_PREVENTION_WINDOW``），会跨过逗号把上一句的
# "避免"算到下一句的真实风险头上，也会因为句子稍长而漏判。
# 注意这里要作用在**保留标点**的文本上（``_clause_text``），
# 归一化文本会把标点换成空格，分不出句。
_CLAUSE_BOUNDARY_RE = re.compile(r"[，。；！？、：,\.;:!?]|但|但是|然而|不过|同时|另外")


def _clause_text(merged_text: str) -> str:
    """保留标点的小写文本，只用于短句分段与预防语境判断。"""
    return _SPACE_RE.sub(" ", (merged_text or "").lower()).strip()


def _clause_prefix(clause_text: str, idx: int) -> str:
    """返回 ``idx`` 所在短句里，``idx`` 之前的那一段文本。"""
    start = 0
    for match in _CLAUSE_BOUNDARY_RE.finditer(clause_text):
        if match.end() <= idx:
            start = match.end()
        else:
            break
    return clause_text[start:idx]


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
        # 保留标点的版本，只给预防语境 / 短句分段用。
        clause_text = _clause_text(merged_text)

        event_type, event_reasons = self._infer_event_type(normalized_text, clause_text)
        target_goal, goal_reasons = self._infer_target_goal(normalized_text, raw_event["target_goal"])
        target_audience, audience_reasons = self._infer_target_audience(normalized_text, raw_event["target_audience"])
        extracted_keywords = self._extract_keywords(raw_event, normalized_text, raw_event["target_audience"])
        constraints = self._normalize_constraints(raw_event.get("constraints", {}), normalized_text, clause_text)
        # 第十一轮：隐私语境（要保护 vs 已泄露）与"被预防表达覆盖的风险"。
        privacy_context = self._infer_privacy_context(normalized_text, clause_text)
        preventive_risk = self._has_preventive_risk(normalized_text, clause_text)

        reasoning = _dedupe_preserve_order(
            event_reasons
            + goal_reasons
            + audience_reasons
            + [
                f"risk_level={constraints.risk_level}",
                f"privacy_context={privacy_context}",
                f"preventive_risk={str(preventive_risk).lower()}",
            ]
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
                privacy_context=privacy_context,
                preventive_risk=preventive_risk,
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

    def _infer_event_type(
        self,
        normalized_text: str,
        clause_text: str,
    ) -> tuple[str, list[str]]:
        best_type = "general_influence_event"
        best_score = 0.0
        reasons: list[str] = []

        for event_type, keywords in EVENT_TYPE_RULES.items():
            # 第十一轮：舆情响应是唯一一类"靠风险词驱动"的事件类型，
            # 因此它必须先过一遍预防语境判断——「开展校园交流活动，
            # 避免引发争议」里的"争议"是预防措施，不是已发生的舆情。
            # 其余类型（活动/英语/亲子…）描述的是事件本体，不做这层过滤。
            if event_type == "public_opinion_response":
                matched = [
                    keyword
                    for keyword in self._matched_keywords(normalized_text, keywords)
                    if not self._keyword_prevented(clause_text, keyword)
                ]
            else:
                matched = self._matched_keywords(normalized_text, keywords)
            if not matched:
                continue
            score = len(matched) / len(keywords)
            if score > best_score:
                best_type = event_type
                best_score = score
                reasons = [f"event_type={event_type}", f"matched_keywords={','.join(matched)}"]

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
            if self._keyword_score(normalized_text, keywords) <= 0.0:
                continue
            # 第十一轮：校园受众要有上下文依据。出现"校园/学校"并不自动等于
            # 受众是学生——如果同时只出现教职工专属词（老师/教师/教研…）
            # 而没有任何学生专属词（学生/班级/社团/同学），说明说的是
            # 学校这个单位，此时不推定 campus_student。
            if audience_tag == "campus_student" and not self._has_campus_student_context(
                normalized_text
            ):
                continue
            inferred.append(audience_tag)
        if not inferred:
            inferred.append("general_public")
            reasons.append("target_audience=inferred:general_public")
        else:
            reasons.append(f"target_audience=inferred:{','.join(_dedupe_preserve_order(inferred))}")
        return _dedupe_preserve_order(inferred), reasons

    def _has_campus_student_context(self, normalized_text: str) -> bool:
        """校园受众是否有明确的"学生群体"上下文依据。"""
        has_student_cue = any(cue in normalized_text for cue in _STUDENT_SPECIFIC_CUES)
        if has_student_cue:
            return True
        # 只有教职工专属词、没有学生词 → 说的是学校/教职工，不是学生群体。
        return not any(cue in normalized_text for cue in _PRACTITIONER_ONLY_CUES)

    def _normalize_constraints(
        self,
        raw_constraints: dict[str, Any],
        normalized_text: str,
        clause_text: str,
    ) -> EventConstraints:
        constraints = dict(raw_constraints)
        if "risk_level" not in constraints:
            constraints["risk_level"] = self._infer_risk_level(normalized_text, clause_text)
        if "allowed_platforms" not in constraints:
            constraints["allowed_platforms"] = list(DATASET_PLATFORM_VALUES)
        return EventConstraints.model_validate(constraints)

    def _infer_risk_level(self, normalized_text: str, clause_text: str) -> str:
        if self._has_effective_risk_keyword(normalized_text, clause_text, HIGH_RISK_KEYWORDS):
            return "high"
        if self._has_effective_risk_keyword(normalized_text, clause_text, MEDIUM_RISK_KEYWORDS):
            return "medium"
        return "low"

    def _infer_privacy_context(self, normalized_text: str, clause_text: str) -> str:
        """隐私语境：``none`` / ``preventive``（要保护）/ ``occurred``（已泄露）。

        区分依据是**泄露类词是否真实出现**：「不得泄露参与者个人信息」里的
        "泄露"处于禁止语境 → preventive；「学校已经发生学生个人信息泄露事件」
        里的"泄露"是客观事实 → occurred。只有文本真的提到隐私才返回非 none。
        """
        if not self._matched_keywords(normalized_text, TOPIC_RULES["privacy"]):
            return "none"
        for term in PRIVACY_LEAK_TERMS:
            if self._fuzzy_contains(normalized_text, term) and not self._keyword_prevented(
                clause_text, term
            ):
                return "occurred"
        return "preventive"

    def _has_preventive_risk(self, normalized_text: str, clause_text: str) -> bool:
        """是否存在"被预防表达覆盖"的风险词。

        这类事件的风险等级按既有规则不抬高，但**仍然需要风险约束**——
        「避免引发争议」不代表可以随意发言。
        """
        return any(
            self._fuzzy_contains(normalized_text, keyword)
            and self._keyword_prevented(clause_text, keyword)
            for keyword in HIGH_RISK_KEYWORDS + MEDIUM_RISK_KEYWORDS
        )

    def _has_effective_risk_keyword(
        self,
        normalized_text: str,
        clause_text: str,
        keywords: tuple[str, ...],
    ) -> bool:
        """判断是否存在"非预防性"的风险关键词命中。

        「避免引发争议」「防止刷屏」「杜绝投诉」「不得泄露」表达的是
        **希望这些事不发生**，按原逻辑会平白把风险等级抬上去。
        """
        return any(
            self._fuzzy_contains(normalized_text, keyword)
            and not self._keyword_prevented(clause_text, keyword)
            for keyword in keywords
        )

    def _keyword_prevented(self, clause_text: str, keyword: str) -> bool:
        """关键词是否**只**在预防/否定语境中出现。

        与旧实现（固定 8 字窗口）的两点区别：

        1. **按短句分段**而不是按字符窗口：逗号/句号/转折词之后就是新的一句，
           上一句的"避免"不该压掉下一句的真实风险
           （「已经引发争议，但希望避免发酵」里的争议仍算真实风险）。
        2. **已发生线索优先**：同一个短句里如果同时出现"已经/已有/出现/发生"
           这类词，说明风险客观存在（「为避免已出现的争议进一步发酵」），
           此时不再按预防处理——也就是"不能因为出现避免就整句判成无风险"。

        只要存在**任一**命中不在预防语境里，就认为该风险词真实存在。
        """
        start = 0
        hit_any = False
        while True:
            idx = clause_text.find(keyword, start)
            if idx < 0:
                break
            hit_any = True
            preceding = _clause_prefix(clause_text, idx)
            has_occurred_cue = any(cue in preceding for cue in _OCCURRED_CUES)
            has_prevention_cue = any(cue in preceding for cue in _PREVENTION_CUES)
            if has_occurred_cue or not has_prevention_cue:
                return False
            start = idx + len(keyword)
        return hit_any

    def _extract_keywords(
        self,
        raw_event: dict[str, Any],
        normalized_text: str,
        explicit_audience: list[str],
    ) -> list[str]:
        keywords: list[str] = []
        # 第十一轮：主题词（校园 / 环保 / 隐私）与前三套词表走完全相同的
        # 命中流程，因此会自然进入 extracted_keywords → semantic_tags。
        for rule_set in (EVENT_TYPE_RULES, GOAL_RULES, AUDIENCE_RULES, TOPIC_RULES):
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
        return self._valid_cjk_generic_term(token)

    def _valid_cjk_generic_term(self, token: str) -> bool:
        """只约束通用正则抓到的**纯中文**片段；ASCII 词（如 KOL/SUV）直接放行。

        按三层顺序判断，避免退化成"超过 N 字就删"：

        1. **词表覆盖**：片段里若已含三套固定词表中的关键词，说明该关键词
           本身已经被词表命中收录了（``_extract_keywords`` 先走词表），
           再保留整段只是引入从句噪声，因此丢弃整段。
        2. **从句标记**：含连词 / 指代 / 叙述动词 / 助词 → 这是句子不是词。
        3. **长度兜底**：无分词器时，超过 4 个汉字的连续串几乎必然是短语
           或从句。
        """
        if not _CJK_ONLY_RE.match(token):
            return True
        if any(keyword in token for keyword in _RULE_KEYWORD_SET):
            return False
        if any(marker in token for marker in _CLAUSE_MARKERS):
            return False
        return len(token) <= _MAX_GENERIC_CJK_TERM

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

    def _default_negative_constraints(
        self,
        *,
        risk_level: str,
        event_type: str,
        privacy_context: str = "none",
        preventive_risk: bool = False,
    ) -> list[str]:
        constraints = ["避免节点内容完全同质化"]
        if risk_level in {"medium", "high"}:
            constraints.append("避免情绪化和对立化表达")
        if risk_level == "high" or event_type == "public_opinion_response":
            constraints.append("避免未经证实的推测性扩写")
        # 第十一轮：预防语境不抬风险等级，但要留下明确的风险约束，
        # 不能因为写了"避免"就当成没有风险。
        if preventive_risk:
            constraints.append("避免引发争议或误解，表述保持中立克制")
        # 第十一轮：隐私区分"要保护"与"已泄露"，只写与语境相符的约束。
        if privacy_context == "preventive":
            constraints.append("避免收集、展示或传播参与者的个人信息")
        elif privacy_context == "occurred":
            constraints.append("不得二次传播当事人个人信息与未经核实的细节")
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
            extracted_keywords = self._normalize_str_list(response.get("extracted_keywords")) or parsed_event.extracted_keywords
            risk_level = str(response.get("risk_level") or parsed_event.constraints.risk_level).strip().lower()
            if risk_level not in {"low", "medium", "high"}:
                risk_level = parsed_event.constraints.risk_level
            semantic_tags = self._normalize_str_list(response.get("semantic_tags")) or parsed_event.semantic_tags
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

    def _normalize_roles(self, value: Any) -> list[str]:
        allowed = {"core_publish_node", "interaction_response_node", "amplification_node", "support_node"}
        return [item for item in self._normalize_str_list(value) if item in allowed]
