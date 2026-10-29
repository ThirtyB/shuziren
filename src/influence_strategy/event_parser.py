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

#: 关键词词表（时事政治领域）。
#:
#: **注意**：以下三套词表现在**只用于关键词提取**（``_extract_keywords``）与节点
#: 兴趣标签构建（``scripts/build_political_domain_nodes.py``），**不再参与分类判定**。
#: 事件类型 / 目标 / 受众的分类已改为纯语义相似度（见 ``EVENT_TYPE_ANCHORS`` 等），
#: 由 ``SemanticClassifier`` 在解析时用本地编码器比对锚文本得到。
EVENT_TYPE_RULES: dict[str, tuple[str, ...]] = {
    # military / defense_policy / defense_industry / security* / border_security /
    # peacekeeping / peace_process / regional_security
    "military_conflict": (
        "军事", "军演", "导弹", "核武器", "核试验", "战争", "军事冲突", "停火",
        "撤军", "维和", "边境", "领空", "领海", "战机", "航母", "无人机",
        "空袭", "防务", "军费", "武装",
    ),
    # international_relations / international_law / regional_relations /
    # global_governance / humanitarian_crisis / human_rights / rule_of_law
    "international_relations": (
        "外交", "双边", "多边", "峰会", "会谈", "出访", "访华", "建交",
        "国际组织", "联合国", "安理会", "盟友", "伙伴", "协议", "签署",
        "声明", "谴责", "召见", "大使", "地缘政治", "主权", "领土", "断交",
    ),
    # economic_growth / economic_crisis / economic_policy / inflation /
    # monetary_policy / fiscal_policy / tax_policy / debt / household_debt / inflation
    "macro_economy": (
        "经济增长", "衰退", "滞胀", "通胀", "通缩", "物价", "利率", "降息",
        "加息", "央行", "货币政策", "财政", "税收", "国债", "债务", "赤字",
        "内需", "消费", "出口", "进口",
    ),
    # finance / financial_market / financial_risk / financial_regulation /
    # banking_regulation / insurance / payments / real_estate / housing / international_finance
    "financial_market": (
        "股市", "债市", "银行", "信贷", "保险", "投资", "基金", "资本",
        "上市", "破产", "房地产", "房价", "楼市", "理财", "证券", "股民",
    ),
    # artificial_intelligence / ai_governance / ai_policy / ai_risk / technology /
    # space / space_policy / intellectual_property
    "tech_ai": (
        "人工智能", "大模型", "算法", "算力", "芯片", "半导体", "光刻机",
        "开源", "自动驾驶", "机器人", "量子", "航天", "卫星", "火箭",
        "空间站", "知识产权",
    ),
    # cybersecurity / cybersecurity_regulation / data_breach / data_governance /
    # digital_governance / platform_governance / platform_regulation / antitrust / online_safety
    "cyber_data_governance": (
        "网络安全", "黑客", "泄露", "数据泄露", "漏洞", "勒索", "攻击",
        "木马", "隐私", "个人信息", "数据", "合规", "监管", "平台",
        "反垄断", "内容审核",
    ),
    # energy / energy_* / oil_market / critical_minerals / battery_regulation /
    # hydrogen / utility_regulation / renewable_energy
    "energy_resources": (
        "能源", "原油", "油价", "天然气", "煤炭", "石油", "核电", "光伏",
        "风电", "新能源", "储能", "电池", "稀土", "矿产", "电力", "电价",
        "限电",
    ),
    # climate / climate_* / environment* / biodiversity / volcano / earthquake /
    # wildfire / water_security / disaster_response
    "climate_disaster": (
        "气候", "变暖", "碳排放", "碳中和", "碳市场", "减排", "极端天气",
        "高温", "暴雨", "洪涝", "台风", "干旱", "地震", "火山", "山火",
        "海啸", "灾害", "应急",
    ),
    # public_health / healthcare / pharmaceuticals / healthcare_cybersecurity /
    # environment_health
    "public_health": (
        "疫情", "传染病", "流感", "病毒", "疫苗", "接种", "核酸", "隔离",
        "防控", "公共卫生", "医疗", "医保", "药品", "药价", "集采", "医院",
    ),
    # politics / election / election_dispute / election_integrity /
    # political_crisis / european_politics
    "politics_election": (
        "选举", "大选", "投票", "选民", "民调", "候选人", "议会", "国会",
        "参议院", "众议院", "政府", "内阁", "换届", "就职", "弹劾",
        "罢免", "修宪", "公投", "政党", "执政", "法案", "立法", "预算",
    ),
    # social_unrest / security_crisis / political_crisis
    "social_unrest": (
        "抗议", "示威", "游行", "罢工", "罢课", "骚乱", "暴乱", "镇压",
        "戒严", "动荡", "动乱", "政变", "暴动", "集会",
    ),
    # supply_chain / logistics / shipping / aviation_safety / industrial_policy /
    # infrastructure / infrastructure_finance / transport_policy
    "industry_supply_chain": (
        "供应链", "物流", "航运", "港口", "货运", "集装箱", "航线", "空运",
        "航空", "航班", "铁路", "高铁", "基建", "基础设施", "制造业",
        "工厂", "产能", "订单", "产业链", "汽车",
    ),
    # agriculture_policy / food_market
    "agriculture_food": (
        "农业", "粮食", "农产品", "丰收", "减产", "耕地", "种业", "养殖",
        "渔业", "粮价", "粮食安全", "农村", "农民", "食品",
    ),
    # labor_policy / migration_policy / demographics
    "labor_demographics": (
        "用工", "招工", "裁员", "失业", "劳工", "工会", "社保", "养老金",
        "人口", "生育", "老龄化", "户籍", "移民", "难民",
    ),
    # public_policy / consumer_policy / corporate_governance / sports_policy
    "social_livelihood": (
        "教育", "学校", "学生", "教师", "考试", "高考", "学费", "住房",
        "租房", "民生", "治安", "犯罪", "诈骗", "事故", "社区", "城市治理",
    ),
}

GOAL_RULES: dict[str, tuple[str, ...]] = {
    "engagement": ("讨论", "互动", "评论", "参与", "交流"),
    "awareness": ("传播", "曝光", "扩散", "覆盖", "触达"),
    "response": ("回应", "澄清", "解释", "辟谣", "稳定"),
    "conversion": ("报名", "转化", "参与报名", "下载", "注册"),
}

#: 目标人群判定词典（时事政治领域）。
#: 注意 ``general_public`` 是 ``_event_match_keywords`` 里的 generic tag 会被过滤，
#: 因此这里一律用带领域后缀的键名，避免受众标签在下游被整类丢掉。
AUDIENCE_RULES: dict[str, tuple[str, ...]] = {
    "policy_observer": ("政策", "政务", "时政", "公共事务", "改革", "体制改革"),
    "investor_finance": ("投资", "财经", "股市", "基金", "理财", "证券", "股民"),
    "industry_professional": ("行业", "产业", "企业", "职场", "从业", "工商界"),
    "international_audience": ("国际", "海外", "全球", "跨境", "外贸", "涉外"),
    "tech_audience": ("科技", "数码", "互联网", "开发者", "程序员", "极客"),
    "security_strategy": ("安全", "战略", "防务", "军事", "情报", "国际安全"),
    "academic_expert": ("专家", "学者", "研究", "智库", "学术", "分析师"),
    "livelihood_public": ("民生", "市民", "居民", "公众", "社区", "消费者"),
}

#: 分类锚文本（纯语义相似度）。
#:
#: 取代原先的关键词命中计数。每个类别用一段自然语言描述其语义中心，解析时由
#: ``SemanticClassifier`` 把事件文本与各锚文本编码到同一语义空间，取余弦相似度
#: 最高的类别。锚文本刻意复用关键词词表里的核心术语，以让本地编码器（BGE 或
#: TF-IDF+SVD）能在现有向量空间里给出有意义的近邻。
EVENT_TYPE_ANCHORS: dict[str, str] = {
    "military_conflict": "军事冲突、战争、导弹、核武器、核试验、军演、撤军、维和、边境与领空领海安全、战机航母无人机空袭、防务军费与武装力量。",
    "international_relations": "外交关系、双边与多边峰会会谈、出访建交、国际组织与联合国安理会、盟友伙伴协议声明、谴责召见大使、地缘政治主权领土与断交。",
    "macro_economy": "经济增长、衰退滞胀、通胀通缩物价、利率降息加息、央行货币政策、财政税收国债债务赤字、内需消费进出口。",
    "financial_market": "股市债市、银行信贷保险、投资基金融资、上市破产、房地产房价楼市、理财证券股民。",
    "tech_ai": "人工智能大模型、算法算力、芯片半导体光刻机、开源自动驾驶机器人、量子航天卫星火箭空间站、知识产权。",
    "cyber_data_governance": "网络安全黑客、数据泄露漏洞勒索攻击、木马隐私个人信息、数据合规监管平台、反垄断与内容审核。",
    "energy_resources": "能源原油油价、天然气煤炭石油、核电光伏风电新能源、储能电池稀土矿产、电力电价与限电。",
    "climate_disaster": "气候变暖碳排放碳中和、碳市场减排、极端天气高温暴雨洪涝台风干旱、地震火山山火海啸灾害应急。",
    "public_health": "疫情传染病流感、病毒疫苗核酸隔离、防控公共卫生、医疗医保药品药价集采医院。",
    "politics_election": "选举大选投票选民、民调候选人与议会国会、政府内阁换届就职、弹劾罢免修宪公投、政党执政法案立法预算。",
    "social_unrest": "抗议示威游行、罢工罢课骚乱暴乱、镇压戒严动荡、动乱政变暴动集会。",
    "industry_supply_chain": "供应链物流航运、港口货运集装箱航线、空运航空航班、铁路高铁基建基础设施、制造业工厂产能订单产业链汽车。",
    "agriculture_food": "农业粮食农产品、丰收减产耕地种业、养殖渔业粮价粮食安全、农村农民食品。",
    "labor_demographics": "用工招工裁员失业、劳工工会社保养老金、人口生育老龄化、户籍移民难民。",
    "social_livelihood": "教育学校学生教师考试高考、学费住房租房民生、治安犯罪诈骗事故、社区城市治理。",
    # 兜底类别：事件文本在语义空间里最贴近它时，回落到通用影响力事件。
    "general_influence_event": "一般性的社会热点与公共话题、产品与品牌传播、日常讨论与通用影响力事件。",
}

GOAL_ANCHORS: dict[str, str] = {
    "engagement": "提升讨论度、互动率与用户参与，鼓励评论、交流、问答与社群活跃。",
    "awareness": "扩大事件触达范围、建立基础认知、提升传播曝光与扩散覆盖。",
    "response": "快速回应公众疑问、澄清误读、解释事实、稳定舆论节奏。",
    "conversion": "引导目标人群完成后续行动，如报名、下载、注册、转化。",
}

AUDIENCE_ANCHORS: dict[str, str] = {
    "policy_observer": "关注政策、政务、时政与公共事务、体制改革的观察者。",
    "investor_finance": "关注投资、财经、股市、基金、理财、证券的投资者与股民。",
    "industry_professional": "产业、行业、企业、职场与工商界从业者。",
    "international_audience": "关注国际、海外、全球、跨境与外贸的涉外受众。",
    "tech_audience": "科技、数码、互联网、开发者、程序员与极客群体。",
    "security_strategy": "关注安全、战略、防务、军事、情报与国际安全的群体。",
    "academic_expert": "专家、学者、研究人员、智库与分析师等学术群体。",
    "livelihood_public": "关注民生、市民、居民、公众、社区与消费者的普通大众。",
}

#: 风险等级触发词。命中即抬高风险等级，进而收紧准入门槛、加大扣分系数、
#: 压低发布频次。时政场景下的战争/制裁/骚乱等应判高风险，这是预期行为。
HIGH_RISK_KEYWORDS = (
    "危机", "投诉", "举报", "造假", "翻车", "维权", "道歉",
    "战争", "袭击", "空袭", "制裁", "骚乱", "暴乱", "政变", "镇压",
    "弹劾", "泄漏", "崩盘", "遇难", "伤亡", "封锁", "断交", "暴动",
)
MEDIUM_RISK_KEYWORDS = (
    "争议", "质疑", "刷屏", "负面", "误解", "冲突", "罢工", "抗议",
    "分歧", "摩擦", "波动", "裁员", "紧张", "对峙", "谴责", "警告",
    "下滑", "亏损", "调查", "审查",
)


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
        # 语义分类器懒加载缓存：第一次解析时按 workspace_root 取本地编码器构建，
        # 之后复用，避免每个事件都重新编码锚文本。
        self._classifiers: dict[str, Any] = {}
        self._classifiers_ready = False

    def parse(
        self,
        payload: str | dict[str, Any],
        *,
        workspace_root: str | Path | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
    ) -> ParsedEvent:
        raw_event = self._normalize_payload(payload)
        parsed_event = self._parse_rule_based(raw_event, workspace_root=workspace_root)
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

    def _parse_rule_based(
        self,
        raw_event: dict[str, Any],
        workspace_root: str | Path | None = None,
    ) -> ParsedEvent:
        merged_text = " ".join(
            [
                raw_event["event_title"],
                raw_event["event_description"],
                raw_event["target_goal"],
                " ".join(raw_event["target_audience"]),
            ]
        )
        normalized_text = _normalize_text(merged_text)

        classifiers = self._ensure_classifiers(workspace_root)
        event_type, event_reasons = self._infer_event_type(merged_text, classifiers)
        target_goal, goal_reasons = self._infer_target_goal(merged_text, classifiers, raw_event["target_goal"])
        target_audience, audience_reasons = self._infer_target_audience(merged_text, classifiers, raw_event["target_audience"])
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

    def _ensure_classifiers(
        self,
        workspace_root: str | Path | None,
    ) -> dict[str, Any]:
        """按 workspace_root 取本地编码器，构建（并缓存）三个语义分类器。

        本地模型产物缺失时返回全部为 ``None`` 的分类器字典，调用方据此退化为
        默认类别，保证没有预计算产物也能解析（只是分类退化为兜底）。
        """

        if self._classifiers_ready:
            return self._classifiers

        encoder = None
        try:
            from .local_models import load_local_models

            if workspace_root is not None:
                models = load_local_models(workspace_root)
                encoder = models.encoder if models else None
        except Exception:  # noqa: BLE001 - 取不到编码器不应中断解析
            encoder = None

        from .local_semantic import SemanticClassifier

        self._classifiers = {
            "event_type": SemanticClassifier.build(encoder, EVENT_TYPE_ANCHORS),
            "target_goal": SemanticClassifier.build(encoder, GOAL_ANCHORS),
            "target_audience": SemanticClassifier.build(encoder, AUDIENCE_ANCHORS),
        }
        self._classifiers_ready = True
        return self._classifiers

    def _infer_event_type(self, text: str, classifiers: dict[str, Any]) -> tuple[str, list[str]]:
        clf = classifiers.get("event_type")
        if clf is None:
            return "general_influence_event", [
                "event_type=general_influence_event",
                "semantic_encoder_unavailable",
            ]
        label, score = clf.classify(text, default_label="general_influence_event", margin=0.03)
        return label, [f"event_type={label}", f"semantic_sim={score:.3f}"]

    def _infer_target_goal(self, text: str, classifiers: dict[str, Any], explicit_goal: str) -> tuple[str, list[str]]:
        if explicit_goal:
            return explicit_goal, [f"target_goal=explicit:{explicit_goal}"]

        clf = classifiers.get("target_goal")
        if clf is None:
            return "awareness", ["target_goal=inferred:awareness", "semantic_encoder_unavailable"]
        label, score = clf.classify(text)
        return label, [f"target_goal=inferred:{label}", f"semantic_sim={score:.3f}"]

    def _infer_target_audience(
        self,
        text: str,
        classifiers: dict[str, Any],
        explicit_audience: list[str],
    ) -> tuple[list[str], list[str]]:
        inferred = list(explicit_audience)
        reasons = [f"target_audience=explicit:{','.join(explicit_audience)}"] if explicit_audience else []
        clf = classifiers.get("target_audience")
        if clf is not None:
            # 取语义最贴近的前两类，避免只给单一标签导致下游话题匹配信号过窄。
            for label, score in clf.rank(text, 2):
                if label not in inferred:
                    inferred.append(label)
                    reasons.append(f"target_audience=inferred:{label}:{score:.3f}")
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
