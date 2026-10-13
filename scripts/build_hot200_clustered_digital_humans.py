from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EVENTS_PATH = "eval/hot_event_opinion_variants_200.json"
DEFAULT_PROFILE_PATH = "data/derived/weibo_profile_reworked_with_neighbors.graph.anon"
DEFAULT_MANIFEST_PATH = "data/derived/weibo_profile_reworked_hot200_clustered_manifest.jsonl"
DEFAULT_SUMMARY_PATH = "data/derived/weibo_profile_reworked_hot200_clustered_summary.json"
DEFAULT_CLUSTER_SUMMARY_PATH = "data/derived/hot200_topic_cluster_summary.json"

GRAPH_ATTRIBUTE_FIELDS = (
    "neighbor_count",
    "engaged_by_neighbor_count",
    "engaged_to_neighbor_count",
    "mutual_neighbor_count",
    "self_interaction_count",
    "self_comment_count",
    "self_repost_count",
    "received_interaction_count",
    "received_comment_count",
    "received_repost_count",
    "made_interaction_count",
    "made_comment_count",
    "made_repost_count",
    "isolated",
)

NEIGHBOR_STAT_FIELDS = (
    "received_comment_count",
    "received_repost_count",
    "made_comment_count",
    "made_repost_count",
)

GENERIC_TERMS = {
    "event",
    "hot",
    "topic",
    "general",
    "热点",
    "事件",
    "传播",
    "讨论",
    "影响",
    "关注",
    "推动",
    "风险",
    "政策",
    "治理",
    "市场",
    "持续",
    "成为",
    "相关",
    "问题",
    "公众",
}

CLUSTER_SPECS: list[dict[str, Any]] = [
    {
        "cluster": "public_health_medical",
        "label": "公共卫生与医疗系统",
        "domain_tokens": ["public_health", "healthcare", "pharmaceuticals", "climate_health", "environment_health"],
        "keywords": ["公共卫生", "医疗", "疫苗", "健康", "药品", "医保", "医疗系统"],
        "terms": ["公共卫生", "医疗体系", "疫苗", "健康风险", "医保支付", "药品供应", "医院运营"],
    },
    {
        "cluster": "climate_disaster_emergency",
        "label": "气候灾害与城市应急",
        "domain_tokens": ["climate_disaster", "disaster_response", "wildfire", "volcano", "earthquake", "climate_health"],
        "keywords": ["极端天气", "洪灾", "山火", "地震", "火山", "应急", "救援"],
        "terms": ["气候灾害", "极端天气", "城市应急", "灾害救援", "基础设施", "公共安全"],
    },
    {
        "cluster": "climate_environment_policy",
        "label": "气候政策与环境治理",
        "domain_tokens": ["climate", "climate_policy", "climate_regulation", "environment", "environment_policy", "biodiversity", "climate_security"],
        "keywords": ["气候", "减排", "塑料污染", "生物多样性", "碳", "生态", "环境"],
        "terms": ["气候政策", "环境治理", "减排转型", "生态保护", "塑料污染", "生物多样性"],
    },
    {
        "cluster": "cybersecurity_data_governance",
        "label": "网络安全与数据治理",
        "domain_tokens": ["cybersecurity", "data_governance", "data_breach", "healthcare_cybersecurity", "cybersecurity_regulation"],
        "keywords": ["网络安全", "数据安全", "数据泄露", "隐私", "合规", "勒索软件"],
        "terms": ["网络安全", "数据安全", "数据泄露", "隐私保护", "数字韧性", "合规风险"],
    },
    {
        "cluster": "ai_platform_governance",
        "label": "AI治理与平台监管",
        "domain_tokens": ["ai_governance", "ai_policy", "ai_risk", "platform_governance", "platform_regulation", "online_safety", "technology_regulation", "artificial_intelligence"],
        "keywords": ["人工智能", "AI", "算法", "平台", "TikTok", "深度伪造", "模型"],
        "terms": ["人工智能", "AI治理", "平台治理", "算法责任", "生成式AI", "内容审核", "模型安全"],
    },
    {
        "cluster": "technology_industry_chip",
        "label": "科技产业与芯片供应",
        "domain_tokens": ["technology", "technology_regulation", "critical_minerals", "intellectual_property"],
        "keywords": ["芯片", "半导体", "算力", "科技", "知识产权", "关键矿产"],
        "terms": ["科技产业", "芯片供应", "半导体", "算力", "关键矿产", "知识产权", "产业链"],
    },
    {
        "cluster": "finance_monetary_market",
        "label": "金融市场与货币政策",
        "domain_tokens": ["finance", "financial_market", "monetary_policy", "international_finance", "inflation", "financial_risk"],
        "keywords": ["利率", "美联储", "黄金", "美元", "通胀", "金融市场", "资产"],
        "terms": ["金融市场", "货币政策", "利率路径", "通胀预期", "资产价格", "黄金", "美元"],
    },
    {
        "cluster": "financial_regulation_crypto_banking",
        "label": "金融监管与加密资产",
        "domain_tokens": ["financial_regulation", "crypto_regulation", "banking_regulation", "consumer_finance", "payments", "insurance"],
        "keywords": ["监管", "银行", "加密资产", "支付", "保险", "消费金融", "资本规则"],
        "terms": ["金融监管", "银行资本", "加密资产", "支付体系", "消费金融", "保险市场", "投资者保护"],
    },
    {
        "cluster": "fiscal_tax_debt",
        "label": "财政税收与债务风险",
        "domain_tokens": ["fiscal_policy", "tax_policy", "debt", "economic_crisis", "infrastructure_finance", "household_debt"],
        "keywords": ["财政", "税", "债务", "赤字", "融资", "IMF", "地方债"],
        "terms": ["财政政策", "税收政策", "债务风险", "赤字约束", "地方债", "融资压力", "公共支出"],
    },
    {
        "cluster": "real_estate_housing",
        "label": "房地产与住房政策",
        "domain_tokens": ["real_estate", "housing", "infrastructure"],
        "keywords": ["房地产", "住房", "房贷", "首付", "租赁", "楼市"],
        "terms": ["房地产", "住房政策", "房贷利率", "市场预期", "居民负担", "库存去化"],
    },
    {
        "cluster": "energy_security_oil",
        "label": "能源安全与油气市场",
        "domain_tokens": ["energy", "energy_security", "oil_market", "energy_market", "energy_regulation", "energy_tax", "energy_demand"],
        "keywords": ["能源", "油价", "原油", "天然气", "电力", "能源安全"],
        "terms": ["能源安全", "油价", "天然气", "能源通胀", "供需预期", "电力保障", "能源市场"],
    },
    {
        "cluster": "clean_energy_transition",
        "label": "清洁能源与绿色转型",
        "domain_tokens": ["clean_energy", "renewable_energy", "energy_transition", "hydrogen", "battery_regulation", "utility_regulation"],
        "keywords": ["清洁能源", "新能源", "氢能", "电池", "电网", "绿色转型", "可再生"],
        "terms": ["清洁能源", "绿色转型", "新能源", "氢能", "电池法规", "电网建设", "可再生能源"],
    },
    {
        "cluster": "international_trade_supply_chain",
        "label": "国际贸易与供应链",
        "domain_tokens": ["trade", "international_trade", "supply_chain", "corporate_governance", "corporate_reporting", "industrial_policy"],
        "keywords": ["贸易", "关税", "补贴", "供应链", "企业", "产业政策"],
        "terms": ["国际贸易", "关税政策", "产业补贴", "供应链", "跨国企业", "产业政策", "企业治理"],
    },
    {
        "cluster": "shipping_logistics_transport",
        "label": "航运物流与交通运输",
        "domain_tokens": ["shipping", "logistics", "transport_policy", "aviation_safety"],
        "keywords": ["航运", "港口", "物流", "海事", "运输", "航空", "波音"],
        "terms": ["国际航运", "港口物流", "海事规则", "交通运输", "航空安全", "供应链韧性"],
    },
    {
        "cluster": "military_security_defense",
        "label": "军事安全与防务政策",
        "domain_tokens": ["military", "security_policy", "security", "security_crisis", "regional_security", "defense_policy", "defense_industry", "peacekeeping", "border_security"],
        "keywords": ["军事", "安全", "防务", "北约", "边境", "停火", "维和"],
        "terms": ["军事安全", "防务政策", "地缘政治", "区域安全", "停火执行", "边境安全", "安全支援"],
    },
    {
        "cluster": "international_relations_law",
        "label": "国际关系与国际法",
        "domain_tokens": ["international_relations", "regional_relations", "international_law", "global_governance", "rule_of_law", "human_rights"],
        "keywords": ["国际关系", "国际法", "人权", "全球治理", "外交", "地区关系"],
        "terms": ["国际关系", "国际法", "全球治理", "地区关系", "外交协调", "人权议题", "规则秩序"],
    },
    {
        "cluster": "migration_humanitarian_crisis",
        "label": "移民政策与人道危机",
        "domain_tokens": ["migration_policy", "humanitarian_crisis", "peace_process", "water_security"],
        "keywords": ["移民", "难民", "庇护", "人道", "加沙", "救援", "停火"],
        "terms": ["移民政策", "庇护规则", "人道危机", "援助准入", "难民保护", "停火安排", "水安全"],
    },
    {
        "cluster": "politics_election_institution",
        "label": "政治选举与制度信任",
        "domain_tokens": ["politics", "political_crisis", "election", "election_dispute", "election_integrity", "european_politics"],
        "keywords": ["选举", "总统", "弹劾", "政治", "制度", "信任", "权力交接"],
        "terms": ["政治选举", "制度信任", "权力交接", "政治危机", "选举安全", "政策预期"],
    },
    {
        "cluster": "public_policy_livelihood",
        "label": "公共政策与民生服务",
        "domain_tokens": ["public_policy", "economic_policy", "labor_policy", "consumer_policy", "digital_governance"],
        "keywords": ["公共政策", "民生", "退休", "就业", "消费者", "民营经济", "规则"],
        "terms": ["公共政策", "民生服务", "就业市场", "消费者权益", "民营经济", "数字治理", "政策解读"],
    },
    {
        "cluster": "social_unrest_governance",
        "label": "社会抗议与治理重建",
        "domain_tokens": ["social_unrest", "political_crisis", "security_crisis"],
        "keywords": ["抗议", "临时政府", "治理重建", "社会秩序", "问责", "冲突"],
        "terms": ["社会抗议", "治理重建", "公共秩序", "问责机制", "制度恢复", "社会稳定"],
    },
    {
        "cluster": "agriculture_food_market",
        "label": "农业政策与食品市场",
        "domain_tokens": ["agriculture_policy", "food_market"],
        "keywords": ["农业", "农民", "食品", "可可", "粮食", "农产品"],
        "terms": ["农业政策", "食品市场", "农产品价格", "农民收入", "供应安全", "消费成本"],
    },
    {
        "cluster": "consumer_tourism_sports",
        "label": "消费旅游与体育治理",
        "domain_tokens": ["tourism", "sports_policy", "consumer_policy"],
        "keywords": ["旅游", "体育", "消费", "赛事", "维修权"],
        "terms": ["旅游市场", "消费政策", "体育治理", "赛事服务", "维修权", "消费体验"],
    },
    {
        "cluster": "demographics_social_services",
        "label": "人口结构与社会服务",
        "domain_tokens": ["demographics", "labor_policy", "public_policy"],
        "keywords": ["人口", "养老", "退休", "就业", "社保"],
        "terms": ["人口结构", "养老压力", "延迟退休", "就业市场", "社保体系", "公共服务"],
    },
    {
        "cluster": "space_satellite_policy",
        "label": "航天科技与卫星政策",
        "domain_tokens": ["space", "space_policy"],
        "keywords": ["航天", "卫星", "月球", "嫦娥", "NASA", "轨道", "频谱"],
        "terms": ["航天科技", "卫星互联网", "深空探测", "轨道资源", "频谱协调", "科研进展"],
    },
    {
        "cluster": "water_infrastructure_resilience",
        "label": "水安全与基础设施韧性",
        "domain_tokens": ["water_security", "infrastructure", "disaster_response"],
        "keywords": ["水安全", "基础设施", "电力", "城市", "公共服务"],
        "terms": ["水安全", "基础设施", "城市韧性", "公共服务", "电力保障", "灾后恢复"],
    },
    {
        "cluster": "corporate_compliance_reporting",
        "label": "企业合规与可持续披露",
        "domain_tokens": ["corporate_governance", "corporate_reporting", "intellectual_property"],
        "keywords": ["企业", "披露", "尽职调查", "合规", "知识产权", "可持续"],
        "terms": ["企业合规", "可持续披露", "尽职调查", "供应链责任", "知识产权", "治理透明度"],
    },
    {
        "cluster": "regional_economic_growth",
        "label": "区域经济与增长压力",
        "domain_tokens": ["economic_growth", "economic_crisis", "regional_relations"],
        "keywords": ["经济增长", "区域", "IMF", "产业", "发展"],
        "terms": ["区域经济", "增长压力", "产业发展", "财政空间", "市场信心", "国际机构"],
    },
    {
        "cluster": "antitrust_market_order",
        "label": "反垄断与市场秩序",
        "domain_tokens": ["antitrust", "platform_governance", "consumer_policy"],
        "keywords": ["反垄断", "竞争", "市场秩序", "大型科技平台", "应用商店"],
        "terms": ["反垄断", "市场秩序", "平台竞争", "应用商店", "监管执法", "消费者选择"],
    },
    {
        "cluster": "online_information_integrity",
        "label": "网络内容安全与信息真实性",
        "domain_tokens": ["online_safety", "election_integrity", "ai_risk"],
        "keywords": ["虚假信息", "深度伪造", "内容安全", "信息真实性", "网络干预"],
        "terms": ["内容安全", "信息真实性", "深度伪造", "网络干预", "事实核查", "平台责任"],
    },
    {
        "cluster": "global_policy_general",
        "label": "全球公共议题综合",
        "domain_tokens": [],
        "keywords": [],
        "terms": ["公共议题", "政策解读", "社会影响", "风险沟通", "事实核查", "背景解释"],
    },
]

ROLE_SPECS: list[dict[str, Any]] = [
    {
        "role": "core_publish_node",
        "alias": "core",
        "followers": (150_000, 800_000),
        "friends": (500, 2_000),
        "anchor_edges": (35, 60),
        "description": "长期关注{cluster_label}，喜欢把{focus_terms}里的复杂脉络讲清楚。",
    },
    {
        "role": "interaction_response_node",
        "alias": "qa",
        "followers": (30_000, 250_000),
        "friends": (800, 3_000),
        "anchor_edges": (25, 45),
        "description": "常刷{cluster_label}相关讨论，看到争议点会顺手查资料，也爱记录{focus_terms}的新变化。",
    },
    {
        "role": "amplification_node",
        "alias": "amplify",
        "followers": (80_000, 600_000),
        "friends": (400, 1_800),
        "anchor_edges": (40, 70),
        "description": "每天整理{cluster_label}动态，收藏{focus_terms}相关资料，偶尔写点简短观察。",
    },
    {
        "role": "support_node",
        "alias": "support",
        "followers": (10_000, 150_000),
        "friends": (400, 2_200),
        "anchor_edges": (20, 35),
        "description": "偏好资料核查、背景补充和风险提示，关注{cluster_label}里的细节和上下文。",
    },
    {
        "role": "data_explain_node",
        "alias": "data",
        "followers": (20_000, 200_000),
        "friends": (300, 1_600),
        "anchor_edges": (25, 45),
        "description": "习惯用数据、时间线和政策文件理解{cluster_label}，关注{focus_terms}背后的指标变化。",
    },
]

BRIDGE_CLUSTER_PAIRS = {
    "ai_platform_governance": ["cybersecurity_data_governance", "online_information_integrity", "antitrust_market_order"],
    "cybersecurity_data_governance": ["ai_platform_governance", "financial_regulation_crypto_banking", "online_information_integrity"],
    "climate_disaster_emergency": ["climate_environment_policy", "water_infrastructure_resilience", "public_health_medical"],
    "energy_security_oil": ["clean_energy_transition", "finance_monetary_market", "international_trade_supply_chain"],
    "shipping_logistics_transport": ["international_trade_supply_chain", "military_security_defense", "energy_security_oil"],
    "finance_monetary_market": ["financial_regulation_crypto_banking", "fiscal_tax_debt", "energy_security_oil"],
    "international_relations_law": ["military_security_defense", "migration_humanitarian_crisis", "politics_election_institution"],
    "public_policy_livelihood": ["demographics_social_services", "fiscal_tax_debt", "real_estate_housing"],
    "technology_industry_chip": ["ai_platform_governance", "international_trade_supply_chain", "clean_energy_transition"],
}


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [event for event in payload if isinstance(event, dict)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def stable_hash_int(text: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{text}".encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def dedupe_texts(items: list[Any], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def event_text(event: dict[str, Any]) -> str:
    variants = event.get("opinion_variants")
    variant_text = " ".join(str(item) for item in variants[:4]) if isinstance(variants, list) else ""
    return " ".join(
        [
            str(event.get("domain") or ""),
            str(event.get("event_title") or ""),
            str(event.get("event_summary") or ""),
            str(event.get("target") or ""),
            variant_text,
        ]
    ).lower()


def event_terms(event: dict[str, Any], *, limit: int = 10) -> list[str]:
    text = event_text(event)
    terms: list[str] = []
    for spec in CLUSTER_SPECS:
        for term in spec.get("terms", []):
            if term and term.lower() in text:
                terms.append(term)
        for keyword in spec.get("keywords", []):
            if keyword and keyword.lower() in text:
                terms.append(keyword)
    for token in str(event.get("domain") or "").replace("-", "_").split("_"):
        if token and token not in GENERIC_TERMS:
            terms.append(token)
    title = str(event.get("event_title") or "")
    for chunk in title.replace("，", " ").replace("、", " ").replace("和", " ").split():
        if len(chunk) >= 2 and chunk not in GENERIC_TERMS:
            terms.append(chunk[:12])
    return dedupe_texts(terms, limit=limit)


def assign_cluster(event: dict[str, Any]) -> str:
    domain = str(event.get("domain") or "").lower()
    text = event_text(event)
    best_cluster = "global_policy_general"
    best_score = -1
    for spec in CLUSTER_SPECS:
        score = 0
        for token in spec.get("domain_tokens", []):
            if token and (token == domain or token in domain):
                score += 8
            elif token and any(part and part in token for part in domain.replace("-", "_").split("_")):
                score += 3
        for keyword in spec.get("keywords", []):
            if keyword and keyword.lower() in text:
                score += 4
        for term in spec.get("terms", []):
            if term and term.lower() in text:
                score += 3
        if score > best_score:
            best_score = score
            best_cluster = str(spec["cluster"])
    return best_cluster


def build_event_clusters(events: list[dict[str, Any]], *, cluster_limit: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    events_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_cluster[assign_cluster(event)].append(event)

    specs_by_key = {str(spec["cluster"]): spec for spec in CLUSTER_SPECS}
    ranked_keys = sorted(
        events_by_cluster,
        key=lambda key: (len(events_by_cluster[key]), key),
        reverse=True,
    )
    selected_keys = ranked_keys[:cluster_limit]

    clusters: list[dict[str, Any]] = []
    for key in selected_keys:
        spec = specs_by_key.get(key, specs_by_key["global_policy_general"])
        cluster_events = events_by_cluster[key]
        domain_counter = Counter(str(event.get("domain") or "general") for event in cluster_events)
        terms = dedupe_texts(
            [
                *spec.get("terms", []),
                *[
                    term
                    for event in cluster_events
                    for term in event_terms(event, limit=8)
                ],
            ],
            limit=16,
        )
        clusters.append(
            {
                "cluster": key,
                "label": spec.get("label", key),
                "terms": terms,
                "event_ids": [str(event.get("event_id") or "") for event in cluster_events],
                "domains": dict(domain_counter.most_common()),
                "event_count": len(cluster_events),
            }
        )
    return clusters, events_by_cluster


def normalize_profile_text(record: dict[str, Any]) -> str:
    interests = " ".join(str(item).strip() for item in record.get("user_interests", []) if str(item).strip())
    description = str(record.get("user_description") or "").strip()
    return f"{interests} {description}".lower()


def low_value_score(user_id: str, record: dict[str, Any]) -> float:
    graph = record.get("graph_attributes") if isinstance(record.get("graph_attributes"), dict) else {}
    description = str(record.get("user_description") or "").strip()
    interests = [item for item in record.get("user_interests", []) if str(item).strip()]
    return (
        len(description) * 10.0
        + len(interests) * 80.0
        + math.log1p(max(0, safe_int(record.get("user_followers")))) * 25.0
        + math.log1p(max(0, safe_int(record.get("user_friends")))) * 6.0
        + safe_int(graph.get("neighbor_count")) * 2.0
        + safe_int(graph.get("received_interaction_count")) * 0.8
        + safe_int(graph.get("made_interaction_count")) * 0.7
        + (stable_hash_int(user_id, 97) % 1000) / 1000
    )


def is_rewrite_candidate(record: dict[str, Any]) -> bool:
    description = str(record.get("user_description") or "").strip()
    interests = [str(item).strip() for item in record.get("user_interests", []) if str(item).strip()]
    graph = record.get("graph_attributes") if isinstance(record.get("graph_attributes"), dict) else {}
    followers = safe_int(record.get("user_followers"))
    if str(record.get("user_name") or "").startswith("topic_"):
        return False
    return (
        len(description) <= 8
        and len(interests) <= 1
        and safe_int(graph.get("neighbor_count")) >= 1
        and followers < 50_000
    )


def select_rewrite_ids(records: dict[str, dict[str, Any]], *, count: int) -> list[str]:
    candidates = [
        (low_value_score(user_id, record), user_id)
        for user_id, record in records.items()
        if is_rewrite_candidate(record)
    ]
    candidates.sort()
    if len(candidates) < count:
        raise ValueError(f"Not enough low-value connected nodes: required={count}, available={len(candidates)}")
    return [user_id for _score, user_id in candidates[:count]]


def build_topic_profile(
    *,
    original: dict[str, Any],
    user_id: str,
    cluster: dict[str, Any],
    role_spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(stable_hash_int(f"profile:{user_id}:{cluster['cluster']}:{role_spec['role']}", seed))
    focus_terms = "、".join(cluster["terms"][:4])
    updated = dict(original)
    updated["user_id"] = str(original.get("user_id") or user_id)
    updated["user_name"] = f"topic_{cluster['cluster']}_{role_spec['alias']}_{user_id}"
    updated["user_followers"] = rng.randint(*role_spec["followers"])
    updated["user_friends"] = rng.randint(*role_spec["friends"])
    updated["user_interests"] = dedupe_texts(
        [*cluster["terms"], role_spec["role"], str(cluster["label"])],
        limit=14,
    )
    updated["user_description"] = str(role_spec["description"]).format(
        cluster_label=cluster["label"],
        focus_terms=focus_terms or cluster["label"],
    )
    updated["topic_profile_metadata"] = {
        "source": "hot_event_opinion_variants_200",
        "cluster": cluster["cluster"],
        "cluster_label": cluster["label"],
        "role": role_spec["role"],
        "covered_event_ids": cluster["event_ids"][:20],
        "version": "hot200_clustered_digital_human_v1",
    }
    return updated


def make_neighbor_stats() -> dict[str, int]:
    return {field: 0 for field in NEIGHBOR_STAT_FIELDS}


def relation_label(stats: dict[str, int]) -> str:
    received_total = stats["received_comment_count"] + stats["received_repost_count"]
    made_total = stats["made_comment_count"] + stats["made_repost_count"]
    if received_total and made_total:
        return "mutual"
    if received_total:
        return "engaged_by"
    if made_total:
        return "engaged_to"
    return "isolated"


def add_neighbor_stats(
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    user_id: str,
    neighbor_id: str,
    stats: dict[str, Any],
) -> None:
    if user_id == neighbor_id:
        return
    target = relation_map[user_id][neighbor_id]
    for field in NEIGHBOR_STAT_FIELDS:
        target[field] += safe_int(stats.get(field))


def add_interaction(
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    post_owner_id: str,
    actor_id: str,
    interact_type: str,
    count: int,
) -> None:
    if post_owner_id == actor_id or count <= 0:
        return
    owner_stats = relation_map[post_owner_id][actor_id]
    actor_stats = relation_map[actor_id][post_owner_id]
    if interact_type == "comment":
        owner_stats["received_comment_count"] += count
        actor_stats["made_comment_count"] += count
    elif interact_type == "reposts":
        owner_stats["received_repost_count"] += count
        actor_stats["made_repost_count"] += count


def original_neighbor_lookup(records: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for user_id, record in records.items():
        for neighbor in record.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            neighbor_id = str(neighbor.get("neighbor_id") or "").strip()
            if neighbor_id:
                lookup[(user_id, neighbor_id)] = neighbor
    return lookup


def preserve_original_relations(
    *,
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    records: dict[str, dict[str, Any]],
    rewritten_ids: set[str],
    preserve_edges_per_node: int,
) -> dict[str, int]:
    preserved_by_rewritten: Counter[str] = Counter()
    lookup = original_neighbor_lookup(records)
    preserve_targets: dict[str, set[str]] = defaultdict(set)

    for rewritten_id in rewritten_ids:
        original_neighbors = [
            neighbor
            for neighbor in records[rewritten_id].get("neighbors", [])
            if isinstance(neighbor, dict)
            and str(neighbor.get("neighbor_id") or "") not in rewritten_ids
            and str(neighbor.get("neighbor_id") or "") in records
        ]
        original_neighbors.sort(
            key=lambda item: safe_int(item.get("total_interaction_count"))
            or safe_int(item.get("received_interaction_count")) + safe_int(item.get("made_interaction_count")),
            reverse=True,
        )
        for neighbor in original_neighbors[:preserve_edges_per_node]:
            neighbor_id = str(neighbor.get("neighbor_id"))
            preserve_targets[rewritten_id].add(neighbor_id)
            preserve_targets[neighbor_id].add(rewritten_id)

    for user_id, record in records.items():
        if user_id in rewritten_ids:
            continue
        for neighbor in record.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            neighbor_id = str(neighbor.get("neighbor_id") or "").strip()
            if not neighbor_id or neighbor_id in rewritten_ids:
                continue
            add_neighbor_stats(relation_map, user_id, neighbor_id, neighbor)

    for rewritten_id, targets in preserve_targets.items():
        if rewritten_id not in rewritten_ids:
            continue
        for target_id in targets:
            forward = lookup.get((rewritten_id, target_id))
            reverse = lookup.get((target_id, rewritten_id))
            if forward:
                add_neighbor_stats(relation_map, rewritten_id, target_id, forward)
            if reverse:
                add_neighbor_stats(relation_map, target_id, rewritten_id, reverse)
            preserved_by_rewritten[rewritten_id] += 1
    return dict(preserved_by_rewritten)


def score_anchor(record: dict[str, Any], terms: list[str], user_id: str, seed: int) -> float:
    text = normalize_profile_text(record)
    exact_hits = sum(1 for term in terms if term and term.lower() in text)
    graph = record.get("graph_attributes") if isinstance(record.get("graph_attributes"), dict) else {}
    return (
        exact_hits * 10.0
        + math.log1p(max(0, safe_int(record.get("user_followers")))) * 0.30
        + math.log1p(max(0, safe_int(graph.get("received_interaction_count")) + safe_int(graph.get("made_interaction_count")))) * 0.35
        + min(len(record.get("user_interests", [])), 6) * 0.18
        + (0.5 if str(record.get("user_description") or "").strip() else 0.0)
        + (stable_hash_int(f"anchor:{user_id}", seed) % 1000) / 10000
    )


def build_anchor_rankings(
    records: dict[str, dict[str, Any]],
    clusters: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    seed: int,
) -> dict[str, list[str]]:
    candidate_ids = [user_id for user_id in records if user_id not in excluded_ids]
    rankings: dict[str, list[str]] = {}
    for cluster in clusters:
        scored = [
            (score_anchor(records[user_id], cluster["terms"], user_id, seed), user_id)
            for user_id in candidate_ids
        ]
        scored.sort(reverse=True)
        rankings[cluster["cluster"]] = [user_id for score, user_id in scored if score > 0.35]
    return rankings


def role_counts(role: str, *, peer_link: bool, seed_text: str, seed: int) -> dict[str, int]:
    rng = random.Random(stable_hash_int(seed_text, seed))
    boost = 2 if peer_link else 0
    if role == "core_publish_node":
        return {
            "incoming_comment": rng.randint(4, 9 + boost),
            "incoming_repost": rng.randint(2, 6 + boost),
            "outgoing_comment": rng.randint(1, 3 + boost),
            "outgoing_repost": rng.randint(0, 2 + boost),
        }
    if role == "interaction_response_node":
        return {
            "incoming_comment": rng.randint(2, 6 + boost),
            "incoming_repost": rng.randint(0, 2 + boost),
            "outgoing_comment": rng.randint(5, 11 + boost),
            "outgoing_repost": rng.randint(0, 2 + boost),
        }
    if role == "amplification_node":
        return {
            "incoming_comment": rng.randint(1, 4 + boost),
            "incoming_repost": rng.randint(1, 3 + boost),
            "outgoing_comment": rng.randint(1, 4 + boost),
            "outgoing_repost": rng.randint(4, 10 + boost),
        }
    if role == "support_node":
        return {
            "incoming_comment": rng.randint(2, 5 + boost),
            "incoming_repost": rng.randint(0, 2 + boost),
            "outgoing_comment": rng.randint(3, 7 + boost),
            "outgoing_repost": rng.randint(0, 2 + boost),
        }
    return {
        "incoming_comment": rng.randint(2, 5 + boost),
        "incoming_repost": rng.randint(0, 2 + boost),
        "outgoing_comment": rng.randint(3, 7 + boost),
        "outgoing_repost": rng.randint(1, 3 + boost),
    }


def add_role_relationship(
    *,
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    rewritten_id: str,
    target_id: str,
    role: str,
    seed: int,
    peer_link: bool,
) -> None:
    if rewritten_id == target_id:
        return
    counts = role_counts(
        role,
        peer_link=peer_link,
        seed_text=f"{rewritten_id}:{target_id}:{role}:{peer_link}",
        seed=seed,
    )
    add_interaction(relation_map, rewritten_id, target_id, "comment", counts["incoming_comment"])
    add_interaction(relation_map, rewritten_id, target_id, "reposts", counts["incoming_repost"])
    add_interaction(relation_map, target_id, rewritten_id, "comment", counts["outgoing_comment"])
    add_interaction(relation_map, target_id, rewritten_id, "reposts", counts["outgoing_repost"])


def cluster_bridge_targets(
    *,
    cluster_key: str,
    node_ids_by_cluster: dict[str, list[str]],
    current_id: str,
    count: int,
    seed: int,
) -> list[str]:
    related_clusters = BRIDGE_CLUSTER_PAIRS.get(cluster_key, [])
    pool: list[str] = []
    for related in related_clusters:
        pool.extend(node_ids_by_cluster.get(related, []))
    if len(pool) < count:
        for other_cluster, node_ids in node_ids_by_cluster.items():
            if other_cluster != cluster_key:
                pool.extend(node_ids)
    pool = [user_id for user_id in dict.fromkeys(pool) if user_id != current_id]
    pool.sort(key=lambda item: stable_hash_int(f"bridge:{current_id}:{item}", seed))
    return pool[:count]


def inject_cluster_relations(
    *,
    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]],
    rewritten_items: list[dict[str, Any]],
    anchor_rankings: dict[str, list[str]],
    same_cluster_edges: tuple[int, int],
    bridge_edges: tuple[int, int],
    seed: int,
) -> list[dict[str, Any]]:
    node_ids_by_cluster: dict[str, list[str]] = defaultdict(list)
    item_by_user_id: dict[str, dict[str, Any]] = {}
    for item in rewritten_items:
        node_ids_by_cluster[item["cluster"]].append(item["user_id"])
        item_by_user_id[item["user_id"]] = item

    connection_summaries: list[dict[str, Any]] = []
    for item in rewritten_items:
        user_id = item["user_id"]
        cluster_key = item["cluster"]
        role = item["role"]
        rng = random.Random(stable_hash_int(f"relations:{user_id}", seed))

        same_candidates = [
            candidate_id
            for candidate_id in node_ids_by_cluster[cluster_key]
            if candidate_id != user_id
        ]
        same_candidates.sort(key=lambda candidate_id: stable_hash_int(f"same:{user_id}:{candidate_id}", seed))
        same_count = min(len(same_candidates), rng.randint(*same_cluster_edges))
        same_targets = same_candidates[:same_count]

        anchor_candidates = [
            candidate_id
            for candidate_id in anchor_rankings.get(cluster_key, [])
            if candidate_id != user_id
        ]
        anchor_count = rng.randint(*item["anchor_edges"])
        anchor_targets = anchor_candidates[:anchor_count]

        bridge_count = rng.randint(*bridge_edges)
        bridge_targets = cluster_bridge_targets(
            cluster_key=cluster_key,
            node_ids_by_cluster=node_ids_by_cluster,
            current_id=user_id,
            count=bridge_count,
            seed=seed,
        )

        for target_id in same_targets:
            add_role_relationship(
                relation_map=relation_map,
                rewritten_id=user_id,
                target_id=target_id,
                role=role,
                seed=seed,
                peer_link=True,
            )
        for target_id in anchor_targets:
            add_role_relationship(
                relation_map=relation_map,
                rewritten_id=user_id,
                target_id=target_id,
                role=role,
                seed=seed,
                peer_link=False,
            )
        for target_id in bridge_targets:
            target_role = item_by_user_id.get(target_id, {}).get("role", role)
            add_role_relationship(
                relation_map=relation_map,
                rewritten_id=user_id,
                target_id=target_id,
                role=target_role,
                seed=seed,
                peer_link=True,
            )

        connection_summaries.append(
            {
                "user_id": user_id,
                "added_same_cluster_neighbor_count": len(same_targets),
                "added_anchor_neighbor_count": len(anchor_targets),
                "added_bridge_neighbor_count": len(bridge_targets),
                "same_cluster_neighbor_ids": same_targets,
                "anchor_neighbor_ids": anchor_targets,
                "bridge_neighbor_ids": bridge_targets,
            }
        )
    return connection_summaries


def build_profile_with_recomputed_graph(
    *,
    user_id: str,
    record: dict[str, Any],
    neighbor_stats: dict[str, dict[str, int]],
) -> dict[str, Any]:
    neighbors: list[dict[str, Any]] = []
    graph_counter = Counter()
    for neighbor_id, stats in sorted(
        neighbor_stats.items(),
        key=lambda item: (
            item[1]["received_comment_count"]
            + item[1]["received_repost_count"]
            + item[1]["made_comment_count"]
            + item[1]["made_repost_count"],
            item[0],
        ),
        reverse=True,
    ):
        received_interaction = stats["received_comment_count"] + stats["received_repost_count"]
        made_interaction = stats["made_comment_count"] + stats["made_repost_count"]
        total_interaction = received_interaction + made_interaction
        if total_interaction <= 0:
            continue
        relation = relation_label(stats)
        if relation == "engaged_by":
            graph_counter["engaged_by_neighbor_count"] += 1
        elif relation == "engaged_to":
            graph_counter["engaged_to_neighbor_count"] += 1
        elif relation == "mutual":
            graph_counter["mutual_neighbor_count"] += 1
        graph_counter["received_comment_count"] += stats["received_comment_count"]
        graph_counter["received_repost_count"] += stats["received_repost_count"]
        graph_counter["made_comment_count"] += stats["made_comment_count"]
        graph_counter["made_repost_count"] += stats["made_repost_count"]
        neighbors.append(
            {
                "neighbor_id": neighbor_id,
                "relation": relation,
                "received_comment_count": stats["received_comment_count"],
                "received_repost_count": stats["received_repost_count"],
                "made_comment_count": stats["made_comment_count"],
                "made_repost_count": stats["made_repost_count"],
                "received_interaction_count": received_interaction,
                "made_interaction_count": made_interaction,
                "total_interaction_count": total_interaction,
            }
        )

    graph_attributes = {
        "neighbor_count": len(neighbors),
        "engaged_by_neighbor_count": graph_counter["engaged_by_neighbor_count"],
        "engaged_to_neighbor_count": graph_counter["engaged_to_neighbor_count"],
        "mutual_neighbor_count": graph_counter["mutual_neighbor_count"],
        "self_interaction_count": safe_int((record.get("graph_attributes") or {}).get("self_interaction_count")),
        "self_comment_count": safe_int((record.get("graph_attributes") or {}).get("self_comment_count")),
        "self_repost_count": safe_int((record.get("graph_attributes") or {}).get("self_repost_count")),
        "received_interaction_count": graph_counter["received_comment_count"] + graph_counter["received_repost_count"],
        "received_comment_count": graph_counter["received_comment_count"],
        "received_repost_count": graph_counter["received_repost_count"],
        "made_interaction_count": graph_counter["made_comment_count"] + graph_counter["made_repost_count"],
        "made_comment_count": graph_counter["made_comment_count"],
        "made_repost_count": graph_counter["made_repost_count"],
        "isolated": len(neighbors) == 0,
    }

    updated = dict(record)
    updated["user_id"] = str(updated.get("user_id") or user_id)
    updated["graph_attributes"] = graph_attributes
    updated["neighbors"] = neighbors
    return updated


def build_rewritten_items(
    *,
    records: dict[str, dict[str, Any]],
    rewrite_ids: list[str],
    clusters: list[dict[str, Any]],
    target_rewrite_count: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    updated_records = dict(records)
    items: list[dict[str, Any]] = []
    cursor = 0
    role_count = len(ROLE_SPECS)
    for cluster in clusters:
        for role_spec in ROLE_SPECS:
            if cursor >= target_rewrite_count:
                break
            user_id = rewrite_ids[cursor]
            original = records[user_id]
            updated = build_topic_profile(
                original=original,
                user_id=user_id,
                cluster=cluster,
                role_spec=role_spec,
                seed=seed,
            )
            updated_records[user_id] = updated
            items.append(
                {
                    "user_id": user_id,
                    "cluster": cluster["cluster"],
                    "cluster_label": cluster["label"],
                    "role": role_spec["role"],
                    "anchor_edges": role_spec["anchor_edges"],
                    "covered_event_ids": cluster["event_ids"],
                    "covered_domains": cluster["domains"],
                    "original_profile": {
                        "user_name": original.get("user_name"),
                        "user_followers": original.get("user_followers"),
                        "user_friends": original.get("user_friends"),
                        "user_interests": original.get("user_interests", []),
                        "user_description": original.get("user_description", ""),
                        "graph_attributes": original.get("graph_attributes", {}),
                    },
                    "rewritten_profile": {
                        "user_name": updated.get("user_name"),
                        "user_followers": updated.get("user_followers"),
                        "user_friends": updated.get("user_friends"),
                        "user_interests": updated.get("user_interests", []),
                        "user_description": updated.get("user_description", ""),
                        "topic_profile_metadata": updated.get("topic_profile_metadata", {}),
                    },
                }
            )
            cursor += 1
        if cursor >= target_rewrite_count:
            break

    if cursor < target_rewrite_count:
        # Distribute remaining roles over the largest clusters.
        extra_clusters = sorted(clusters, key=lambda item: (item["event_count"], item["cluster"]), reverse=True)
        while cursor < target_rewrite_count:
            cluster = extra_clusters[cursor % len(extra_clusters)]
            role_spec = ROLE_SPECS[cursor % role_count]
            user_id = rewrite_ids[cursor]
            original = records[user_id]
            updated = build_topic_profile(
                original=original,
                user_id=user_id,
                cluster=cluster,
                role_spec=role_spec,
                seed=seed + cursor,
            )
            updated_records[user_id] = updated
            items.append(
                {
                    "user_id": user_id,
                    "cluster": cluster["cluster"],
                    "cluster_label": cluster["label"],
                    "role": role_spec["role"],
                    "anchor_edges": role_spec["anchor_edges"],
                    "covered_event_ids": cluster["event_ids"],
                    "covered_domains": cluster["domains"],
                    "original_profile": {
                        "user_name": original.get("user_name"),
                        "user_followers": original.get("user_followers"),
                        "user_friends": original.get("user_friends"),
                        "user_interests": original.get("user_interests", []),
                        "user_description": original.get("user_description", ""),
                        "graph_attributes": original.get("graph_attributes", {}),
                    },
                    "rewritten_profile": {
                        "user_name": updated.get("user_name"),
                        "user_followers": updated.get("user_followers"),
                        "user_friends": updated.get("user_friends"),
                        "user_interests": updated.get("user_interests", []),
                        "user_description": updated.get("user_description", ""),
                        "topic_profile_metadata": updated.get("topic_profile_metadata", {}),
                    },
                }
            )
            cursor += 1
    return updated_records, items


def validate_bidirectional_new_edges(records: dict[str, dict[str, Any]], rewritten_ids: set[str]) -> dict[str, Any]:
    missing_reverse = 0
    checked_edges = 0
    for user_id in rewritten_ids:
        record = records.get(user_id, {})
        for neighbor in record.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            neighbor_id = str(neighbor.get("neighbor_id") or "")
            if not neighbor_id or neighbor_id not in records:
                continue
            checked_edges += 1
            reverse_neighbors = records[neighbor_id].get("neighbors", [])
            if not any(str(item.get("neighbor_id") or "") == user_id for item in reverse_neighbors if isinstance(item, dict)):
                missing_reverse += 1
    return {
        "checked_rewritten_edges": checked_edges,
        "missing_reverse_edges": missing_reverse,
        "bidirectional_ok": missing_reverse == 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite low-value existing nodes into clustered digital humans for the 200 hot-event dataset and update graph relations in place.",
    )
    parser.add_argument("--events", default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--profile", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--cluster-summary-output", default=DEFAULT_CLUSTER_SUMMARY_PATH)
    parser.add_argument("--cluster-limit", type=int, default=30)
    parser.add_argument("--target-rewrite-count", type=int, default=150)
    parser.add_argument("--preserve-original-edges-per-node", type=int, default=5)
    parser.add_argument("--same-cluster-edges-min", type=int, default=3)
    parser.add_argument("--same-cluster-edges-max", type=int, default=5)
    parser.add_argument("--bridge-edges-min", type=int, default=3)
    parser.add_argument("--bridge-edges-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--no-backup", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    events_path = Path(args.events)
    profile_path = Path(args.profile)
    manifest_output_path = Path(args.manifest_output)
    summary_output_path = Path(args.summary_output)
    cluster_summary_output_path = Path(args.cluster_summary_output)

    events = load_events(events_path)
    records = load_json_object(profile_path)
    records = {str(user_id): record for user_id, record in records.items() if isinstance(record, dict)}

    clusters, events_by_cluster = build_event_clusters(events, cluster_limit=args.cluster_limit)
    rewrite_ids = select_rewrite_ids(records, count=args.target_rewrite_count)
    rewritten_records, rewritten_items = build_rewritten_items(
        records=records,
        rewrite_ids=rewrite_ids,
        clusters=clusters,
        target_rewrite_count=args.target_rewrite_count,
        seed=args.seed,
    )
    rewritten_ids = {item["user_id"] for item in rewritten_items}

    relation_map: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(make_neighbor_stats)
    )
    preserved_counts = preserve_original_relations(
        relation_map=relation_map,
        records=records,
        rewritten_ids=rewritten_ids,
        preserve_edges_per_node=args.preserve_original_edges_per_node,
    )
    anchor_rankings = build_anchor_rankings(
        rewritten_records,
        clusters,
        excluded_ids=rewritten_ids,
        seed=args.seed,
    )
    connection_summaries = inject_cluster_relations(
        relation_map=relation_map,
        rewritten_items=rewritten_items,
        anchor_rankings=anchor_rankings,
        same_cluster_edges=(args.same_cluster_edges_min, args.same_cluster_edges_max),
        bridge_edges=(args.bridge_edges_min, args.bridge_edges_max),
        seed=args.seed,
    )
    connection_by_user_id = {item["user_id"]: item for item in connection_summaries}

    output_records: dict[str, dict[str, Any]] = {}
    for user_id, record in rewritten_records.items():
        output_records[user_id] = build_profile_with_recomputed_graph(
            user_id=user_id,
            record=record,
            neighbor_stats=relation_map.get(user_id, {}),
        )

    validation = validate_bidirectional_new_edges(output_records, rewritten_ids)
    if not validation["bidirectional_ok"]:
        raise RuntimeError(f"Generated graph has missing reverse edges: {validation}")

    for item in rewritten_items:
        connection_update = connection_by_user_id.get(item["user_id"], {})
        item["connection_update"] = {
            "preserved_original_neighbor_count": preserved_counts.get(item["user_id"], 0),
            "added_same_cluster_neighbor_count": connection_update.get("added_same_cluster_neighbor_count", 0),
            "added_anchor_neighbor_count": connection_update.get("added_anchor_neighbor_count", 0),
            "added_bridge_neighbor_count": connection_update.get("added_bridge_neighbor_count", 0),
            "same_cluster_neighbor_ids": connection_update.get("same_cluster_neighbor_ids", []),
            "anchor_neighbor_ids": connection_update.get("anchor_neighbor_ids", []),
            "bridge_neighbor_ids": connection_update.get("bridge_neighbor_ids", []),
        }

    if not args.no_backup:
        backup_path = profile_path.with_suffix(profile_path.suffix + ".bak")
        shutil.copy2(profile_path, backup_path)
    else:
        backup_path = None

    write_json(profile_path, output_records)
    write_jsonl(manifest_output_path, rewritten_items)

    rewritten_neighbor_counts = [
        output_records[user_id]["graph_attributes"]["neighbor_count"]
        for user_id in rewritten_ids
    ]
    rewritten_followers = [
        safe_int(output_records[user_id].get("user_followers"))
        for user_id in rewritten_ids
    ]
    touched_original_ids = {
        neighbor["neighbor_id"]
        for user_id in rewritten_ids
        for neighbor in output_records[user_id].get("neighbors", [])
        if neighbor.get("neighbor_id") not in rewritten_ids
    }
    summary = {
        "events_path": str(events_path),
        "profile_path": str(profile_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "manifest_output_path": str(manifest_output_path),
        "summary_output_path": str(summary_output_path),
        "cluster_summary_output_path": str(cluster_summary_output_path),
        "input_profile_count": len(records),
        "output_profile_count": len(output_records),
        "event_count": len(events),
        "selected_cluster_count": len(clusters),
        "target_rewrite_count": args.target_rewrite_count,
        "rewritten_profile_count": len(rewritten_items),
        "touched_original_neighbor_count": len(touched_original_ids),
        "rewritten_neighbor_count_min": min(rewritten_neighbor_counts) if rewritten_neighbor_counts else 0,
        "rewritten_neighbor_count_max": max(rewritten_neighbor_counts) if rewritten_neighbor_counts else 0,
        "rewritten_neighbor_count_avg": round(sum(rewritten_neighbor_counts) / max(len(rewritten_neighbor_counts), 1), 4),
        "rewritten_follower_min": min(rewritten_followers) if rewritten_followers else 0,
        "rewritten_follower_max": max(rewritten_followers) if rewritten_followers else 0,
        "role_distribution": dict(Counter(item["role"] for item in rewritten_items).most_common()),
        "cluster_distribution": dict(Counter(item["cluster"] for item in rewritten_items).most_common()),
        "validation": validation,
        "parameters": {
            "preserve_original_edges_per_node": args.preserve_original_edges_per_node,
            "same_cluster_edges": [args.same_cluster_edges_min, args.same_cluster_edges_max],
            "bridge_edges": [args.bridge_edges_min, args.bridge_edges_max],
            "seed": args.seed,
        },
        "notes": [
            "existing low-value node ids are reused; no new user ids are created",
            "original profile file is updated in place after backup",
            "rewritten nodes preserve a small number of original relations and gain topic-cluster, anchor, and bridge relations",
            "graph_attributes and neighbors are recomputed from the relation map",
        ],
    }
    cluster_summary = {
        "selected_clusters": clusters,
        "unselected_cluster_count": max(0, len(events_by_cluster) - len(clusters)),
        "all_cluster_event_distribution": {
            key: len(value)
            for key, value in sorted(events_by_cluster.items(), key=lambda item: (-len(item[1]), item[0]))
        },
    }
    write_json(summary_output_path, summary)
    write_json(cluster_summary_output_path, cluster_summary)

    print(f"updated profile dataset in place: {profile_path}")
    if backup_path is not None:
        print(f"backup saved: {backup_path}")
    print(f"manifest saved: {manifest_output_path}")
    print(f"summary saved: {summary_output_path}")
    print(f"cluster summary saved: {cluster_summary_output_path}")
    print(f"rewritten_profile_count: {len(rewritten_items)}")
    print(f"selected_cluster_count: {len(clusters)}")
    print(f"rewritten_neighbor_count_avg: {summary['rewritten_neighbor_count_avg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
