from __future__ import annotations

import json
from typing import Any

from .models import ParsedEvent


def build_event_parser_system_prompt() -> str:
    return (
        "你是影响力事件分发策略系统中的事件解析助手。"
        "请基于输入事件信息，输出严格 JSON。"
        "不要编造未提供的外部事实。"
        "输出重点是传播策略建模，不是商品推荐。"
    )


def build_event_parser_user_prompt(
    *,
    raw_event: dict[str, Any],
    fallback_parse: dict[str, Any],
) -> str:
    payload = {
        "task": "将事件解析为影响力分发策略所需的结构化字段。",
        "raw_event": raw_event,
        "fallback_parse": fallback_parse,
        "requirements": [
            "保留事件传播语境，不要改写成商品推荐任务。",
            "event_type、target_goal、target_audience 尽量贴合传播任务。",
            "semantic_tags 用于后续候选节点语义匹配，建议 3-8 个。",
            "narrative_frames 用于区分叙事角度，建议 2-5 个。",
            "target_roles 用于说明该事件更依赖哪些角色，例如 core_publish_node、interaction_response_node、amplification_node、support_node。",
            "negative_constraints 用于说明避免使用的传播方式或表达方式。",
            "dispatch_preferences.candidate_pool_size 取值 8-60。",
            "dispatch_preferences.rerank_top_k 取值 3-20。",
        ],
        "required_json_schema": {
            "event_type": "general_influence_event",
            "target_goal": "awareness",
            "target_audience": ["general_public"],
            "extracted_keywords": ["关键词1", "关键词2"],
            "risk_level": "low",
            "semantic_tags": ["标签1", "标签2"],
            "narrative_frames": ["叙事角度1", "叙事角度2"],
            "sensitive_entities": ["敏感实体"],
            "target_roles": ["core_publish_node", "interaction_response_node"],
            "negative_constraints": ["避免情绪化表达"],
            "dispatch_preferences": {
                "candidate_pool_size": 24,
                "rerank_top_k": 12,
                "semantic_weight": 0.35,
                "diversity_weight": 0.20,
                "risk_weight": 0.20,
                "preferred_roles": ["core_publish_node", "interaction_response_node"],
                "preferred_narrative": ["事实解释", "问答承接"],
            },
            "reasoning": ["简短说明1", "简短说明2"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_feature_enrichment_system_prompt() -> str:
    return (
        "你是数字人候选节点语义评估助手。"
        "请针对事件与候选节点的匹配度返回严格 JSON。"
        "评分范围统一为 0 到 1。"
        "不要输出 Markdown。"
    )


def build_feature_enrichment_user_prompt(
    *,
    event: ParsedEvent,
    candidate_cards: list[dict[str, Any]],
) -> str:
    payload = {
        "task": "针对候选节点生成事件适配评分，供后续打分和选人使用。",
        "event": {
            "event_id": event.event_id,
            "event_title": event.event_title,
            "event_description": event.event_description,
            "target_goal": event.target_goal,
            "event_type": event.event_type,
            "target_audience": event.target_audience,
            "semantic_tags": event.semantic_tags,
            "narrative_frames": event.narrative_frames,
            "target_roles": event.target_roles,
            "negative_constraints": event.negative_constraints,
            "risk_level": event.constraints.risk_level,
        },
        "requirements": [
            "结合节点画像、兴趣、图结构摘要，判断其是否适合参与该事件分发。",
            "节点的高粉丝量或泛公众覆盖能力，不能替代事件主题相关性。",
            "如果节点画像与事件主题明显不相关，例如旅游/美食/生活方式账号去传播军事、航运、金融风险事件，则 semantic_relevance_score 应明显偏低。",
            "semantic_relevance_score 反映事件主题匹配度。",
            "audience_fit_score 反映该节点触达目标受众的适配度。",
            "role_fit_score 反映该节点承担当前事件传播角色的适配度。",
            "narrative_fit_score 反映其是否适合当前叙事框架。",
            "risk_conflict_score 越高表示越可能与事件约束冲突。",
            "novelty_score 越高表示该节点能提供与其他高影响力节点不同的视角。",
            "semantic_tags 用于概括该节点与事件的语义关联，建议 1-4 个；如果关联度低，可以返回空列表。",
            "reasoning 只需短语，不要长段落。",
        ],
        "candidate_cards": candidate_cards,
        "required_json_schema": {
            "nodes": {
                "id81584": {
                    "semantic_relevance_score": 0.72,
                    "audience_fit_score": 0.68,
                    "role_fit_score": 0.74,
                    "narrative_fit_score": 0.61,
                    "risk_conflict_score": 0.15,
                    "novelty_score": 0.44,
                    "semantic_tags": ["国际时事", "供应链"],
                    "reasoning": ["受众覆盖广", "适合首发解释"],
                }
            }
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_selector_system_prompt() -> str:
    return (
        "你是影响力事件分发策略中的候选组合选择助手。"
        "请在给定候选中选择更适合该事件的一组匿名节点。"
        "必须返回严格 JSON。"
        "不能编造不存在的节点。"
    )


def build_selector_user_prompt(
    *,
    event: ParsedEvent,
    max_selected: int,
    shortlist: list[dict[str, Any]],
) -> str:
    payload = {
        "task": "在候选节点中给出主选顺序，兼顾事件适配、角色覆盖、风险控制和语义多样性。",
        "event": {
            "event_id": event.event_id,
            "event_title": event.event_title,
            "target_goal": event.target_goal,
            "event_type": event.event_type,
            "risk_level": event.constraints.risk_level,
            "target_audience": event.target_audience,
            "semantic_tags": event.semantic_tags,
            "narrative_frames": event.narrative_frames,
            "target_roles": event.target_roles,
            "negative_constraints": event.negative_constraints,
        },
        "constraints": {
            "max_selected_nodes": max_selected,
            "preferred_roles": event.dispatch_preferences.preferred_roles or event.target_roles,
        },
        "requirements": [
            "selected_order 只包含候选中的 user_id。",
            "尽量覆盖核心首发、互动承接、扩散等不同角色。",
            "优先选择语义贴合且彼此不过度同质的节点。",
            "理由保持简短，每个节点 1-3 个短语。",
            "recommended_role 只能是 core_publish_node、interaction_response_node、amplification_node、support_node 之一。",
        ],
        "shortlist": shortlist,
        "required_json_schema": {
            "selected_order": [
                {
                    "user_id": "81584",
                    "recommended_role": "core_publish_node",
                    "reasoning": ["适合首发", "主题贴合"],
                }
            ],
            "fallback_order": ["162909", "73085"],
            "global_notes": ["需要角色覆盖", "避免同质化"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
