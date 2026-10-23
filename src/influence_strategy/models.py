from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProductContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_name: str
    domain: str = "unknown"
    ads: str = ""
    influencer_ids: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    user_name: str
    user_followers: int = 0
    user_friends: int = 0
    user_interests: list[str] = Field(default_factory=list)
    user_description: str = ""
    # 数字人所属模拟平台（facebook_simulated / instagram_simulated / twitter_simulated）。
    # 空字符串表示旧数据集未提供平台归属，选人时应视为"平台未知"。
    platform: str = ""


class GraphAttributes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    neighbor_count: int = 0
    engaged_by_neighbor_count: int = 0
    engaged_to_neighbor_count: int = 0
    mutual_neighbor_count: int = 0
    self_interaction_count: int = 0
    self_comment_count: int = 0
    self_repost_count: int = 0
    received_interaction_count: int = 0
    received_comment_count: int = 0
    received_repost_count: int = 0
    made_interaction_count: int = 0
    made_comment_count: int = 0
    made_repost_count: int = 0
    isolated: bool = True


class NeighborRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    neighbor_id: str
    relation: Literal["engaged_by", "engaged_to", "mutual", "isolated"]
    received_comment_count: int = 0
    received_repost_count: int = 0
    made_comment_count: int = 0
    made_repost_count: int = 0
    received_interaction_count: int = 0
    made_interaction_count: int = 0
    total_interaction_count: int = 0


class EnrichedUserProfile(UserProfile):
    graph_attributes: GraphAttributes = Field(default_factory=GraphAttributes)
    neighbors: list[NeighborRelation] = Field(default_factory=list)


class InteractionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_user_id: str
    target_user_id: str
    interact_type: str
    text_raw: str = ""
    text_comment: str = ""


class DatasetSummary(BaseModel):
    product_name: str
    profile_count: int
    interaction_source_count: int
    interaction_record_count: int
    enriched_profile_count: int | None = None


class DatasetBundle(BaseModel):
    product_context: ProductContext
    profiles: dict[str, UserProfile]
    interactions: dict[str, list[InteractionRecord]]
    enriched_profiles: dict[str, EnrichedUserProfile] | None = None
    summary: DatasetSummary


class EventConstraints(BaseModel):
    risk_level: Literal["low", "medium", "high"] = "medium"
    max_selected_nodes: int = Field(default=10, ge=1, le=100)
    max_frequency_per_day: int = Field(default=3, ge=1, le=24)
    campaign_window_hours: int = Field(default=24, ge=1, le=168)
    allowed_platforms: list[str] = Field(
        default_factory=lambda: [
            "facebook_simulated",
            "instagram_simulated",
            "twitter_simulated",
        ]
    )


class EventDispatchPreferences(BaseModel):
    candidate_pool_size: int = Field(default=24, ge=6, le=120)
    rerank_top_k: int = Field(default=12, ge=3, le=40)
    semantic_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    risk_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    preferred_roles: list[str] = Field(default_factory=list)
    preferred_narrative: list[str] = Field(default_factory=list)


class ParsedEvent(BaseModel):
    event_id: str
    product_name: str = "abc_reading"
    event_title: str
    event_description: str
    target_goal: str
    event_type: str
    target_audience: list[str] = Field(default_factory=list)
    extracted_keywords: list[str] = Field(default_factory=list)
    constraints: EventConstraints = Field(default_factory=EventConstraints)
    semantic_tags: list[str] = Field(default_factory=list)
    narrative_frames: list[str] = Field(default_factory=list)
    sensitive_entities: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    dispatch_preferences: EventDispatchPreferences = Field(default_factory=EventDispatchPreferences)
    parser_name: str = "rule_based_v1"
    reasoning: list[str] = Field(default_factory=list)
    llm_metadata: dict[str, Any] = Field(default_factory=dict)


class NodeFeature(BaseModel):
    user_id: str
    user_name: str
    event_id: str
    event_type: str
    role_hint: str
    # 数字人所属模拟平台（来自 UserProfile.platform，随特征传递到选人与分发阶段）。
    platform: str = ""
    influencer_flag: bool = False
    is_interaction_source: bool = False
    follower_count: int = 0
    friend_count: int = 0
    interest_count: int = 0
    has_description: bool = False
    neighbor_count: int = 0
    mutual_neighbor_count: int = 0
    received_interaction_count: int = 0
    made_interaction_count: int = 0
    received_comment_count: int = 0
    received_repost_count: int = 0
    made_comment_count: int = 0
    made_repost_count: int = 0
    self_interaction_count: int = 0
    comment_ratio: float = 0.0
    repost_ratio: float = 0.0
    mutual_neighbor_ratio: float = 0.0
    profile_completeness_score: float = 0.0
    topic_match_score: float = 0.0
    influence_score: float = 0.0
    diffusion_score: float = 0.0
    activity_score: float = 0.0
    stability_score: float = 0.0
    feature_ready_score: float = 0.0
    semantic_profile: str = ""
    semantic_tags: list[str] = Field(default_factory=list)
    event_semantic_relevance_score: float = 0.0
    audience_fit_score: float = 0.0
    role_fit_score: float = 0.0
    narrative_fit_score: float = 0.0
    risk_conflict_score: float = 0.0
    novelty_score: float = 0.0
    llm_feature_score: float = 0.0
    llm_feature_used: bool = False
    keyword_hit_count: int = 0
    matched_keywords: list[str] = Field(default_factory=list)
    candidate_reasoning: list[str] = Field(default_factory=list)


class FeatureBuildSummary(BaseModel):
    event_id: str
    event_type: str
    node_count: int
    source_node_count: int
    matched_node_count: int
    avg_feature_ready_score: float
    avg_topic_match_score: float
    # ---- 召回层审计字段（FeatureBuilder 前置召回） ----
    # 口径区分（展示/报告必须靠这两个字段，不能只看 node_count）：
    #   node_count            = 本 result 里实际参与精算的人数（召回时为 200）
    #   total_candidate_count = 召回前的全量候选人数（35,636），恒为全量口径
    #   recall_size / recall_mode = None 表示关闭召回，此时两者相等。
    # 5 个 log_score 的分母（follower/friend/neighbor/received/made max_log）
    # **始终**取自 total_candidate_count 的全量集合，
    # 保证召回子集上的 score 与全量逐位相等。
    recall_size: int | None = None
    recall_mode: Literal["platform_aware"] | None = None
    total_candidate_count: int | None = None


class FeatureBuildResult(BaseModel):
    event: ParsedEvent
    product_context: ProductContext
    summary: FeatureBuildSummary
    node_features: list[NodeFeature]


class NodeScore(NodeFeature):
    raw_dispatch_score: float = 0.0
    risk_score: float = 0.0
    risk_level: Literal["low", "medium", "high"] = "low"
    final_score: float = 0.0
    eligible: bool = False
    manual_review_required: bool = False
    priority_tier: Literal["high", "medium", "low"] = "low"
    risk_flags: list[str] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)


class ScoreSummary(BaseModel):
    event_id: str
    event_type: str
    node_count: int
    eligible_count: int
    manual_review_count: int
    high_priority_count: int
    avg_final_score: float
    avg_risk_score: float


class ScoreResult(BaseModel):
    event: ParsedEvent
    product_context: ProductContext
    summary: ScoreSummary
    node_scores: list[NodeScore]


class SelectedNode(NodeScore):
    selection_rank: int = 0
    selected_role: str = ""
    dispatch_stage: str = ""
    dispatch_priority: str = "p3"
    selection_bucket: Literal["primary", "fallback"] = "primary"


class SelectionSummary(BaseModel):
    event_id: str
    event_type: str
    max_selected_nodes: int
    selected_count: int
    fallback_count: int
    selected_role_distribution: dict[str, int] = Field(default_factory=dict)
    selected_stage_distribution: dict[str, int] = Field(default_factory=dict)
    avg_selected_final_score: float = 0.0
    selection_mode: Literal["automatic", "targeted"] = "automatic"
    # 多人分发指定的平台（multi_user 模式下候选池只保留该平台的数字人）。
    dispatch_platform: str | None = None
    target_user_id: str | None = None
    target_role: str | None = None
    # 单人分发目标数字人所属平台（targeted 模式下直接取自目标画像的 platform 字段）。
    target_platform: str | None = None
    target_final_score: float | None = None
    target_topic_match_score: float | None = None
    target_warning: list[str] = Field(default_factory=list)


class SelectionResult(BaseModel):
    event: ParsedEvent
    product_context: ProductContext
    summary: SelectionSummary
    selected_nodes: list[SelectedNode]
    fallback_nodes: list[SelectedNode]


class StrategyNodePlan(BaseModel):
    user_id: str
    user_name: str
    # 该数字人所属模拟平台（来自 NodeScore.platform）。
    platform: str = ""
    selection_rank: int = 0
    selection_bucket: Literal["primary", "fallback"] = "primary"
    selected_role: str
    dispatch_stage: str
    dispatch_priority: str = "p3"
    final_score: float = 0.0
    risk_level: Literal["low", "medium", "high"] = "low"
    manual_review_required: bool = False
    follower_count: int = 0
    friend_count: int = 0
    neighbor_count: int = 0
    mutual_neighbor_count: int = 0
    received_interaction_count: int = 0
    made_interaction_count: int = 0
    influence_score: float = 0.0
    diffusion_score: float = 0.0
    topic_match_score: float = 0.0
    stability_score: float = 0.0
    llm_feature_score: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    semantic_tags: list[str] = Field(default_factory=list)
    timing_window: str = ""
    frequency_per_day: int = Field(default=1, ge=1, le=24)
    recommended_action: str = ""
    suggested_content_style: str = ""
    rationale: list[str] = Field(default_factory=list)


class StrategyStagePlan(BaseModel):
    stage_name: str
    stage_label: str
    time_window: str
    objective: str
    node_count: int = 0
    node_ids: list[str] = Field(default_factory=list)
    selected_roles: list[str] = Field(default_factory=list)
    content_focus: str = ""
    success_signal: str = ""


class StrategyFrequencyPlan(BaseModel):
    global_cap_per_day: int = 3
    core_publish_node_per_day: int = 1
    interaction_response_node_per_day: int = 2
    amplification_node_per_day: int = 1
    support_node_per_day: int = 1
    notes: list[str] = Field(default_factory=list)


class StrategyPlatformPlan(BaseModel):
    primary_platform: str = "facebook_simulated"
    secondary_platforms: list[str] = Field(default_factory=list)
    execution_mode: str = "single_platform_simulated"
    coordination_notes: list[str] = Field(default_factory=list)


class StrategyContentPlan(BaseModel):
    core_publish_node: str = ""
    interaction_response_node: str = ""
    amplification_node: str = ""
    support_node: str = ""
    general_guardrails: list[str] = Field(default_factory=list)


class StrategyRiskControl(BaseModel):
    risk_level: Literal["low", "medium", "high"] = "medium"
    review_required: bool = False
    manual_review_node_ids: list[str] = Field(default_factory=list)
    fallback_trigger: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DispatchStrategy(BaseModel):
    target_object: list[str] = Field(default_factory=list)
    objective: str = ""
    time_plan: dict[str, str] = Field(default_factory=dict)
    frequency_plan: StrategyFrequencyPlan = Field(default_factory=StrategyFrequencyPlan)
    platform_plan: StrategyPlatformPlan = Field(default_factory=StrategyPlatformPlan)
    content_plan: StrategyContentPlan = Field(default_factory=StrategyContentPlan)
    risk_control: StrategyRiskControl = Field(default_factory=StrategyRiskControl)
    explainability: list[str] = Field(default_factory=list)


class StrategySummary(BaseModel):
    event_id: str
    event_type: str
    selected_count: int
    fallback_count: int
    primary_platform: str
    estimated_total_dispatches: int = 0
    review_required: bool = False
    avg_selected_final_score: float = 0.0
    # 策略整体状态：
    #   * ``ok``                 主选 + 备选都齐，无需人工
    #   * ``manual_required``    主选存在但需要人工复核（manual_review / 高风险）
    #   * ``no_eligible_nodes``  没有主选，需要人工挑选首发账号
    status: Literal["ok", "manual_required", "no_eligible_nodes"] = "ok"
    status_reason: str = ""


class StrategyResult(BaseModel):
    event: ParsedEvent
    product_context: ProductContext
    selection_summary: SelectionSummary
    summary: StrategySummary
    stage_plans: list[StrategyStagePlan]
    selected_nodes: list[StrategyNodePlan]
    fallback_nodes: list[StrategyNodePlan]
    strategy: DispatchStrategy
