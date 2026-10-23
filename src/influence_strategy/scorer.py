from __future__ import annotations

import pandas as pd

from .models import FeatureBuildResult, NodeFeature, NodeScore, ProductContext, ScoreResult, ScoreSummary


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, value))


#: 图结构传播潜力对 diffusion 信号的**增量**上限。
#: 用「加法增量」而不是「加权混合」：没有模型产物时 graph_potential 恒为 0，
#: 加法形式让公式逐字等同于原版，历史结果与既有测试不受任何影响；有产物时
#: 最多再加 0.35，代表"处在互动密集社群中心"带来的额外传播能力。
DIFFUSION_GRAPH_BONUS = 0.35

#: 本地语义**分位数**门槛（0~1，1 表示最相关）。
#:
#: 为什么用分位数而不是余弦绝对值：BGE 的余弦 p50≈0.30、p95≈0.43，而 SVD 的
#: p95 才≈0.11 —— 同一个绝对阈值（如 0.12）在 BGE 下会放行 100% 的节点、在 SVD
#: 下只放行 5%，完全不可移植。分位数则恒定表示「是否进入本次事件的前 N%」：
#: low 放行约 10%（3.5 万节点即 3500+ 候选，足够选人）、medium 5%、high 2%，
#: 风险越高要求语义越靠前。
_LOCAL_SEMANTIC_THRESHOLD = {
    "low": 0.90,
    "medium": 0.95,
    "high": 0.98,
}


class Scorer:
    def score(self, feature_result: FeatureBuildResult) -> ScoreResult:
        risk_limit = self._risk_limit(feature_result.event.constraints.risk_level)
        risk_penalty_factor = self._risk_penalty_factor(feature_result.event.constraints.risk_level)
        dispatch_weights = self._dispatch_weights(feature_result.event)

        node_scores: list[NodeScore] = []
        for feature in feature_result.node_features:
            risk_flags, risk_score = self._compute_risk(feature)
            semantic_gate = self._semantic_gate(feature_result.event.constraints.risk_level, feature)
            # diffusion 信号 = 节点自身互动表现 + 图结构传播潜力（增量）。
            # 后者由 GraphFeatureBuilder 预计算（SGC 邻居传播 + PageRank 等），
            # 补上"粉丝不多但处在互动密集社群中心"这类单看计数拿不到的信息。
            diffusion_signal = _clip_score(
                feature.diffusion_score + DIFFUSION_GRAPH_BONUS * feature.graph_potential
            )
            raw_dispatch_score = _clip_score(
                dispatch_weights["influence"] * feature.influence_score
                + dispatch_weights["diffusion"] * diffusion_signal
                + dispatch_weights["topic"] * feature.topic_match_score
                + dispatch_weights["stability"] * feature.stability_score
                # 语义项改用本地模型分：LLM 关闭后 llm_feature_score 恒为 0，
                # 而这部分权重最高可占 0.45，空着等于整条打分链路丢掉近半信号。
                + dispatch_weights["semantic"] * feature.local_semantic_score
            )
            final_score = _clip_score(raw_dispatch_score * (1 - risk_penalty_factor * risk_score))
            risk_level = self._risk_level(risk_score)
            manual_review_required = (
                risk_score >= 0.75
                or (feature_result.event.constraints.risk_level == "low" and risk_score > 0.35)
            )
            eligible = (
                risk_score <= risk_limit
                and feature.stability_score >= 0.20
                and semantic_gate
                and self._has_semantic_support(
                    risk_level=feature_result.event.constraints.risk_level,
                    feature=feature,
                )
            )
            priority_tier = self._priority_tier(final_score)
            selection_reasons = self._selection_reasons(
                feature=feature,
                risk_flags=risk_flags,
                final_score=final_score,
            )

            # 原写法 ``NodeScore(**feature.model_dump(), ...)`` 会对每个节点做一次
            # Pydantic 序列化 + 一次全字段校验，35k 节点实测 1.16s/轮（占 scorer
            # 总耗时约 2/3）。这里沿用 ``FeatureBuilder.build_features`` 已验证的
            # 范式：接管 __dict__ 并手工补齐 Pydantic 2 的内部槽位，完全跳过校验。
            # 输出与旧写法逐字节一致（model_dump / model_dump_json /
            # exclude_unset / model_fields_set / __dict__ 已全量比对验证）。
            node_score = NodeScore.__new__(NodeScore)
            node_score.__dict__.update(feature.__dict__)
            node_score.__dict__.update(
                {
                    "raw_dispatch_score": round(raw_dispatch_score, 6),
                    "risk_score": round(risk_score, 6),
                    "risk_level": risk_level,
                    "final_score": round(final_score, 6),
                    "eligible": eligible,
                    "manual_review_required": manual_review_required,
                    "priority_tier": priority_tier,
                    "risk_flags": risk_flags,
                    "selection_reasons": selection_reasons,
                }
            )
            # 缺这些内部槽位时，下游 model_copy / model_dump(exclude_unset) /
            # 重新校验都会抛错或行为不一致。
            object.__setattr__(
                node_score, "__pydantic_fields_set__", set(node_score.__dict__.keys())
            )
            object.__setattr__(node_score, "__pydantic_extra__", None)
            object.__setattr__(node_score, "__pydantic_private__", None)
            node_scores.append(node_score)

        node_scores.sort(
            key=lambda item: (
                item.eligible,
                item.final_score,
                item.topic_match_score,
                item.influence_score,
                -item.risk_score,
            ),
            reverse=True,
        )

        summary = ScoreSummary(
            event_id=feature_result.event.event_id,
            event_type=feature_result.event.event_type,
            node_count=len(node_scores),
            eligible_count=sum(1 for item in node_scores if item.eligible),
            manual_review_count=sum(1 for item in node_scores if item.manual_review_required),
            high_priority_count=sum(1 for item in node_scores if item.priority_tier == "high"),
            avg_final_score=round(
                sum(item.final_score for item in node_scores) / max(len(node_scores), 1),
                6,
            ),
            avg_risk_score=round(
                sum(item.risk_score for item in node_scores) / max(len(node_scores), 1),
                6,
            ),
        )
        return ScoreResult(
            event=feature_result.event,
            product_context=feature_result.product_context,
            summary=summary,
            node_scores=node_scores,
        )

    def to_frame(self, score_result: ScoreResult) -> pd.DataFrame:
        return pd.DataFrame(item.model_dump() for item in score_result.node_scores)

    def _compute_risk(self, feature: NodeFeature) -> tuple[list[str], float]:
        total_interactions = feature.received_interaction_count + feature.made_interaction_count
        self_loop_ratio = feature.self_interaction_count / total_interactions if total_interactions else 0.0

        risk_components: list[float] = []
        risk_flags: list[str] = []

        profile_sparse_component = (1 - feature.profile_completeness_score) * 0.35
        if profile_sparse_component > 0.10:
            risk_flags.append("profile_sparse")
        risk_components.append(profile_sparse_component)

        if feature.neighbor_count == 0:
            risk_flags.append("isolated_node")
            risk_components.append(0.20)

        if self_loop_ratio > 0.10:
            risk_flags.append("self_interaction_high")
            risk_components.append(min(self_loop_ratio, 1.0) * 0.35)

        if feature.topic_match_score == 0.0:
            risk_flags.append("topic_mismatch")
            risk_components.append(0.18)
        elif feature.topic_match_score < 0.10:
            risk_flags.append("topic_match_weak")
            risk_components.append(0.08)

        if (
            feature.influence_score >= 0.65
            and feature.topic_match_score == 0.0
            and feature.event_semantic_relevance_score < 0.20
        ):
            risk_flags.append("generic_high_influence_mismatch")
            risk_components.append(0.22)

        if feature.llm_feature_used and feature.event_semantic_relevance_score < 0.30:
            risk_flags.append("event_semantic_mismatch")
            mismatch_component = 0.24 if feature.topic_match_score < 0.25 else 0.16
            if feature.influence_score >= 0.60:
                mismatch_component += 0.08
            risk_components.append(mismatch_component)

        if feature.risk_conflict_score >= 0.25:
            risk_flags.append("llm_risk_conflict")
            risk_components.append(min(feature.risk_conflict_score, 1.0) * 0.25)

        if total_interactions > 0 and feature.mutual_neighbor_ratio < 0.05 and feature.diffusion_score > 0.40:
            risk_flags.append("interaction_concentration")
            risk_components.append(0.12)

        if not feature.has_description and feature.interest_count == 0:
            risk_flags.append("profile_context_missing")
            risk_components.append(0.15)

        risk_score = _clip_score(sum(risk_components))
        deduped_flags: list[str] = []
        for flag in risk_flags:
            if flag not in deduped_flags:
                deduped_flags.append(flag)
        return deduped_flags, risk_score

    def _selection_reasons(
        self,
        feature: NodeFeature,
        risk_flags: list[str],
        final_score: float,
    ) -> list[str]:
        reasons: list[str] = []
        if feature.influence_score >= 0.60:
            reasons.append("high_influence")
        if feature.diffusion_score >= 0.45:
            reasons.append("strong_diffusion")
        if feature.topic_match_score >= 0.20:
            reasons.append("good_topic_match")
        elif feature.topic_match_score > 0.0:
            reasons.append("some_topic_match")
        if feature.llm_feature_score >= 0.55:
            reasons.append("strong_semantic_fit")
        elif feature.llm_feature_score >= 0.30:
            reasons.append("some_semantic_fit")
        if feature.audience_fit_score >= 0.55:
            reasons.append("audience_fit")
        if feature.role_fit_score >= 0.55:
            reasons.append("role_fit")
        if feature.stability_score >= 0.70:
            reasons.append("stable_profile")
        if feature.influencer_flag:
            reasons.append("known_influencer")
        if feature.semantic_tags:
            reasons.append(f"semantic_tags={','.join(feature.semantic_tags[:3])}")
        reasons.append(f"role_hint={feature.role_hint}")
        reasons.append(f"final_score={final_score:.3f}")
        if risk_flags:
            reasons.append(f"risk_flags={','.join(risk_flags)}")
        return reasons

    def _risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.75:
            return "high"
        if risk_score >= 0.40:
            return "medium"
        return "low"

    def _priority_tier(self, final_score: float) -> str:
        if final_score >= 0.55:
            return "high"
        if final_score >= 0.35:
            return "medium"
        return "low"

    def _risk_limit(self, risk_level: str) -> float:
        mapping = {
            "low": 0.35,
            "medium": 0.55,
            "high": 0.75,
        }
        return mapping.get(risk_level, 0.55)

    def _risk_penalty_factor(self, risk_level: str) -> float:
        mapping = {
            "low": 0.65,
            "medium": 0.50,
            "high": 0.35,
        }
        return mapping.get(risk_level, 0.50)

    def _semantic_gate(self, risk_level: str, feature: NodeFeature) -> bool:
        # 本地语义分是关键词/话题字面匹配之外的新通道：换了说法描述同一话题
        # （「亲子共读」vs「陪孩子读绘本」）时字面命中可能为 0，但语义向量仍然接近。
        # 这正是原来由 LLM 判断、现在交给本地模型承担的部分。
        local_match = feature.local_semantic_rank >= _LOCAL_SEMANTIC_THRESHOLD.get(
            risk_level, 0.95
        )
        base_match = (
            feature.keyword_hit_count > 0
            or feature.topic_match_score > 0.0
            or local_match
        )
        if not feature.llm_feature_used:
            return base_match
        if risk_level == "high":
            if feature.event_semantic_relevance_score < 0.30 and feature.llm_feature_score < 0.50:
                return False
            return (
                feature.event_semantic_relevance_score >= 0.45
                or feature.llm_feature_score >= 0.55
                or feature.topic_match_score >= 0.30
            )
        if risk_level == "medium":
            return (
                feature.event_semantic_relevance_score >= 0.28
                or feature.topic_match_score >= 0.15
                or feature.llm_feature_score >= 0.35
            )
        return (
            feature.event_semantic_relevance_score >= 0.18
            or feature.topic_match_score > 0.0
            or feature.llm_feature_score >= 0.25
        )

    def _has_semantic_support(self, *, risk_level: str, feature: NodeFeature) -> bool:
        # 每个分支都补一条本地语义通道：字面命中不足时，语义排名足够靠前也算有支撑。
        local = feature.local_semantic_rank
        if risk_level == "high":
            if feature.llm_feature_used:
                return (
                    feature.event_semantic_relevance_score >= 0.35
                    or feature.llm_feature_score >= 0.50
                    or feature.topic_match_score >= 0.30
                )
            return (
                feature.keyword_hit_count >= 2
                or feature.topic_match_score >= 0.18
                or local >= _LOCAL_SEMANTIC_THRESHOLD["high"]
            )
        if risk_level == "medium":
            return (
                feature.event_semantic_relevance_score >= 0.25
                or feature.llm_feature_score >= 0.20
                or feature.keyword_hit_count >= 1
                or feature.topic_match_score >= 0.08
                or local >= _LOCAL_SEMANTIC_THRESHOLD["medium"]
            )
        return (
            feature.event_semantic_relevance_score >= 0.18
            or feature.llm_feature_score >= 0.15
            or feature.keyword_hit_count >= 1
            or feature.topic_match_score > 0.0
            or local >= _LOCAL_SEMANTIC_THRESHOLD["low"]
        )

    def _dispatch_weights(self, event) -> dict[str, float]:
        semantic_weight = float(event.dispatch_preferences.semantic_weight)
        risk_level = event.constraints.risk_level
        target_goal = event.target_goal

        influence = 0.18
        diffusion = 0.17
        topic = 0.24
        stability = 0.21

        if risk_level == "high":
            influence = 0.10
            diffusion = 0.10
            topic = 0.24
            stability = 0.26
        elif target_goal == "engagement":
            influence = 0.14
            diffusion = 0.20
            topic = 0.20
            stability = 0.16
        elif target_goal == "response":
            influence = 0.12
            diffusion = 0.10
            topic = 0.24
            stability = 0.24

        semantic = min(max(semantic_weight, 0.0), 0.45)
        remaining = max(0.55, 1.0 - semantic)
        base_total = influence + diffusion + topic + stability
        scale = remaining / base_total if base_total else 1.0
        return {
            "influence": influence * scale,
            "diffusion": diffusion * scale,
            "topic": topic * scale,
            "stability": stability * scale,
            "semantic": semantic,
        }
