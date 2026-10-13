from __future__ import annotations

import math
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from .data_loader import DataLoader
from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import (
    EnrichedUserProfile,
    FeatureBuildResult,
    FeatureBuildSummary,
    NodeFeature,
    ParsedEvent,
    ProductContext,
    UserProfile,
)
from .prompts import build_feature_enrichment_system_prompt, build_feature_enrichment_user_prompt

_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


EVENT_KEYWORD_EXPANSIONS = (
    (
        ("\u822a\u8fd0", "\u7ea2\u6d77", "\u62a4\u822a", "\u6d77\u8fd0", "\u5168\u7403\u7269\u6d41", "\u4f9b\u5e94\u94fe"),
        (
            "\u822a\u8fd0",
            "\u6d77\u8fd0",
            "\u7269\u6d41",
            "\u4f9b\u5e94\u94fe",
            "\u56fd\u9645\u8d38\u6613",
            "\u5168\u7403\u5e02\u573a",
            "\u6e2f\u53e3",
            "\u80fd\u6e90",
            "\u56fd\u9645\u5173\u7cfb",
            "\u5730\u7f18\u653f\u6cbb",
            "\u8d38\u6613\u6210\u672c",
        ),
    ),
    (
        ("\u6cb9\u4ef7", "\u80fd\u6e90", "\u539f\u6cb9", "\u5730\u7f18\u98ce\u9669"),
        ("\u80fd\u6e90\u4ef7\u683c", "\u80fd\u6e90", "\u56fd\u9645\u8d38\u6613", "\u4f9b\u5e94\u94fe", "\u5168\u7403\u5e02\u573a", "\u5730\u7f18\u653f\u6cbb"),
    ),
    (
        ("\u6781\u7aef\u5929\u6c14", "\u57ce\u5e02\u5e94\u6025", "\u7535\u529b\u4fdd\u969c", "\u516c\u5171\u5b89\u5168"),
        ("\u57ce\u5e02\u6cbb\u7406", "\u516c\u5171\u5b89\u5168", "\u6c11\u751f", "\u516c\u5171\u670d\u52a1", "\u57fa\u5c42\u670d\u52a1", "\u5e94\u6025", "\u7535\u529b"),
    ),
    (
        ("\u6570\u636e\u6cc4\u9732", "\u7f51\u7edc\u5b89\u5168", "\u5408\u89c4", "\u9690\u79c1"),
        ("\u7f51\u7edc\u5b89\u5168", "\u9690\u79c1\u4fdd\u62a4", "\u5408\u89c4\u98ce\u9669", "\u6cd5\u5f8b\u6cd5\u89c4", "\u4e8b\u5b9e\u6838\u67e5", "\u516c\u5171\u89c4\u5219"),
    ),
)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = _NON_TEXT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, value))


class FeatureBuilder:
    def __init__(self, product_name: str = "abc_reading") -> None:
        self.product_name = product_name

    def build_from_loader(
        self,
        workspace_root: str | Path,
        event: ParsedEvent,
        profile_limit: int | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
    ) -> FeatureBuildResult:
        loader = DataLoader(workspace_root, product_name=self.product_name)
        product_context = loader.load_product_context()
        profiles = loader.load_profiles(limit=profile_limit)
        try:
            enriched_profiles = loader.load_enriched_profiles(limit=profile_limit)
        except (JSONDecodeError, ValueError):
            enriched_profiles = {}
        try:
            source_user_ids = set(
                loader.load_interactions(limit_sources=None, limit_records_per_source=0).keys()
            )
        except (JSONDecodeError, ValueError):
            source_user_ids = set()
        return self.build_features(
            product_context=product_context,
            profiles=profiles,
            event=event,
            enriched_profiles=enriched_profiles,
            source_user_ids=source_user_ids,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=llm_client,
        )

    def build_features(
        self,
        product_context: ProductContext,
        profiles: dict[str, UserProfile],
        event: ParsedEvent,
        enriched_profiles: dict[str, EnrichedUserProfile] | None = None,
        source_user_ids: set[str] | None = None,
        workspace_root: str | Path | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
    ) -> FeatureBuildResult:
        enriched_profiles = enriched_profiles or {}
        source_user_ids = source_user_ids or set()
        user_ids = list(profiles.keys())
        influencer_ids = set(product_context.influencer_ids)

        follower_max_log = self._max_log(
            profiles[user_id].user_followers for user_id in user_ids
        )
        friend_max_log = self._max_log(
            profiles[user_id].user_friends for user_id in user_ids
        )
        neighbor_max_log = self._max_log(
            enriched_profiles.get(user_id, EnrichedUserProfile(
                user_id=user_id,
                user_name=profiles[user_id].user_name,
                user_followers=profiles[user_id].user_followers,
                user_friends=profiles[user_id].user_friends,
                user_interests=profiles[user_id].user_interests,
                user_description=profiles[user_id].user_description,
                platform=profiles[user_id].platform,
            )).graph_attributes.neighbor_count
            for user_id in user_ids
        )
        received_max_log = self._max_log(
            enriched_profiles.get(user_id, EnrichedUserProfile(
                user_id=user_id,
                user_name=profiles[user_id].user_name,
                user_followers=profiles[user_id].user_followers,
                user_friends=profiles[user_id].user_friends,
                user_interests=profiles[user_id].user_interests,
                user_description=profiles[user_id].user_description,
                platform=profiles[user_id].platform,
            )).graph_attributes.received_interaction_count
            for user_id in user_ids
        )
        made_max_log = self._max_log(
            enriched_profiles.get(user_id, EnrichedUserProfile(
                user_id=user_id,
                user_name=profiles[user_id].user_name,
                user_followers=profiles[user_id].user_followers,
                user_friends=profiles[user_id].user_friends,
                user_interests=profiles[user_id].user_interests,
                user_description=profiles[user_id].user_description,
                platform=profiles[user_id].platform,
            )).graph_attributes.made_interaction_count
            for user_id in user_ids
        )

        node_features: list[NodeFeature] = []
        keywords = self._event_match_keywords(event)
        normalized_keywords = self._prepare_keywords(keywords)

        for user_id, profile in profiles.items():
            enriched = enriched_profiles.get(user_id)
            graph_attributes = enriched.graph_attributes if enriched is not None else None

            follower_score = self._log_score(profile.user_followers, follower_max_log)
            friend_score = self._log_score(profile.user_friends, friend_max_log)
            neighbor_count = graph_attributes.neighbor_count if graph_attributes else 0
            mutual_neighbor_count = graph_attributes.mutual_neighbor_count if graph_attributes else 0
            received_interaction_count = graph_attributes.received_interaction_count if graph_attributes else 0
            made_interaction_count = graph_attributes.made_interaction_count if graph_attributes else 0
            received_comment_count = graph_attributes.received_comment_count if graph_attributes else 0
            received_repost_count = graph_attributes.received_repost_count if graph_attributes else 0
            made_comment_count = graph_attributes.made_comment_count if graph_attributes else 0
            made_repost_count = graph_attributes.made_repost_count if graph_attributes else 0
            self_interaction_count = graph_attributes.self_interaction_count if graph_attributes else 0
            has_graph_signal = bool(
                graph_attributes
                and (
                    neighbor_count
                    or received_interaction_count
                    or made_interaction_count
                    or mutual_neighbor_count
                )
            )

            neighbor_score = self._log_score(neighbor_count, neighbor_max_log)
            received_score = self._log_score(received_interaction_count, received_max_log)
            made_score = self._log_score(made_interaction_count, made_max_log)

            total_interaction_count = received_interaction_count + made_interaction_count
            total_comment_count = received_comment_count + made_comment_count
            total_repost_count = received_repost_count + made_repost_count
            comment_ratio = total_comment_count / total_interaction_count if total_interaction_count else 0.0
            repost_ratio = total_repost_count / total_interaction_count if total_interaction_count else 0.0
            mutual_neighbor_ratio = mutual_neighbor_count / neighbor_count if neighbor_count else 0.0

            matched_keywords = self._match_keywords(
                profile=profile,
                normalized_keywords=normalized_keywords,
            )
            semantic_profile = DataLoader.summarize_profile_for_semantics(
                profile,
                enriched_profile=enriched,
                is_interaction_source=user_id in source_user_ids,
            )
            topic_match_score = len(matched_keywords) / len(normalized_keywords) if normalized_keywords else 0.0
            profile_completeness_score = self._profile_completeness_score(profile)
            stability_score = self._stability_score(
                profile_completeness_score=profile_completeness_score,
                total_interaction_count=total_interaction_count,
                self_interaction_count=self_interaction_count,
            )
            if has_graph_signal:
                influence_score = _clip_score(
                    0.45 * follower_score
                    + 0.30 * received_score
                    + 0.15 * neighbor_score
                    + 0.10 * float(user_id in influencer_ids)
                )
                diffusion_score = _clip_score(
                    0.35 * neighbor_score
                    + 0.25 * made_score
                    + 0.25 * mutual_neighbor_ratio
                    + 0.15 * repost_ratio
                )
                activity_score = _clip_score(
                    0.45 * received_score
                    + 0.35 * made_score
                    + 0.10 * comment_ratio
                    + 0.10 * friend_score
                )
            else:
                influence_score = _clip_score(
                    0.75 * follower_score
                    + 0.15 * friend_score
                    + 0.10 * float(user_id in influencer_ids)
                )
                diffusion_score = _clip_score(0.45 * friend_score + 0.35 * follower_score)
                activity_score = _clip_score(0.55 * friend_score + 0.25 * follower_score)
            feature_ready_score = _clip_score(
                0.35 * influence_score
                + 0.25 * diffusion_score
                + 0.25 * topic_match_score
                + 0.15 * stability_score
            )

            role_hint = self._infer_role_hint(
                influence_score=influence_score,
                diffusion_score=diffusion_score,
                activity_score=activity_score,
                topic_match_score=topic_match_score,
                comment_ratio=comment_ratio,
            )

            node_features.append(
                NodeFeature(
                    user_id=user_id,
                    user_name=profile.user_name,
                    event_id=event.event_id,
                    event_type=event.event_type,
                    role_hint=role_hint,
                    platform=profile.platform,
                    influencer_flag=user_id in influencer_ids,
                    is_interaction_source=user_id in source_user_ids,
                    follower_count=profile.user_followers,
                    friend_count=profile.user_friends,
                    interest_count=len(profile.user_interests),
                    has_description=bool(profile.user_description.strip()),
                    neighbor_count=neighbor_count,
                    mutual_neighbor_count=mutual_neighbor_count,
                    received_interaction_count=received_interaction_count,
                    made_interaction_count=made_interaction_count,
                    received_comment_count=received_comment_count,
                    received_repost_count=received_repost_count,
                    made_comment_count=made_comment_count,
                    made_repost_count=made_repost_count,
                    self_interaction_count=self_interaction_count,
                    comment_ratio=round(comment_ratio, 6),
                    repost_ratio=round(repost_ratio, 6),
                    mutual_neighbor_ratio=round(mutual_neighbor_ratio, 6),
                    profile_completeness_score=round(profile_completeness_score, 6),
                    topic_match_score=round(topic_match_score, 6),
                    influence_score=round(influence_score, 6),
                    diffusion_score=round(diffusion_score, 6),
                    activity_score=round(activity_score, 6),
                    stability_score=round(stability_score, 6),
                    feature_ready_score=round(feature_ready_score, 6),
                    semantic_profile=semantic_profile,
                    semantic_tags=list(matched_keywords[:4]),
                    keyword_hit_count=len(matched_keywords),
                    matched_keywords=matched_keywords,
                )
            )

        node_features = self._sort_features(node_features)
        node_features = self._augment_with_llm(
            node_features=node_features,
            event=event,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=llm_client,
        )

        matched_node_count = sum(1 for item in node_features if item.keyword_hit_count > 0)
        summary = FeatureBuildSummary(
            event_id=event.event_id,
            event_type=event.event_type,
            node_count=len(node_features),
            source_node_count=sum(1 for item in node_features if item.is_interaction_source),
            matched_node_count=matched_node_count,
            avg_feature_ready_score=round(
                sum(item.feature_ready_score for item in node_features) / max(len(node_features), 1),
                6,
            ),
            avg_topic_match_score=round(
                sum(item.topic_match_score for item in node_features) / max(len(node_features), 1),
                6,
            ),
        )
        return FeatureBuildResult(
            event=event,
            product_context=product_context,
            summary=summary,
            node_features=node_features,
        )

    def to_frame(self, result: FeatureBuildResult) -> pd.DataFrame:
        rows = [item.model_dump() for item in result.node_features]
        return pd.DataFrame(rows)

    def _prepare_keywords(self, keywords: list[str]) -> list[str]:
        prepared: list[str] = []
        for keyword in keywords:
            normalized = _normalize_text(keyword)
            if not normalized:
                continue
            if normalized not in prepared:
                prepared.append(normalized)
        return prepared

    def _event_match_keywords(self, event: ParsedEvent) -> list[str]:
        generic_tags = {"general_public", "awareness", "engagement", "response", "conversion"}
        audience_keywords = [
            item
            for item in event.target_audience
            if item not in generic_tags
        ]
        semantic_keywords = [
            item
            for item in event.semantic_tags
            if item not in generic_tags
        ]
        keywords = [
            event.event_title,
            *event.extracted_keywords,
            *semantic_keywords,
            *audience_keywords,
        ]
        event_text = _normalize_text(
            " ".join(
                [
                    event.event_title,
                    event.event_description,
                    *event.extracted_keywords,
                    *semantic_keywords,
                ]
            )
        ).replace(" ", "")
        for triggers, expansions in EVENT_KEYWORD_EXPANSIONS:
            if any(trigger in event_text for trigger in triggers):
                keywords.extend(expansions)
        return keywords

    def _match_keywords(self, profile: UserProfile, normalized_keywords: list[str]) -> list[str]:
        if not normalized_keywords:
            return []

        search_text = _normalize_text(
            " ".join([profile.user_description, *profile.user_interests])
        )
        if not search_text:
            return []

        matches: list[str] = []
        tokens = search_text.split()
        compact_text = search_text.replace(" ", "")
        for keyword in normalized_keywords:
            if keyword in compact_text:
                matches.append(keyword)
                continue
            if any(self._token_fuzzy_matches(keyword, token) for token in tokens):
                matches.append(keyword)
        return matches

    def _token_fuzzy_matches(self, keyword: str, token: str) -> bool:
        if not keyword or not token:
            return False
        normalized_keyword = keyword.replace(" ", "")
        normalized_token = token.replace(" ", "")
        if len(normalized_keyword) <= 1 or len(normalized_token) <= 1:
            return False
        if normalized_keyword.isascii() or normalized_token.isascii():
            if len(normalized_keyword) < 4 or len(normalized_token) < 4:
                return False
        else:
            if len(normalized_keyword) < 2 or len(normalized_token) < 2:
                return False
            shorter = min(len(normalized_keyword), len(normalized_token))
            longer = max(len(normalized_keyword), len(normalized_token))
            if longer >= 4 and (shorter < 3 or shorter / longer < 0.50):
                return False
        return fuzz.partial_ratio(keyword, token) >= 90

    def _profile_completeness_score(self, profile: UserProfile) -> float:
        has_description = 1.0 if profile.user_description.strip() else 0.0
        interest_score = min(len(profile.user_interests) / 4, 1.0)
        follower_presence = 1.0 if profile.user_followers > 0 else 0.0
        return _clip_score((has_description + interest_score + follower_presence) / 3)

    def _stability_score(
        self,
        profile_completeness_score: float,
        total_interaction_count: int,
        self_interaction_count: int,
    ) -> float:
        self_loop_ratio = self_interaction_count / total_interaction_count if total_interaction_count else 0.0
        self_loop_penalty = min(self_loop_ratio, 1.0)
        return _clip_score(0.75 * profile_completeness_score + 0.25 * (1 - self_loop_penalty))

    def _infer_role_hint(
        self,
        influence_score: float,
        diffusion_score: float,
        activity_score: float,
        topic_match_score: float,
        comment_ratio: float,
    ) -> str:
        if influence_score >= 0.6 and topic_match_score >= 0.2:
            return "core_broadcast"
        if diffusion_score >= 0.45:
            return "amplification"
        if activity_score >= 0.35 or comment_ratio >= 0.6:
            return "interaction_response"
        return "general_support"

    def _max_log(self, values) -> float:
        max_value = max((math.log1p(max(0, int(value))) for value in values), default=0.0)
        return max(max_value, 1.0)

    def _log_score(self, value: int, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return _clip_score(math.log1p(max(0, int(value))) / denominator)

    def _sort_features(self, node_features: list[NodeFeature]) -> list[NodeFeature]:
        node_features.sort(
            key=lambda item: (
                item.keyword_hit_count > 0,
                item.topic_match_score,
                item.feature_ready_score,
                item.activity_score,
                item.stability_score,
                item.influence_score,
                item.follower_count,
            ),
            reverse=True,
        )
        return node_features

    def _augment_with_llm(
        self,
        *,
        node_features: list[NodeFeature],
        event: ParsedEvent,
        workspace_root: str | Path | None,
        use_llm: bool,
        llm_client: Any | None,
    ) -> list[NodeFeature]:
        if not use_llm or not node_features:
            return node_features

        client = llm_client
        if client is None and workspace_root is not None:
            client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
        if client is None:
            return node_features

        candidate_pool_size = event.dispatch_preferences.candidate_pool_size
        top_candidates = self._candidate_pool_for_llm(
            node_features=node_features,
            candidate_pool_size=candidate_pool_size,
        )
        candidate_cards = [
            {
                "user_id": feature.user_id,
                "user_name": feature.user_name,
                "role_hint": feature.role_hint,
                "semantic_profile": feature.semantic_profile,
                "matched_keywords": feature.matched_keywords,
                "baseline_scores": {
                    "feature_ready_score": feature.feature_ready_score,
                    "topic_match_score": feature.topic_match_score,
                    "influence_score": feature.influence_score,
                    "diffusion_score": feature.diffusion_score,
                    "activity_score": feature.activity_score,
                    "stability_score": feature.stability_score,
                },
                "signals": {
                    "follower_count": feature.follower_count,
                    "friend_count": feature.friend_count,
                    "neighbor_count": feature.neighbor_count,
                    "mutual_neighbor_count": feature.mutual_neighbor_count,
                    "received_interaction_count": feature.received_interaction_count,
                    "made_interaction_count": feature.made_interaction_count,
                    "is_interaction_source": feature.is_interaction_source,
                    "influencer_flag": feature.influencer_flag,
                },
            }
            for feature in top_candidates
        ]

        try:
            response = client.generate_json(
                system_prompt=build_feature_enrichment_system_prompt(),
                user_prompt=build_feature_enrichment_user_prompt(
                    event=event,
                    candidate_cards=candidate_cards,
                ),
            )
        except (LLMClientError, OSError, ValueError, TypeError):
            return node_features

        raw_nodes = self._extract_llm_node_mapping(
            response=response,
            expected_ids=[feature.user_id for feature in top_candidates],
        )
        if not raw_nodes:
            return node_features

        llm_updates: dict[str, dict[str, Any]] = {}
        for feature in top_candidates:
            raw_node = (
                raw_nodes.get(feature.user_id)
                or raw_nodes.get(str(feature.user_id))
                or raw_nodes.get(f"id{feature.user_id}")
                or raw_nodes.get(f"digital_human_{feature.user_id}")
            )
            if not isinstance(raw_node, dict):
                continue

            event_relevance = self._coerce_score(
                self._pick_first(raw_node, "semantic_relevance_score", "event_semantic_relevance_score", "event_relevance_score", "relevance_score")
            )
            audience_fit = self._coerce_score(
                self._pick_first(raw_node, "audience_fit_score", "audience_score")
            )
            role_fit = self._coerce_score(
                self._pick_first(raw_node, "role_fit_score", "role_score")
            )
            narrative_fit = self._coerce_score(
                self._pick_first(raw_node, "narrative_fit_score", "narrative_score")
            )
            risk_conflict = self._coerce_score(
                self._pick_first(raw_node, "risk_conflict_score", "risk_score", "conflict_score")
            )
            novelty = self._coerce_score(
                self._pick_first(raw_node, "novelty_score", "diversity_score")
            )
            llm_feature_score = _clip_score(
                0.35 * event_relevance
                + 0.20 * audience_fit
                + 0.20 * role_fit
                + 0.15 * narrative_fit
                + 0.10 * novelty
            )
            llm_updates[feature.user_id] = {
                "event_semantic_relevance_score": event_relevance,
                "audience_fit_score": audience_fit,
                "role_fit_score": role_fit,
                "narrative_fit_score": narrative_fit,
                "risk_conflict_score": risk_conflict,
                "novelty_score": novelty,
                "llm_feature_score": llm_feature_score,
                "semantic_tags": self._normalize_semantic_tags(
                    self._pick_first(raw_node, "semantic_tags", "tags")
                ) or feature.semantic_tags,
                "candidate_reasoning": self._normalize_reasoning(
                    self._pick_first(raw_node, "reasoning", "reasons", "analysis")
                ),
                "llm_feature_used": True,
            }

        if not llm_updates:
            return node_features

        semantic_weight = max(event.dispatch_preferences.semantic_weight, 0.50)
        base_weight = max(0.0, 1.0 - semantic_weight)
        updated_features: list[NodeFeature] = []
        for feature in node_features:
            update = llm_updates.get(feature.user_id)
            if update is None:
                updated_features.append(feature)
                continue

            augmented_ready_score = _clip_score(
                base_weight * feature.feature_ready_score
                + semantic_weight * update["llm_feature_score"]
            )
            updated_features.append(
                feature.model_copy(
                    update={
                        **update,
                        "feature_ready_score": round(augmented_ready_score, 6),
                    }
                )
            )

        return self._sort_features(updated_features)

    def _coerce_score(self, value: Any) -> float:
        try:
            return _clip_score(float(value))
        except (TypeError, ValueError):
            return 0.0

    def _pick_first(self, payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    def _candidate_pool_for_llm(
        self,
        *,
        node_features: list[NodeFeature],
        candidate_pool_size: int,
    ) -> list[NodeFeature]:
        diversified: list[NodeFeature] = []
        used_ids: set[str] = set()

        def add_bucket(features: list[NodeFeature], *, generic_high_influence_cap: int | None = None) -> None:
            generic_high_influence_count = 0
            for feature in features:
                if len(diversified) >= candidate_pool_size:
                    return
                if feature.user_id in used_ids:
                    continue
                is_generic_high_influence = (
                    feature.keyword_hit_count == 0
                    and feature.topic_match_score == 0.0
                    and feature.influence_score >= 0.65
                )
                if (
                    generic_high_influence_cap is not None
                    and is_generic_high_influence
                    and generic_high_influence_count >= generic_high_influence_cap
                ):
                    continue
                diversified.append(feature)
                used_ids.add(feature.user_id)
                if is_generic_high_influence:
                    generic_high_influence_count += 1

        topic_bucket = sorted(
            node_features,
            key=lambda item: (
                item.keyword_hit_count,
                item.topic_match_score,
                item.stability_score,
                item.activity_score,
            ),
            reverse=True,
        )
        interaction_bucket = sorted(
            [
                item
                for item in node_features
                if item.is_interaction_source or item.role_hint == "interaction_response"
            ],
            key=lambda item: (
                item.activity_score,
                item.diffusion_score,
                item.stability_score,
                item.influence_score,
            ),
            reverse=True,
        )
        diffusion_bucket = sorted(
            node_features,
            key=lambda item: (
                item.diffusion_score,
                item.activity_score,
                item.stability_score,
                item.influence_score,
            ),
            reverse=True,
        )
        baseline_bucket = list(node_features)

        add_bucket(topic_bucket)
        add_bucket(interaction_bucket, generic_high_influence_cap=max(2, candidate_pool_size // 6))
        add_bucket(diffusion_bucket, generic_high_influence_cap=max(2, candidate_pool_size // 6))
        add_bucket(baseline_bucket, generic_high_influence_cap=max(2, candidate_pool_size // 6))
        return diversified[:candidate_pool_size]

    def _extract_llm_node_mapping(
        self,
        *,
        response: dict[str, Any],
        expected_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(response, dict):
            return {}

        alias_keys = ("nodes", "candidates", "results", "items", "data")
        for alias in alias_keys:
            normalized = self._normalize_llm_container(response.get(alias), expected_ids)
            if normalized:
                return normalized

        normalized_response = self._normalize_llm_container(response, expected_ids)
        if normalized_response:
            return normalized_response

        for value in response.values():
            normalized = self._normalize_llm_container(value, expected_ids)
            if normalized:
                return normalized
        return {}

    def _normalize_llm_container(
        self,
        container: Any,
        expected_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        if isinstance(container, list):
            normalized: dict[str, dict[str, Any]] = {}
            for item in container:
                if not isinstance(item, dict):
                    continue
                user_id = str(
                    item.get("user_id")
                    or item.get("id")
                    or item.get("node_id")
                    or item.get("digital_human_id")
                    or ""
                ).strip()
                if user_id:
                    normalized[user_id] = item
            return normalized

        if not isinstance(container, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for expected_id in expected_ids:
            raw_node = (
                container.get(expected_id)
                or container.get(str(expected_id))
                or container.get(f"id{expected_id}")
                or container.get(f"digital_human_{expected_id}")
            )
            if isinstance(raw_node, dict):
                normalized[str(expected_id)] = raw_node

        if normalized:
            return normalized

        for key, value in container.items():
            if not isinstance(value, dict):
                continue
            key_text = str(key).strip()
            candidate_id = key_text.removeprefix("id").removeprefix("digital_human_")
            if candidate_id and candidate_id in expected_ids:
                normalized[candidate_id] = value
        return normalized

    def _normalize_semantic_tags(self, value: Any) -> list[str]:
        if isinstance(value, str):
            value = re.split(r"[,;，；]\s*", value)
        if not isinstance(value, list):
            return []
        tags: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in tags:
                tags.append(text)
        return tags[:4]

    def _normalize_reasoning(self, value: Any) -> list[str]:
        if isinstance(value, str):
            value = re.split(r"[\n,;，；]\s*", value)
        if not isinstance(value, list):
            return []
        reasoning: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in reasoning:
                reasoning.append(text)
        return reasoning[:4]
