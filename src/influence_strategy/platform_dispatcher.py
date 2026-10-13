"""Multi-platform parallel dispatch.

Design notes (why this module exists and what it deliberately avoids)
--------------------------------------------------------------------
*The existing ``Selector`` cannot serve per-platform head-counts.*
``Selector.select`` reads ``score_result.event.constraints.max_selected_nodes``
and truncates every platform to that single global number. ``Selector`` also
applies a *single* role vocabulary (``core_publish_node`` /
``interaction_response_node`` / ``amplification_node`` / ``support_node``).
Reusing it for "Facebook 3 / Twitter 5 / Instagram 2 with platform-specific
roles" would require mutating the shared ``ScoreResult``/``Event`` between
calls. Therefore this dispatcher implements a **lightweight, transparent
per-platform selection** that only *reads* ``ScoreResult`` and reuses the
existing normalised ``NodeScore`` fields and risk flags. ``Selector``,
``Scorer`` and ``StrategyGenerator`` are not modified.

*Scores.* ``Scorer._clip_score`` normalises ``final_score``,
``topic_match_score``, ``influence_score``, ``diffusion_score``,
``stability_score`` and ``activity_score`` to ``[0, 1]``. Because
``platform_fit_score`` is also produced in ``[0, 1]``, blending it with
``final_score`` is a **convex combination of two same-scale values** – not a
raw addition of differently-scaled numbers. The blend weights live in
``PlatformProfile.dispatch_score_weights`` and are documented per platform.

*Safety.* ``platform_fit_score`` / ``platform_dispatch_score`` are *ranking*
scores only. ``eligible``, ``risk_level``, ``manual_review_required`` and
``risk_flags`` are copied verbatim from ``NodeScore`` and are never rewritten.
Every assignment carries an explicit ``safety_tier`` so a non-eligible or
review-required node can never be mistaken for a safety-approved one.

*Time plan.* ``campaign_window_hours`` is a **hard upper bound** for every
stage: ``0 <= start_hour <= end_hour <= campaign_window_hours``. Stage
boundaries are derived from ``PlatformTimeStageSpec.ratio_start/ratio_end``, so
a short window rounds two boundaries onto the same hour. Each stage keeps its
``>= _STAGE_MIN_HOURS`` duration only while the window can afford it; the plan
never extends the activity window to satisfy that preference. Overlapping and
zero-length stages are therefore an accepted degradation for very short
windows, and ``PlatformTimePlan.stage_schedule_note`` states this explicitly.
``PlatformTimePlan`` validates the bound, so a future blueprint change cannot
silently produce an out-of-window plan.

*No fabrication.* Every platform is a **simulated** platform. Each digital
human belongs to exactly one simulated platform via ``NodeScore.platform``
(derived from the dataset ``platform`` field), and a platform can only draw
from the digital humans that already belong to it. No fake follower counts or
account identifiers are produced. See ``policy_disclaimer`` /
``fit_limitations``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import NodeScore, ScoreResult
from .platform_profiles import (
    PlatformProfile,
    PlatformTimeStageSpec,
    UnsupportedPlatformError,
    get_platform_profile,
    list_supported_platforms,
    normalize_platform_key,
)

# Normalised NodeScore signals used to estimate platform fit.
_FIT_SIGNAL_FIELDS: tuple[str, ...] = (
    "topic_match_score",
    "stability_score",
    "influence_score",
    "diffusion_score",
    "activity_score",
)

_SAFETY_TIER_ORDER: dict[str, int] = {
    "safe": 0,
    "review_required": 1,
    "not_eligible": 2,
}

_FIT_METHOD = (
    "platform_fit_score = Σ(signal_weight × normalised_signal) + "
    "role_weight × role_hint_affinity，全部输入均为 [0,1] 归一化值；"
    "platform_dispatch_score = w_final × final_score + w_fit × platform_fit_score（凸组合）。"
)

# 每个时间阶段「希望」达到的最短时长（小时）。
# 这只是偏好，不是硬约束：当活动窗口短到无法让所有阶段各自达到该时长时，
# 该偏好会被放弃（阶段发生重叠甚至退化为零长度），而不会延长活动窗口。
_STAGE_MIN_HOURS = 1

STAGE_SCHEDULE_NOTE = (
    "时间阶段边界由 time_blueprint 的比例 × 活动窗口小时数四舍五入得到，"
    "因此阶段只是内容节奏建议，并不保证各自独占一段完整时段。"
    "窗口足够长时每个阶段至少 1 小时并按顺序覆盖整个窗口；"
    "窗口过短时（例如 1 小时）比例取整后阶段会重叠或退化为零长度，"
    "这是刻意保留的降级行为——不会为了让每个阶段占满 1 小时而延长活动窗口。"
    "任何阶段都满足 0 <= start_hour <= end_hour <= campaign_window_hours，"
    "不会出现负时长，也不会超出活动窗口。"
)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN guard
        return default
    return number


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _render(template: str, context: dict[str, str]) -> str:
    """Substitute ``{token}`` placeholders without ever raising on unknown keys."""

    rendered = template
    for key, value in context.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _count_safety_tiers(
    assignments: Sequence["PlatformNodeAssignment"],
) -> SafetyTierCounts:
    """Count selected nodes per safety tier (single source of truth)."""

    counts = Counter(item.safety_tier for item in assignments)
    return SafetyTierCounts(
        safe=counts.get("safe", 0),
        review_required=counts.get("review_required", 0),
        not_eligible=counts.get("not_eligible", 0),
    )


class PlatformSpec(BaseModel):
    """Caller-provided request for one platform."""

    model_config = ConfigDict(extra="forbid")

    platform_key: str
    requested_count: int = Field(ge=1)


class SafetyTierCounts(BaseModel):
    """How many selected nodes fall into each safety tier.

    ``shortfall_count`` (requested - selected) answers *"did we get enough
    people?"*. It does **not** answer *"were they qualified?"*. When the
    qualified pool is smaller than ``requested_count`` the dispatcher fills the
    gap with ``not_eligible`` nodes while preserving their risk state. These
    counters plus ``safe_shortfall_count`` make that gap explicit so a
    non-eligible filler can never be read as a qualified node.

    Tiers:

    * ``safe``           – ``eligible=true`` 且 ``manual_review_required=false``
    * ``review_required``– ``eligible=true`` 但 ``manual_review_required=true``
    * ``not_eligible``   – ``eligible=false``（含 ``risk_level=high``）
    """

    model_config = ConfigDict(extra="forbid")

    safe: int = 0
    review_required: int = 0
    not_eligible: int = 0


class PlatformContentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_key: str
    label: str
    guidance: str


class PlatformContentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform_key: str
    content_modes: list[str] = Field(default_factory=list)
    sections: list[PlatformContentSection] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)


class PlatformTimeStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_key: str
    label: str
    window: str
    # 机读边界：保证 0 <= start_hour <= end_hour <= campaign_window_hours。
    # window 字符串与这两个字段同源，保留字符串是为了不破坏既有输出契约。
    start_hour: int = 0
    end_hour: int = 0
    focus: str


class PlatformTimePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform_key: str
    campaign_window_hours: int
    stages: list[PlatformTimeStage] = Field(default_factory=list)
    cadence_notes: list[str] = Field(default_factory=list)
    # 明确说明阶段语义，避免把 stage 误读为「各自独占的完整时段」。
    stage_schedule_note: str = STAGE_SCHEDULE_NOTE

    @model_validator(mode="after")
    def _validate_stages_fit_window(self) -> "PlatformTimePlan":
        """所有阶段必须落在 ``[0, campaign_window_hours]`` 之内。

        这是输出契约的硬约束：即使将来 ``time_blueprint`` 的比例变化，
        也不会静默产出超出活动窗口的时间计划。
        """

        for stage in self.stages:
            if stage.start_hour < 0:
                raise ValueError(
                    f"平台 '{self.platform_key}' 阶段 '{stage.stage_key}' 的 "
                    f"start_hour={stage.start_hour} 不能为负数"
                )
            if stage.end_hour < stage.start_hour:
                raise ValueError(
                    f"平台 '{self.platform_key}' 阶段 '{stage.stage_key}' 出现负时长："
                    f"start_hour={stage.start_hour} > end_hour={stage.end_hour}"
                )
            if stage.end_hour > self.campaign_window_hours:
                raise ValueError(
                    f"平台 '{self.platform_key}' 阶段 '{stage.stage_key}' 的 "
                    f"end_hour={stage.end_hour} 超出活动窗口 "
                    f"{self.campaign_window_hours} 小时"
                )
        return self


class PlatformNodeAssignment(BaseModel):
    """One selected digital-human node, scoped to a single platform."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    user_name: str
    # 该数字人所属模拟平台（来自 NodeScore.platform，与所在平台一致）。
    platform: str = ""
    selection_rank: int
    platform_role: str
    source_role_hint: str
    safety_tier: Literal["safe", "review_required", "not_eligible"]

    # base scores copied verbatim from NodeScore (never overwritten)
    final_score: float
    topic_match_score: float
    influence_score: float
    diffusion_score: float
    stability_score: float
    activity_score: float

    # platform-specific estimated scores
    platform_fit_score: float
    platform_dispatch_score: float
    fit_basis: list[str] = Field(default_factory=list)
    # 输入中未被平台配置声明的项（例如 role_hint 不在 role_affinity 中），
    # 显式记录以避免默认值 0 被静默当成真实信号。
    undeclared_inputs: list[str] = Field(default_factory=list)

    # risk information preserved verbatim
    risk_level: Literal["low", "medium", "high"]
    eligible: bool
    manual_review_required: bool
    risk_flags: list[str] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)


class PlatformDispatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform_key: str
    display_name: str
    requested_count: int
    selected_count: int
    shortfall_count: int
    # 名额“够不够”和“够不够格”是两件事，分开记录：
    #   shortfall_count      = requested_count - selected_count
    #   safe_shortfall_count = requested_count - safe 层节点数
    # 即使人数选满（shortfall_count == 0），只要用 not_eligible 节点补位，
    # safe_shortfall_count 仍然 > 0，从而不会被读成“凑够了合格的人”。
    selected_safety_tiers: SafetyTierCounts = Field(default_factory=SafetyTierCounts)
    safe_shortfall_count: int = 0
    selected_nodes: list[PlatformNodeAssignment] = Field(default_factory=list)
    platform_roles: list[str] = Field(default_factory=list)
    content_plan: PlatformContentPlan
    time_plan: PlatformTimePlan
    warnings: list[str] = Field(default_factory=list)
    policy_disclaimer: str = ""
    fit_limitations: list[str] = Field(default_factory=list)
    fit_method: str = _FIT_METHOD


class MultiPlatformDispatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_mode: Literal["cross_platform"] = "cross_platform"
    event_id: str
    event_type: str
    campaign_window_hours: int
    platforms: list[PlatformDispatchResult] = Field(default_factory=list)
    total_requested_count: int = 0
    total_selected_count: int = 0
    total_selected_safety_tiers: SafetyTierCounts = Field(
        default_factory=SafetyTierCounts
    )
    total_safe_shortfall_count: int = 0
    unique_user_count: int = 0
    overlapping_user_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    policy_disclaimer: str = ""
    fit_limitations: list[str] = Field(default_factory=list)


class PlatformDispatcher:
    """Dispatch one ``ScoreResult`` across several platforms independently."""

    def dispatch(
        self,
        *,
        score_result: ScoreResult,
        platform_specs: Sequence[PlatformSpec],
        campaign_window_hours: int | None = None,
    ) -> MultiPlatformDispatchResult:
        if not platform_specs:
            raise ValueError("platform_specs 不能为空，请至少指定一个平台。")

        seen_keys: set[str] = set()
        resolved: list[tuple[PlatformProfile, int]] = []
        for spec in platform_specs:
            profile = get_platform_profile(spec.platform_key)
            if profile.platform_key in seen_keys:
                raise ValueError(
                    f"平台 '{profile.platform_key}' 在同一批调度中重复出现，请合并为一条配置。"
                )
            seen_keys.add(profile.platform_key)
            count = int(spec.requested_count)
            if count < 1:
                raise ValueError(
                    f"平台 '{profile.platform_key}' 的 requested_count 必须是正整数，当前为 {count}。"
                )
            resolved.append((profile, count))

        window_hours = (
            int(campaign_window_hours)
            if campaign_window_hours is not None
            else int(score_result.event.constraints.campaign_window_hours)
        )
        window_hours = max(window_hours, 1)

        platform_results = [
            self._dispatch_one(
                profile=profile,
                requested_count=count,
                score_result=score_result,
                window_hours=window_hours,
            )
            for profile, count in resolved
        ]

        user_id_counts = Counter(
            assignment.user_id
            for result in platform_results
            for assignment in result.selected_nodes
        )
        unique_user_ids = sorted(user_id_counts)
        overlapping_user_ids = sorted(
            user_id for user_id, count in user_id_counts.items() if count > 1
        )

        total_requested = sum(result.requested_count for result in platform_results)
        total_selected = sum(result.selected_count for result in platform_results)
        total_tiers = SafetyTierCounts(
            safe=sum(result.selected_safety_tiers.safe for result in platform_results),
            review_required=sum(
                result.selected_safety_tiers.review_required
                for result in platform_results
            ),
            not_eligible=sum(
                result.selected_safety_tiers.not_eligible
                for result in platform_results
            ),
        )
        total_safe_shortfall = sum(
            result.safe_shortfall_count for result in platform_results
        )

        top_level_warnings: list[str] = []
        for result in platform_results:
            for warning in result.warnings:
                top_level_warnings.append(f"[{result.platform_key}] {warning}")
        if total_safe_shortfall > 0:
            top_level_warnings.append(
                f"跨平台合格节点缺口合计 {total_safe_shortfall} 个"
                f"（合格 {total_tiers.safe} 个 / 待人工复核 {total_tiers.review_required} 个 / "
                f"未通过资格筛选 {total_tiers.not_eligible} 个）；"
                "缺口由未通过资格筛选的节点补位，这些节点不会被标记为合格。"
            )
        if overlapping_user_ids:
            # 每个数字人只归属唯一模拟平台，理论上不会触发；保留为防御性告警，
            # 一旦出现说明平台归属字段与调度逻辑不一致，需要排查。
            top_level_warnings.append(
                "检测到同一数字人被分配到多个平台（不应发生，请检查平台归属数据）："
                + ", ".join(overlapping_user_ids)
            )
        top_level_warnings.append(
            "所有平台均为模拟平台；每个数字人只归属唯一模拟平台，"
            "平台只从属于本平台的数字人中选取，不会跨平台重复选人。"
        )

        # 顶层免责声明 / 限制取所有平台的并集，避免只反映第一个平台。
        disclaimers = _dedupe_preserve_order(
            [
                result.policy_disclaimer
                for result in platform_results
                if result.policy_disclaimer
            ]
        )
        limitations = _dedupe_preserve_order(
            [
                note
                for result in platform_results
                for note in result.fit_limitations
            ]
        )

        return MultiPlatformDispatchResult(
            event_id=score_result.event.event_id,
            event_type=score_result.event.event_type,
            campaign_window_hours=window_hours,
            platforms=platform_results,
            total_requested_count=total_requested,
            total_selected_count=total_selected,
            total_selected_safety_tiers=total_tiers,
            total_safe_shortfall_count=total_safe_shortfall,
            unique_user_count=len(unique_user_ids),
            overlapping_user_ids=overlapping_user_ids,
            warnings=_dedupe_preserve_order(top_level_warnings),
            policy_disclaimer=" ".join(disclaimers),
            fit_limitations=limitations,
        )

    # ------------------------------------------------------------------ #
    # single platform
    # ------------------------------------------------------------------ #
    def _dispatch_one(
        self,
        *,
        profile: PlatformProfile,
        requested_count: int,
        score_result: ScoreResult,
        window_hours: int,
    ) -> PlatformDispatchResult:
        ranked = self._rank_candidates(profile=profile, score_result=score_result)
        selected = ranked[:requested_count]
        shortfall = max(0, requested_count - len(selected))

        assignments = [
            self._build_assignment(
                profile=profile,
                node=node,
                fit_score=fit_score,
                dispatch_score=dispatch_score,
                fit_basis=fit_basis,
                rank=index,
            )
            for index, (node, fit_score, dispatch_score, fit_basis) in enumerate(
                selected, start=1
            )
        ]

        safety_tiers = _count_safety_tiers(assignments)
        safe_shortfall = max(0, requested_count - safety_tiers.safe)

        warnings = self._build_platform_warnings(
            profile=profile,
            requested_count=requested_count,
            assignments=assignments,
            shortfall=shortfall,
            safety_tiers=safety_tiers,
            safe_shortfall=safe_shortfall,
        )

        return PlatformDispatchResult(
            platform_key=profile.platform_key,
            display_name=profile.display_name,
            requested_count=requested_count,
            selected_count=len(assignments),
            shortfall_count=shortfall,
            selected_safety_tiers=safety_tiers,
            safe_shortfall_count=safe_shortfall,
            selected_nodes=assignments,
            platform_roles=list(profile.platform_roles),
            content_plan=self._build_content_plan(profile, score_result),
            time_plan=self._build_time_plan(profile, window_hours),
            warnings=warnings,
            policy_disclaimer=profile.policy_disclaimer,
            fit_limitations=list(profile.fit_limitations),
        )

    # ------------------------------------------------------------------ #
    # ranking
    # ------------------------------------------------------------------ #
    def _rank_candidates(
        self,
        *,
        profile: PlatformProfile,
        score_result: ScoreResult,
    ) -> list[tuple[NodeScore, float, float, list[str]]]:
        scored: list[tuple[NodeScore, float, float, list[str]]] = []
        for node in self._candidates_for_platform(profile=profile, score_result=score_result):
            fit_score, fit_basis = self._platform_fit_score(profile=profile, node=node)
            dispatch_score = self._platform_dispatch_score(
                profile=profile, node=node, fit_score=fit_score
            )
            scored.append((node, fit_score, dispatch_score, fit_basis))

        scored.sort(
            key=lambda item: (
                _SAFETY_TIER_ORDER[self._safety_tier(item[0])],
                -item[2],  # platform_dispatch_score
                -_coerce_float(item[0].final_score),
                -_coerce_float(item[0].topic_match_score),
                item[0].user_id,
            )
        )
        return scored

    @staticmethod
    def _candidates_for_platform(
        *,
        profile: PlatformProfile,
        score_result: ScoreResult,
    ) -> list[NodeScore]:
        """只保留归属 ``profile.platform_key`` 的候选数字人。

        数据集为每个数字人写入唯一的模拟平台（``NodeScore.platform``），
        因此同一数字人不可能同时属于多个平台，跨平台重复选人在此被彻底排除。
        归属为空（旧数据集）的节点不参与任何平台的分发，避免把平台未知
        的数字人误分到某个平台。
        """

        target_key = normalize_platform_key(profile.platform_key)
        return [
            node
            for node in score_result.node_scores
            if normalize_platform_key(getattr(node, "platform", "")) == target_key
        ]

    def _platform_fit_score(
        self,
        *,
        profile: PlatformProfile,
        node: NodeScore,
    ) -> tuple[float, list[str]]:
        signals: dict[str, float] = {
            field: _coerce_float(getattr(node, field, 0.0))
            for field in _FIT_SIGNAL_FIELDS
        }
        role_hint = str(getattr(node, "role_hint", "") or "")
        affinity = _coerce_float(profile.role_affinity.get(role_hint, 0.0))

        basis: list[str] = []
        total = 0.0
        for field in _FIT_SIGNAL_FIELDS:
            weight = _coerce_float(profile.signal_weights.get(field, 0.0))
            contribution = weight * signals[field]
            total += contribution
            if weight > 0.0:
                basis.append(f"{field}={signals[field]:.3f}×{weight:.2f}")

        role_weight = _coerce_float(profile.signal_weights.get("role_affinity", 0.0))
        total += role_weight * affinity
        # 未声明的 role_hint 会在查表时静默落到 0，这里显式标注，
        # 使「真实亲和度为 0」与「该 role 未在平台配置中声明」不再无法区分。
        undeclared_marker = (
            ""
            if role_hint in profile.role_affinity
            else "，未在平台 role_affinity 中声明，按0计入"
        )
        basis.append(
            f"role_hint={role_hint or 'unknown'}"
            f"(affinity={affinity:.2f}×{role_weight:.2f}{undeclared_marker})"
        )

        fit_score = _clip(total)
        basis.append(f"platform_fit_score={fit_score:.4f}")
        return fit_score, basis

    def _platform_dispatch_score(
        self,
        *,
        profile: PlatformProfile,
        node: NodeScore,
        fit_score: float,
    ) -> float:
        weights = profile.dispatch_score_weights
        w_final = _coerce_float(weights.get("final_score", 0.0))
        w_fit = _coerce_float(weights.get("platform_fit_score", 0.0))
        final_score = _clip(_coerce_float(node.final_score))
        return _clip(w_final * final_score + w_fit * fit_score)

    @staticmethod
    def _safety_tier(node: NodeScore) -> str:
        if not bool(getattr(node, "eligible", False)):
            return "not_eligible"
        if bool(getattr(node, "manual_review_required", False)):
            return "review_required"
        return "safe"

    def _build_assignment(
        self,
        *,
        profile: PlatformProfile,
        node: NodeScore,
        fit_score: float,
        dispatch_score: float,
        fit_basis: list[str],
        rank: int,
    ) -> PlatformNodeAssignment:
        platform_role = profile.platform_roles[(rank - 1) % len(profile.platform_roles)]
        safety_tier = self._safety_tier(node)

        source_role_hint = str(getattr(node, "role_hint", "") or "")
        undeclared_inputs: list[str] = []
        if source_role_hint not in profile.role_affinity:
            undeclared_inputs.append(
                f"role_hint={source_role_hint or 'unknown'}"
                "（未在平台 role_affinity 中声明，按 0 计入 platform_fit_score）"
            )

        selection_reasons = list(getattr(node, "selection_reasons", []) or [])
        selection_reasons.extend(
            [
                f"platform={profile.platform_key}",
                f"platform_role={platform_role}",
                f"platform_fit_score={fit_score:.4f}",
                f"platform_dispatch_score={dispatch_score:.4f}",
                f"safety_tier={safety_tier}",
            ]
        )

        return PlatformNodeAssignment(
            user_id=str(getattr(node, "user_id", "")),
            user_name=str(getattr(node, "user_name", "") or ""),
            platform=str(getattr(node, "platform", "") or ""),
            selection_rank=rank,
            platform_role=platform_role,
            source_role_hint=source_role_hint,
            safety_tier=safety_tier,
            final_score=_coerce_float(node.final_score),
            topic_match_score=_coerce_float(node.topic_match_score),
            influence_score=_coerce_float(node.influence_score),
            diffusion_score=_coerce_float(node.diffusion_score),
            stability_score=_coerce_float(node.stability_score),
            activity_score=_coerce_float(node.activity_score),
            platform_fit_score=round(fit_score, 6),
            platform_dispatch_score=round(dispatch_score, 6),
            fit_basis=fit_basis,
            undeclared_inputs=undeclared_inputs,
            risk_level=getattr(node, "risk_level", "low"),
            eligible=bool(getattr(node, "eligible", False)),
            manual_review_required=bool(getattr(node, "manual_review_required", False)),
            risk_flags=list(getattr(node, "risk_flags", []) or []),
            selection_reasons=_dedupe_preserve_order(selection_reasons),
        )

    # ------------------------------------------------------------------ #
    # warnings
    # ------------------------------------------------------------------ #
    def _build_platform_warnings(
        self,
        *,
        profile: PlatformProfile,
        requested_count: int,
        assignments: list[PlatformNodeAssignment],
        shortfall: int,
        safety_tiers: SafetyTierCounts,
        safe_shortfall: int,
    ) -> list[str]:
        warnings: list[str] = []

        if shortfall > 0:
            warnings.append(
                f"候选不足：requested_count={requested_count}，实际选出 "
                f"{len(assignments)} 个，缺口 {shortfall} 个；未虚构节点补齐。"
            )

        not_eligible_ids = [
            item.user_id for item in assignments if item.safety_tier == "not_eligible"
        ]
        if not_eligible_ids:
            warnings.append(
                f"为满足人数要求，纳入了 {len(not_eligible_ids)} 个未通过资格筛选的节点："
                + ", ".join(not_eligible_ids)
                + "；这些节点保持 eligible=false，风险标记未被修改。"
            )
            # 「人数够了」不等于「合格的人够了」，单独把合格缺口写清楚。
            warnings.append(
                f"合格节点不足：requested_count={requested_count}，合格且无需人工复核的节点仅 "
                f"{safety_tiers.safe} 个（待人工复核 {safety_tiers.review_required} 个），"
                f"合格缺口 {safe_shortfall} 个；"
                "缺口由未通过资格筛选的节点补位，这些节点不会被当作合格节点。"
            )

        review_ids = [
            item.user_id for item in assignments if item.manual_review_required
        ]
        if review_ids:
            warnings.append(
                "纳入的节点中包含需人工复核的节点：" + ", ".join(review_ids) + "。"
            )

        high_risk_ids = [
            item.user_id
            for item in assignments
            if str(item.risk_level) == "high"
        ]
        if high_risk_ids:
            warnings.append(
                "纳入的节点中包含风险等级 high 的节点：" + ", ".join(high_risk_ids) + "。"
            )

        warnings.append(
            f"平台适配分为保守估计，依据：{_FIT_METHOD}"
            "所有平台均为模拟平台，候选数字人均来自本平台的模拟账号。"
        )
        for note in profile.cadence_notes:
            warnings.append(note)
        return _dedupe_preserve_order(warnings)

    # ------------------------------------------------------------------ #
    # content / time plans
    # ------------------------------------------------------------------ #
    def _build_content_plan(
        self,
        profile: PlatformProfile,
        score_result: ScoreResult,
    ) -> PlatformContentPlan:
        event = score_result.event
        keywords = [str(item) for item in getattr(event, "extracted_keywords", []) or []]
        audience = [
            str(item) for item in getattr(event, "target_audience", []) or []
        ]
        context = {
            "event_title": str(getattr(event, "event_title", "") or "当前事件"),
            "target_goal": str(getattr(event, "target_goal", "") or "awareness"),
            "audience": "、".join(audience) if audience else "泛公众",
            "topic": keywords[0] if keywords else str(getattr(event, "event_type", "") or "general"),
            "risk_level": str(getattr(event.constraints, "risk_level", "medium")),
            "platform": profile.display_name,
        }

        sections = [
            PlatformContentSection(
                section_key=spec.section_key,
                label=spec.label,
                guidance=_render(spec.guidance, context),
            )
            for spec in profile.content_template.sections
        ]

        guardrails = list(profile.content_template.guardrails)
        guardrails.extend(
            str(item) for item in getattr(event, "negative_constraints", []) or []
        )
        if context["risk_level"] in {"medium", "high"}:
            guardrails.append("避免情绪化、对立化或未经证实的表述。")
        if context["risk_level"] == "high":
            guardrails.append("高风险事件仅做事实性陈述，不追加推测性扩写。")

        return PlatformContentPlan(
            platform_key=profile.platform_key,
            content_modes=list(profile.content_template.content_modes),
            sections=sections,
            guardrails=_dedupe_preserve_order(guardrails),
        )

    def _build_time_plan(
        self,
        profile: PlatformProfile,
        window_hours: int,
    ) -> PlatformTimePlan:
        stages = [
            self._build_time_stage(spec=spec, window_hours=window_hours)
            for spec in profile.time_blueprint
        ]
        return PlatformTimePlan(
            platform_key=profile.platform_key,
            campaign_window_hours=window_hours,
            stages=stages,
            cadence_notes=list(profile.cadence_notes),
        )

    @staticmethod
    def _build_time_stage(
        *,
        spec: PlatformTimeStageSpec,
        window_hours: int,
    ) -> PlatformTimeStage:
        """把一条比例蓝图映射成窗口内的整数小时阶段。

        ``campaign_window_hours`` 是硬上限，优先级高于「每个阶段至少
        ``_STAGE_MIN_HOURS`` 小时」这一偏好：

        * ``start`` / ``end`` 先按比例取整并夹取到 ``[0, window]``；
        * 再尝试把 ``end`` 抬到 ``start + _STAGE_MIN_HOURS``，
          但最终仍会被 ``window`` 截断。

        由于比例取整可能让相邻边界落到同一小时（窗口越短越容易发生），
        本函数会产生**重叠或零长度**阶段。原实现缺少最后的 ``window`` 截断，
        因此 ``window=1`` 时会产出 ``T+1h - T+2h`` 这种超出窗口的阶段；
        这里选择「窗口上限优先」，而不是延长活动窗口。
        可证明不变式：``start <= end <= window``——
        当 ``start < window`` 时 ``min(max(end, start+1), window) >= start+1``；
        当 ``start == window`` 时结果为 ``window``，即零长度阶段。
        """

        window = max(int(window_hours), 1)
        start = min(max(int(round(window * spec.ratio_start)), 0), window)
        end = int(round(window * spec.ratio_end))
        end = max(end, start + _STAGE_MIN_HOURS)
        # 窗口上限优先：宁可阶段重叠/零长度，也不越过活动窗口。
        end = min(max(end, 0), window)

        return PlatformTimeStage(
            stage_key=spec.stage_key,
            label=spec.label,
            window=f"T+{start}h - T+{end}h",
            start_hour=start,
            end_hour=end,
            focus=spec.focus,
        )


def to_json_dict(result: MultiPlatformDispatchResult) -> dict[str, Any]:
    """Serialize a dispatch result to a plain JSON-safe dict."""

    return result.model_dump(mode="json")


__all__ = [
    "STAGE_SCHEDULE_NOTE",
    "MultiPlatformDispatchResult",
    "PlatformContentPlan",
    "PlatformContentSection",
    "PlatformDispatcher",
    "PlatformDispatchResult",
    "PlatformNodeAssignment",
    "PlatformSpec",
    "PlatformTimePlan",
    "PlatformTimeStage",
    "SafetyTierCounts",
    "UnsupportedPlatformError",
    "get_platform_profile",
    "list_supported_platforms",
    "to_json_dict",
]
