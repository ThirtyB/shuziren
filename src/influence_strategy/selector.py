from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import (
    NodeScore,
    ParsedEvent,
    ScoreResult,
    SelectedNode,
    SelectionResult,
    SelectionSummary,
    UserProfile,
)
from .platform_profiles import normalize_platform_key
from .prompts import build_selector_system_prompt, build_selector_user_prompt

# --------------------------------------------------------------------------- #
# role_hint ↔ selected_role / dispatch_stage
# --------------------------------------------------------------------------- #
# ``FeatureBuilder`` 为每个数字人推断 ``role_hint``，``Selector`` 把它翻译成
# 策略层角色 ``selected_role`` 与执行阶段 ``dispatch_stage``。
#
# ``PlatformDispatcher`` 复用**同一套** role_hint 词汇表（见
# ``PlatformNodeAssignment.source_role_hint``），因此跨平台模式的选人结果只要
# 查这几张表就能回到五维调度策略的角色/阶段口径，不需要在调用方另立一套映射。
# 把映射提到模块级，就是为了让"多人分发选的人"和"跨平台选的人"用同一个来源，
# 避免两处各写一份而慢慢漂移。
ROLE_HINT_TO_SELECTED_ROLE: dict[str, str] = {
    "core_broadcast": "core_publish_node",
    "interaction_response": "interaction_response_node",
    "amplification": "amplification_node",
    "general_support": "support_node",
}

#: ``selected_role`` → ``role_hint``（重平衡角色后用于回填 ``dispatch_stage``）。
SELECTED_ROLE_TO_ROLE_HINT: dict[str, str] = {
    "core_publish_node": "core_broadcast",
    "interaction_response_node": "interaction_response",
    "amplification_node": "amplification",
    "support_node": "general_support",
}

ROLE_HINT_TO_DISPATCH_STAGE: dict[str, str] = {
    "core_broadcast": "stage_node_activation",
    "interaction_response": "stage_node_activation",
    "amplification": "stage_info_diffusion",
    "general_support": "stage_consensus",
}

#: 未识别的 role_hint 兜底（与历史实现保持一致，不引入新的行为）。
DEFAULT_SELECTED_ROLE = "support_node"
DEFAULT_ROLE_HINT = "general_support"
DEFAULT_DISPATCH_STAGE = "stage_consensus"

# --------------------------------------------------------------------------- #
# 多人协同（单平台）身份配额
# --------------------------------------------------------------------------- #
#: 身份人数占比。支持不设占比——它拿走分配完前三类之后的剩余名额。
ROLE_QUOTA_RATIO: dict[str, float] = {
    "core_publish_node": 0.20,
    "amplification_node": 0.40,
    "interaction_response_node": 0.20,
}

#: 人数下限。首发与扩散至少 1 人（总人数 >= 2 时必定满足）；
#: 互动与支持允许为 0。
ROLE_MIN_COUNT: dict[str, int] = {
    "core_publish_node": 1,
    "amplification_node": 1,
    "interaction_response_node": 0,
    "support_node": 0,
}

#: 身份分配顺序：先定首发，再从**剩下的人**里定扩散，然后互动，最后剩余归支持。
#: 注意这与执行时期顺序（首发 → 互动 → 扩散 → 支持）不同：选人按重要性，执行按时间。
ROLE_ASSIGN_ORDER: tuple[str, ...] = (
    "core_publish_node",
    "amplification_node",
    "interaction_response_node",
    "support_node",
)

#: 首发挑选的"影响力 / 话题匹配度"门槛。
#:
#: 实测候选池分布（facebook 平台、189 个 eligible 节点）：
#:   influence_score   p25=0.33  p50=0.40  p75=0.52
#:   topic_match_score p50=0.18  max=0.36
#: topic_match_score 是「命中词数 / 关键词总数」，命中 2 个词才 0.18，几乎无人能到
#: 0.3 —— 门槛按 0.3 定会让第一档永远为空、每次都降级。这里取 0.15，
#: 大致对应"至少命中 2 个关键词"，既有区分度又够得着。
INFLUENCE_FLOOR = 0.35
TOPIC_FLOOR = 0.15


def rebalance_selected_roles(
    selected_nodes: list[SelectedNode],
) -> list[SelectedNode]:
    """保证多人/跨平台选出的节点里有合理的角色分布。

    规则：核心发布节点必须存在；角色种类至少 ``min(3, 人数)``。
    不足时按 ``core_publish_node → interaction_response_node → amplification_node →
    support_node`` 的顺序补齐，最后一段回退到 ``interaction_response_node``，避免
    同一个平台的若干人全是 amplification 之类、让所有互动动作都退化成"自指"。

    原本只有 :meth:`Selector._rebalance_selected_roles` 在多人分发里调用，
    跨平台分发（``PlatformDispatcher``）走的是另一条选人路径，没有这个保障——
    结果就是核心发布节点缺失、动作生成阶段找不到协同目标，整张清单退化成
    "自发自评"。提取为模块级函数就是让两条路径共用同一份角色再平衡逻辑。
    """
    if len(selected_nodes) <= 1:
        return selected_nodes
    existing_roles = [node.selected_role for node in selected_nodes]
    required_unique_roles = min(3, len(selected_nodes))
    needs_rebalance = (
        "core_publish_node" not in existing_roles
        or len(set(existing_roles)) < required_unique_roles
    )
    if not needs_rebalance:
        return selected_nodes

    desired_roles = [
        "core_publish_node",
        "interaction_response_node",
        "amplification_node",
        "support_node",
        "interaction_response_node",
    ]
    rebalanced: list[SelectedNode] = []
    for index, node in enumerate(selected_nodes):
        selected_role = desired_roles[min(index, len(desired_roles) - 1)]
        role_hint = SELECTED_ROLE_TO_ROLE_HINT.get(selected_role, DEFAULT_ROLE_HINT)
        reasons = [
            reason
            for reason in node.selection_reasons
            if not reason.startswith("selected_role=")
            and not reason.startswith("dispatch_stage=")
        ]
        if selected_role != node.selected_role:
            reasons.append(f"role_rebalanced={node.selected_role}->{selected_role}")
        reasons.append(f"selected_role={selected_role}")
        reasons.append(
            f"dispatch_stage={ROLE_HINT_TO_DISPATCH_STAGE.get(role_hint, DEFAULT_DISPATCH_STAGE)}"
        )
        rebalanced.append(
            node.model_copy(
                update={
                    "selected_role": selected_role,
                    "dispatch_stage": ROLE_HINT_TO_DISPATCH_STAGE.get(
                        role_hint, DEFAULT_DISPATCH_STAGE
                    ),
                    "selection_reasons": reasons,
                }
            )
        )
    return rebalanced


class Selector:
    def select(
        self,
        score_result: ScoreResult,
        *,
        workspace_root: str | Path | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
        target_user_id: str | None = None,
        dispatch_platform: str | None = None,
        single_target_mode: str = "self_operate",
        target_old_post_id: str | None = None,
        water_army_enabled: bool = False,
        water_army_count: int = 0,
        water_army_profiles: dict[str, UserProfile] | None = None,
    ) -> SelectionResult:
        if target_user_id:
            if single_target_mode == "attention":
                result = self._select_attention(score_result, target_user_id, target_old_post_id)
            else:
                result = self._select_targeted(score_result, target_user_id)
        else:
            result = self._select_automatic(score_result, dispatch_platform)

        # 白库水军：仅在启用时参与，不计入选人上限，也不影响角色编排。
        if water_army_enabled and water_army_profiles:
            result.water_army_nodes = self._build_water_army_nodes(
                water_army_profiles,
                event=score_result.event,
                platforms=list(score_result.event.constraints.allowed_platforms),
                max_count=water_army_count,
                exclude_user_ids={
                    n.user_id for n in result.selected_nodes
                }
                | ({result.summary.target_user_id} if result.summary.target_user_id else set()),
            )
            result.summary.water_army_enabled = True
            result.summary.water_army_count = len(result.water_army_nodes)
            result.summary.water_army_platforms = sorted(
                {n.platform for n in result.water_army_nodes if n.platform}
            )
        return result

    def _select_automatic(
        self,
        score_result: ScoreResult,
        dispatch_platform: str | None = None,
        exclude_user_ids: set[str] | None = None,
    ) -> SelectionResult:
        max_selected = score_result.event.constraints.max_selected_nodes

        # 多人分发必须指定平台：只在候选池中保留归属该平台的数字人，
        # 保证"能且只能选择该平台上已有的数字人"。
        candidate_nodes = self._filter_nodes_by_platform(
            score_result.node_scores, dispatch_platform
        )
        if exclude_user_ids:
            candidate_nodes = [
                node for node in candidate_nodes if node.user_id not in exclude_user_ids
            ]

        safe_pool = [
            node for node in candidate_nodes if node.eligible and not node.manual_review_required
        ]
        review_pool = [
            node for node in candidate_nodes if node.eligible and node.manual_review_required
        ]
        # fallback_pool 仅保留"不进入 safe_pool / review_pool"的节点
        # （即 not eligible 且没被 primary 拿走）。与 review_pool 不重叠，
        # 避免同一个 manual_review_required 节点被 fallback 池和 review 池
        # 双计入、最终被两条分支重复列举。
        fallback_pool = [
            node for node in candidate_nodes if not node.eligible
        ]

        # ---- 按身份依次定量挑选 ----
        #
        # 与旧实现的两点本质差别：
        #   1) 选人与定身份合成一步：先定首发，再从**剩下的人**里定扩散，依次类推，
        #      因此不再需要事后 rebalance（它会用固定顺序覆盖这里的身份）；
        #   2) 风险的门槛仍然生效：先 safe（合格且不需复核），不够退到 review，
        #      最后才动用 fallback。
        #
        # LLM 选人指导暂时不参与（见 _make_selected_node 的 llm_selection_plan=None）。
        # LLM 选人指导暂时不参与：身份完全由指标决定，保证结果可复现。
        llm_selection_plan: dict[str, Any] | None = None
        tiered_pool: list[NodeScore] = safe_pool + review_pool + fallback_pool
        quotas = self._role_quotas(max_selected)

        remaining: list[NodeScore] = list(tiered_pool)
        picks: dict[str, list[NodeScore]] = {}

        # ① 首发：影响力 + 话题匹配度，两者都过线优先，人数不够再逐级降级
        picks["core_publish_node"] = self._pick_launch_nodes(
            remaining, quotas["core_publish_node"]
        )

        # ② 扩散：从剩下的人里按传播力取
        remaining = self._without(remaining, picks["core_publish_node"])
        picks["amplification_node"] = self._pick_top(
            remaining, quotas["amplification_node"], lambda node: node.diffusion_score
        )

        # ③ 互动：活跃度 + 归一化评论数
        remaining = self._without(remaining, picks["amplification_node"])
        picks["interaction_response_node"] = self._pick_top(
            remaining,
            quotas["interaction_response_node"],
            self._interaction_score(remaining),
        )

        # ④ 支持：最后剩下的人
        remaining = self._without(remaining, picks["interaction_response_node"])
        picks["support_node"] = remaining[: quotas["support_node"]]

        selected_nodes: list[SelectedNode] = []
        for role in ROLE_ASSIGN_ORDER:
            for node in picks.get(role, []):
                selected_nodes.append(
                    self._make_selected_node(
                        node,
                        len(selected_nodes) + 1,
                        "primary",
                        forced_role=role,
                    )
                )
        used_ids = {node.user_id for node in selected_nodes}

        # fallback 数量与 max_selected 挂钩（最多 max_selected*2，下限 5）。
        # 同时去重：同一 user_id 只出现一次。
        fallback_limit = max(5, max_selected * 2)
        seen_ids: set[str] = set(used_ids)
        fallback_candidates: list[NodeScore] = []
        for pool in (safe_pool, review_pool, fallback_pool):
            for node in pool:
                if node.user_id in seen_ids:
                    continue
                seen_ids.add(node.user_id)
                fallback_candidates.append(node)
                if len(fallback_candidates) >= fallback_limit:
                    break
            if len(fallback_candidates) >= fallback_limit:
                break

        fallback_nodes: list[SelectedNode] = []
        for index, candidate in enumerate(fallback_candidates, start=1):
            fallback_nodes.append(
                self._make_selected_node(
                    candidate,
                    index,
                    "fallback",
                    llm_selection_plan=llm_selection_plan,
                )
            )

        role_distribution = Counter(node.selected_role for node in selected_nodes)
        stage_distribution = Counter(node.dispatch_stage for node in selected_nodes)
        summary = SelectionSummary(
            event_id=score_result.event.event_id,
            event_type=score_result.event.event_type,
            max_selected_nodes=max_selected,
            selected_count=len(selected_nodes),
            fallback_count=len(fallback_nodes),
            selected_role_distribution=dict(role_distribution),
            selected_stage_distribution=dict(stage_distribution),
            avg_selected_final_score=round(
                sum(node.final_score for node in selected_nodes) / max(len(selected_nodes), 1),
                6,
            ),
            selection_mode="automatic",
            dispatch_platform=normalize_platform_key(dispatch_platform) or None,
        )

        return SelectionResult(
            event=score_result.event,
            product_context=score_result.product_context,
            summary=summary,
            selected_nodes=selected_nodes,
            fallback_nodes=fallback_nodes,
        )

    # ---------------- 多人协同身份分配 ----------------

    def _role_quotas(self, total: int) -> dict[str, int]:
        """按占比切分各身份人数。

        * 首发 20%、扩散 40%、互动 20%，支持拿走剩余名额；
        * 首发与扩散保底 1 人，互动与支持允许为 0；
        * 任一步都不允许超出 ``total``——总人数不足时后面的身份让位。
        """

        total = max(int(total), 2)  # N 不允许小于 2
        quotas: dict[str, int] = {}
        assigned = 0
        for role in ("core_publish_node", "amplification_node", "interaction_response_node"):
            want = int(round(total * ROLE_QUOTA_RATIO[role]))
            want = max(want, ROLE_MIN_COUNT[role])
            want = min(want, total - assigned)
            quotas[role] = max(0, want)
            assigned += quotas[role]
        quotas["support_node"] = max(0, total - assigned)
        return quotas

    @staticmethod
    def _without(pool: list[NodeScore], picked: list[NodeScore]) -> list[NodeScore]:
        """从池子里剔除已选中的人——保证一个人只会拿到一个身份。"""

        picked_ids = {node.user_id for node in picked}
        return [node for node in pool if node.user_id not in picked_ids]

    def _pick_by_tiers(
        self,
        pool: list[NodeScore],
        count: int,
        score_key,
        tiers,
    ) -> list[NodeScore]:
        """按档位依次补足：高档位选不够就往低档位退。

        ``tiers`` 按"严格 → 宽松"排列，每一项是 ``(node) -> bool`` 过滤器。
        """

        chosen: list[NodeScore] = []
        used: set[str] = set()
        for tier_filter in tiers:
            if len(chosen) >= count:
                break
            candidates = [n for n in pool if n.user_id not in used and tier_filter(n)]
            candidates.sort(key=score_key, reverse=True)
            for node in candidates:
                if len(chosen) >= count:
                    break
                chosen.append(node)
                used.add(node.user_id)
        return chosen

    def _pick_launch_nodes(self, pool: list[NodeScore], count: int) -> list[NodeScore]:
        """首发：影响力 + 话题匹配度（直接相加排序）。

        三档降级：
          1) 两者都过线 —— 最理想；
          2) 只有影响力过线 —— 话题不够但人够强；
          3) 不看门槛 —— 兜底，保证人数一定凑得齐。
        """

        def score(node: NodeScore) -> float:
            return node.influence_score + node.topic_match_score

        tiers = [
            lambda n: n.influence_score > INFLUENCE_FLOOR
            and n.topic_match_score > TOPIC_FLOOR,
            lambda n: n.influence_score > INFLUENCE_FLOOR,
            lambda n: True,
        ]
        return self._pick_by_tiers(pool, count, score, tiers)

    def _pick_top(self, pool: list[NodeScore], count: int, score_key) -> list[NodeScore]:
        """按单一指标取 top count。"""

        if count <= 0:
            return []
        return sorted(pool, key=score_key, reverse=True)[:count]

    @staticmethod
    def _interaction_score(pool: list[NodeScore]):
        """互动排序键：活跃度 + 归一化评论数。

        评论数是原始计数（实测 p50=77、max=5926），与 0~1 的活跃度量纲不同，
        直接相加会被长尾主导，因此先按池内最大值归一化到 0~1 再相加。
        """

        max_comment = max((node.received_comment_count for node in pool), default=0)

        def score(node: NodeScore) -> float:
            normalized = node.received_comment_count / max_comment if max_comment else 0.0
            return node.activity_score + normalized

        return score

    def _filter_nodes_by_platform(
        self,
        nodes: list[NodeScore],
        dispatch_platform: str | None,
    ) -> list[NodeScore]:
        """按平台过滤候选数字人。

        * ``dispatch_platform`` 为空时不过滤（保持旧调用方行为）；
        * 否则只保留 ``NodeScore.platform`` 归一化后等于该平台的数字人，
          从而保证多人分发"能且只能选择该平台上已有的数字人"。
        """

        target_key = normalize_platform_key(dispatch_platform)
        if not target_key:
            return list(nodes)
        return [
            node
            for node in nodes
            if normalize_platform_key(node.platform) == target_key
        ]

    def _select_targeted(
        self,
        score_result: ScoreResult,
        target_user_id: str,
    ) -> SelectionResult:
        """Single-user targeted selection.

        Keeps the original scorer output for the target user untouched and
        only emits diagnostics (no fallback nodes, no role rebalancing). The
        target user is forced to ``core_publish_node`` / ``stage_node_activation`` so
        that downstream StrategyGenerator treats them as the core publisher.
        The target's simulated platform (``NodeScore.platform``) is surfaced in
        both ``selection_reasons`` and ``SelectionSummary.target_platform`` so
        the caller can see which platform the chosen person belongs to.

        Fail-fast when the target is missing from the candidate pool — we do
        not want to silently fall back to automatic selection.
        """
        target_role = "core_publish_node"
        dispatch_stage = "stage_node_activation"

        target_node = next(
            (node for node in score_result.node_scores if node.user_id == target_user_id),
            None,
        )
        if target_node is None:
            raise ValueError(
                f"target_user_id '{target_user_id}' 不在当前事件的候选数字人画像中，"
                "请检查该用户是否存在于 profiles 数据中或扩大 profile_limit。"
            )

        warnings = self._evaluate_target_warnings(target_node, score_result.node_scores)

        target_platform = normalize_platform_key(target_node.platform) or None

        selection_reasons = list(target_node.selection_reasons)
        if warnings:
            selection_reasons.append("target_warning=" + "; ".join(warnings))
        selection_reasons.extend(
            [
                "targeted_selection=true",
                f"platform={target_platform or 'unknown'}",
                f"selected_role={target_role}",
                f"dispatch_stage={dispatch_stage}",
                "selection_bucket=primary",
            ]
        )

        selected_node = SelectedNode(
            **target_node.model_dump(exclude={"selection_reasons"}),
            selection_reasons=selection_reasons,
            selection_rank=1,
            selected_role=target_role,
            dispatch_stage=dispatch_stage,
            dispatch_priority=self._dispatch_priority(target_node.priority_tier),
            selection_bucket="primary",
        )

        summary = SelectionSummary(
            event_id=score_result.event.event_id,
            event_type=score_result.event.event_type,
            max_selected_nodes=score_result.event.constraints.max_selected_nodes,
            selected_count=1,
            fallback_count=0,
            selected_role_distribution={target_role: 1},
            selected_stage_distribution={dispatch_stage: 1},
            avg_selected_final_score=round(target_node.final_score, 6),
            selection_mode="targeted",
            target_user_id=target_user_id,
            target_role=target_role,
            target_platform=target_platform,
            target_final_score=target_node.final_score,
            target_topic_match_score=target_node.topic_match_score,
            target_warning=warnings,
        )

        return SelectionResult(
            event=score_result.event,
            product_context=score_result.product_context,
            summary=summary,
            selected_nodes=[selected_node],
            fallback_nodes=[],
        )

    def _select_attention(
        self,
        score_result: ScoreResult,
        target_user_id: str,
        target_old_post_id: str | None,
    ) -> SelectionResult:
        """思路二（数字人注意）：多数字人协同，全程 @ 指定路人（被动、不反应）。

        选人逻辑与多人协同一致（按角色配额从候选池挑演员），但强制排除被点名的
        路人本人；selection_mode 标记为 "attention"，并把路人身份写入 summary，
        供下游在帖文里注入 @ 点名内容。被点名路人本身不分配任何动作。
        """
        target_node = next(
            (node for node in score_result.node_scores if node.user_id == target_user_id),
            None,
        )
        if target_node is None:
            raise ValueError(
                f"target_user_id '{target_user_id}' 不在当前事件的候选数字人画像中，"
                "请检查该用户是否存在于 profiles 数据中或扩大 profile_limit。"
            )
        target_user_name = target_node.user_name
        target_platform = normalize_platform_key(target_node.platform) or None
        warnings = self._evaluate_target_warnings(target_node, score_result.node_scores)

        # 复用多人协同选人逻辑，但演员中排除被点名的路人。
        result = self._select_automatic(
            score_result, None, exclude_user_ids={target_user_id}
        )
        result.summary.selection_mode = "attention"
        result.summary.target_user_id = target_user_id
        result.summary.target_user_name = target_user_name
        result.summary.target_platform = target_platform
        result.summary.target_old_post_id = target_old_post_id
        result.summary.target_warning = warnings
        return result

    def _build_water_army_nodes(
        self,
        water_army_profiles: dict[str, UserProfile],
        *,
        event: ParsedEvent,
        platforms: list[str],
        max_count: int,
        exclude_user_ids: set[str],
    ) -> list[SelectedNode]:
        """从白库挑水军：只在事件平台上的低活跃账号，且不与真实演员/目标路人重复。

        水军只做点赞 / 转发（不附带文本），因此无需语义打分，随机抽取即可；
        返回 SelectedNode 列表，is_water_army=True，标记 role/stage 为 water_army。
        """
        if not water_army_profiles or max_count <= 0:
            return []
        platform_set = {normalize_platform_key(p) for p in platforms}
        candidates = [
            prof
            for prof in water_army_profiles.values()
            if normalize_platform_key(prof.platform) in platform_set
            and prof.user_id not in exclude_user_ids
        ]
        chosen = candidates[: max_count]
        nodes: list[SelectedNode] = []
        for rank, prof in enumerate(chosen, start=1):
            node_score = NodeScore(
                user_id=prof.user_id,
                user_name=prof.user_name,
                event_id=event.event_id,
                event_type=event.event_type,
                role_hint="special_support",
                platform=prof.platform,
                follower_count=prof.user_followers,
                friend_count=prof.user_friends,
                is_water_army=True,
                eligible=True,
                final_score=0.0,
            )
            nodes.append(
                SelectedNode(
                    **node_score.model_dump(),
                    selection_rank=rank,
                    selected_role="special_support",
                    dispatch_stage="special_support",
                    dispatch_priority="p3",
                    selection_bucket="primary",
                )
            )
        return nodes

    def _evaluate_target_warnings(
        self,
        target_node: NodeScore,
        pool: list[NodeScore],
    ) -> list[str]:
        """Inspect the target node without mutating them; only collect diagnostics."""
        warnings: list[str] = []

        if not target_node.eligible:
            reasons = "; ".join(target_node.risk_flags) if target_node.risk_flags else "未通过资格筛选"
            warnings.append(f"目标数字人未通过资格筛选（{reasons}）")

        if target_node.manual_review_required:
            warnings.append("目标数字人触发人工复核标记")

        if target_node.risk_level == "high":
            warnings.append("目标数字人风险等级为 high")

        topic_scores = [node.topic_match_score for node in pool]
        if topic_scores:
            median_topic = sorted(topic_scores)[len(topic_scores) // 2]
            if target_node.topic_match_score < median_topic:
                warnings.append("目标数字人话题匹配度低于候选池中位数")

        final_scores = [node.final_score for node in pool if node.final_score > 0]
        if final_scores:
            avg_final = sum(final_scores) / len(final_scores)
            if target_node.final_score < avg_final:
                warnings.append("目标数字人综合评分低于候选池平均水平")

        return warnings

    def to_frame(self, selection_result: SelectionResult, bucket: str = "primary") -> pd.DataFrame:
        if bucket == "fallback":
            rows = [item.model_dump() for item in selection_result.fallback_nodes]
        else:
            rows = [item.model_dump() for item in selection_result.selected_nodes]
        return pd.DataFrame(rows, columns=list(SelectedNode.model_fields.keys()))

    def _first_by_role(
        self,
        pool: list[NodeScore],
        role_hint: str,
        used_ids: set[str],
        llm_selection_plan: dict[str, Any] | None = None,
    ) -> NodeScore | None:
        for node in pool:
            if self._resolved_role_hint(node, llm_selection_plan) == role_hint and node.user_id not in used_ids:
                return node
        return None

    def _make_selected_node(
        self,
        node: NodeScore,
        selection_rank: int,
        selection_bucket: str,
        llm_selection_plan: dict[str, Any] | None = None,
        *,
        forced_role: str | None = None,
    ) -> SelectedNode:
        role_hint = self._resolved_role_hint(node, llm_selection_plan)
        if forced_role:
            # 身份分配阶段定好的身份就是最终身份，不再让特征层的 role_hint 猜测覆盖。
            selected_role = forced_role
            role_hint = self._role_hint_for_selected_role(forced_role)
        else:
            selected_role = self._selected_role(role_hint)
        dispatch_stage = self._dispatch_stage(role_hint)
        dispatch_priority = self._dispatch_priority(node.priority_tier)
        selection_reasons = list(node.selection_reasons)
        llm_reasoning = self._llm_reasoning(node.user_id, llm_selection_plan)
        selection_reasons.extend(llm_reasoning)
        selection_reasons.append(f"selected_role={selected_role}")
        selection_reasons.append(f"dispatch_stage={dispatch_stage}")
        selection_reasons.append(f"selection_bucket={selection_bucket}")

        return SelectedNode(
            **node.model_dump(exclude={"selection_reasons"}),
            selection_reasons=selection_reasons,
            selection_rank=selection_rank,
            selected_role=selected_role,
            dispatch_stage=dispatch_stage,
            dispatch_priority=dispatch_priority,
            selection_bucket=selection_bucket,
        )

    def _selected_role(self, role_hint: str) -> str:
        return ROLE_HINT_TO_SELECTED_ROLE.get(role_hint, DEFAULT_SELECTED_ROLE)

    def _rebalance_selected_roles(self, selected_nodes: list[SelectedNode]) -> list[SelectedNode]:
        return rebalance_selected_roles(selected_nodes)

    def _role_hint_for_selected_role(self, selected_role: str) -> str:
        return SELECTED_ROLE_TO_ROLE_HINT.get(selected_role, DEFAULT_ROLE_HINT)

    def _dispatch_stage(self, role_hint: str) -> str:
        return ROLE_HINT_TO_DISPATCH_STAGE.get(role_hint, DEFAULT_DISPATCH_STAGE)

    def _dispatch_priority(self, priority_tier: str) -> str:
        mapping = {
            "high": "p1",
            "medium": "p2",
            "low": "p3",
        }
        return mapping.get(priority_tier, "p3")

    def _preferred_roles(
        self,
        score_result: ScoreResult,
        llm_selection_plan: dict[str, Any] | None = None,
    ) -> tuple[str, ...]:
        role_map = {
            "core_publish_node": "core_broadcast",
            "interaction_response_node": "interaction_response",
            "amplification_node": "amplification",
            "support_node": "general_support",
        }
        if llm_selection_plan:
            ordered_roles = []
            raw_role_map = llm_selection_plan.get("role_map", {})
            order_map = llm_selection_plan.get("order_map", {})
            for user_id, _index in sorted(order_map.items(), key=lambda item: item[1]):
                recommended_role = raw_role_map.get(user_id)
                mapped_role = role_map.get(recommended_role)
                if mapped_role and mapped_role not in ordered_roles:
                    ordered_roles.append(mapped_role)
            if ordered_roles:
                return tuple(ordered_roles)

        preferred = [
            role_map[item]
            for item in score_result.event.target_roles
            if item in role_map
        ]
        if not preferred:
            preferred = ["core_broadcast", "interaction_response", "amplification"]
        return tuple(dict.fromkeys(preferred))

    def _llm_selection_plan(
        self,
        *,
        score_result: ScoreResult,
        safe_pool: list[NodeScore],
        max_selected: int,
        workspace_root: str | Path | None,
        use_llm: bool,
        llm_client: Any | None,
    ) -> dict[str, Any] | None:
        if not use_llm or not safe_pool:
            return None

        client = llm_client
        if client is None and workspace_root is not None:
            client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
        if client is None:
            return None

        shortlist = safe_pool[: score_result.event.dispatch_preferences.rerank_top_k]
        shortlist_payload = [
            {
                "user_id": node.user_id,
                "user_name": node.user_name,
                "role_hint": node.role_hint,
                "final_score": node.final_score,
                "llm_feature_score": node.llm_feature_score,
                "topic_match_score": node.topic_match_score,
                "risk_level": node.risk_level,
                "semantic_tags": node.semantic_tags,
                "matched_keywords": node.matched_keywords,
                "selection_reasons": node.selection_reasons[:6],
            }
            for node in shortlist
        ]
        try:
            response = client.generate_json(
                system_prompt=build_selector_system_prompt(),
                user_prompt=build_selector_user_prompt(
                    event=score_result.event,
                    max_selected=max_selected,
                    shortlist=shortlist_payload,
                ),
            )
        except (LLMClientError, OSError, ValueError, TypeError):
            return None

        selected_order = response.get("selected_order")
        fallback_order = response.get("fallback_order")
        global_notes = response.get("global_notes")
        if not isinstance(selected_order, list):
            return None

        order_map: dict[str, int] = {}
        role_map: dict[str, str] = {}
        reasoning_map: dict[str, list[str]] = {}
        for index, item in enumerate(selected_order):
            if not isinstance(item, dict):
                continue
            user_id = str(item.get("user_id", "")).strip()
            if not user_id:
                continue
            order_map[user_id] = index
            recommended_role = str(item.get("recommended_role", "")).strip()
            if recommended_role in {"core_publish_node", "interaction_response_node", "amplification_node", "support_node"}:
                role_map[user_id] = recommended_role
            reasoning_map[user_id] = [
                str(reason).strip()
                for reason in item.get("reasoning", [])
                if str(reason).strip()
            ][:4]

        normalized_fallback = [
            str(item).strip()
            for item in fallback_order
            if str(item).strip()
        ] if isinstance(fallback_order, list) else []
        normalized_notes = [
            str(item).strip()
            for item in global_notes
            if str(item).strip()
        ] if isinstance(global_notes, list) else []
        if not order_map:
            return None
        return {
            "order_map": order_map,
            "role_map": role_map,
            "reasoning_map": reasoning_map,
            "fallback_order": normalized_fallback,
            "global_notes": normalized_notes,
        }

    def _reorder_pool_with_llm(
        self,
        pool: list[NodeScore],
        llm_selection_plan: dict[str, Any] | None,
    ) -> list[NodeScore]:
        if not llm_selection_plan:
            return pool
        order_map = llm_selection_plan.get("order_map", {})
        return sorted(
            pool,
            key=lambda node: (
                order_map.get(node.user_id, len(order_map) + 1000),
                -node.final_score,
            ),
        )

    def _resolved_role_hint(
        self,
        node: NodeScore,
        llm_selection_plan: dict[str, Any] | None,
    ) -> str:
        if not llm_selection_plan:
            return node.role_hint
        role_map = llm_selection_plan.get("role_map", {})
        recommended_role = role_map.get(node.user_id)
        reverse_map = {
            "core_publish_node": "core_broadcast",
            "interaction_response_node": "interaction_response",
            "amplification_node": "amplification",
            "support_node": "general_support",
        }
        return reverse_map.get(recommended_role, node.role_hint)

    def _llm_reasoning(
        self,
        user_id: str,
        llm_selection_plan: dict[str, Any] | None,
    ) -> list[str]:
        if not llm_selection_plan:
            return []
        reasoning_map = llm_selection_plan.get("reasoning_map", {})
        reasoning = reasoning_map.get(user_id, [])
        return [f"llm_selector={item}" for item in reasoning]
