from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import NodeScore, ScoreResult, SelectedNode, SelectionResult, SelectionSummary
from .platform_profiles import normalize_platform_key
from .prompts import (
    build_selector_id_order_user_prompt,
    build_selector_system_prompt,
    build_selector_user_prompt,
)

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
    "core_broadcast": "stage_1_launch",
    "interaction_response": "stage_2_engage",
    "amplification": "stage_3_amplify",
    "general_support": "stage_2_support",
}

#: 未识别的 role_hint 兜底（与历史实现保持一致，不引入新的行为）。
DEFAULT_SELECTED_ROLE = "support_node"
DEFAULT_ROLE_HINT = "general_support"
DEFAULT_DISPATCH_STAGE = "stage_2_support"


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


# --------------------------------------------------------------------------- #
# 规则初选（第七轮）：把 LLM 重排的输入从整个 safe_pool 收敛到 N 人
# --------------------------------------------------------------------------- #
# 原实现是 ``safe_pool[:rerank_top_k]``（规则模式下 rerank_top_k 恒为 30），
# 一次性把 30 人的完整卡片交给本地 qwen3:8b，提示词 13.8KB、实测 17~18s。
# 第七轮改成"规则初选 + 小范围重排"：先用 scorer 已经算好的分数和既有
# role_hint 词汇表收敛候选，**再**把这批人交给原来的 ``_llm_selection_plan``，
# 沿用原有的提示词构造、输出 schema、解析流程与 fallback 机制。
#
# 要注意的是这里**真的缩小了候选列表**，不是把提示词字符串截断到 15 人：
# ``safe_pool`` 本身被替换掉，后面的 ``_reorder_pool_with_llm``、按角色选人循环、
# fallback 池构建都只在这批人（外加未被初选影响的 review_pool / fallback_pool）
# 上进行。

#: 规则初选的默认候选上限。
DEFAULT_SELECTOR_CANDIDATE_LIMIT = 15

# --------------------------------------------------------------------------- #
# 第十三轮：selector 提示词模式
# --------------------------------------------------------------------------- #
# ``id_order``（默认）：模型只输出排序后的 user_id，角色与理由由既有规则补齐。
#   实测（固定事件 + 15 人候选池 + 本地 qwen3:8b，预热后 3 次中位数）：
#   单次 LLM 调用 11.85s → 3.90s，**模型仍然决定顺序**，只是不再复述角色与理由。
# ``full``：第十三轮之前的模式，模型还要输出 recommended_role / reasoning。
#   保留它是为了让对照实验与提示词回归测试随时可切回，能力并未删除。
#
# 无论哪种模式，解析逻辑、候选池校验、去重、安全过滤与角色平衡**完全相同**。
DEFAULT_SELECTOR_OUTPUT_MODE = "id_order"

#: 初选阶段用于保证角色多样性的 role_hint 顺序，与
#: ``ROLE_HINT_TO_SELECTED_ROLE`` 的词汇表一致，不新造角色。
DEFAULT_PREFILTER_ROLE_HINTS: tuple[str, ...] = (
    "core_broadcast",
    "interaction_response",
    "amplification",
    "general_support",
)


def prefilter_selector_candidates(
    safe_pool: list[NodeScore],
    *,
    limit: int,
    preferred_role_hints: tuple[str, ...] = (),
) -> list[NodeScore]:
    """用规则把候选池收敛到 ``limit`` 人，供 LLM 小范围重排。

    复用的都是既有产出：

    * **排序依据**是 scorer 已经算好的 ``final_score``，这里不引入任何新的
      评分公式，也不改动它；
    * **角色多样性**用 FeatureBuilder 已经推断好的 ``role_hint``，配合上面
      的既有词汇表，不新造角色；
    * **风险约束**由调用方保证：传进来的已经是
      ``eligible and not manual_review_required`` 的 safe_pool。

    ``limit <= 0`` 或候选本来就不足时原样返回，绝不报错。
    """
    if limit <= 0 or len(safe_pool) <= limit:
        return list(safe_pool)

    # 同分时用 user_id 兜底，保证同一份输入得到同一份初选结果（可复现）。
    ordered = sorted(safe_pool, key=lambda node: (-node.final_score, node.user_id))

    roles = tuple(preferred_role_hints) or DEFAULT_PREFILTER_ROLE_HINTS
    # 每个角色先预留均分名额，避免"分数最高的 15 人全都是同一个角色"，
    # 让 LLM 无论怎么重排都选不出合理的角色组合。
    per_role = max(1, limit // max(1, len(roles)))

    picked: list[NodeScore] = []
    used_ids: set[str] = set()
    taken_per_role: dict[str, int] = {}

    for role_hint in roles:
        for node in ordered:
            if taken_per_role.get(role_hint, 0) >= per_role:
                break
            if node.role_hint != role_hint or node.user_id in used_ids:
                continue
            picked.append(node)
            used_ids.add(node.user_id)
            taken_per_role[role_hint] = taken_per_role.get(role_hint, 0) + 1

    # 名额没占满时用剩下的高分节点补齐（含上面没覆盖到的角色）。
    for node in ordered:
        if len(picked) >= limit:
            break
        if node.user_id in used_ids:
            continue
        picked.append(node)
        used_ids.add(node.user_id)

    # 最后仍按 final_score 排序，保证进入 prompt 与下游的顺序是"高分在前"，
    # 也保证 LLM 不返回时纯规则路径拿到的顺序与既有行为一致。
    return sorted(picked, key=lambda node: (-node.final_score, node.user_id))


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
        candidate_limit: int | None = None,
    ) -> SelectionResult:
        if target_user_id:
            return self._select_targeted(score_result, target_user_id)

        max_selected = score_result.event.constraints.max_selected_nodes

        # 多人分发必须指定平台：只在候选池中保留归属该平台的数字人，
        # 保证"能且只能选择该平台上已有的数字人"。
        candidate_nodes = self._filter_nodes_by_platform(
            score_result.node_scores, dispatch_platform
        )

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

        # 第七轮：规则初选。
        # 只在 LLM 重排真的要跑时才收敛候选池——纯规则路径（use_llm=False）
        # 保持原有候选池不变，行为与第七轮之前完全一致。
        # 位置在 _llm_selection_plan 之前，因此后续 prompt 里出现的人
        # 就是这批初选结果，不存在"先拼 30 人再截断字符串"的情况。
        if use_llm:
            safe_pool = prefilter_selector_candidates(
                safe_pool,
                limit=(
                    DEFAULT_SELECTOR_CANDIDATE_LIMIT
                    if candidate_limit is None
                    else candidate_limit
                ),
                preferred_role_hints=self._rule_preferred_roles(score_result),
            )

        selected_nodes: list[SelectedNode] = []
        used_ids: set[str] = set()
        llm_selection_plan = self._llm_selection_plan(
            score_result=score_result,
            safe_pool=safe_pool,
            max_selected=max_selected,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=llm_client,
        )
        preferred_roles = self._preferred_roles(score_result, llm_selection_plan)
        safe_pool = self._reorder_pool_with_llm(safe_pool, llm_selection_plan)

        for role in preferred_roles:
            if len(selected_nodes) >= max_selected:
                break
            candidate = self._first_by_role(
                safe_pool,
                role,
                used_ids,
                llm_selection_plan=llm_selection_plan,
            )
            if candidate is not None:
                selected_nodes.append(
                    self._make_selected_node(
                        candidate,
                        len(selected_nodes) + 1,
                        "primary",
                        llm_selection_plan=llm_selection_plan,
                    )
                )
                used_ids.add(candidate.user_id)

        for candidate in safe_pool:
            if len(selected_nodes) >= max_selected:
                break
            if candidate.user_id in used_ids:
                continue
            selected_nodes.append(
                self._make_selected_node(
                    candidate,
                    len(selected_nodes) + 1,
                    "primary",
                    llm_selection_plan=llm_selection_plan,
                )
            )
            used_ids.add(candidate.user_id)

        for candidate in review_pool:
            if len(selected_nodes) >= max_selected:
                break
            if candidate.user_id in used_ids:
                continue
            selected_nodes.append(
                self._make_selected_node(
                    candidate,
                    len(selected_nodes) + 1,
                    "primary",
                    llm_selection_plan=llm_selection_plan,
                )
            )
            used_ids.add(candidate.user_id)

        selected_nodes = self._rebalance_selected_roles(selected_nodes)

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
        target user is forced to ``core_publish_node`` / ``stage_1_launch`` so
        that downstream StrategyGenerator treats them as the core publisher.
        The target's simulated platform (``NodeScore.platform``) is surfaced in
        both ``selection_reasons`` and ``SelectionSummary.target_platform`` so
        the caller can see which platform the chosen person belongs to.

        Fail-fast when the target is missing from the candidate pool — we do
        not want to silently fall back to automatic selection.
        """
        target_role = "core_publish_node"
        dispatch_stage = "stage_1_launch"

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
    ) -> SelectedNode:
        role_hint = self._resolved_role_hint(node, llm_selection_plan)
        selected_role = self._selected_role(role_hint)
        dispatch_stage = self._dispatch_stage(role_hint)
        dispatch_priority = self._dispatch_priority(node.priority_tier)
        selection_reasons = list(node.selection_reasons)
        llm_reasoning = self._llm_reasoning(node.user_id, llm_selection_plan)
        selection_reasons.extend(llm_reasoning)
        # 第十三轮：把"模型排序是否真的生效"写进结果，便于审计。
        # * ``llm_reorder=`` 记录本次重排状态（applied / partial(N/M) / invalid）；
        # * 模型只给了顺序、没给文字理由时，补一条 ``llm_order_rank=`` 说明
        #   该节点的位次来自模型输出（不是规则分数冒充的）。
        if llm_selection_plan:
            status = llm_selection_plan.get("llm_reorder_status")
            if status:
                selection_reasons.append(f"llm_reorder={status}")
            order_map = llm_selection_plan.get("order_map", {})
            if order_map and not llm_reasoning and node.user_id in order_map:
                selection_reasons.append(f"llm_order_rank={order_map[node.user_id] + 1}")
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

    def _rule_preferred_roles(self, score_result: ScoreResult) -> tuple[str, ...]:
        """规则初选阶段用的角色偏好。

        此时 LLM 结果还不存在，因此刻意传 ``llm_selection_plan=None``，
        让 :meth:`_preferred_roles` 走它原有的"取事件 target_roles"分支。
        这样初选和最终选人用的是同一份角色词汇表，不会两套口径。
        """
        return self._preferred_roles(score_result, None)

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
        # 第七轮：模型可能返回候选集之外的 user_id（幻觉或 ID 漂移）。
        # 记录本次真实候选集，解析阶段只接受其中的 ID。
        shortlist_ids = {node.user_id for node in shortlist}
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
        # 第十三轮：默认只让模型输出排序后的 user_id（见 DEFAULT_SELECTOR_OUTPUT_MODE）。
        # 两种模式共用同一套解析、校验、去重与安全过滤。
        if DEFAULT_SELECTOR_OUTPUT_MODE == "id_order":
            user_prompt = build_selector_id_order_user_prompt(
                event=score_result.event,
                max_selected=max_selected,
                shortlist=shortlist_payload,
            )
        else:
            user_prompt = build_selector_user_prompt(
                event=score_result.event,
                max_selected=max_selected,
                shortlist=shortlist_payload,
            )
        try:
            response = client.generate_json(
                system_prompt=build_selector_system_prompt(),
                user_prompt=user_prompt,
            )
        except (LLMClientError, OSError, ValueError, TypeError):
            return None

        selected_order = response.get("selected_order")
        fallback_order = response.get("fallback_order")
        global_notes = response.get("global_notes")
        # 第十三轮：连 ``selected_order`` 都不是数组（空响应 / 键名漂移 / 非 JSON）
        # 同样要标记为 invalid，而不是静默退回规则路径。
        if not isinstance(selected_order, list):
            return {
                "order_map": {},
                "role_map": {},
                "reasoning_map": {},
                "fallback_order": [],
                "global_notes": [],
                "llm_reorder_status": "invalid",
            }

        order_map: dict[str, int] = {}
        role_map: dict[str, str] = {}
        reasoning_map: dict[str, list[str]] = {}
        for index, item in enumerate(selected_order):
            # 第八轮（续）：模型有时会把 selected_order 退化成纯 ID 数组
            # （实测"只要求少写"的提示词会让 qwen3:8b 直接输出 ["96961", ...]）。
            # 以前这种条目会被整批丢弃 → order_map 为空 → 静默退回纯规则路径，
            # 一次 6 秒的调用完全不起作用且没有任何报错。这里改成接纳裸 ID：
            # 它只提供**顺序**，角色与理由仍回退到规则侧（role_hint / 空理由），
            # 安全校验不变——ID 仍必须在本次候选集内。
            if isinstance(item, (str, int)):
                item = {"user_id": str(item)}
            if not isinstance(item, dict):
                continue
            user_id = str(item.get("user_id", "")).strip()
            # 第七轮：只接受本次初选候选集内的 user_id。集合外的 ID 直接丢弃，
            # 不会污染 order_map / role_map / reasoning_map。若全部无效，
            # 下方 order_map 为空会沿用原有的"返回 None → 走纯规则"兜底路径。
            if not user_id or user_id not in shortlist_ids:
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

        # fallback_order 同样只保留候选集内的 ID，避免把不存在的候选
        # 当成"可补位节点"透出给下游。
        normalized_fallback = [
            str(item).strip()
            for item in fallback_order
            if str(item).strip() and str(item).strip() in shortlist_ids
        ] if isinstance(fallback_order, list) else []
        normalized_notes = [
            str(item).strip()
            for item in global_notes
            if str(item).strip()
        ] if isinstance(global_notes, list) else []

        # 第十三轮：LLM 重排的**结果状态**必须可观测。
        # 过去 order_map 为空就直接 ``return None``，调用方无从区分
        # "没跑 LLM" 与 "跑了但模型输出无效"，容易出现"静默回退却当成成功重排"。
        # 这里改成：只要真的发出过请求，就返回一份带状态的 plan。
        valid_count = len(order_map)
        if valid_count == 0:
            status = "invalid"
        elif valid_count < max_selected:
            status = f"partial({valid_count}/{max_selected})"
        else:
            status = "applied"
        return {
            "order_map": order_map,
            "role_map": role_map,
            "reasoning_map": reasoning_map,
            "fallback_order": normalized_fallback,
            "global_notes": normalized_notes,
            "llm_reorder_status": status,
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
