from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import NodeScore, ScoreResult, SelectedNode, SelectionResult, SelectionSummary
from .platform_profiles import normalize_platform_key
from .prompts import build_selector_system_prompt, build_selector_user_prompt


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
        mapping = {
            "core_broadcast": "core_publish_node",
            "interaction_response": "interaction_response_node",
            "amplification": "amplification_node",
            "general_support": "support_node",
        }
        return mapping.get(role_hint, "support_node")

    def _rebalance_selected_roles(self, selected_nodes: list[SelectedNode]) -> list[SelectedNode]:
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
            role_hint = self._role_hint_for_selected_role(selected_role)
            reasons = [
                reason
                for reason in node.selection_reasons
                if not reason.startswith("selected_role=") and not reason.startswith("dispatch_stage=")
            ]
            if selected_role != node.selected_role:
                reasons.append(f"role_rebalanced={node.selected_role}->{selected_role}")
            reasons.append(f"selected_role={selected_role}")
            reasons.append(f"dispatch_stage={self._dispatch_stage(role_hint)}")
            rebalanced.append(
                node.model_copy(
                    update={
                        "selected_role": selected_role,
                        "dispatch_stage": self._dispatch_stage(role_hint),
                        "selection_reasons": reasons,
                    }
                )
            )
        return rebalanced

    def _role_hint_for_selected_role(self, selected_role: str) -> str:
        mapping = {
            "core_publish_node": "core_broadcast",
            "interaction_response_node": "interaction_response",
            "amplification_node": "amplification",
            "support_node": "general_support",
        }
        return mapping.get(selected_role, "general_support")

    def _dispatch_stage(self, role_hint: str) -> str:
        mapping = {
            "core_broadcast": "stage_1_launch",
            "interaction_response": "stage_2_engage",
            "amplification": "stage_3_amplify",
            "general_support": "stage_2_support",
        }
        return mapping.get(role_hint, "stage_2_support")

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
