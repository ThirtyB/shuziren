from __future__ import annotations

from collections import Counter

import pandas as pd

from .models import (
    DispatchStrategy,
    SelectionResult,
    SelectedNode,
    StrategyContentPlan,
    StrategyFrequencyPlan,
    StrategyNodePlan,
    StrategyPlatformPlan,
    StrategyResult,
    StrategyRiskControl,
    StrategyStagePlan,
    StrategySummary,
)
from .platform_profiles import normalize_platform_key


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class StrategyGenerator:
    def generate(self, selection_result: SelectionResult) -> StrategyResult:
        time_plan = self._build_time_plan(selection_result.event.constraints.campaign_window_hours)
        frequency_plan = self._build_frequency_plan(
            max_frequency_per_day=selection_result.event.constraints.max_frequency_per_day,
            risk_level=selection_result.event.constraints.risk_level,
        )
        platform_plan = self._build_platform_plan(
            selection_result.event.constraints.allowed_platforms,
            dispatch_platform=self._resolved_dispatch_platform(selection_result),
        )
        content_plan = self._build_content_plan(
            event_type=selection_result.event.event_type,
            target_goal=selection_result.event.target_goal,
            risk_level=selection_result.event.constraints.risk_level,
        )

        selected_nodes = [
            self._build_node_plan(
                node=node,
                time_plan=time_plan,
                frequency_plan=frequency_plan,
                content_plan=content_plan,
                bucket="primary",
            )
            for node in selection_result.selected_nodes
        ]
        fallback_nodes = [
            self._build_node_plan(
                node=node,
                time_plan=time_plan,
                frequency_plan=frequency_plan,
                content_plan=content_plan,
                bucket="fallback",
            )
            for node in selection_result.fallback_nodes
        ]
        stage_plans = self._build_stage_plans(
            selected_nodes=selected_nodes,
            time_plan=time_plan,
            target_goal=selection_result.event.target_goal,
        )
        risk_control = self._build_risk_control(
            selection_result=selection_result,
            selected_nodes=selected_nodes,
            fallback_nodes=fallback_nodes,
        )
        explainability = self._build_explainability(
            selection_result=selection_result,
            selected_nodes=selected_nodes,
            platform_plan=platform_plan,
            risk_control=risk_control,
        )

        strategy = DispatchStrategy(
            target_object=self._build_target_object(selection_result),
            objective=self._objective_text(selection_result.event.target_goal),
            time_plan=time_plan,
            frequency_plan=frequency_plan,
            platform_plan=platform_plan,
            content_plan=content_plan,
            risk_control=risk_control,
            explainability=explainability,
        )

        # 推导整体状态：
        #   * selected_count == 0  →  no_eligible_nodes
        #   * 有主选 + 需要人工复核  →  manual_required
        #   * 有主选 + 风险中等 / 全部通过  →  ok
        if not selected_nodes:
            status = "no_eligible_nodes"
            status_reason = (
                f"候选池中没有任何数字人通过主选阈值（事件 {selection_result.event.event_id}），"
                "需人工挑选首发账号"
            )
        elif risk_control.review_required:
            status = "manual_required"
            status_reason = "主选节点含需人工复核项"
        else:
            status = "ok"
            status_reason = ""

        summary = StrategySummary(
            event_id=selection_result.event.event_id,
            event_type=selection_result.event.event_type,
            selected_count=len(selected_nodes),
            fallback_count=len(fallback_nodes),
            primary_platform=platform_plan.primary_platform,
            estimated_total_dispatches=sum(node.frequency_per_day for node in selected_nodes),
            review_required=risk_control.review_required,
            avg_selected_final_score=round(
                sum(node.final_score for node in selected_nodes) / max(len(selected_nodes), 1),
                6,
            ),
            status=status,
            status_reason=status_reason,
            dispatch_platform=selection_result.summary.dispatch_platform,
            target_platform=selection_result.summary.target_platform,
        )

        return StrategyResult(
            event=selection_result.event,
            product_context=selection_result.product_context,
            selection_summary=selection_result.summary,
            summary=summary,
            stage_plans=stage_plans,
            selected_nodes=selected_nodes,
            fallback_nodes=fallback_nodes,
            strategy=strategy,
        )

    def build_node_plan_from_strategy(
        self,
        *,
        node: SelectedNode,
        strategy: DispatchStrategy,
        bucket: str = "primary",
    ) -> StrategyNodePlan:
        """用已有策略的时间/频率/内容计划补全单个节点的行动级字段。

        开多平台分发时，每个平台的选人由 ``PlatformDispatcher`` 独立完成、并不
        经过 ``Selector``；但"被选中的数字人该在哪个阶段、以什么频率、用哪种
        内容风格执行"必须与多人分发完全同源。这里复用 ``generate`` 使用的同一段
        节点规划逻辑，避免调用方各自推导而使两种模式的动作口径漂移。
        """

        return self._build_node_plan(
            node=node,
            time_plan=strategy.time_plan,
            frequency_plan=strategy.frequency_plan,
            content_plan=strategy.content_plan,
            bucket=bucket,
        )

    def to_frame(self, strategy_result: StrategyResult, bucket: str = "primary") -> pd.DataFrame:
        if bucket == "fallback":
            rows = [item.model_dump() for item in strategy_result.fallback_nodes]
        else:
            rows = [item.model_dump() for item in strategy_result.selected_nodes]
        return pd.DataFrame(rows, columns=list(StrategyNodePlan.model_fields.keys()))

    def stage_frame(self, strategy_result: StrategyResult) -> pd.DataFrame:
        rows = [item.model_dump() for item in strategy_result.stage_plans]
        return pd.DataFrame(rows, columns=list(StrategyStagePlan.model_fields.keys()))

    def _build_time_plan(self, campaign_window_hours: int) -> dict[str, str]:
        total_hours = max(int(campaign_window_hours), 1)
        if total_hours == 1:
            launch_end = 1
            engage_end = 1
        elif total_hours == 2:
            launch_end = 1
            engage_end = 2
        else:
            launch_end = max(1, min(total_hours - 1, round(total_hours * 0.10)))
            engage_end = max(launch_end + 1, min(total_hours, round(total_hours * 0.35)))
            if engage_end <= launch_end:
                engage_end = min(total_hours, launch_end + 1)

        return {
            "stage_1_launch": f"T0 - T+{launch_end}h",
            "stage_2_engage": f"T+{launch_end}h - T+{engage_end}h",
            "stage_2_support": f"T+{launch_end}h - T+{engage_end}h",
            "stage_3_amplify": f"T+{engage_end}h - T+{total_hours}h",
        }

    def _build_frequency_plan(
        self,
        max_frequency_per_day: int,
        risk_level: str,
    ) -> StrategyFrequencyPlan:
        global_cap = max(1, max_frequency_per_day)
        core_limit = min(2, global_cap)
        interaction_limit = min(global_cap, 3)
        amplification_limit = min(2, global_cap)
        support_limit = 1 if risk_level == "high" else min(2, global_cap)

        notes = [
            f"single_node_daily_cap<={global_cap}",
            "avoid_identical_messages_in_short_window",
        ]
        if risk_level == "high":
            core_limit = 1
            interaction_limit = min(2, global_cap)
            amplification_limit = 1
            notes.append("high_risk_event_reduce_clustered_dispatch")
        elif risk_level == "low":
            notes.append("low_risk_event_allows_full_three_stage_schedule")

        return StrategyFrequencyPlan(
            global_cap_per_day=global_cap,
            core_publish_node_per_day=core_limit,
            interaction_response_node_per_day=interaction_limit,
            amplification_node_per_day=amplification_limit,
            support_node_per_day=support_limit,
            notes=notes,
        )

    def _resolved_dispatch_platform(self, selection_result: SelectionResult) -> str | None:
        """确定本次策略的主平台。

        * 单人分发：使用目标数字人所属平台（``target_platform``）；
        * 多人分发：使用显式指定的 ``dispatch_platform``；
        * 都没有时返回 ``None``，由 :meth:`_build_platform_plan` 回退到事件允许平台。
        """

        summary = selection_result.summary
        return (
            summary.dispatch_platform
            or summary.target_platform
            or None
        )

    def _build_platform_plan(
        self,
        allowed_platforms: list[str],
        dispatch_platform: str | None = None,
    ) -> StrategyPlatformPlan:
        platforms = allowed_platforms or ["facebook_simulated"]
        primary = dispatch_platform or platforms[0]
        secondary = [
            platform
            for platform in platforms
            if normalize_platform_key(platform) != normalize_platform_key(primary)
        ]
        execution_mode = "multi_platform_simulated" if secondary else "single_platform_simulated"

        notes = ["keep_core_message_consistent_across_nodes"]
        if dispatch_platform:
            notes.append(f"dispatch_platform={dispatch_platform}")
        if secondary:
            notes.append("stagger_secondary_platforms_after_primary_launch")
        else:
            notes.append("current_baseline_focuses_on_single_platform_execution")

        return StrategyPlatformPlan(
            primary_platform=primary,
            secondary_platforms=secondary,
            execution_mode=execution_mode,
            coordination_notes=notes,
        )

    def _build_content_plan(
        self,
        event_type: str,
        target_goal: str,
        risk_level: str,
    ) -> StrategyContentPlan:
        if event_type == "public_opinion_response":
            core_text = "事实澄清与立场说明型内容，优先给出边界、结论和可验证信息。"
            interaction_text = "问答式回应内容，集中处理高频疑问并纠正误读。"
            amplification_text = "摘要式复述核心事实，避免情绪化放大与未经证实的信息扩散。"
            support_text = "补充背景材料、权威引用或案例脉络，帮助稳定讨论方向。"
        elif event_type == "activity_announcement":
            core_text = "活动公告型内容，突出时间、参与方式、核心亮点和行动入口。"
            interaction_text = "答疑与规则说明型内容，处理报名、参与门槛和流程问题。"
            amplification_text = "亮点摘录与提醒型内容，适合做二次扩散和阶段性提醒。"
            support_text = "补充体验反馈、活动价值和用户视角，增强可信度。"
        elif target_goal == "response":
            core_text = "说明型首发内容，先统一事实边界，再给出明确回应。"
            interaction_text = "高频问题回应内容，重点压缩误解和冲突升级空间。"
            amplification_text = "统一口径的简版转述内容，帮助外圈用户快速获取主信息。"
            support_text = "补充背景和上下文，减少断章取义。"
        elif target_goal == "engagement":
            core_text = "观点引导型首发内容，抛出明确议题并设置讨论锚点。"
            interaction_text = "提问式、跟帖式互动内容，承接评论区讨论并回收反馈。"
            amplification_text = "摘要式扩散内容，提炼亮点并引导二次参与。"
            support_text = "补充案例、阅读体验或用户视角，维持长尾互动。"
        else:
            core_text = "主信息发布型内容，先完成首轮触达和统一表达。"
            interaction_text = "轻量互动跟进内容，用于承接问题和提升停留讨论。"
            amplification_text = "摘要转述型内容，用于扩大外圈覆盖。"
            support_text = "辅助补充内容，用于增强信息完整度。"

        guardrails = [
            "避免高频重复表达",
            "避免同一时间窗口内节点内容完全同质化",
            "确保核心事实和口径一致",
        ]
        if risk_level in {"medium", "high"}:
            guardrails.append("避免使用情绪化、对立化或未经证实的表述")
        if risk_level == "high":
            guardrails.append("放大节点仅做事实摘要，不追加推测性扩写")

        return StrategyContentPlan(
            core_publish_node=core_text,
            interaction_response_node=interaction_text,
            amplification_node=amplification_text,
            support_node=support_text,
            general_guardrails=guardrails,
        )

    def _build_node_plan(
        self,
        node: SelectedNode,
        time_plan: dict[str, str],
        frequency_plan: StrategyFrequencyPlan,
        content_plan: StrategyContentPlan,
        bucket: str,
    ) -> StrategyNodePlan:
        # 行动级字段（selected_role / dispatch_stage / timing_window /
        # frequency_per_day / recommended_action / suggested_content_style）
        # **只在 primary 桶填充**。fallback 节点是"备选候选名单"，它们
        # 不会在流水线中执行任何动作——任何把这些字段填进 fallback 节点
        # 都会让消费者误以为它们会"待命执行"，是组装错误。
        if bucket == "primary":
            timing_window = time_plan.get(node.dispatch_stage, "")
            frequency = self._frequency_for_role(node.selected_role, frequency_plan)
            recommended_action = self._recommended_action(node.selected_role, bucket)
            suggested_content_style = self._content_style_for_role(node.selected_role, content_plan)
            selected_role = node.selected_role
            dispatch_stage = node.dispatch_stage
            dispatch_priority = node.dispatch_priority
        else:
            timing_window = ""
            # ``frequency_per_day`` 模型要求 ``ge=1``，因此保留 1 作为
            # "占位"；调用方应通过 ``selection_bucket == "fallback"`` 识别。
            frequency = 1
            recommended_action = "（备选候选 — 等待主选补位决策，由人工指定动作）"
            suggested_content_style = ""
            selected_role = "unspecified"
            dispatch_stage = "unspecified"
            dispatch_priority = "p4"

        rationale = _dedupe_preserve_order(
            [
                *node.selection_reasons[:4],
                *[reason for reason in node.selection_reasons if reason.startswith("llm_selector=")],
                f"matched_keywords={','.join(node.matched_keywords)}" if node.matched_keywords else "",
                f"risk_level={node.risk_level}",
                "manual_review_required" if node.manual_review_required else "",
            ]
        )

        return StrategyNodePlan(
            user_id=node.user_id,
            user_name=node.user_name,
            platform=node.platform,
            selection_rank=node.selection_rank,
            selection_bucket=bucket,
            selected_role=selected_role,
            dispatch_stage=dispatch_stage,
            dispatch_priority=dispatch_priority,
            final_score=node.final_score,
            risk_level=node.risk_level,
            manual_review_required=node.manual_review_required,
            follower_count=node.follower_count,
            friend_count=node.friend_count,
            neighbor_count=node.neighbor_count,
            mutual_neighbor_count=node.mutual_neighbor_count,
            received_interaction_count=node.received_interaction_count,
            made_interaction_count=node.made_interaction_count,
            influence_score=node.influence_score,
            diffusion_score=node.diffusion_score,
            topic_match_score=node.topic_match_score,
            stability_score=node.stability_score,
            llm_feature_score=node.llm_feature_score,
            risk_flags=node.risk_flags,
            matched_keywords=node.matched_keywords,
            semantic_tags=node.semantic_tags,
            timing_window=timing_window,
            frequency_per_day=frequency,
            recommended_action=recommended_action,
            suggested_content_style=suggested_content_style,
            rationale=rationale,
        )

    def _build_stage_plans(
        self,
        selected_nodes: list[StrategyNodePlan],
        time_plan: dict[str, str],
        target_goal: str,
    ) -> list[StrategyStagePlan]:
        stage_specs = {
            "stage_1_launch": ("启动期", "建立首轮叙事锚点并完成首发触达。"),
            "stage_2_engage": ("互动期", "承接评论与问答，提升讨论深度与反馈回收。"),
            "stage_2_support": ("支持期", "补充背景与长尾内容，稳定讨论连续性。"),
            "stage_3_amplify": ("扩散期", "推动二次传播，扩展外圈覆盖范围。"),
        }

        result: list[StrategyStagePlan] = []
        for stage_name in ("stage_1_launch", "stage_2_engage", "stage_2_support", "stage_3_amplify"):
            nodes = [node for node in selected_nodes if node.dispatch_stage == stage_name]
            if not nodes:
                continue
            stage_label, objective = stage_specs[stage_name]
            result.append(
                StrategyStagePlan(
                    stage_name=stage_name,
                    stage_label=stage_label,
                    time_window=time_plan.get(stage_name, ""),
                    objective=objective,
                    node_count=len(nodes),
                    node_ids=[node.user_id for node in nodes],
                    selected_roles=_dedupe_preserve_order([node.selected_role for node in nodes]),
                    content_focus=self._stage_content_focus(stage_name, target_goal),
                    success_signal=self._stage_success_signal(stage_name, target_goal),
                )
            )
        return result

    def _build_risk_control(
        self,
        selection_result: SelectionResult,
        selected_nodes: list[StrategyNodePlan],
        fallback_nodes: list[StrategyNodePlan],
    ) -> StrategyRiskControl:
        manual_review_node_ids = [node.user_id for node in selected_nodes if node.manual_review_required]
        review_required = bool(manual_review_node_ids) or selection_result.event.constraints.risk_level == "high"
        if not selected_nodes:
            review_required = True

        # 当主选为空时，禁止把 fallback 描述成"可补位"——它们只是
        # 候补名单，必须由人工指定才能升级为主选；否则消费者会误以为
        # 系统会自动切换到 fallback 节点。
        if not selected_nodes:
            fallback_trigger = [
                "当前没有主选节点可被自动补位；如需继续推进，请人工指定首发账号或调整选人阈值",
            ]
        else:
            fallback_trigger = [
                "stage_1_launch 未按计划执行时，启用前 1-2 个 fallback 节点补位",
                "评论区出现明显负向升级时，暂停 amplification 节点并回到 interaction_response 节点",
            ]
            if manual_review_node_ids:
                fallback_trigger.append("人工复核未通过时，优先替换为低风险 fallback 节点")

        notes = [
            f"事件风险等级={selection_result.event.constraints.risk_level}",
            "避免同一批节点在短时间内重复触发相同内容",
        ]
        if selected_nodes:
            notes.append("优先观察 stage_1_launch 的首轮反馈，再决定是否扩大扩散范围")
        if manual_review_node_ids:
            notes.append(f"需人工复核节点={','.join(manual_review_node_ids)}")
        if selection_result.event.constraints.risk_level == "high":
            notes.append("高风险事件建议先以核心发布与互动回应为主，放大节点从严控制")
        if selected_nodes and fallback_nodes:
            notes.append(f"已准备 {len(fallback_nodes)} 个备选节点用于替换或补位")
        if not selected_nodes:
            notes.append(
                "当前规则下没有安全主选节点；下方 fallback_nodes 只是供人工挑选的候补名单，"
                "不会在自动流水线中执行"
            )

        return StrategyRiskControl(
            risk_level=selection_result.event.constraints.risk_level,
            review_required=review_required,
            manual_review_node_ids=manual_review_node_ids,
            fallback_trigger=fallback_trigger,
            notes=notes,
        )

    def _build_explainability(
        self,
        selection_result: SelectionResult,
        selected_nodes: list[StrategyNodePlan],
        platform_plan: StrategyPlatformPlan,
        risk_control: StrategyRiskControl,
    ) -> list[str]:
        role_distribution = Counter(node.selected_role for node in selected_nodes)
        distribution_text = ", ".join(f"{role}={count}" for role, count in role_distribution.items())
        target_object = ", ".join(self._build_target_object(selection_result))

        explainability = []
        if selected_nodes:
            explainability.append(
                f"本次策略为事件 {selection_result.event.event_id} 选出 "
                f"{len(selected_nodes)} 个主选节点和 {len(selection_result.fallback_nodes)} 个备选节点。"
            )
            explainability.append(
                f"角色分布为 {distribution_text}，用于覆盖首发、互动承接和二次扩散三个环节。"
            )
        else:
            explainability.append(
                f"本次策略为事件 {selection_result.event.event_id} 选出 0 个主选节点；"
                f"下方 {len(selection_result.fallback_nodes)} 个 fallback 节点仅作为"
                "人工挑选首发账号的候补名单，不会被自动执行。"
            )
        explainability.append(
            f"主平台为 {platform_plan.primary_platform}，传播窗口为 "
            f"{selection_result.event.constraints.campaign_window_hours} 小时。"
        )
        explainability.append(f"目标圈层聚焦于 {target_object}。")
        explainability.append(
            f"当前事件风险等级为 {selection_result.event.constraints.risk_level}，"
            f"人工复核需求={str(risk_control.review_required).lower()}。"
        )
        if selected_nodes:
            explainability.append(
                "主选节点的平均 final_score="
                f"{sum(node.final_score for node in selected_nodes) / len(selected_nodes):.3f}。"
            )
        return explainability

    def _build_target_object(self, selection_result: SelectionResult) -> list[str]:
        audience_map = {
            "parent_child": "亲子阅读相关群体",
            "reading_interest": "阅读兴趣相关群体",
            "english_learning": "英语启蒙相关群体",
            "education_practitioner": "教育从业者",
            "general_public": "泛公众",
        }

        target_object: list[str] = []
        for audience in selection_result.event.target_audience:
            target_object.append(audience_map.get(audience, audience))
        if not target_object:
            target_object.extend(selection_result.event.extracted_keywords[:3])
        if not target_object:
            target_object.append("泛公众")
        return _dedupe_preserve_order(target_object)

    def _objective_text(self, target_goal: str) -> str:
        goal_map = {
            "awareness": "扩大事件触达范围并建立基础认知。",
            "engagement": "提升讨论度、互动率与反馈回收效率。",
            "response": "快速回应疑问并稳定舆情节奏。",
            "conversion": "引导目标人群完成后续行动。",
        }
        return goal_map.get(target_goal, target_goal or "形成可执行的影响力事件分发策略。")

    def _frequency_for_role(
        self,
        selected_role: str,
        frequency_plan: StrategyFrequencyPlan,
    ) -> int:
        mapping = {
            "core_publish_node": frequency_plan.core_publish_node_per_day,
            "interaction_response_node": frequency_plan.interaction_response_node_per_day,
            "amplification_node": frequency_plan.amplification_node_per_day,
            "support_node": frequency_plan.support_node_per_day,
        }
        return mapping.get(selected_role, 1)

    def _recommended_action(self, selected_role: str, bucket: str) -> str:
        base_mapping = {
            "core_publish_node": "发布首条核心内容，统一事件框架并设置讨论锚点。",
            "interaction_response_node": "承接评论区问题，组织答疑、讨论与反馈回收。",
            "amplification_node": "转述核心信息并扩展到相邻圈层，完成二次扩散。",
            "support_node": "补充背景、案例或用户视角，维持长尾可见度。",
        }
        action = base_mapping.get(selected_role, "执行补充传播任务。")
        if bucket == "fallback":
            return f"作为备选节点待命：{action}"
        return action

    def _content_style_for_role(
        self,
        selected_role: str,
        content_plan: StrategyContentPlan,
    ) -> str:
        mapping = {
            "core_publish_node": content_plan.core_publish_node,
            "interaction_response_node": content_plan.interaction_response_node,
            "amplification_node": content_plan.amplification_node,
            "support_node": content_plan.support_node,
        }
        return mapping.get(selected_role, content_plan.support_node)

    def _stage_content_focus(self, stage_name: str, target_goal: str) -> str:
        if stage_name == "stage_1_launch":
            return "首发主信息、核心观点和初始讨论锚点。"
        if stage_name in {"stage_2_engage", "stage_2_support"} and target_goal == "response":
            return "回应高频问题、纠正误读并稳定讨论节奏。"
        if stage_name in {"stage_2_engage", "stage_2_support"}:
            return "回收互动反馈、引导追问并补充案例。"
        return "提炼亮点并推动二次传播，扩大外圈触达。"

    def _stage_success_signal(self, stage_name: str, target_goal: str) -> str:
        if stage_name == "stage_1_launch":
            return "核心信息被稳定引用，首轮触达完成。"
        if stage_name in {"stage_2_engage", "stage_2_support"} and target_goal == "response":
            return "高频疑问得到收敛，负向升级得到控制。"
        if stage_name in {"stage_2_engage", "stage_2_support"}:
            return "评论、问答或讨论深度明显提升。"
        return "二次扩散持续发生，外圈用户开始跟进。"
