from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.eval_hot_events import (
    _build_llm_node_texts,
    _post_content_text,
    hot_event_to_pipeline_payload,
    run_hot_event_dict_evaluations,
    run_hot_event_evaluation,
    run_hot_event_evaluations,
    run_hot_event_multiplatform_evaluations,
)
from influence_strategy.platform_dispatcher import PlatformSpec
from influence_strategy.models import (
    DispatchStrategy,
    EventConstraints,
    ParsedEvent,
    ProductContext,
    SelectionSummary,
    StrategyNodePlan,
    StrategyResult,
    StrategySummary,
)


class FakeLLMClient:
    provider = "deepseek"
    model = "deepseek-test"
    base_url = "https://api.deepseek.com"

    def describe(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, dict]:
        payload = json.loads(user_prompt)
        if "raw_event" in payload and "fallback_parse" in payload:
            return {
                "event_type": "general_influence_event",
                "target_goal": "awareness",
                "target_audience": ["general_public", "technology"],
                "extracted_keywords": ["chip", "supply", "industry_chain"],
                "risk_level": "medium",
                "semantic_tags": ["芯片供应", "产业链影响"],
                "narrative_frames": ["行业解释", "风险提示"],
                "target_roles": [
                    "core_publish_node",
                    "interaction_response_node",
                    "amplification_node",
                ],
                "negative_constraints": ["避免绝对化表达"],
                "dispatch_preferences": {
                    "candidate_pool_size": 12,
                    "rerank_top_k": 6,
                    "semantic_weight": 0.35,
                    "diversity_weight": 0.20,
                    "risk_weight": 0.20,
                },
                "reasoning": ["llm event parser"],
            }
        if "candidate_cards" in payload:
            nodes = {}
            for index, node in enumerate(payload["candidate_cards"]):
                user_id = node["user_id"]
                nodes[f"id{user_id}"] = {
                    "semantic_relevance_score": max(0.2, 0.92 - index * 0.08),
                    "audience_fit_score": max(0.2, 0.88 - index * 0.06),
                    "role_fit_score": max(0.2, 0.84 - index * 0.04),
                    "narrative_fit_score": max(0.2, 0.80 - index * 0.04),
                    "risk_conflict_score": 0.08,
                    "novelty_score": 0.45 + 0.03 * index,
                    "semantic_tags": [f"tag_{user_id}", "event_fit"],
                    "reasoning": [f"candidate_{user_id}_fit"],
                }
            return {"nodes": nodes}
        if "shortlist" in payload and "constraints" in payload:
            shortlist = payload["shortlist"]
            selected_order = []
            role_cycle = [
                "core_publish_node",
                "interaction_response_node",
                "amplification_node",
            ]
            for index, node in enumerate(shortlist[: payload["constraints"]["max_selected_nodes"]]):
                selected_order.append(
                    {
                        "user_id": node["user_id"],
                        "recommended_role": role_cycle[index % len(role_cycle)],
                        "reasoning": [f"selector_pick_{node['user_id']}"],
                    }
                )
            return {
                "selected_order": selected_order,
                "fallback_order": [
                    node["user_id"]
                    for node in shortlist[payload["constraints"]["max_selected_nodes"] :]
                ],
                "global_notes": ["llm selector used"],
            }

        nodes = {}
        for node in payload["eligible_nodes"]:
            user_id = node["user_id"]
            nodes[f"id{user_id}"] = {
                "post_content": f"LLM generated post content {user_id}",
                "audience_profile": f"LLM generated audience profile {user_id}",
                "audience_interaction_strategy": f"LLM generated interaction strategy {user_id}",
                "cross_digital_human_strategy": f"LLM generated collaboration strategy {user_id}",
            }
        return {"nodes": nodes}


class EvalHotEventsTest(unittest.TestCase):
    def test_hot_event_payload_keeps_ten_opinion_variants(self) -> None:
        hot_event = {
            "event_id": "hot_event_test",
            "domain": "technology",
            "event_title": "AI supply attention",
            "event_summary": "Chip supply tension continues.",
            "opinion_variants": [f"Variant {index}" for index in range(1, 11)],
        }

        payload = hot_event_to_pipeline_payload(hot_event)

        self.assertEqual(payload["event_id"], "hot_event_test")
        self.assertEqual(payload["constraints"]["max_selected_nodes"], 5)
        self.assertEqual(payload["constraints"]["risk_level"], "low")
        self.assertIn("10. Variant 10", payload["event_description"])
        self.assertIn("热点领域: technology", payload["event_description"])
        self.assertEqual(payload["target_audience"], ["general_public"])

    def test_run_hot_event_evaluation_writes_structured_json_and_trace_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            trace_dir = root / "tests" / "pipeline_step_outputs"
            input_path = eval_dir / "hot_event_opinion_variants.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "event_id": "hot_event_eval_001",
                            "domain": "technology",
                            "event_title": "AI chip supply focus",
                            "event_summary": "High-end chip supply is tightening.",
                            "target": "Explain the impact of chip supply changes on the industry chain.",
                            "is_synthetic": True,
                            "opinion_variants": [
                                f"AI chip supply variant {index}"
                                for index in range(1, 11)
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_path, payload = run_hot_event_evaluation(
                workspace_root=root,
                input_path=input_path,
                output_dir=eval_dir / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(payload["事件名称"], "AI chip supply focus")
            self.assertEqual(payload["输出格式版本"], "action_schema_v5_five_dimensions_minimal")
            self.assertEqual(set(payload.keys()), {"事件名称", "输出格式版本", "五维调度策略"})

            five_dimension_strategy = payload["五维调度策略"]
            for dimension in ("目标对象", "时间", "频率", "平台", "内容"):
                self.assertIn(dimension, five_dimension_strategy)

            selected_ids = five_dimension_strategy["目标对象"]["选取数字人id组"]
            self.assertTrue(selected_ids)
            # 三人来自三个不同模拟平台 -> cross_platform（不再硬编码单一平台）
            self.assertEqual(
                five_dimension_strategy["平台"]["平台模式"], "cross_platform"
            )
            platform_items = five_dimension_strategy["平台"]["数字人平台分发"]
            self.assertEqual(len(platform_items), len(selected_ids))
            for item in platform_items:
                self.assertTrue(item["分发平台"].endswith("_simulated"), item)
            self.assertEqual(
                len(five_dimension_strategy["内容"]["数字人发帖内容"]),
                len(selected_ids),
            )
            self.assertTrue(five_dimension_strategy["内容"]["动作清单"])
            first_action = five_dimension_strategy["内容"]["动作清单"][0]
            self.assertIn("动作编号", first_action)
            self.assertIn("执行主体", first_action)
            self.assertIn("动作类型", first_action)
            self.assertIn("目标帖子", first_action)
            self.assertNotIn("目标定位", first_action)
            self.assertNotIn("执行参数", first_action)

            event_trace_dir = trace_dir / "hot_event_eval_001"
            self.assertTrue((event_trace_dir / "00_hot_event_input.json").exists())
            self.assertTrue((event_trace_dir / "02_event_parser_output.json").exists())
            self.assertTrue((event_trace_dir / "03_feature_builder_output.json").exists())
            self.assertTrue((event_trace_dir / "04_scorer_output.json").exists())
            self.assertTrue((event_trace_dir / "05_selector_output.json").exists())
            self.assertTrue((event_trace_dir / "06_strategy_generator_output.json").exists())
            self.assertTrue((event_trace_dir / "07_final_output.json").exists())

    def _make_node(self, *, user_id: str, role: str) -> StrategyNodePlan:
        return StrategyNodePlan(
            user_id=user_id,
            user_name=user_id,
            selected_role=role,
            dispatch_stage="stage_1_launch",
        )

    def test_post_content_fallback_is_natural_narrative(self) -> None:
        node = self._make_node(user_id="1", role="core_publish_node")
        variants = [
            "红海航运紧张使海运成本承压。",
            "多国加强护航说明关键通道安全影响供应链。",
        ]
        post = _post_content_text(
            node=node,
            node_index=1,
            event_title="红海航运安全",
            event_summary="红海及周边海域航运风险上升。",
            variants=variants,
        )
        # 自然叙述：直接复用变体句，不出现内部策略标签
        self.assertEqual(post, "红海航运紧张使海运成本承压。")
        for label in ("传播目标", "内容风格", "执行动作", "参考叙述变体"):
            self.assertNotIn(label, post)

    def test_post_content_fallback_distinct_per_node_and_synthesizes_without_variants(self) -> None:
        variants = ["第一条叙述变体。", "第二条叙述变体。", "第三条叙述变体。"]
        node = self._make_node(user_id="1", role="core_publish_node")
        posts = [
            _post_content_text(
                node=node,
                node_index=index,
                event_title="事件",
                event_summary="事件摘要。",
                variants=variants,
            )
            for index in range(1, len(variants) + 1)
        ]
        # 不同节点按顺序轮转拿到不同变体
        self.assertEqual(len(set(posts)), len(variants))
        self.assertEqual(posts[0], "第一条叙述变体。")

        # 无变体时基于摘要合成，互动节点带自然问句
        interaction_node = self._make_node(user_id="9", role="interaction_response_node")
        synthesized = _post_content_text(
            node=interaction_node,
            node_index=1,
            event_title="事件",
            event_summary="红海及周边海域航运风险上升。",
            variants=[],
        )
        self.assertIn("红海及周边海域航运风险上升。", synthesized)
        self.assertNotIn("传播目标", synthesized)

    def test_content_llm_meta_surfaced_in_trace_and_payload(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            trace_dir = root / "tests" / "pipeline_step_outputs"
            input_path = eval_dir / "hot_event_opinion_variants.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "event_id": "hot_event_meta_001",
                            "domain": "technology",
                            "event_title": "AI chip supply focus",
                            "event_summary": "High-end chip supply is tightening.",
                            "target": "Explain the impact of chip supply changes.",
                            "opinion_variants": [
                                f"AI chip supply variant {index}"
                                for index in range(1, 11)
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_path, payload = run_hot_event_evaluation(
                workspace_root=root,
                input_path=input_path,
                output_dir=eval_dir / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )

            content_dimension = payload["五维调度策略"]["内容"]
            diagnostic = content_dimension["内容生成诊断"]
            self.assertFalse(diagnostic["used"])
            self.assertEqual(diagnostic["fallback_reason"], "llm_disabled")

            # 离线兜底：帖子正文是自然叙述，不是标签模板
            first_post = content_dimension["数字人发帖内容"][0]["发帖内容"]
            self.assertIn("AI chip supply variant", first_post)
            for label in ("传播目标", "内容风格", "执行动作", "参考叙述变体"):
                self.assertNotIn(label, first_post)

            event_trace_dir = trace_dir / "hot_event_meta_001"
            meta_trace = json.loads(
                (event_trace_dir / "08_content_generation_meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta_trace["post_content_source"], "rule_fallback")
            self.assertEqual(
                meta_trace["content_generation"]["fallback_reason"], "llm_disabled"
            )

    def test_run_hot_event_evaluation_replaces_text_fields_with_llm_content(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            input_path = eval_dir / "hot_event_opinion_variants.json"
            input_path.write_text(
                json.dumps(
                    {
                        "event_id": "hot_event_eval_llm",
                        "domain": "technology",
                        "event_title": "AI chip supply test",
                        "event_summary": "High-end chip supply is tightening.",
                        "target": "Test replacing content with LLM output.",
                        "opinion_variants": [f"Variant {index}" for index in range(1, 11)],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_path, payload = run_hot_event_evaluation(
                workspace_root=root,
                input_path=input_path,
                output_dir=eval_dir / "output",
                llm_client=FakeLLMClient(),
                trace_dir=root / "tests" / "pipeline_step_outputs",
            )

            self.assertTrue(output_path.exists())
            content_items = payload["五维调度策略"]["内容"]["数字人发帖内容"]
            self.assertTrue(content_items[0]["发帖内容"].startswith("LLM generated post content"))
            action_types = {
                action["动作类型"]
                for action in payload["五维调度策略"]["内容"]["动作清单"]
            }
            self.assertTrue(action_types & {"reply_comment", "comment_post", "quote_repost", "like_post"})

    def test_llm_text_generation_includes_high_risk_image_nodes(self) -> None:
        node = StrategyNodePlan(
            user_id="risk-node-1",
            user_name="risk_node",
            selected_role="core_publish_node",
            dispatch_stage="stage_1_launch",
            final_score=0.12,
            stability_score=0.05,
            risk_level="high",
            manual_review_required=True,
            recommended_action="publish guarded explanation",
            suggested_content_style="careful factual update",
        )
        parsed_event = ParsedEvent(
            event_id="hot_event_image_risk",
            event_title="Image risk event",
            event_description="A high risk image event.",
            target_goal="Use LLM text while keeping risk controls.",
            event_type="general_influence_event",
            constraints=EventConstraints(risk_level="high"),
        )
        strategy_result = StrategyResult(
            event=parsed_event,
            product_context=ProductContext(product_name="abc_reading"),
            selection_summary=SelectionSummary(
                event_id=parsed_event.event_id,
                event_type=parsed_event.event_type,
                max_selected_nodes=1,
                selected_count=1,
                fallback_count=0,
            ),
            summary=StrategySummary(
                event_id=parsed_event.event_id,
                event_type=parsed_event.event_type,
                selected_count=1,
                fallback_count=0,
                primary_platform="facebook_simulated",
            ),
            stage_plans=[],
            selected_nodes=[node],
            fallback_nodes=[],
            strategy=DispatchStrategy(
                target_object=["general_public"],
                objective="Use LLM text while keeping risk controls.",
            ),
        )

        overrides, metadata = _build_llm_node_texts(
            hot_event={
                "event_id": parsed_event.event_id,
                "source_type": "image",
                "domain": "politics",
                "event_title": parsed_event.event_title,
                "event_summary": parsed_event.event_description,
            },
            strategy_result=strategy_result,
            platform="facebook_simulated",
            target=parsed_event.target_goal,
            workspace_root=None,
            use_llm=True,
            llm_client=FakeLLMClient(),
        )

        self.assertTrue(metadata["used"])
        self.assertIn(node.user_id, overrides)
        self.assertEqual(
            overrides[node.user_id]["post_content"],
            "LLM generated post content risk-node-1",
        )

    def test_run_hot_event_evaluations_writes_first_n_events_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            input_path = eval_dir / "hot_event_opinion_variants.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "event_id": f"hot_event_eval_{index:03d}",
                            "domain": "technology",
                            "event_title": f"Batch event {index}",
                            "event_summary": "Batch evaluation event.",
                            "target": "Test batch output.",
                            "opinion_variants": [
                                f"Batch variant {variant_index}"
                                for variant_index in range(1, 11)
                            ],
                        }
                        for index in range(1, 4)
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            results = run_hot_event_evaluations(
                workspace_root=root,
                input_path=input_path,
                output_dir=eval_dir / "output",
                event_limit=2,
                use_llm=False,
                trace_dir=root / "tests" / "pipeline_step_outputs",
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(
                [output_path.name for output_path, _payload in results],
                [
                    "hot_event_eval_001_strategy_output.json",
                    "hot_event_eval_002_strategy_output.json",
                ],
            )
            self.assertFalse(
                (eval_dir / "output" / "hot_event_eval_003_strategy_output.json").exists()
            )

    def test_run_hot_event_dict_evaluations_writes_image_trace_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            trace_dir = root / "tests" / "pipeline_step_outputs"
            image_path = eval_dir / "image" / "4.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")

            events = [
                {
                    "event_id": "hot_event_005",
                    "domain": "technology",
                    "event_title": "AI chip supply image event",
                    "event_summary": "High-end chip supply is tightening.",
                    "target": "Explain the impact of chip supply changes.",
                    "source_type": "image",
                    "source_image": str(image_path),
                    "image_recognition": {
                        "method": "vision_llm",
                        "confidence": 0.91,
                        "matched_event_id": "hot_event_005",
                        "match_score": 0.87,
                        "match_threshold": 0.72,
                        "fallback_used": False,
                        "warnings": [],
                        "recognized_event": {"event_title": "AI chip supply image event"},
                    },
                    "opinion_variants": ["AI chip supply variant"],
                }
            ]

            results = run_hot_event_dict_evaluations(
                workspace_root=root,
                events=events,
                output_dir=eval_dir / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )

            self.assertEqual(len(results), 1)
            event_trace_dir = trace_dir / "hot_event_005"
            image_input = json.loads((event_trace_dir / "00_image_input.json").read_text(encoding="utf-8"))
            recognition = json.loads((event_trace_dir / "00_image_recognition.json").read_text(encoding="utf-8"))
            self.assertEqual(image_input["source_image"], str(image_path))
            self.assertEqual(recognition["matched_event_id"], "hot_event_005")
            self.assertEqual(recognition["match_threshold"], 0.72)
            self.assertTrue((event_trace_dir / "00_hot_event_input.json").exists())
            self.assertTrue((event_trace_dir / "07_final_output.json").exists())

    def _write_minimal_event_input(
        self, eval_dir: Path, event_id: str = "hot_event_eval_001"
    ) -> Path:
        eval_dir.mkdir(parents=True, exist_ok=True)
        input_path = eval_dir / "hot_event_opinion_variants.json"
        input_path.write_text(
            json.dumps(
                [
                    {
                        "event_id": event_id,
                        "domain": "technology",
                        "event_title": "AI chip supply focus",
                        "event_summary": "High-end chip supply is tightening.",
                        "target": "Explain the impact of chip supply changes on the industry chain.",
                        "is_synthetic": True,
                        "opinion_variants": [
                            f"AI chip supply variant {index}"
                            for index in range(1, 11)
                        ],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return input_path

    def test_trace_of_previous_run_is_preserved_when_modes_share_trace_dir(self) -> None:
        """自动模式与多平台模式共用 trace_dir 时，前一次运行的 trace 必须仍可取回。"""

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)
            input_path = self._write_minimal_event_input(root / "eval")
            trace_dir = root / "tests" / "pipeline_step_outputs"
            event_trace_dir = trace_dir / "hot_event_eval_001"

            run_hot_event_evaluations(
                workspace_root=root,
                input_path=input_path,
                output_dir=root / "eval" / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )
            first_final = json.loads(
                (event_trace_dir / "07_final_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_final["run_label"], "automatic")
            self.assertIsNone(first_final["previous_run_archive"])
            first_payload = first_final["payload"]
            # 多人分发（automatic）保持旧版五维调度策略输出
            self.assertEqual(
                set(first_payload.keys()),
                {"事件名称", "五维调度策略", "输出格式版本"},
            )

            run_hot_event_multiplatform_evaluations(
                workspace_root=root,
                input_path=input_path,
                platform_specs=[
                    PlatformSpec(platform_key="facebook", requested_count=1),
                    PlatformSpec(platform_key="twitter", requested_count=2),
                ],
                output_dir=root / "eval" / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )
            second_final = json.loads(
                (event_trace_dir / "07_final_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(second_final["run_label"], "multiplatform_facebook-twitter")
            self.assertIn("多平台分发策略", second_final["payload"])
            # 多平台最终产物是独立构造的：不含旧版五维调度策略
            self.assertEqual(
                set(second_final["payload"].keys()),
                {"事件名称", "多平台分发策略", "输出格式版本"},
            )
            self.assertNotIn("五维调度策略", second_final["payload"])

            archive_name = second_final["previous_run_archive"]
            self.assertIsNotNone(
                archive_name,
                "第二次运行必须归档而不是直接覆盖第一次的 trace",
            )
            self.assertTrue(
                str(archive_name).endswith("_automatic"),
                f"归档目录名应标明它保存的是哪一次运行，实际为 {archive_name}",
            )
            archive_dir = event_trace_dir / "runs" / str(archive_name)
            self.assertTrue(archive_dir.is_dir())

            # 归档必须包含完整步骤文件，而不是只有最终输出
            for name in (
                "00_hot_event_input.json",
                "02_event_parser_output.json",
                "04_scorer_output.json",
                "05_selector_output.json",
                "07_final_output.json",
            ):
                self.assertTrue((archive_dir / name).exists(), name)

            archived_final = json.loads(
                (archive_dir / "07_final_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(archived_final["run_label"], "automatic")
            self.assertEqual(archived_final["payload"], first_payload)

            # 兼容性：既有 canonical 路径继续指向最新一次运行
            self.assertTrue((event_trace_dir / "07_final_output.json").exists())
            self.assertTrue((event_trace_dir / "00_hot_event_input.json").exists())
            # 归档目录不会被下一次归档再次嵌套
            self.assertFalse((archive_dir / "runs").exists())

    def test_repeated_runs_accumulate_archives_without_losing_earlier_ones(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)
            input_path = self._write_minimal_event_input(root / "eval")
            trace_dir = root / "tests" / "pipeline_step_outputs"
            event_trace_dir = trace_dir / "hot_event_eval_001"

            run_hot_event_evaluations(
                workspace_root=root,
                input_path=input_path,
                output_dir=root / "eval" / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )
            for specs in (
                [PlatformSpec(platform_key="facebook", requested_count=1)],
                [PlatformSpec(platform_key="twitter", requested_count=1)],
            ):
                run_hot_event_multiplatform_evaluations(
                    workspace_root=root,
                    input_path=input_path,
                    platform_specs=specs,
                    output_dir=root / "eval" / "output",
                    use_llm=False,
                    trace_dir=trace_dir,
                )

            runs_dir = event_trace_dir / "runs"
            self.assertTrue(runs_dir.is_dir())
            archive_names = sorted(path.name for path in runs_dir.iterdir())
            self.assertEqual(
                len(archive_names), 2, f"两次历史运行都应被保留，实际为 {archive_names}"
            )
            self.assertTrue(any(name.endswith("_automatic") for name in archive_names))
            self.assertTrue(
                any(name.endswith("_multiplatform_facebook") for name in archive_names),
                archive_names,
            )
            # 被归档的运行仍可读，且内容形态正确
            labels = set()
            for name in archive_names:
                archived = json.loads(
                    (runs_dir / name / "07_final_output.json").read_text(encoding="utf-8")
                )
                labels.add(archived["run_label"])
            self.assertEqual(labels, {"automatic", "multiplatform_facebook"})

            current = json.loads(
                (event_trace_dir / "07_final_output.json").read_text(encoding="utf-8")
            )
            self.assertEqual(current["run_label"], "multiplatform_twitter")
            self.assertIn("多平台分发策略", current["payload"])
            self.assertNotIn("五维调度策略", current["payload"])

    def _write_minimal_workspace(self, root: Path) -> None:
        raw_dir = root / "data" / "raw"
        derived_dir = root / "data" / "derived"
        raw_dir.mkdir(parents=True)
        derived_dir.mkdir(parents=True)

        product_info = {
            "product_name": "abc_reading",
            "domain": "reading",
            "ads": "reading context",
            "influencer_ids": ["1"],
        }
        profiles = {
            "1": {
                "user_id": 1,
                "user_name": "core_industry_observer",
                "user_followers": 2400,
                "user_friends": 80,
                "user_interests": ["chip_supply", "industry_chain", "ai_infrastructure"],
                "user_description": "Focus on chip supply, semiconductor industry chain, and AI infrastructure updates.",
                "platform": "facebook_simulated",
            },
            "2": {
                "user_id": 2,
                "user_name": "amplifier_supply_chain",
                "user_followers": 650,
                "user_friends": 90,
                "user_interests": ["semiconductor", "market_update", "content_diffusion"],
                "user_description": "Suitable for secondary diffusion of semiconductor and supply chain discussions.",
                "platform": "twitter_simulated",
            },
            "3": {
                "user_id": 3,
                "user_name": "responder_qa",
                "user_followers": 380,
                "user_friends": 50,
                "user_interests": ["industry_qa", "technology_policy", "education_exchange"],
                "user_description": "Good at comment interaction and answering technology policy questions.",
                "platform": "instagram_simulated",
            },
        }
        interactions = {
            "1": [{"interact_id": 2, "interact_type": "comment"}],
            "2": [{"interact_id": 1, "interact_type": "reposts"}],
            "3": [{"interact_id": 1, "interact_type": "comment"}],
        }
        enriched_profiles = {
            "1": {
                **profiles["1"],
                "graph_attributes": {
                    "neighbor_count": 3,
                    "mutual_neighbor_count": 1,
                    "received_interaction_count": 8,
                    "received_comment_count": 6,
                    "received_repost_count": 2,
                    "made_interaction_count": 2,
                    "made_repost_count": 2,
                    "isolated": False,
                },
                "neighbors": [],
            },
            "2": {
                **profiles["2"],
                "graph_attributes": {
                    "neighbor_count": 4,
                    "mutual_neighbor_count": 1,
                    "received_interaction_count": 2,
                    "received_comment_count": 1,
                    "received_repost_count": 1,
                    "made_interaction_count": 8,
                    "made_comment_count": 2,
                    "made_repost_count": 6,
                    "isolated": False,
                },
                "neighbors": [],
            },
            "3": {
                **profiles["3"],
                "graph_attributes": {
                    "neighbor_count": 2,
                    "mutual_neighbor_count": 1,
                    "received_interaction_count": 4,
                    "received_comment_count": 4,
                    "made_interaction_count": 3,
                    "made_comment_count": 3,
                    "isolated": False,
                },
                "neighbors": [],
            },
        }

        (raw_dir / "abc_reading_product_info.json").write_text(
            json.dumps(product_info, ensure_ascii=False),
            encoding="utf-8",
        )
        (raw_dir / "abc_reading_profile.graph.anon").write_text(
            json.dumps(profiles, ensure_ascii=False),
            encoding="utf-8",
        )
        (raw_dir / "abc_reading_interaction.graph.anon").write_text(
            json.dumps(interactions, ensure_ascii=False),
            encoding="utf-8",
        )
        (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
            json.dumps(enriched_profiles, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
