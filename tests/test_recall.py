"""分层召回前移到 FeatureBuilder 之前的回归测试。

覆盖：
  * 5 个全量 max_log 分母在召回子集上**与全量逐字节相等**
  * ``FeatureBuildSummary.recall_size`` / ``recall_mode`` / ``node_count`` 正确
  * platform_aware 召回遵守平台过滤
  * 全量 vs 召回 N=200 跑 Scorer+Selector 后：selected_ids / selected_role /
    risk_level / final_score 一致
  * 真实入口（``eval_hot_events._run_hot_event_evaluation``）正确透传
    dispatch_platform / recall_size=200 / recall_mode=platform_aware
  * summary 展示口径：``total_candidate_count``(35,636) vs ``node_count``(200)
"""

from __future__ import annotations

import json
import sys
import time
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.data_loader import DataLoader  # noqa: E402
from influence_strategy.event_parser import RuleBasedEventParser  # noqa: E402
from influence_strategy.feature_builder import FeatureBuilder  # noqa: E402
from influence_strategy.models import ParsedEvent  # noqa: E402
from influence_strategy.platform_profiles import normalize_platform_key  # noqa: E402
from influence_strategy.scorer import Scorer  # noqa: E402
from influence_strategy.selector import Selector  # noqa: E402

DISPATCH_PLATFORM = "facebook_simulated"


# --------------------------------------------------------------------------- #
# 合成 workspace（足够异构，能跑出有意义的 4 桶排序）
# --------------------------------------------------------------------------- #
def _build_synthetic_workspace(
    tmpdir: str,
    *,
    n_profiles: int = 12,
) -> tuple[Path, FeatureBuilder, dict[str, dict]]:
    """构造 12 个异构 profile：粉丝/邻居/接收/发出/平台都分散。"""
    root = Path(tmpdir)
    raw_dir = root / "data" / "raw"
    derived_dir = root / "data" / "derived"
    raw_dir.mkdir(parents=True)
    derived_dir.mkdir(parents=True)

    product_info = {
        "product_name": "abc_reading",
        "domain": "reading",
        "influencer_ids": ["1"],
    }
    # 12 个异构 profile：覆盖 facebook / weibo / x，followers 0-100k，
    # neighbors 0-50，received 0-200，made 0-200，self_interaction 0-10
    platforms = ["facebook_simulated", "weibo_simulated", "x_simulated",
                 "facebook_simulated", "weibo_simulated", "x_simulated",
                 "facebook_simulated", "weibo_simulated", "x_simulated",
                 "facebook_simulated", "facebook_simulated", "facebook_simulated"]
    profiles: dict[str, dict] = {}
    enriched: dict[str, dict] = {}
    for i in range(1, n_profiles + 1):
        uid = str(i)
        follower = [0, 100, 500, 2000, 5000, 10000, 30000, 60000, 80000, 100000, 200000, 0][i - 1]
        profiles[uid] = {
            "user_id": int(uid),
            "user_name": f"node_{uid}",
            "user_followers": follower,
            "user_friends": i * 10,
            "user_interests": (
                ["亲子阅读", "英语启蒙"] if i in (2, 5, 8, 11) else
                ["general topic"] if i in (1, 4) else
                ["其他话题"]
            ),
            "user_description": (
                "聚焦亲子阅读和英语启蒙的内容创作者" if i in (2, 5, 8, 11) else
                "general interest account" if i in (1, 4) else
                "其他方向创作者"
            ),
            "platform": platforms[i - 1],
        }
        enriched[uid] = {
            **profiles[uid],
            "graph_attributes": {
                "neighbor_count": [0, 1, 5, 10, 20, 30, 50, 5, 15, 25, 40, 3][i - 1],
                "mutual_neighbor_count": [0, 0, 2, 5, 8, 10, 20, 2, 6, 10, 15, 1][i - 1],
                "received_interaction_count": [0, 2, 10, 30, 60, 100, 200, 8, 40, 80, 150, 3][i - 1],
                "made_interaction_count": [0, 1, 5, 15, 30, 50, 100, 4, 20, 40, 80, 2][i - 1],
                "received_comment_count": [0, 1, 5, 15, 30, 50, 100, 4, 20, 40, 80, 2][i - 1],
                "received_repost_count": [0, 1, 5, 15, 30, 50, 100, 4, 20, 40, 70, 1][i - 1],
                "made_comment_count": [0, 0, 3, 8, 15, 25, 50, 2, 10, 20, 40, 1][i - 1],
                "made_repost_count": [0, 1, 2, 7, 15, 25, 50, 2, 10, 20, 40, 1][i - 1],
                "self_interaction_count": [0, 0, 1, 2, 3, 5, 8, 0, 1, 2, 3, 0][i - 1],
            },
            "neighbors": [],
        }
    (raw_dir / "abc_reading_product_info.json").write_text(
        json.dumps(product_info, ensure_ascii=False), encoding="utf-8"
    )
    (raw_dir / "abc_reading_profile.graph.anon").write_text(
        json.dumps(profiles, ensure_ascii=False), encoding="utf-8"
    )
    (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
        json.dumps(enriched, ensure_ascii=False), encoding="utf-8"
    )
    return root, FeatureBuilder(), profiles


def _make_event() -> ParsedEvent:
    return RuleBasedEventParser().parse(
        "围绕亲子阅读和英语启蒙做一次传播活动，提升讨论度。"
    )


# --------------------------------------------------------------------------- #
# 单元测试
# --------------------------------------------------------------------------- #
class RecallUnitTest(unittest.TestCase):
    """合成数据上的 4 个关键不变量。"""

    def test_recall_candidates_respects_platform(self) -> None:
        """召回严格遵守 dispatch_platform。"""
        with TemporaryDirectory() as tmp:
            root, builder, _ = _build_synthetic_workspace(tmp)
            loader = DataLoader(root)
            recall = builder.recall_candidates(
                profiles=loader.load_profiles(),
                enriched_graph_attributes=loader.load_enriched_graph_attributes(),
                event=_make_event(),
                dispatch_platform=DISPATCH_PLATFORM,
                size=200,
                source_user_ids=set(),
            )
            # 召回出来的每个 user_id 在原始 profiles 里都是 facebook 平台
            profiles = loader.load_profiles()
            for uid in recall:
                self.assertEqual(
                    normalize_platform_key(profiles[uid].platform),
                    normalize_platform_key(DISPATCH_PLATFORM),
                    f"user_id={uid} 不在 {DISPATCH_PLATFORM} 平台",
                )

    def test_recall_candidates_size_respects_cap(self) -> None:
        """召回结果 size ≤ 实际平台内人数。"""
        with TemporaryDirectory() as tmp:
            root, builder, _ = _build_synthetic_workspace(tmp)
            loader = DataLoader(root)
            recall = builder.recall_candidates(
                profiles=loader.load_profiles(),
                enriched_graph_attributes=loader.load_enriched_graph_attributes(),
                event=_make_event(),
                dispatch_platform=DISPATCH_PLATFORM,
                size=200,
                source_user_ids=set(),
            )
            # 12 个 profile 中 facebook_simulated 占 6（uid 1,4,7,10,11,12）
            self.assertLessEqual(len(recall), 6)
            self.assertGreater(len(recall), 0)

    def test_summary_recall_fields_off(self) -> None:
        """未开启召回时：recall_size=None, recall_mode=None, node_count=全量。"""
        with TemporaryDirectory() as tmp:
            root, builder, _ = _build_synthetic_workspace(tmp)
            event = _make_event()
            res = builder.build_from_loader(
                workspace_root=root, event=event, use_llm=False
            )
            self.assertIsNone(res.summary.recall_size)
            self.assertIsNone(res.summary.recall_mode)
            self.assertEqual(res.summary.node_count, 12)

    def test_summary_recall_fields_on(self) -> None:
        """开启召回时：recall_size=实际召回数, recall_mode="platform_aware"。"""
        with TemporaryDirectory() as tmp:
            root, builder, _ = _build_synthetic_workspace(tmp)
            event = _make_event()
            res = builder.build_from_loader(
                workspace_root=root,
                event=event,
                use_llm=False,
                recall_size=200,
                recall_mode="platform_aware",
                dispatch_platform=DISPATCH_PLATFORM,
            )
            self.assertEqual(res.summary.recall_mode, "platform_aware")
            self.assertIsNotNone(res.summary.recall_size)
            self.assertGreater(res.summary.recall_size, 0)
            self.assertEqual(res.summary.node_count, res.summary.recall_size)


# --------------------------------------------------------------------------- #
# 真实数据集成测试（35,636 vs 200，5s 上下，可反复跑）
# --------------------------------------------------------------------------- #
class RecallIntegrationTest(unittest.TestCase):
    """真实数据下验证：召回 N=200 与全量在 Scorer+Selector 输出上**逐项一致**。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline_root = PROJECT_ROOT
        cls.events_path = cls.pipeline_root / "eval" / "hot_event_opinion_variants_200.json"
        if not cls.events_path.exists():
            raise unittest.SkipTest(f"缺数据：{cls.events_path}")

    def _load_event_payload(self, event_id: str) -> dict:
        from influence_strategy.eval_hot_events import load_hot_events, hot_event_to_pipeline_payload
        events = load_hot_events(self.events_path)
        hot = next(e for e in events if str(e.get("event_id")) == event_id)
        return hot_event_to_pipeline_payload(
            hot,
            max_selected_nodes=5,
            risk_level=None,
            campaign_window_hours=24,
            max_frequency_per_day=3,
            allowed_platforms=None,
        )

    def _run(self, event_id: str, *, recall_size: int | None) -> tuple[list[dict], float]:
        from influence_strategy.pipeline import StrategyPipeline
        payload = self._load_event_payload(event_id)
        pipeline = StrategyPipeline(product_name="abc_reading")
        t0 = time.monotonic()
        if recall_size is None:
            arts = pipeline.run_with_artifacts(
                workspace_root=self.pipeline_root,
                event_input=payload,
                profile_limit=None,
                use_llm=False,
                dispatch_platform=DISPATCH_PLATFORM,
            )
        else:
            arts = pipeline.run_with_artifacts(
                workspace_root=self.pipeline_root,
                event_input=payload,
                profile_limit=None,
                use_llm=False,
                dispatch_platform=DISPATCH_PLATFORM,
                recall_size=recall_size,
                recall_mode="platform_aware",
            )
        elapsed = time.monotonic() - t0
        snap = [
            {
                "user_id": n.user_id,
                "selected_role": n.selected_role,
                "dispatch_stage": n.dispatch_stage,
                "platform": n.platform,
                "risk_level": n.risk_level,
                "final_score": round(n.final_score, 6),
            }
            for n in arts.selection_result.selected_nodes
        ]
        return snap, elapsed

    def test_max_log_dominator_preserved(self) -> None:
        """200 人子集上的 score 与全量**逐字节相等**（max_log 分母保住了）。"""
        from influence_strategy.feature_builder import FeatureBuilder
        from influence_strategy.event_parser import RuleBasedEventParser
        from influence_strategy.eval_hot_events import hot_event_to_pipeline_payload, load_hot_events

        event_id = "hot_event_001"
        events = load_hot_events(self.events_path)
        hot = next(e for e in events if str(e.get("event_id")) == event_id)
        payload = hot_event_to_pipeline_payload(
            hot,
            max_selected_nodes=5,
            risk_level=None,
            campaign_window_hours=24,
            max_frequency_per_day=3,
            allowed_platforms=None,
        )
        event = RuleBasedEventParser().parse(payload, workspace_root=self.pipeline_root, use_llm=False)
        builder = FeatureBuilder()

        # 全量
        res_full = builder.build_from_loader(
            workspace_root=self.pipeline_root, event=event, use_llm=False
        )
        # 召回 200
        res_recall = builder.build_from_loader(
            workspace_root=self.pipeline_root,
            event=event,
            use_llm=False,
            recall_size=200,
            recall_mode="platform_aware",
            dispatch_platform=DISPATCH_PLATFORM,
        )

        full_by_id = {nf.user_id: nf for nf in res_full.node_features}
        recall_ids = {nf.user_id for nf in res_recall.node_features}

        # 每个召回节点都要在全量里
        missing = recall_ids - set(full_by_id)
        self.assertEqual(missing, set(), f"召回节点不在全量中: {missing}")

        # 5 个核心 score + 比例字段在两者上**逐位相等**
        score_fields = (
            "topic_match_score",
            "influence_score",
            "diffusion_score",
            "activity_score",
            "stability_score",
            "feature_ready_score",
            "mutual_neighbor_ratio",
            "comment_ratio",
            "repost_ratio",
            "profile_completeness_score",
        )
        max_diff = 0.0
        worst_field = ""
        for nf in res_recall.node_features:
            full = full_by_id[nf.user_id]
            for f in score_fields:
                d = abs(getattr(nf, f) - getattr(full, f))
                if d > max_diff:
                    max_diff = d
                    worst_field = f
        self.assertLess(
            max_diff, 1e-9,
            f"召回与全量 score 漂移：max_diff={max_diff} field={worst_field}",
        )

    def test_full_vs_recall200_selection_identical(self) -> None:
        """Scorer+Selector 全量 vs 召回 200：selected_ids/role/platform/risk/score 全等。"""
        full_snap, full_sec = self._run("hot_event_001", recall_size=None)
        rec_snap, rec_sec = self._run("hot_event_001", recall_size=200)
        self.assertEqual(
            [s["user_id"] for s in full_snap],
            [s["user_id"] for s in rec_snap],
            f"selected_ids 不一致:\n full={full_snap}\n recall={rec_snap}",
        )
        self.assertEqual(full_snap, rec_snap, "字段级不一致")
        # 端到端时间显著下降
        self.assertLess(
            rec_sec, full_sec * 0.8,
            f"召回未明显加速：full={full_sec:.2f}s recall={rec_sec:.2f}s",
        )

    def test_full_vs_recall200_e2e_timing(self) -> None:
        """端到端时间：全量 vs 召回 200（仅打印报告用，不强制阈值）。"""
        _, full_sec = self._run("hot_event_001", recall_size=None)
        _, rec_sec = self._run("hot_event_001", recall_size=200)
        print(
            f"\n[recall e2e] hot_event_001 facebook_simulated: "
            f"full={full_sec:.3f}s recall200={rec_sec:.3f}s "
            f"speedup={full_sec / max(rec_sec, 0.001):.2f}x"
        )


# --------------------------------------------------------------------------- #
# 真实入口接线（eval_hot_events 三个入口 + CLI）
# --------------------------------------------------------------------------- #
class RecallEntryWiringTest(unittest.TestCase):
    """真实入口必须透传 dispatch_platform / recall_size=200 / recall_mode。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.events_path = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
        if not cls.events_path.exists():
            raise unittest.SkipTest(f"缺数据：{cls.events_path}")
        from influence_strategy.eval_hot_events import load_hot_events
        cls.events = load_hot_events(cls.events_path)

    def _hot_event(self, event_id: str) -> dict:
        return next(e for e in self.events if str(e.get("event_id")) == event_id)

    def test_entry_single_platform_defaults(self) -> None:
        """默认路径：recall_size=200 / recall_mode=platform_aware 必须生效。"""
        from influence_strategy.eval_hot_events import _run_hot_event_evaluation
        from influence_strategy.pipeline import StrategyPipeline

        captured: dict = {}

        def _spy(**kwargs):
            captured.update(kwargs)
            return _run_hot_event_evaluation(**kwargs)

        # 直接调真实入口，但把 pipeline 换成探针以捕获入参
        with TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(
                StrategyPipeline,
                "run_with_artifacts",
                side_effect=lambda **kw: captured.update(kw) or _make_fake_artifacts(),
            ):
                try:
                    _run_hot_event_evaluation(
                        workspace_root=PROJECT_ROOT,
                        output_dir=Path(tmp),
                        hot_event=self._hot_event("hot_event_001"),
                        use_llm=False,
                        trace_dir=Path(tmp) / "trace",
                        target_user_id=None,
                        dispatch_platform=DISPATCH_PLATFORM,
                    )
                except Exception:  # noqa: BLE001  # 探针返回的假 artifacts 必然抛错
                    pass

        self.assertEqual(captured.get("recall_size"), 200)
        self.assertEqual(captured.get("recall_mode"), "platform_aware")
        self.assertEqual(captured.get("dispatch_platform"), DISPATCH_PLATFORM)

    def test_entry_targeted_disables_recall(self) -> None:
        """单人 target_user_id 强制唯一节点 → 召回必须关掉。"""
        from influence_strategy.eval_hot_events import _run_hot_event_evaluation
        from influence_strategy.pipeline import StrategyPipeline

        captured: dict = {}
        with TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(
                StrategyPipeline,
                "run_with_artifacts",
                side_effect=lambda **kw: captured.update(kw) or _make_fake_artifacts(),
            ):
                try:
                    _run_hot_event_evaluation(
                        workspace_root=PROJECT_ROOT,
                        output_dir=Path(tmp),
                        hot_event=self._hot_event("hot_event_001"),
                        use_llm=False,
                        trace_dir=Path(tmp) / "trace",
                        target_user_id="121351",
                        dispatch_platform=DISPATCH_PLATFORM,
                    )
                except Exception:  # noqa: BLE001
                    pass

        self.assertIsNone(captured.get("recall_size"))
        self.assertIsNone(captured.get("recall_mode"))

    def test_entry_multiplatform_passes_recall(self) -> None:
        """多平台入口透传召回，但不做单平台 platform_aware。"""
        from influence_strategy.eval_hot_events import (
            _run_hot_event_multiplatform_evaluation,
        )
        from influence_strategy.pipeline import StrategyPipeline
        from influence_strategy.platform_dispatcher import PlatformSpec

        captured: dict = {}
        with TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(
                StrategyPipeline,
                "run_with_artifacts",
                side_effect=lambda **kw: captured.update(kw) or _make_fake_artifacts(),
            ):
                try:
                    _run_hot_event_multiplatform_evaluation(
                        workspace_root=PROJECT_ROOT,
                        output_dir=Path(tmp),
                        hot_event=self._hot_event("hot_event_001"),
                        platform_specs=[
                            PlatformSpec(platform_key="facebook_simulated", requested_count=3),
                            PlatformSpec(platform_key="weibo_simulated", requested_count=2),
                        ],
                        use_llm=False,
                        trace_dir=Path(tmp) / "trace",
                    )
                except Exception:  # noqa: BLE001
                    pass

        self.assertEqual(captured.get("recall_size"), 200)
        self.assertEqual(captured.get("recall_mode"), "platform_aware")
        # 多平台不能给单平台 dispatch_platform，否则会误删其它平台节点
        self.assertIsNone(captured.get("dispatch_platform"))

    def test_entry_off_switch(self) -> None:
        """recall_size=0 时入口必须传 None（等价关闭）。"""
        from influence_strategy.eval_hot_events import _run_hot_event_evaluation
        from influence_strategy.pipeline import StrategyPipeline

        captured: dict = {}
        with TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(
                StrategyPipeline,
                "run_with_artifacts",
                side_effect=lambda **kw: captured.update(kw) or _make_fake_artifacts(),
            ):
                try:
                    _run_hot_event_evaluation(
                        workspace_root=PROJECT_ROOT,
                        output_dir=Path(tmp),
                        hot_event=self._hot_event("hot_event_001"),
                        use_llm=False,
                        trace_dir=Path(tmp) / "trace",
                        target_user_id=None,
                        dispatch_platform=DISPATCH_PLATFORM,
                        recall_size=0,
                    )
                except Exception:  # noqa: BLE001
                    pass

        self.assertIsNone(captured.get("recall_size"))
        self.assertIsNone(captured.get("recall_mode"))


class RecallSummaryDisplayTest(unittest.TestCase):
    """summary 必须能区分「召回 200 / 全量 35,636」。"""

    def test_total_candidate_count_is_full(self) -> None:
        """召回开启时 node_count=200 但 total_candidate_count 必须是全量。"""
        from influence_strategy.event_parser import RuleBasedEventParser
        from influence_strategy.feature_builder import FeatureBuilder
        from influence_strategy.eval_hot_events import (
            _candidate_counts_view,
            hot_event_to_pipeline_payload,
            load_hot_events,
        )

        events = load_hot_events(
            PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
        )
        hot = next(e for e in events if str(e.get("event_id")) == "hot_event_001")
        payload = hot_event_to_pipeline_payload(
            hot,
            max_selected_nodes=5,
            risk_level=None,
            campaign_window_hours=24,
            max_frequency_per_day=3,
            allowed_platforms=None,
        )
        event = RuleBasedEventParser().parse(
            payload, workspace_root=PROJECT_ROOT, use_llm=False
        )
        builder = FeatureBuilder()

        res = builder.build_from_loader(
            workspace_root=PROJECT_ROOT,
            event=event,
            use_llm=False,
            recall_size=200,
            recall_mode="platform_aware",
            dispatch_platform=DISPATCH_PLATFORM,
        )
        # 关键：node_count 是召回人数，total_candidate_count 才是全量
        self.assertEqual(res.summary.node_count, 200)
        self.assertEqual(res.summary.total_candidate_count, 35636)
        self.assertEqual(res.summary.recall_size, 200)

        view = _candidate_counts_view(res.summary)
        self.assertEqual(view["candidate_pool_total"], 35636)
        self.assertEqual(view["candidate_pool_used"], 200)
        self.assertTrue(view["recall_enabled"])
        self.assertEqual(view["display_text"], "召回 200 / 全量 35636")

    def test_view_without_recall(self) -> None:
        """关闭召回时两个口径相等，display_text 不提召回。"""
        from influence_strategy.models import FeatureBuildSummary

        summary = FeatureBuildSummary(
            event_id="e",
            event_type="general",
            node_count=35636,
            source_node_count=0,
            matched_node_count=0,
            avg_feature_ready_score=0.0,
            avg_topic_match_score=0.0,
            total_candidate_count=35636,
        )
        from influence_strategy.eval_hot_events import _candidate_counts_view

        view = _candidate_counts_view(summary)
        self.assertFalse(view["recall_enabled"])
        self.assertEqual(view["candidate_pool_total"], 35636)
        self.assertEqual(view["candidate_pool_used"], 35636)
        self.assertEqual(view["display_text"], "全量 35636")


def _make_fake_artifacts():
    """入口接线探针用的空壳 artifacts（只为捕获入参，后续必然抛错）。"""
    raise RuntimeError("__PROBE__")


if __name__ == "__main__":
    unittest.main()
