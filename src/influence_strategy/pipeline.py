from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .event_parser import RuleBasedEventParser
from .feature_builder import FeatureBuilder
from .llm_client import OpenAICompatibleLLMClient
from .models import FeatureBuildResult, ParsedEvent, ScoreResult, SelectionResult, StrategyResult
from .scorer import Scorer
from .selector import Selector
from .strategy_generator import StrategyGenerator


@dataclass(slots=True)
class PipelineArtifacts:
    event: ParsedEvent
    feature_result: FeatureBuildResult
    score_result: ScoreResult
    selection_result: SelectionResult
    strategy_result: StrategyResult
    # 各阶段耗时（秒）。键名稳定，便于上游聚合面板读取。
    stage_times: dict[str, float] = field(default_factory=dict)


class StrategyPipeline:
    def __init__(self, product_name: str = "abc_reading") -> None:
        self.product_name = product_name
        self.event_parser = RuleBasedEventParser(default_product_name=product_name)
        self.feature_builder = FeatureBuilder(product_name=product_name)
        self.scorer = Scorer()
        self.selector = Selector()
        self.strategy_generator = StrategyGenerator()

    def run(
        self,
        workspace_root: str | Path,
        event_input: str | dict[str, Any],
        profile_limit: int | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
        target_user_id: str | None = None,
        dispatch_platform: str | None = None,
    ) -> StrategyResult:
        return self.run_with_artifacts(
            workspace_root=workspace_root,
            event_input=event_input,
            profile_limit=profile_limit,
            use_llm=use_llm,
            llm_client=llm_client,
            target_user_id=target_user_id,
            dispatch_platform=dispatch_platform,
        ).strategy_result

    def run_with_artifacts(
        self,
        workspace_root: str | Path,
        event_input: str | dict[str, Any],
        profile_limit: int | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
        target_user_id: str | None = None,
        dispatch_platform: str | None = None,
    ) -> PipelineArtifacts:
        client = llm_client
        if client is None and use_llm:
            client = OpenAICompatibleLLMClient.from_env_files(workspace_root)

        # 每个阶段独立计时，便于上游做"5 次压测"的平均用时聚合。
        # 用 perf_counter 而不是 time.time，避免系统时钟回拨影响。
        stage_times: dict[str, float] = {}
        pipeline_start = time.perf_counter()

        t0 = time.perf_counter()
        event = self.event_parser.parse(
            event_input,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=client,
        )
        stage_times["event_parser"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        feature_result = self.feature_builder.build_from_loader(
            workspace_root=workspace_root,
            event=event,
            profile_limit=profile_limit,
            use_llm=use_llm,
            llm_client=client,
        )
        stage_times["feature_builder"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        score_result = self.scorer.score(feature_result)
        stage_times["scorer"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        selection_result = self.selector.select(
            score_result,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=client,
            target_user_id=target_user_id,
            dispatch_platform=dispatch_platform,
        )
        stage_times["selector"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        strategy_result = self.strategy_generator.generate(selection_result)
        stage_times["strategy_generator"] = time.perf_counter() - t0

        stage_times["pipeline_total"] = time.perf_counter() - pipeline_start
        return PipelineArtifacts(
            event=event,
            feature_result=feature_result,
            score_result=score_result,
            selection_result=selection_result,
            strategy_result=strategy_result,
            stage_times=stage_times,
        )
