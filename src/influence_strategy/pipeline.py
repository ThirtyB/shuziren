from __future__ import annotations

from dataclasses import dataclass
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

        event = self.event_parser.parse(
            event_input,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=client,
        )
        feature_result = self.feature_builder.build_from_loader(
            workspace_root=workspace_root,
            event=event,
            profile_limit=profile_limit,
            use_llm=use_llm,
            llm_client=client,
        )
        score_result = self.scorer.score(feature_result)
        selection_result = self.selector.select(
            score_result,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=client,
            target_user_id=target_user_id,
            dispatch_platform=dispatch_platform,
        )
        strategy_result = self.strategy_generator.generate(selection_result)
        return PipelineArtifacts(
            event=event,
            feature_result=feature_result,
            score_result=score_result,
            selection_result=selection_result,
            strategy_result=strategy_result,
        )
