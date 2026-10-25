from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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


class _TimingLLMClient:
    """第九轮：只给 LLM 客户端套一层计时壳，**不改任何请求与返回值**。

    之前 ``stage_times`` 只能看到"某阶段共花了多久"，无法把"模型等待"和
    "本地计算"分开；第九轮实测中模型等待占全链路约 93%，后续每一轮都需要
    这个口径来判断优化到底有没有碰到真正的瓶颈。这里用委托而不是改
    ``llm_client.py``，计时改动与业务逻辑完全隔离。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.wait_seconds: list[float] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return self._inner.generate_json(
                system_prompt=system_prompt, user_prompt=user_prompt
            )
        finally:
            self.wait_seconds.append(time.perf_counter() - started)


# 第十轮：事件解析阶段 LLM 的**生产默认**开关。
# 依据（第九轮实测，本地 qwen3:8b）：事件解析那一次调用占全链路约 47%
# （约 12.5s），而在固定事件上它的产出与纯规则解析只差一个语义标签；
# 规则解析器处理的是自由文本，关掉它**不影响**接收任意新事件。
# 因此生产运行默认关闭事件解析 LLM，只保留 selector 的小范围重排 LLM。
# 需要对照实验时显式传 use_llm_event_parser=True 即可恢复，能力并未删除。
DEFAULT_USE_LLM_EVENT_PARSER = False


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
        recall_size: int | None = None,
        recall_mode: Literal["platform_aware"] | None = None,
        use_llm_event_parser: bool | None = None,
        use_llm_selector: bool | None = None,
        selector_candidate_limit: int | None = None,
    ) -> StrategyResult:
        return self.run_with_artifacts(
            workspace_root=workspace_root,
            event_input=event_input,
            profile_limit=profile_limit,
            use_llm=use_llm,
            llm_client=llm_client,
            target_user_id=target_user_id,
            dispatch_platform=dispatch_platform,
            recall_size=recall_size,
            recall_mode=recall_mode,
            use_llm_event_parser=use_llm_event_parser,
            use_llm_selector=use_llm_selector,
            selector_candidate_limit=selector_candidate_limit,
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
        recall_size: int | None = None,
        recall_mode: Literal["platform_aware"] | None = None,
        use_llm_event_parser: bool | None = None,
        use_llm_selector: bool | None = None,
        selector_candidate_limit: int | None = None,
    ) -> PipelineArtifacts:
        # 第十轮：事件解析阶段的 LLM 默认**关闭**（DEFAULT_USE_LLM_EVENT_PARSER）。
        # 显式 True/False 仍然优先，所以"开事件解析 LLM / 关 selector LLM"
        # 这类组合以及对照实验都不受影响。
        event_parser_use_llm = (
            DEFAULT_USE_LLM_EVENT_PARSER
            if use_llm_event_parser is None
            else use_llm_event_parser
        )
        # 第七轮：selector 阶段同样独立开关，语义与事件解析开关保持一致。
        # use_llm_selector=False 时走原有纯规则 selector，且**不做**候选初选。
        selector_use_llm = use_llm if use_llm_selector is None else use_llm_selector

        client = llm_client
        # 只要任一阶段需要 LLM 就要构造客户端：过去这里只看 use_llm，
        # 会导致 use_llm=False + use_llm_event_parser=True 时拿不到 client。
        if client is None and (use_llm or event_parser_use_llm or selector_use_llm):
            client = OpenAICompatibleLLMClient.from_env_files(workspace_root)

        # 每个阶段独立计时，便于上游做"5 次压测"的平均用时聚合。
        # 用 perf_counter 而不是 time.time，避免系统时钟回拨影响。
        stage_times: dict[str, float] = {}
        pipeline_start = time.perf_counter()

        # 第九轮：所有阶段共用这个计时壳，才能把"模型等待"从阶段耗时里剥离出来。
        timed_client = _TimingLLMClient(client) if client is not None else None
        stage_client = timed_client if timed_client is not None else client

        t0 = time.perf_counter()
        event = self.event_parser.parse(
            event_input,
            workspace_root=workspace_root,
            # 只用事件解析自己的开关，不再跟随 selector 的 use_llm。
            use_llm=event_parser_use_llm,
            llm_client=stage_client,
        )
        stage_times["event_parser"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # 第一轮性能优化：特征构建阶段固定走纯规则，不再参与 --use-llm。
        # 该阶段会把整个候选池（24~48 人）拼成 56~71KB 的提示词交给本地
        # qwen3:8b，实测单次 50~122s，且多数运行直接超时；而它的产出只是
        # 覆盖 6 个子分并把 feature_ready_score 与 llm_feature_score 混合。
        # 这里传 False 而不是删掉参数：feature_builder 的 LLM 增强能力仍在，
        # 需要时改这一个开关即可恢复。event_parser / selector 的 LLM 调用保留。
        feature_result = self.feature_builder.build_from_loader(
            workspace_root=workspace_root,
            event=event,
            profile_limit=profile_limit,
            use_llm=False,
            llm_client=stage_client,
            recall_size=recall_size,
            recall_mode=recall_mode,
            dispatch_platform=dispatch_platform,
        )
        stage_times["feature_builder"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        score_result = self.scorer.score(feature_result)
        stage_times["scorer"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        selection_result = self.selector.select(
            score_result,
            workspace_root=workspace_root,
            # 选人阶段用自己的开关，不再跟随事件解析的开关。
            use_llm=selector_use_llm,
            llm_client=stage_client,
            target_user_id=target_user_id,
            dispatch_platform=dispatch_platform,
            candidate_limit=selector_candidate_limit,
        )
        stage_times["selector"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        strategy_result = self.strategy_generator.generate(selection_result)
        stage_times["strategy_generator"] = time.perf_counter() - t0

        # 第九轮：模型等待总时长（不含解析/落盘）。键名沿用上游"按阶段聚合"的
        # 约定（eval_hot_events 已在同一 dict 里放 llm_client_init 等同类键）。
        if timed_client is not None:
            stage_times["llm_wait"] = sum(timed_client.wait_seconds)
            stage_times["llm_calls"] = float(len(timed_client.wait_seconds))

        stage_times["pipeline_total"] = time.perf_counter() - pipeline_start
        return PipelineArtifacts(
            event=event,
            feature_result=feature_result,
            score_result=score_result,
            selection_result=selection_result,
            strategy_result=strategy_result,
            stage_times=stage_times,
        )
