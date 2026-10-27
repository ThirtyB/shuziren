from __future__ import annotations

import logging
import math
import re
import threading
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .data_loader import DataLoader
from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .local_models import load_local_models
from .local_semantic import build_event_text
from .models import (
    EnrichedUserProfile,
    FeatureBuildResult,
    FeatureBuildSummary,
    GraphAttributes,
    NodeFeature,
    ParsedEvent,
    ProductContext,
    UserProfile,
)
from .prompts import build_feature_enrichment_system_prompt, build_feature_enrichment_user_prompt

logger = logging.getLogger(__name__)

#: 各语义后端下「值得考虑」的余弦绝对下限。
#: 低于它的节点即使排进本次事件前 10% 也置 0 —— 当整个库与话题都不相关时，
#: 纯分位数会强行选出「最不无关」的一批人，绝对下限就是拦住这种情况。
#: BGE 无关文本的余弦约 0.15~0.25，取 0.30；SVD 的 p95 才 0.11，取 0.10。
LOCAL_SEMANTIC_FLOOR = {"bge": 0.30, "svd": 0.10}

_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)

# ---------------------------------------------------------------------------
# LLM 特征增强的可观测性：仅做统计/日志，不重试、不改评分/选人/风险口径
# ---------------------------------------------------------------------------
#
# 设计目标：让 `_augment_with_llm` 不再静默吞掉 `LLMClientError / OSError /
# ValueError / TypeError`。失败时仍按业务语义降级（节点保持原 `feature_ready_score=0`
# 状态），但调用方与运维侧应当能从 metrics 与日志里立刻看到：
#   1) 异常类型（`LLMClientError` / `OSError` / `ValueError` / `TypeError`）
#   2) prompt 字符数（用于排查是否与请求体过大相关）
#   3) 请求耗时（用于排查慢调用与超时）
#   4) 本次是否成功
#
# 不要扩展 `except` 的范围、不要加重试、不要修改候选池/打分/选人/风险公式。
_FEATURE_LLM_METRICS: dict[str, Any] = {
    "calls": 0,
    "success": 0,
    "failure": 0,
    "failure_by_type": {},
    "last_call": None,
}
_FEATURE_LLM_METRICS_LOCK = threading.Lock()


def _record_feature_llm_outcome(
    *,
    success: bool,
    error_type: str | None,
    error_message: str | None,
    prompt_chars: int,
    elapsed_sec: float,
) -> None:
    """记录一次 LLM 特征增强调用的结果。仅在 `_augment_with_llm` 内部调用。"""
    with _FEATURE_LLM_METRICS_LOCK:
        _FEATURE_LLM_METRICS["calls"] += 1
        if success:
            _FEATURE_LLM_METRICS["success"] += 1
        else:
            _FEATURE_LLM_METRICS["failure"] += 1
            if error_type:
                bucket = _FEATURE_LLM_METRICS["failure_by_type"]
                bucket[error_type] = bucket.get(error_type, 0) + 1
        _FEATURE_LLM_METRICS["last_call"] = {
            "success": success,
            "error_type": error_type,
            "error_message": error_message,
            "prompt_chars": prompt_chars,
            "elapsed_sec": elapsed_sec,
        }


def get_feature_llm_metrics() -> dict[str, Any]:
    """获取 LLM 特征增强的成功/失败统计快照（线程安全）。"""
    with _FEATURE_LLM_METRICS_LOCK:
        return {
            "calls": _FEATURE_LLM_METRICS["calls"],
            "success": _FEATURE_LLM_METRICS["success"],
            "failure": _FEATURE_LLM_METRICS["failure"],
            "failure_by_type": dict(_FEATURE_LLM_METRICS["failure_by_type"]),
            "last_call": (
                dict(_FEATURE_LLM_METRICS["last_call"])
                if _FEATURE_LLM_METRICS["last_call"] is not None
                else None
            ),
        }


def reset_feature_llm_metrics() -> None:
    """清零 LLM 特征增强的统计。仅供测试使用。"""
    with _FEATURE_LLM_METRICS_LOCK:
        _FEATURE_LLM_METRICS["calls"] = 0
        _FEATURE_LLM_METRICS["success"] = 0
        _FEATURE_LLM_METRICS["failure"] = 0
        _FEATURE_LLM_METRICS["failure_by_type"] = {}
        _FEATURE_LLM_METRICS["last_call"] = None


EVENT_KEYWORD_EXPANSIONS = (
    (
        ("\u822a\u8fd0", "\u7ea2\u6d77", "\u62a4\u822a", "\u6d77\u8fd0", "\u5168\u7403\u7269\u6d41", "\u4f9b\u5e94\u94fe"),
        (
            "\u822a\u8fd0",
            "\u6d77\u8fd0",
            "\u7269\u6d41",
            "\u4f9b\u5e94\u94fe",
            "\u56fd\u9645\u8d38\u6613",
            "\u5168\u7403\u5e02\u573a",
            "\u6e2f\u53e3",
            "\u80fd\u6e90",
            "\u56fd\u9645\u5173\u7cfb",
            "\u5730\u7f18\u653f\u6cbb",
            "\u8d38\u6613\u6210\u672c",
        ),
    ),
    (
        ("\u6cb9\u4ef7", "\u80fd\u6e90", "\u539f\u6cb9", "\u5730\u7f18\u98ce\u9669"),
        ("\u80fd\u6e90\u4ef7\u683c", "\u80fd\u6e90", "\u56fd\u9645\u8d38\u6613", "\u4f9b\u5e94\u94fe", "\u5168\u7403\u5e02\u573a", "\u5730\u7f18\u653f\u6cbb"),
    ),
    (
        ("\u6781\u7aef\u5929\u6c14", "\u57ce\u5e02\u5e94\u6025", "\u7535\u529b\u4fdd\u969c", "\u516c\u5171\u5b89\u5168"),
        ("\u57ce\u5e02\u6cbb\u7406", "\u516c\u5171\u5b89\u5168", "\u6c11\u751f", "\u516c\u5171\u670d\u52a1", "\u57fa\u5c42\u670d\u52a1", "\u5e94\u6025", "\u7535\u529b"),
    ),
    (
        ("\u6570\u636e\u6cc4\u9732", "\u7f51\u7edc\u5b89\u5168", "\u5408\u89c4", "\u9690\u79c1"),
        ("\u7f51\u7edc\u5b89\u5168", "\u9690\u79c1\u4fdd\u62a4", "\u5408\u89c4\u98ce\u9669", "\u6cd5\u5f8b\u6cd5\u89c4", "\u4e8b\u5b9e\u6838\u67e5", "\u516c\u5171\u89c4\u5219"),
    ),
)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = _NON_TEXT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, value))


class FeatureBuilder:
    def __init__(self, product_name: str = "abc_reading") -> None:
        self.product_name = product_name

    def build_from_loader(
        self,
        workspace_root: str | Path,
        event: ParsedEvent,
        profile_limit: int | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
        *,
        recall_size: int | None = None,
        recall_mode: Literal["platform_aware"] | None = None,
        dispatch_platform: str | None = None,
    ) -> FeatureBuildResult:
        loader = DataLoader(workspace_root, product_name=self.product_name)
        product_context = loader.load_product_context()
        profiles = loader.load_profiles(limit=profile_limit)
        try:
            # 用 load_enriched_graph_attributes 替代 load_enriched_profiles：
            # 只取 graph_attributes 9 个字段，跳过 neighbors 列表和
            # EnrichedUserProfile 整体校验，60MB+ JSON 解析从 ~3.3s 降到 <0.5s。
            enriched_profiles = loader.load_enriched_graph_attributes(limit=profile_limit)
        except (JSONDecodeError, ValueError):
            enriched_profiles = {}
        try:
            source_user_ids = set(
                loader.load_interactions(limit_sources=None, limit_records_per_source=0).keys()
            )
        except (JSONDecodeError, ValueError):
            source_user_ids = set()

        # ---- 召回层：仅在显式开启时跑 ----
        # 不改 ②③④ 的 LLM 逻辑；召回结果仅用于过滤 profiles / enriched_profiles，
        # 5 个全量 max_log 由 build_features 内部在过滤前计算（保证 score 等价）。
        recall_user_ids: set[str] | None = None
        if recall_mode == "platform_aware" and recall_size and recall_size > 0:
            recall_user_ids = self.recall_candidates(
                profiles=profiles,
                enriched_graph_attributes=enriched_profiles,
                event=event,
                dispatch_platform=dispatch_platform,
                size=recall_size,
                source_user_ids=source_user_ids,
                workspace_root=workspace_root,
            )

        return self.build_features(
            product_context=product_context,
            profiles=profiles,
            event=event,
            enriched_profiles=enriched_profiles,
            source_user_ids=source_user_ids,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=llm_client,
            recall_user_ids=recall_user_ids,
            recall_mode=recall_mode,
        )

    def recall_candidates(
        self,
        *,
        profiles: dict[str, UserProfile],
        enriched_graph_attributes: dict[str, Any],
        event: ParsedEvent,
        dispatch_platform: str | None = None,
        size: int = 200,
        source_user_ids: set[str] | None = None,
        workspace_root: str | Path | None = None,
    ) -> set[str]:
        """召回与事件最相关的 size 个 user_id（单一混合排序，不再分桶）。

        排序键（全部降序）：
          1) ``kw_hits``  字面关键词命中数 —— 直接对应"与关键词最相关"；
          2) ``semantic`` BGE 语义相关度 —— 关键词数相同时决定先后，
             同时承担兜底职责：关键词命中人数不足 size 时，
             剩余名额自动由语义最相关的节点填补；
          3) 图活跃度 / 粉丝数 —— 前两者分不出高下时的最终裁决。

        仍然使用**廉价 raw 字段**而非归一化 score（与旧实现同因）：
          1) recall 在 build_features 之前，5 个 log_score 的分母尚未算出；
          2) log 是单调变换，raw 排序与归一化排序等价。

        泛大V（关键词零命中但粉丝量大）保留少量席位 ``max(2, size // 20)``，
        因为它们靠关键词进不来，但对扩散有实际价值。

        ``workspace_root`` 用于取 BGE 语义分；为 None 或本地模型缺失时语义分
        全为 0，自动退化为纯关键词排序，不会报错。
        """
        from .platform_profiles import normalize_platform_key

        target = normalize_platform_key(dispatch_platform) if dispatch_platform else None
        if target:
            pool = [(uid, p) for uid, p in profiles.items() if normalize_platform_key(p.platform) == target]
        else:
            pool = list(profiles.items())

        if not pool or size <= 0:
            return set()

        # 廉价关键词打分（不做 rapidfuzz；用 substring，够用）
        keywords = self._prepare_keywords(self._event_match_keywords(event))
        kw_count = len(keywords) or 1
        source_set = source_user_ids or set()
        #: 泛大V 保留席位：关键词零命中但粉丝量大的账号靠关键词进不来，
        #: 但对扩散有实际价值，预留少量名额（旧配额的 1/3，避免稀释相关性）。
        big_v_cap = max(2, size // 20)

        items: list[dict[str, Any]] = []
        for uid, p in pool:
            ga = enriched_graph_attributes.get(uid) if enriched_graph_attributes else None
            target_ga = getattr(ga, "graph_attributes", None) if ga is not None else None
            if target_ga is None:
                target_ga = ga
            neighbor = int(getattr(target_ga, "neighbor_count", 0)) if target_ga else 0
            received = int(getattr(target_ga, "received_interaction_count", 0)) if target_ga else 0
            made = int(getattr(target_ga, "made_interaction_count", 0)) if target_ga else 0
            followers = int(p.user_followers)
            friends = int(p.user_friends)
            kw_hits = sum(
                1
                for k in keywords
                if k
                and (
                    k in (p.user_description or "")
                    or any(k in i for i in p.user_interests)
                )
            )
            is_source = uid in source_set
            # role_hint == "interaction_response" 的代理：is_source 或 高活动度
            is_interaction_like = is_source or (received + made) >= 50
            # generic_high_influence 的代理：粉丝多 + 无主题匹配
            is_generic_hi = kw_hits == 0 and followers >= 50000
            items.append(
                {
                    "user_id": uid,
                    "kw_hits": kw_hits,
                    "topic_score": kw_hits / kw_count,
                    "neighbor": neighbor,
                    "received": received,
                    "made": made,
                    "followers": followers,
                    "friends": friends,
                    "is_interaction_like": is_interaction_like,
                    "is_generic_hi": is_generic_hi,
                }
            )

        # ---- 混合排序：关键词命中优先，语义相关度兜底 ----
        semantic_map = self._recall_semantic_scores(
            workspace_root=workspace_root,
            user_ids=[item["user_id"] for item in items],
            event=event,
        )

        def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
            return (
                item["kw_hits"],
                semantic_map.get(item["user_id"], 0.0),
                item["neighbor"] + item["received"],
                item["followers"],
            )

        ordered = sorted(items, key=rank_key, reverse=True)

        # 泛大V 单独取：它们 kw_hits 恒为 0，在主榜里永远垫底，
        # 因此单独按 (语义分, 粉丝数) 选 big_v_cap 个，再从主榜腾出等量名额。
        big_v_pool = sorted(
            (item for item in items if item["is_generic_hi"]),
            key=lambda item: (semantic_map.get(item["user_id"], 0.0), item["followers"]),
            reverse=True,
        )[:big_v_cap]
        reserved = {item["user_id"] for item in big_v_pool}

        chosen: list[dict[str, Any]] = []
        used: set[str] = set()
        main_quota = size - len(big_v_pool)
        for item in ordered:
            if len(chosen) >= main_quota:
                break
            if item["user_id"] in used or item["user_id"] in reserved:
                continue
            chosen.append(item)
            used.add(item["user_id"])

        for item in big_v_pool:
            if len(chosen) >= size:
                break
            if item["user_id"] in used:
                continue
            chosen.append(item)
            used.add(item["user_id"])

        return {item["user_id"] for item in chosen[:size]}

    def _recall_semantic_scores(
        self,
        *,
        workspace_root: str | Path | None,
        user_ids: list[str],
        event: ParsedEvent,
    ) -> dict[str, float]:
        """召回阶段的 BGE 语义相关度。

        本地模型缺失或 ``workspace_root`` 为 None 时返回空 dict，召回自动
        退化为纯关键词排序——保证无模型环境（如部分单测）不会报错。

        语义分是对全库做一次矩阵乘法得到的（约 30ms），"算全库"与"只算
        user_ids" 成本几乎相同，因此这里直接复用全量结果。
        """

        if not workspace_root or not user_ids:
            return {}
        try:
            from .local_models import load_local_models
            from .local_semantic import build_event_text
        except ImportError:
            return {}

        local_models = load_local_models(workspace_root)
        if local_models is None:
            return {}
        try:
            return local_models.semantic_scores(build_event_text(event))
        except Exception:  # pragma: no cover - 模型损坏时不应中断召回
            logger.warning("召回阶段语义打分失败，退化为纯关键词排序", exc_info=True)
            return {}

    def build_features(
        self,
        product_context: ProductContext,
        profiles: dict[str, UserProfile],
        event: ParsedEvent,
        enriched_profiles: dict[str, EnrichedUserProfile] | None = None,
        source_user_ids: set[str] | None = None,
        workspace_root: str | Path | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
        *,
        recall_user_ids: set[str] | None = None,
        recall_mode: Literal["platform_aware"] | None = None,
    ) -> FeatureBuildResult:
        enriched_profiles = enriched_profiles or {}
        source_user_ids = source_user_ids or set()
        influencer_ids = set(product_context.influencer_ids)
        # 召回**前**的全量候选人数（35,636）。必须在过滤前取，
        # 供 summary.total_candidate_count 使用，让展示层能区分
        # 「召回 200 / 全量 35,636」，不把 200 误读成全量。
        total_candidate_count = len(profiles)

        # ---- 5 个全量 max_log（必须在召回过滤之前算） ----
        # 这 5 个分母决定 follower_score / friend_score / neighbor_score /
        # received_score / made_score 的归一化结果。为保证召回子集上的 score
        # 与全量**逐字节相等**（验证见 tests/test_feature_builder.py 回归），
        # 必须先扫一遍全量 profiles + enriched_profiles，再做召回过滤。
        # 成本：5 列 × 35k int 读，< 50ms，可忽略。
        def _ga_field(ep: Any, field: str) -> int:
            ga = getattr(ep, "graph_attributes", None)
            target = ga if ga is not None else ep
            return int(getattr(target, field, 0))

        follower_max_log = self._max_log_array(
            np.array(
                [int(p.user_followers) for p in profiles.values()], dtype=np.int64
            )
            if profiles
            else np.array([0], dtype=np.int64)
        )
        friend_max_log = self._max_log_array(
            np.array(
                [int(p.user_friends) for p in profiles.values()], dtype=np.int64
            )
            if profiles
            else np.array([0], dtype=np.int64)
        )
        # 与原版语义严格对齐：df 是 profiles 左连接 enriched_profiles，
        # 所以分母只覆盖 **profiles 里存在的 user_id**；enriched 中多余的
        # user_id 不参与（原版也一样）。
        _ep_by_uid = enriched_profiles
        neighbor_max_log = self._max_log_array(
            np.array(
                [
                    _ga_field(_ep_by_uid[uid], "neighbor_count")
                    for uid in profiles
                    if uid in _ep_by_uid
                ],
                dtype=np.int64,
            )
            if profiles
            else np.array([0], dtype=np.int64)
        )
        received_max_log = self._max_log_array(
            np.array(
                [
                    _ga_field(_ep_by_uid[uid], "received_interaction_count")
                    for uid in profiles
                    if uid in _ep_by_uid
                ],
                dtype=np.int64,
            )
            if profiles
            else np.array([0], dtype=np.int64)
        )
        made_max_log = self._max_log_array(
            np.array(
                [
                    _ga_field(_ep_by_uid[uid], "made_interaction_count")
                    for uid in profiles
                    if uid in _ep_by_uid
                ],
                dtype=np.int64,
            )
            if profiles
            else np.array([0], dtype=np.int64)
        )

        # ---- 召回过滤（profiles + enriched_profiles 都截到 recall_user_ids） ----
        if recall_user_ids is not None:
            profiles = {
                uid: p for uid, p in profiles.items() if uid in recall_user_ids
            }
            enriched_profiles = {
                uid: ep
                for uid, ep in enriched_profiles.items()
                if uid in recall_user_ids
            }

        # ---- 向量化版 build_features：把 N 个 profile 装入 DataFrame，列运算一次算完 ----
        # 与原版逐行 Python 循环的输出在数值上保持一致（< 1e-9），排序结果一致。
        # 当 recall 开启时，N=recall_size（默认 200），否则 N=35,636。

        # 1. profiles -> DataFrame
        profile_rows = [
            {
                "user_id": uid,
                "user_name": p.user_name,
                "platform": p.platform,
                "user_followers": int(p.user_followers),
                "user_friends": int(p.user_friends),
                "user_interests": list(p.user_interests),
                "user_description": p.user_description,
                "interest_count": len(p.user_interests),
                "has_description": bool(p.user_description.strip()),
            }
            for uid, p in profiles.items()
        ]
        df = pd.DataFrame(profile_rows)

        # 2. enriched_profiles -> DataFrame，左连接
        #    ``load_enriched_graph_attributes`` 直接返回 ``dict[user_id, GraphAttributes]``；
        #    但老测试还在用 ``dict[user_id, EnrichedUserProfile]``，自动检测兼容。
        def _ga_of(ep):
            # ``load_enriched_graph_attributes`` 直接返回 ``GraphAttributes``，只有
            # 老测试还在传 ``EnrichedUserProfile``。原来的无条件 ``getattr`` 探测对
            # ``GraphAttributes`` 每次都会走 Pydantic 的 ``__getattr__`` 慢路径；先用
            # isinstance 短路掉绝大多数情况。两者语义完全等价：GraphAttributes 自身
            # 没有 ``graph_attributes`` 字段，所以 getattr 必然返回 None 并回退到 ep。
            if isinstance(ep, GraphAttributes):
                return ep
            ga = getattr(ep, "graph_attributes", None)
            return ga if ga is not None else ep

        # ``_ga_of(ep)`` 必须提到循环外只解析一次：写在下面 9 个字段的推导式里会
        # 让每个用户被解析 9 次，35k 用户实测白花 ~0.8s。
        enriched_rows = []
        for uid, ep in enriched_profiles.items():
            ga = _ga_of(ep)
            enriched_rows.append(
                {
                    "user_id": uid,
                    "neighbor_count": int(ga.neighbor_count),
                    "mutual_neighbor_count": int(ga.mutual_neighbor_count),
                    "received_interaction_count": int(ga.received_interaction_count),
                    "made_interaction_count": int(ga.made_interaction_count),
                    "received_comment_count": int(ga.received_comment_count),
                    "received_repost_count": int(ga.received_repost_count),
                    "made_comment_count": int(ga.made_comment_count),
                    "made_repost_count": int(ga.made_repost_count),
                    "self_interaction_count": int(ga.self_interaction_count),
                }
            )
        _enriched_cols = (
            "neighbor_count",
            "mutual_neighbor_count",
            "received_interaction_count",
            "made_interaction_count",
            "received_comment_count",
            "received_repost_count",
            "made_comment_count",
            "made_repost_count",
            "self_interaction_count",
        )
        if enriched_rows:
            edf = pd.DataFrame(enriched_rows)
            df = df.merge(edf, on="user_id", how="left")
            for col in _enriched_cols:
                df[col] = df[col].fillna(0).astype(np.int64)
        else:
            # enriched_profiles 为空时（测试或边界），直接补 0 列，与原版 .get(uid) 返回 0 一致
            for col in _enriched_cols:
                df[col] = np.int64(0)

        # 3. 集合判定列
        df["influencer_flag"] = df["user_id"].isin(influencer_ids)
        df["is_interaction_source"] = df["user_id"].isin(source_user_ids)

        # 4. _log_score 全部用 numpy（5 个全量 max_log 已在本函数开头算出，
        #    此处直接复用，保证召回子集上的 score 与全量逐字节相等）
        df["follower_score"] = self._log_score_array(
            df["user_followers"].to_numpy(), follower_max_log
        )
        df["friend_score"] = self._log_score_array(
            df["user_friends"].to_numpy(), friend_max_log
        )
        df["neighbor_score"] = self._log_score_array(
            df["neighbor_count"].to_numpy(), neighbor_max_log
        )
        df["received_score"] = self._log_score_array(
            df["received_interaction_count"].to_numpy(), received_max_log
        )
        df["made_score"] = self._log_score_array(
            df["made_interaction_count"].to_numpy(), made_max_log
        )

        # 5. has_graph_signal
        df["has_graph_signal"] = (
            (df["neighbor_count"] > 0)
            | (df["received_interaction_count"] > 0)
            | (df["made_interaction_count"] > 0)
            | (df["mutual_neighbor_count"] > 0)
        )

        # 6. ratios（与原版逐行三元式一致）
        total_int = (
            df["received_interaction_count"] + df["made_interaction_count"]
        ).astype(np.int64)
        df["total_interaction_count"] = total_int
        total_comment = df["received_comment_count"] + df["made_comment_count"]
        total_repost = df["received_repost_count"] + df["made_repost_count"]
        safe_total = np.where(total_int > 0, total_int, 1)
        df["comment_ratio"] = np.where(total_int > 0, total_comment / safe_total, 0.0)
        df["repost_ratio"] = np.where(total_int > 0, total_repost / safe_total, 0.0)
        safe_neighbor = np.where(df["neighbor_count"] > 0, df["neighbor_count"], 1)
        df["mutual_neighbor_ratio"] = np.where(
            df["neighbor_count"] > 0, df["mutual_neighbor_count"] / safe_neighbor, 0.0
        )

        # 7. profile_completeness_score
        has_desc_f = df["has_description"].astype(np.float64)
        interest_score = np.minimum(df["interest_count"].astype(np.float64) / 4.0, 1.0)
        follower_present = (df["user_followers"] > 0).astype(np.float64)
        df["profile_completeness_score"] = np.clip(
            (has_desc_f + interest_score + follower_present) / 3.0, 0.0, 1.0
        )

        # 8. stability_score
        self_loop_ratio = np.where(
            total_int > 0,
            df["self_interaction_count"].to_numpy() / np.where(total_int > 0, total_int.to_numpy(), 1),
            0.0,
        )
        self_loop_penalty = np.minimum(self_loop_ratio, 1.0)
        df["stability_score"] = np.clip(
            0.75 * df["profile_completeness_score"] + 0.25 * (1.0 - self_loop_penalty),
            0.0,
            1.0,
        )

        # 9. 主体三分数（has_graph_signal 两支）
        inf_flag = df["influencer_flag"].astype(np.float64)
        has_graph = df["has_graph_signal"]

        influence_with = (
            0.45 * df["follower_score"]
            + 0.30 * df["received_score"]
            + 0.15 * df["neighbor_score"]
            + 0.10 * inf_flag
        )
        diffusion_with = (
            0.35 * df["neighbor_score"]
            + 0.25 * df["made_score"]
            + 0.25 * df["mutual_neighbor_ratio"]
            + 0.15 * df["repost_ratio"]
        )
        activity_with = (
            0.45 * df["received_score"]
            + 0.35 * df["made_score"]
            + 0.10 * df["comment_ratio"]
            + 0.10 * df["friend_score"]
        )
        influence_without = 0.75 * df["follower_score"] + 0.15 * df["friend_score"] + 0.10 * inf_flag
        diffusion_without = 0.45 * df["friend_score"] + 0.35 * df["follower_score"]
        activity_without = 0.55 * df["friend_score"] + 0.25 * df["follower_score"]

        df["influence_score"] = np.where(
            has_graph, np.clip(influence_with, 0.0, 1.0), np.clip(influence_without, 0.0, 1.0)
        )
        df["diffusion_score"] = np.where(
            has_graph, np.clip(diffusion_with, 0.0, 1.0), np.clip(diffusion_without, 0.0, 1.0)
        )
        df["activity_score"] = np.where(
            has_graph, np.clip(activity_with, 0.0, 1.0), np.clip(activity_without, 0.0, 1.0)
        )

        # 10. _match_keywords 仍逐行（rapidfuzz fuzzy 匹配很难直接向量化，
        #     但这一段 C 实现很快，每用户 ~1ms，整体 ~30-40s 是后续优化点）。
        keywords = self._event_match_keywords(event)
        normalized_keywords = self._prepare_keywords(keywords)
        kw_count = len(normalized_keywords)

        profile_by_id = profiles
        enriched_by_id = enriched_profiles
        source_ids = source_user_ids
        user_ids_ordered = df["user_id"].tolist()

        matched_list: list[list[str]] = []
        semantic_profiles: list[str] = []
        for uid in user_ids_ordered:
            profile = profile_by_id[uid]
            matched_list.append(
                self._match_keywords(
                    profile=profile, normalized_keywords=normalized_keywords
                )
            )
            semantic_profiles.append(
                DataLoader.summarize_profile_for_semantics(
                    profile,
                    enriched_profile=enriched_by_id.get(uid),
                    is_interaction_source=uid in source_ids,
                )
            )
        df["matched_keywords"] = matched_list
        df["keyword_hit_count"] = df["matched_keywords"].apply(len).astype(np.int64)
        if kw_count:
            df["topic_match_score"] = df["keyword_hit_count"].astype(np.float64) / kw_count
        else:
            df["topic_match_score"] = 0.0
        df["semantic_profile"] = semantic_profiles

        # 11. feature_ready_score
        df["feature_ready_score"] = np.clip(
            0.35 * df["influence_score"]
            + 0.25 * df["diffusion_score"]
            + 0.25 * df["topic_match_score"]
            + 0.15 * df["stability_score"],
            0.0,
            1.0,
        )

        # 12. role_hint（与 _infer_role_hint 完全等价：if-else 链）
        role_conditions = [
            (df["influence_score"] >= 0.6) & (df["topic_match_score"] >= 0.2),
            df["diffusion_score"] >= 0.45,
            (df["activity_score"] >= 0.35) | (df["comment_ratio"] >= 0.6),
        ]
        role_choices = ["core_broadcast", "amplification", "interaction_response"]
        df["role_hint"] = np.select(
            role_conditions, role_choices, default="general_support"
        )

        # 与原版语义对齐：原版在生成 NodeFeature 时对各 score 做 round(., 6)，
        # 然后再 list.sort；这里排序前先把相关列四舍五入到 6 位，
        # 避免 un-rounded float 的精度差导致并列键错位。
        for col in (
            "topic_match_score",
            "feature_ready_score",
            "activity_score",
            "stability_score",
            "influence_score",
            "comment_ratio",
            "repost_ratio",
            "mutual_neighbor_ratio",
            "profile_completeness_score",
        ):
            df[col] = df[col].round(6)

        # 13. 排序：与原版 _sort_features 同 key（同 desc、同稳定语义）。
        #     pandas 的 sort_values 在大量并列键时，次级顺序未必与 Python
        #     ``list.sort`` 一致；加一个 _row_id 兜底，强制保留 dict.items() 顺序。
        df = df.reset_index(drop=False).rename(columns={"index": "_row_id"})
        df = df.sort_values(
            by=[
                "keyword_hit_count",
                "topic_match_score",
                "feature_ready_score",
                "activity_score",
                "stability_score",
                "influence_score",
                "user_followers",
                "_row_id",
            ],
            ascending=[False, False, False, False, False, False, False, True],
            kind="mergesort",
        ).drop(columns=["_row_id"]).reset_index(drop=True)

        # 14. 构造 NodeFeature（model_construct 跳过 Pydantic 校验，每条 ~5-10μs）
        user_id_arr = df["user_id"].to_numpy()
        user_name_arr = df["user_name"].to_list()
        platform_arr = df["platform"].to_list()
        influencer_flag_arr = df["influencer_flag"].to_numpy()
        is_source_arr = df["is_interaction_source"].to_numpy()
        follower_count_arr = df["user_followers"].to_numpy()
        friend_count_arr = df["user_friends"].to_numpy()
        interest_count_arr = df["interest_count"].to_numpy()
        has_description_arr = df["has_description"].to_numpy()
        neighbor_count_arr = df["neighbor_count"].to_numpy()
        mutual_neighbor_count_arr = df["mutual_neighbor_count"].to_numpy()
        received_int_arr = df["received_interaction_count"].to_numpy()
        made_int_arr = df["made_interaction_count"].to_numpy()
        received_comment_arr = df["received_comment_count"].to_numpy()
        received_repost_arr = df["received_repost_count"].to_numpy()
        made_comment_arr = df["made_comment_count"].to_numpy()
        made_repost_arr = df["made_repost_count"].to_numpy()
        self_int_arr = df["self_interaction_count"].to_numpy()
        # 9 个 score 列在 numpy 层面统一 round（一次性，~1ms），避免 35k * 9 次
        # Python ``round(float, 6)`` 的 0.5s 浪费。语义与原版 round 仍一致。
        comment_ratio_arr = np.round(df["comment_ratio"].to_numpy(), 6)
        repost_ratio_arr = np.round(df["repost_ratio"].to_numpy(), 6)
        mutual_neighbor_ratio_arr = np.round(df["mutual_neighbor_ratio"].to_numpy(), 6)
        profile_completeness_arr = np.round(df["profile_completeness_score"].to_numpy(), 6)
        topic_match_arr = np.round(df["topic_match_score"].to_numpy(), 6)
        influence_arr = np.round(df["influence_score"].to_numpy(), 6)
        diffusion_arr = np.round(df["diffusion_score"].to_numpy(), 6)
        activity_arr = np.round(df["activity_score"].to_numpy(), 6)
        stability_arr = np.round(df["stability_score"].to_numpy(), 6)
        feature_ready_arr = np.round(df["feature_ready_score"].to_numpy(), 6)
        role_hint_arr = df["role_hint"].to_list()
        semantic_profile_arr = df["semantic_profile"].to_list()
        matched_keywords_arr = df["matched_keywords"].to_list()
        keyword_hit_count_arr = df["keyword_hit_count"].to_numpy()

        # ---- 本地模型分：语义相关度 + 图结构传播潜力 ----
        # 产物由 ``scripts/build_local_models.py`` 预计算；缺失时两个数组保持全 0，
        # 打分链路自动退回纯规则模式（单测的最小数据集就是这种情况）。
        local_models = load_local_models(workspace_root) if workspace_root else None
        semantic_arr = np.zeros(len(user_id_arr), dtype=np.float64)
        potential_arr = np.zeros(len(user_id_arr), dtype=np.float64)
        semantic_rank_arr = np.zeros(len(user_id_arr), dtype=np.float64)
        if local_models is not None:
            # ``build_event_text`` 只编码 1 条话题，随后一次矩阵乘法覆盖全部节点。
            semantic_map = local_models.semantic_scores(build_event_text(event))
            potential_map = local_models.graph_potentials()
            semantic_arr = np.fromiter(
                (semantic_map.get(str(uid), 0.0) for uid in user_id_arr),
                dtype=np.float64,
                count=len(user_id_arr),
            )
            potential_arr = np.fromiter(
                (potential_map.get(str(uid), 0.0) for uid in user_id_arr),
                dtype=np.float64,
                count=len(user_id_arr),
            )

            # 分位数排名（0~1）：让 gate 阈值跨后端、跨事件保持同一含义。
            # 低于后端绝对下限的节点直接置 0，避免"全库都不相关"时仍放行前 10%。
            backend = str(getattr(local_models.encoder, "backend", "svd"))
            floor = LOCAL_SEMANTIC_FLOOR.get(backend, 0.10)
            if len(semantic_arr) > 1:
                order = semantic_arr.argsort().argsort()
                semantic_rank_arr = order / float(len(semantic_arr) - 1)
            semantic_rank_arr = np.where(
                semantic_arr >= floor, semantic_rank_arr, 0.0
            )

        node_features: list[NodeFeature] = []
        # Pydantic 2 的 __setattr__ 每次都要走 _model_field_setattr_handler，
        # 35k * 47 字段 ≈ 1.6s 浪费。用 __new__ + __dict__.update 完全绕过，
        # 再把 Pydantic 2 内部需要的 __pydantic_fields_set__ 等手动补齐。
        # 拿一个用占位符的完整 default 实例 dump（拿到每个字段的真实默认值）。
        # ``user_id/user_name/event_id/event_type/role_hint`` 是 required，循环里再覆盖。
        _default_dict = NodeFeature(
            user_id="", user_name="", event_id=event.event_id,
            event_type=event.event_type, role_hint="general_support",
        ).model_dump()
        for i in range(len(df)):
            mk = matched_keywords_arr[i]
            d = dict(_default_dict)
            d["user_id"] = str(user_id_arr[i])
            d["user_name"] = user_name_arr[i]
            d["event_id"] = event.event_id
            d["event_type"] = event.event_type
            d["role_hint"] = str(role_hint_arr[i])
            d["platform"] = str(platform_arr[i])
            d["influencer_flag"] = bool(influencer_flag_arr[i])
            d["is_interaction_source"] = bool(is_source_arr[i])
            d["follower_count"] = int(follower_count_arr[i])
            d["friend_count"] = int(friend_count_arr[i])
            d["interest_count"] = int(interest_count_arr[i])
            d["has_description"] = bool(has_description_arr[i])
            d["neighbor_count"] = int(neighbor_count_arr[i])
            d["mutual_neighbor_count"] = int(mutual_neighbor_count_arr[i])
            d["received_interaction_count"] = int(received_int_arr[i])
            d["made_interaction_count"] = int(made_int_arr[i])
            d["received_comment_count"] = int(received_comment_arr[i])
            d["received_repost_count"] = int(received_repost_arr[i])
            d["made_comment_count"] = int(made_comment_arr[i])
            d["made_repost_count"] = int(made_repost_arr[i])
            d["self_interaction_count"] = int(self_int_arr[i])
            # 9 个 score 字段已经在 numpy 层面 round 过了，直接 float() 取值。
            d["comment_ratio"] = float(comment_ratio_arr[i])
            d["repost_ratio"] = float(repost_ratio_arr[i])
            d["mutual_neighbor_ratio"] = float(mutual_neighbor_ratio_arr[i])
            d["profile_completeness_score"] = float(profile_completeness_arr[i])
            d["topic_match_score"] = float(topic_match_arr[i])
            d["influence_score"] = float(influence_arr[i])
            d["diffusion_score"] = float(diffusion_arr[i])
            d["activity_score"] = float(activity_arr[i])
            d["stability_score"] = float(stability_arr[i])
            d["feature_ready_score"] = float(feature_ready_arr[i])
            d["semantic_profile"] = str(semantic_profile_arr[i])
            d["semantic_tags"] = list(mk[:4])
            d["keyword_hit_count"] = int(keyword_hit_count_arr[i])
            d["matched_keywords"] = list(mk)
            d["local_semantic_score"] = float(semantic_arr[i])
            d["local_semantic_rank"] = float(semantic_rank_arr[i])
            d["graph_potential"] = float(potential_arr[i])
            nf = NodeFeature.__new__(NodeFeature)
            nf.__dict__.update(d)
            # Pydantic 2 内部状态：缺这些下游 model_copy / model_dump / 校验会出错
            object.__setattr__(nf, "__pydantic_fields_set__", set(d.keys()))
            object.__setattr__(nf, "__pydantic_extra__", None)
            object.__setattr__(nf, "__pydantic_private__", None)
            node_features.append(nf)

        node_features = self._augment_with_llm(
            node_features=node_features,
            event=event,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=llm_client,
        )

        matched_node_count = sum(1 for item in node_features if item.keyword_hit_count > 0)
        source_node_count = sum(1 for item in node_features if item.is_interaction_source)
        avg_feature_ready = round(
            sum(item.feature_ready_score for item in node_features) / max(len(node_features), 1),
            6,
        )
        avg_topic_match = round(
            sum(item.topic_match_score for item in node_features) / max(len(node_features), 1),
            6,
        )
        summary = FeatureBuildSummary(
            event_id=event.event_id,
            event_type=event.event_type,
            node_count=len(node_features),
            source_node_count=source_node_count,
            matched_node_count=matched_node_count,
            avg_feature_ready_score=avg_feature_ready,
            avg_topic_match_score=avg_topic_match,
            recall_size=(
                len(node_features) if recall_user_ids is not None else None
            ),
            recall_mode=recall_mode,
            total_candidate_count=total_candidate_count,
        )
        return FeatureBuildResult(
            event=event,
            product_context=product_context,
            summary=summary,
            node_features=node_features,
        )

    def to_frame(self, result: FeatureBuildResult) -> pd.DataFrame:
        rows = [item.model_dump() for item in result.node_features]
        return pd.DataFrame(rows)

    def _prepare_keywords(self, keywords: list[str]) -> list[str]:
        prepared: list[str] = []
        for keyword in keywords:
            normalized = _normalize_text(keyword)
            if not normalized:
                continue
            if normalized not in prepared:
                prepared.append(normalized)
        return prepared

    def _event_match_keywords(self, event: ParsedEvent) -> list[str]:
        generic_tags = {"general_public", "awareness", "engagement", "response", "conversion"}
        audience_keywords = [
            item
            for item in event.target_audience
            if item not in generic_tags
        ]
        semantic_keywords = [
            item
            for item in event.semantic_tags
            if item not in generic_tags
        ]
        keywords = [
            event.event_title,
            *event.extracted_keywords,
            *semantic_keywords,
            *audience_keywords,
        ]
        event_text = _normalize_text(
            " ".join(
                [
                    event.event_title,
                    event.event_description,
                    *event.extracted_keywords,
                    *semantic_keywords,
                ]
            )
        ).replace(" ", "")
        for triggers, expansions in EVENT_KEYWORD_EXPANSIONS:
            if any(trigger in event_text for trigger in triggers):
                keywords.extend(expansions)
        return keywords

    def _match_keywords(self, profile: UserProfile, normalized_keywords: list[str]) -> list[str]:
        if not normalized_keywords:
            return []

        search_text = _normalize_text(
            " ".join([profile.user_description, *profile.user_interests])
        )
        if not search_text:
            return []

        compact_text = search_text.replace(" ", "")
        # 第一遍：只做子串匹配。如果一个 keyword 都没命中，**完全跳过**
        # fuzzy 阶段（35k profile 中大部分与事件不相关，省 3-7s 的 rapidfuzz）。
        substring_matched = [k for k in normalized_keywords if k in compact_text]
        if not substring_matched:
            return []
        # 至少一个子串命中，再对剩下的 keyword 做 fuzzy
        remaining = [k for k in normalized_keywords if k not in substring_matched]
        if not remaining:
            return substring_matched

        tokens = search_text.split()
        matches = list(substring_matched)
        for keyword in remaining:
            if any(self._token_fuzzy_matches(keyword, token) for token in tokens):
                matches.append(keyword)
        return matches

    def _token_fuzzy_matches(self, keyword: str, token: str) -> bool:
        if not keyword or not token:
            return False
        normalized_keyword = keyword.replace(" ", "")
        normalized_token = token.replace(" ", "")
        if len(normalized_keyword) <= 1 or len(normalized_token) <= 1:
            return False
        if normalized_keyword.isascii() or normalized_token.isascii():
            if len(normalized_keyword) < 4 or len(normalized_token) < 4:
                return False
        else:
            if len(normalized_keyword) < 2 or len(normalized_token) < 2:
                return False
            shorter = min(len(normalized_keyword), len(normalized_token))
            longer = max(len(normalized_keyword), len(normalized_token))
            if longer >= 4 and (shorter < 3 or shorter / longer < 0.50):
                return False
        return fuzz.partial_ratio(keyword, token) >= 90

    def _profile_completeness_score(self, profile: UserProfile) -> float:
        has_description = 1.0 if profile.user_description.strip() else 0.0
        interest_score = min(len(profile.user_interests) / 4, 1.0)
        follower_presence = 1.0 if profile.user_followers > 0 else 0.0
        return _clip_score((has_description + interest_score + follower_presence) / 3)

    def _stability_score(
        self,
        profile_completeness_score: float,
        total_interaction_count: int,
        self_interaction_count: int,
    ) -> float:
        self_loop_ratio = self_interaction_count / total_interaction_count if total_interaction_count else 0.0
        self_loop_penalty = min(self_loop_ratio, 1.0)
        return _clip_score(0.75 * profile_completeness_score + 0.25 * (1 - self_loop_penalty))

    def _infer_role_hint(
        self,
        influence_score: float,
        diffusion_score: float,
        activity_score: float,
        topic_match_score: float,
        comment_ratio: float,
    ) -> str:
        if influence_score >= 0.6 and topic_match_score >= 0.2:
            return "core_broadcast"
        if diffusion_score >= 0.45:
            return "amplification"
        if activity_score >= 0.35 or comment_ratio >= 0.6:
            return "interaction_response"
        return "general_support"

    def _max_log(self, values) -> float:
        max_value = max((math.log1p(max(0, int(value))) for value in values), default=0.0)
        return max(max_value, 1.0)

    def _log_score(self, value: int, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return _clip_score(math.log1p(max(0, int(value))) / denominator)

    # ---- 向量化加速（feature_builder 在 35k 节点规模下是端到端主瓶颈） ----
    @staticmethod
    def _max_log_array(values: np.ndarray) -> float:
        """_max_log 的 numpy 版：``max(log1p(max(0, x)))``，保底 1.0。"""
        if values.size == 0:
            return 1.0
        safe = np.maximum(values.astype(np.int64, copy=False), 0)
        m = float(np.log1p(safe).max())
        return max(m, 1.0)

    @staticmethod
    def _log_score_array(values: np.ndarray, denominator: float) -> np.ndarray:
        """_log_score 的 numpy 版。``denominator <= 0`` 时整列返回 0。"""
        if denominator <= 0:
            return np.zeros_like(values, dtype=np.float64)
        safe = np.maximum(values.astype(np.int64, copy=False), 0)
        return np.clip(np.log1p(safe) / denominator, 0.0, 1.0)

    def _sort_features(self, node_features: list[NodeFeature]) -> list[NodeFeature]:
        node_features.sort(
            key=lambda item: (
                item.keyword_hit_count > 0,
                item.topic_match_score,
                item.feature_ready_score,
                item.activity_score,
                item.stability_score,
                item.influence_score,
                item.follower_count,
            ),
            reverse=True,
        )
        return node_features

    def _augment_with_llm(
        self,
        *,
        node_features: list[NodeFeature],
        event: ParsedEvent,
        workspace_root: str | Path | None,
        use_llm: bool,
        llm_client: Any | None,
    ) -> list[NodeFeature]:
        if not use_llm or not node_features:
            return node_features

        client = llm_client
        if client is None and workspace_root is not None:
            client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
        if client is None:
            return node_features

        candidate_pool_size = event.dispatch_preferences.candidate_pool_size
        top_candidates = self._candidate_pool_for_llm(
            node_features=node_features,
            candidate_pool_size=candidate_pool_size,
        )
        candidate_cards = [
            {
                "user_id": feature.user_id,
                "user_name": feature.user_name,
                "role_hint": feature.role_hint,
                "semantic_profile": feature.semantic_profile,
                "matched_keywords": feature.matched_keywords,
                "baseline_scores": {
                    "feature_ready_score": feature.feature_ready_score,
                    "topic_match_score": feature.topic_match_score,
                    "influence_score": feature.influence_score,
                    "diffusion_score": feature.diffusion_score,
                    "activity_score": feature.activity_score,
                    "stability_score": feature.stability_score,
                },
                "signals": {
                    "follower_count": feature.follower_count,
                    "friend_count": feature.friend_count,
                    "neighbor_count": feature.neighbor_count,
                    "mutual_neighbor_count": feature.mutual_neighbor_count,
                    "received_interaction_count": feature.received_interaction_count,
                    "made_interaction_count": feature.made_interaction_count,
                    "is_interaction_source": feature.is_interaction_source,
                    "influencer_flag": feature.influencer_flag,
                },
            }
            for feature in top_candidates
        ]

        # 提示词预先构造（提到 try 之外）：既能在异常路径记录 prompt 体积，
        # 也避免把字符串拼接的开销算进 LLM 耗时。
        system_prompt = build_feature_enrichment_system_prompt()
        user_prompt = build_feature_enrichment_user_prompt(
            event=event,
            candidate_cards=candidate_cards,
        )
        prompt_chars = len(system_prompt) + len(user_prompt)
        started = time.monotonic()

        try:
            response = client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except (LLMClientError, OSError, ValueError, TypeError) as exc:
            elapsed_sec = time.monotonic() - started
            error_type = type(exc).__name__
            _record_feature_llm_outcome(
                success=False,
                error_type=error_type,
                error_message=str(exc),
                prompt_chars=prompt_chars,
                elapsed_sec=elapsed_sec,
            )
            logger.warning(
                "feature_builder._augment_with_llm failed: "
                "error_type=%s prompt_chars=%d elapsed_sec=%.3f error=%r",
                error_type,
                prompt_chars,
                elapsed_sec,
                exc,
            )
            return node_features

        elapsed_sec = time.monotonic() - started
        _record_feature_llm_outcome(
            success=True,
            error_type=None,
            error_message=None,
            prompt_chars=prompt_chars,
            elapsed_sec=elapsed_sec,
        )
        logger.info(
            "feature_builder._augment_with_llm success: "
            "prompt_chars=%d elapsed_sec=%.3f",
            prompt_chars,
            elapsed_sec,
        )

        raw_nodes = self._extract_llm_node_mapping(
            response=response,
            expected_ids=[feature.user_id for feature in top_candidates],
        )
        if not raw_nodes:
            return node_features

        llm_updates: dict[str, dict[str, Any]] = {}
        for feature in top_candidates:
            raw_node = (
                raw_nodes.get(feature.user_id)
                or raw_nodes.get(str(feature.user_id))
                or raw_nodes.get(f"id{feature.user_id}")
                or raw_nodes.get(f"digital_human_{feature.user_id}")
            )
            if not isinstance(raw_node, dict):
                continue

            event_relevance = self._coerce_score(
                self._pick_first(raw_node, "semantic_relevance_score", "event_semantic_relevance_score", "event_relevance_score", "relevance_score")
            )
            audience_fit = self._coerce_score(
                self._pick_first(raw_node, "audience_fit_score", "audience_score")
            )
            role_fit = self._coerce_score(
                self._pick_first(raw_node, "role_fit_score", "role_score")
            )
            narrative_fit = self._coerce_score(
                self._pick_first(raw_node, "narrative_fit_score", "narrative_score")
            )
            risk_conflict = self._coerce_score(
                self._pick_first(raw_node, "risk_conflict_score", "risk_score", "conflict_score")
            )
            novelty = self._coerce_score(
                self._pick_first(raw_node, "novelty_score", "diversity_score")
            )
            llm_feature_score = _clip_score(
                0.35 * event_relevance
                + 0.20 * audience_fit
                + 0.20 * role_fit
                + 0.15 * narrative_fit
                + 0.10 * novelty
            )
            llm_updates[feature.user_id] = {
                "event_semantic_relevance_score": event_relevance,
                "audience_fit_score": audience_fit,
                "role_fit_score": role_fit,
                "narrative_fit_score": narrative_fit,
                "risk_conflict_score": risk_conflict,
                "novelty_score": novelty,
                "llm_feature_score": llm_feature_score,
                "semantic_tags": self._normalize_semantic_tags(
                    self._pick_first(raw_node, "semantic_tags", "tags")
                ) or feature.semantic_tags,
                "candidate_reasoning": self._normalize_reasoning(
                    self._pick_first(raw_node, "reasoning", "reasons", "analysis")
                ),
                "llm_feature_used": True,
            }

        if not llm_updates:
            return node_features

        semantic_weight = max(event.dispatch_preferences.semantic_weight, 0.50)
        base_weight = max(0.0, 1.0 - semantic_weight)
        updated_features: list[NodeFeature] = []
        for feature in node_features:
            update = llm_updates.get(feature.user_id)
            if update is None:
                updated_features.append(feature)
                continue

            augmented_ready_score = _clip_score(
                base_weight * feature.feature_ready_score
                + semantic_weight * update["llm_feature_score"]
            )
            updated_features.append(
                feature.model_copy(
                    update={
                        **update,
                        "feature_ready_score": round(augmented_ready_score, 6),
                    }
                )
            )

        return self._sort_features(updated_features)

    def _coerce_score(self, value: Any) -> float:
        try:
            return _clip_score(float(value))
        except (TypeError, ValueError):
            return 0.0

    def _pick_first(self, payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    def _candidate_pool_for_llm(
        self,
        *,
        node_features: list[NodeFeature],
        candidate_pool_size: int,
    ) -> list[NodeFeature]:
        diversified: list[NodeFeature] = []
        used_ids: set[str] = set()

        def add_bucket(features: list[NodeFeature], *, generic_high_influence_cap: int | None = None) -> None:
            generic_high_influence_count = 0
            for feature in features:
                if len(diversified) >= candidate_pool_size:
                    return
                if feature.user_id in used_ids:
                    continue
                is_generic_high_influence = (
                    feature.keyword_hit_count == 0
                    and feature.topic_match_score == 0.0
                    and feature.influence_score >= 0.65
                )
                if (
                    generic_high_influence_cap is not None
                    and is_generic_high_influence
                    and generic_high_influence_count >= generic_high_influence_cap
                ):
                    continue
                diversified.append(feature)
                used_ids.add(feature.user_id)
                if is_generic_high_influence:
                    generic_high_influence_count += 1

        topic_bucket = sorted(
            node_features,
            key=lambda item: (
                item.keyword_hit_count,
                item.topic_match_score,
                item.stability_score,
                item.activity_score,
            ),
            reverse=True,
        )
        interaction_bucket = sorted(
            [
                item
                for item in node_features
                if item.is_interaction_source or item.role_hint == "interaction_response"
            ],
            key=lambda item: (
                item.activity_score,
                item.diffusion_score,
                item.stability_score,
                item.influence_score,
            ),
            reverse=True,
        )
        diffusion_bucket = sorted(
            node_features,
            key=lambda item: (
                item.diffusion_score,
                item.activity_score,
                item.stability_score,
                item.influence_score,
            ),
            reverse=True,
        )
        baseline_bucket = list(node_features)

        add_bucket(topic_bucket)
        add_bucket(interaction_bucket, generic_high_influence_cap=max(2, candidate_pool_size // 6))
        add_bucket(diffusion_bucket, generic_high_influence_cap=max(2, candidate_pool_size // 6))
        add_bucket(baseline_bucket, generic_high_influence_cap=max(2, candidate_pool_size // 6))
        return diversified[:candidate_pool_size]

    def _extract_llm_node_mapping(
        self,
        *,
        response: dict[str, Any],
        expected_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(response, dict):
            return {}

        alias_keys = ("nodes", "candidates", "results", "items", "data")
        for alias in alias_keys:
            normalized = self._normalize_llm_container(response.get(alias), expected_ids)
            if normalized:
                return normalized

        normalized_response = self._normalize_llm_container(response, expected_ids)
        if normalized_response:
            return normalized_response

        for value in response.values():
            normalized = self._normalize_llm_container(value, expected_ids)
            if normalized:
                return normalized
        return {}

    def _normalize_llm_container(
        self,
        container: Any,
        expected_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        if isinstance(container, list):
            normalized: dict[str, dict[str, Any]] = {}
            for item in container:
                if not isinstance(item, dict):
                    continue
                user_id = str(
                    item.get("user_id")
                    or item.get("id")
                    or item.get("node_id")
                    or item.get("digital_human_id")
                    or ""
                ).strip()
                if user_id:
                    normalized[user_id] = item
            return normalized

        if not isinstance(container, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for expected_id in expected_ids:
            raw_node = (
                container.get(expected_id)
                or container.get(str(expected_id))
                or container.get(f"id{expected_id}")
                or container.get(f"digital_human_{expected_id}")
            )
            if isinstance(raw_node, dict):
                normalized[str(expected_id)] = raw_node

        if normalized:
            return normalized

        for key, value in container.items():
            if not isinstance(value, dict):
                continue
            key_text = str(key).strip()
            candidate_id = key_text.removeprefix("id").removeprefix("digital_human_")
            if candidate_id and candidate_id in expected_ids:
                normalized[candidate_id] = value
        return normalized

    def _normalize_semantic_tags(self, value: Any) -> list[str]:
        if isinstance(value, str):
            value = re.split(r"[,;，；]\s*", value)
        if not isinstance(value, list):
            return []
        tags: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in tags:
                tags.append(text)
        return tags[:4]

    def _normalize_reasoning(self, value: Any) -> list[str]:
        if isinstance(value, str):
            value = re.split(r"[\n,;，；]\s*", value)
        if not isinstance(value, list):
            return []
        reasoning: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in reasoning:
                reasoning.append(text)
        return reasoning[:4]
