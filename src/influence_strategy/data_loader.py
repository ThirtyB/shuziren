from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    DatasetBundle,
    DatasetSummary,
    EnrichedUserProfile,
    GraphAttributes,
    InteractionRecord,
    NeighborRelation,
    ProductContext,
    UserProfile,
)


# 模块级 JSON 解析缓存：(path, mtime) → 解析结果。
# 一次 Python 进程内同一文件只解析一次，多事件评测省 ~1.2s/条。
_JSON_CACHE: dict[tuple[str, int], Any] = {}
# ``load_enriched_graph_attributes`` 的校验结果缓存：(path, mtime, limit) → dict。
_ENRICHED_GA_CACHE: dict[tuple[str, int, int | None], dict[str, GraphAttributes]] = {}
# 第九轮：``load_profiles`` 的校验结果缓存：(path, mtime, limit) → dict[str, UserProfile]。
_PROFILES_CACHE: dict[tuple[str, int, int | None], dict[str, UserProfile]] = {}


class DataLoader:
    def __init__(self, workspace_root: str | Path, product_name: str = "abc_reading") -> None:
        self.workspace_root = Path(workspace_root)
        self.product_name = product_name
        self.raw_dir = self.workspace_root / "data" / "raw"
        self.derived_dir = self.workspace_root / "data" / "derived"

    @property
    def product_info_path(self) -> Path:
        return self.raw_dir / f"{self.product_name}_product_info.json"

    @property
    def profile_path(self) -> Path:
        return self.raw_dir / f"{self.product_name}_profile.graph.anon"

    @property
    def interaction_path(self) -> Path:
        return self.raw_dir / f"{self.product_name}_interaction.graph.anon"

    @property
    def compact_profile_path(self) -> Path:
        return self.derived_dir / f"{self.product_name}_profile_compact.graph.anon"

    @property
    def reworked_profile_path(self) -> Path:
        return self.derived_dir / "weibo_profile_reworked.graph.anon"

    @property
    def reworked_enriched_profile_path(self) -> Path:
        return self.derived_dir / "weibo_profile_reworked_with_neighbors.graph.anon"

    @property
    def platform_enriched_profile_path(self) -> Path:
        """带模拟平台字段（``platform``）的派生数据集，优先级最高。"""

        return (
            self.derived_dir
            / "weibo_profile_reworked_with_neighbors_with_platform.graph.anon"
        )

    @property
    def enriched_profile_path(self) -> Path:
        return self.derived_dir / f"{self.product_name}_profile_with_neighbors.graph.anon"

    def _preferred_profile_path(self) -> Path:
        if self.platform_enriched_profile_path.exists():
            return self.platform_enriched_profile_path
        if self.reworked_enriched_profile_path.exists():
            return self.reworked_enriched_profile_path
        if self.reworked_profile_path.exists():
            return self.reworked_profile_path
        if self.compact_profile_path.exists():
            return self.compact_profile_path
        if self.profile_path.exists():
            return self.profile_path
        if self.enriched_profile_path.exists():
            return self.enriched_profile_path
        return self.profile_path

    def _preferred_enriched_profile_path(self) -> Path:
        if self.platform_enriched_profile_path.exists():
            return self.platform_enriched_profile_path
        if self.reworked_enriched_profile_path.exists():
            return self.reworked_enriched_profile_path
        if self.reworked_profile_path.exists():
            return self.reworked_profile_path
        if self.compact_profile_path.exists():
            return self.compact_profile_path
        return self.enriched_profile_path

    def _read_json(self, path: Path) -> Any:
        # 模块级缓存：同一路径在一次 Python 进程内只解析一次 JSON。
        # 多事件评测（5 条平均）可省 ~1.2s/条。
        key = (str(path), path.stat().st_mtime_ns if path.exists() else 0)
        cached = _JSON_CACHE.get(key)
        if cached is not None:
            return cached
        with path.open("rb") as handle:
            try:
                import orjson
                data = orjson.loads(handle.read())
            except ImportError:
                handle.seek(0)
                data = json.load(handle)
        _JSON_CACHE[key] = data
        return data

    def _ensure_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    def load_product_context(self) -> ProductContext:
        if not self.product_info_path.exists():
            return ProductContext(product_name=self.product_name)
        payload = self._read_json(self.product_info_path)
        if not isinstance(payload, dict):
            raise ValueError("Product info file must contain one JSON object.")
        payload = dict(payload)
        payload["influencer_ids"] = [str(item) for item in payload.get("influencer_ids", [])]
        return ProductContext.model_validate(payload)

    def load_profiles(self, limit: int | None = None) -> dict[str, UserProfile]:
        selected_path = self._preferred_profile_path()
        if not selected_path.exists():
            self._ensure_file(selected_path)
        payload = self._read_json(selected_path)
        if not isinstance(payload, dict):
            raise ValueError("Profile file must contain a JSON object.")

        # 第九轮：与 ``load_enriched_graph_attributes`` 同款的校验结果缓存。
        # 原始 JSON 早已进 ``_JSON_CACHE``，但这里每次调用仍要把 35,636 条
        # profile 重新跑一遍 ``UserProfile.model_validate``（实测 0.17~0.26s/次），
        # 在多事件评测 / 5 次压测里这份开销会按次数线性重复。
        # 缓存键带 (path, mtime_ns, limit)，文件一变即失效；调用方只读不写
        # profile 对象，共享实例不会引入互相污染（全仓无对 profile 的就地修改）。
        mtime = selected_path.stat().st_mtime_ns if selected_path.exists() else 0
        cache_key = (str(selected_path), mtime, limit)
        cached = _PROFILES_CACHE.get(cache_key)
        if cached is not None:
            return cached

        result: dict[str, UserProfile] = {}
        for index, (user_id, raw_profile) in enumerate(payload.items()):
            if limit is not None and index >= limit:
                break
            item = dict(raw_profile)
            item["user_id"] = str(item.get("user_id", user_id))
            item["user_name"] = item.get("user_name") or f"user_{user_id}"
            result[str(user_id)] = UserProfile.model_validate(item)
        _PROFILES_CACHE[cache_key] = result
        return result

    def load_interactions(
        self,
        limit_sources: int | None = None,
        limit_records_per_source: int | None = None,
    ) -> dict[str, list[InteractionRecord]]:
        if not self.interaction_path.exists():
            return {}
        payload = self._read_json(self.interaction_path)
        if not isinstance(payload, dict):
            raise ValueError("Interaction file must contain a JSON object.")

        result: dict[str, list[InteractionRecord]] = {}
        for source_index, (source_user_id, records) in enumerate(payload.items()):
            if limit_sources is not None and source_index >= limit_sources:
                break
            normalized_records: list[InteractionRecord] = []
            for record_index, raw_record in enumerate(records):
                if limit_records_per_source is not None and record_index >= limit_records_per_source:
                    break
                item = dict(raw_record)
                item["source_user_id"] = str(source_user_id)
                item["target_user_id"] = str(item.pop("interact_id"))
                normalized_records.append(InteractionRecord.model_validate(item))
            result[str(source_user_id)] = normalized_records
        return result

    def iter_interactions(
        self,
        limit_sources: int | None = None,
        limit_records_per_source: int | None = None,
    ):
        interaction_map = self.load_interactions(
            limit_sources=limit_sources,
            limit_records_per_source=limit_records_per_source,
        )
        for records in interaction_map.values():
            for record in records:
                yield record

    def load_enriched_profiles(self, limit: int | None = None) -> dict[str, EnrichedUserProfile]:
        selected_path = self._preferred_enriched_profile_path()
        self._ensure_file(selected_path)
        payload = self._read_json(selected_path)
        if not isinstance(payload, dict):
            raise ValueError("Enriched profile file must contain a JSON object.")

        result: dict[str, EnrichedUserProfile] = {}
        for index, (user_id, raw_profile) in enumerate(payload.items()):
            if limit is not None and index >= limit:
                break
            item = dict(raw_profile)
            item["user_id"] = str(item.get("user_id", user_id))
            item["graph_attributes"] = GraphAttributes.model_validate(item.get("graph_attributes", {}))
            item["neighbors"] = [
                NeighborRelation.model_validate(neighbor)
                for neighbor in item.get("neighbors", [])
            ]
            result[str(user_id)] = EnrichedUserProfile.model_validate(item)
        return result

    def load_enriched_graph_attributes(
        self, limit: int | None = None
    ) -> dict[str, GraphAttributes]:
        """``load_enriched_profiles`` 的快速版：只取 ``graph_attributes`` 9 个字段，
        跳过 ``neighbors`` 列表与 ``EnrichedUserProfile`` 整体校验。
        ``load_enriched_profiles`` 在 35k 节点上需要 ~3.3s（多数花在 neighbors 解析
        + Pydantic 校验），本方法通常 < 0.5s，feature_builder 拿到的
        仍是 ``.graph_attributes.X`` 形式的对象。
        """
        selected_path = self._preferred_enriched_profile_path()
        self._ensure_file(selected_path)
        payload = self._read_json(selected_path)
        if not isinstance(payload, dict):
            raise ValueError("Enriched profile file must contain a JSON object.")

        # 缓存 (path, mtime, limit) → dict[str, GraphAttributes]，省 35k 次 Pydantic
        # model_validate 的开销（约 0.5s/次调用）。多事件评测可显著省时。
        mtime = selected_path.stat().st_mtime_ns if selected_path.exists() else 0
        cache_key = (str(selected_path), mtime, limit)
        cached = _ENRICHED_GA_CACHE.get(cache_key)
        if cached is not None:
            return cached

        result: dict[str, GraphAttributes] = {}
        for index, (user_id, raw_profile) in enumerate(payload.items()):
            if limit is not None and index >= limit:
                break
            ga_raw = raw_profile.get("graph_attributes", {}) if isinstance(raw_profile, dict) else {}
            if not isinstance(ga_raw, dict):
                ga_raw = {}
            result[str(user_id)] = GraphAttributes.model_validate(ga_raw)
        _ENRICHED_GA_CACHE[cache_key] = result
        return result

    def build_summary(self) -> DatasetSummary:
        product_context = self.load_product_context()
        profiles_raw = self._read_json(self._preferred_profile_path())
        interactions_raw = self._read_json(self.interaction_path) if self.interaction_path.exists() else {}
        enriched_count: int | None = None
        preferred_enriched_path = self._preferred_enriched_profile_path()
        if preferred_enriched_path.exists():
            enriched_raw = self._read_json(preferred_enriched_path)
            enriched_count = len(enriched_raw)

        return DatasetSummary(
            product_name=product_context.product_name,
            profile_count=len(profiles_raw),
            interaction_source_count=len(interactions_raw),
            interaction_record_count=sum(len(records) for records in interactions_raw.values()),
            enriched_profile_count=enriched_count,
        )

    def load_dataset_bundle(
        self,
        include_enriched_profiles: bool = True,
        profile_limit: int | None = None,
        interaction_source_limit: int | None = None,
        interaction_record_limit: int | None = None,
    ) -> DatasetBundle:
        product_context = self.load_product_context()
        profiles = self.load_profiles(limit=profile_limit)
        interactions = self.load_interactions(
            limit_sources=interaction_source_limit,
            limit_records_per_source=interaction_record_limit,
        )
        enriched_profiles = None
        if include_enriched_profiles and self._preferred_enriched_profile_path().exists():
            enriched_profiles = self.load_enriched_profiles(limit=profile_limit)

        summary = DatasetSummary(
            product_name=product_context.product_name,
            profile_count=len(profiles),
            interaction_source_count=len(interactions),
            interaction_record_count=sum(len(records) for records in interactions.values()),
            enriched_profile_count=len(enriched_profiles) if enriched_profiles is not None else None,
        )
        return DatasetBundle(
            product_context=product_context,
            profiles=profiles,
            interactions=interactions,
            enriched_profiles=enriched_profiles,
            summary=summary,
        )

    @staticmethod
    def summarize_profile_for_semantics(
        profile: UserProfile,
        *,
        enriched_profile: EnrichedUserProfile | None = None,
        is_interaction_source: bool = False,
    ) -> str:
        # 兼容 ``load_enriched_graph_attributes`` 直接返回 GraphAttributes 的快速路径
        ga_attr = getattr(enriched_profile, "graph_attributes", None)
        if ga_attr is None and isinstance(enriched_profile, GraphAttributes):
            ga_attr = enriched_profile
        graph_attributes = ga_attr if ga_attr is not None else GraphAttributes()
        interests = ", ".join(item for item in profile.user_interests if item)
        description = profile.user_description.strip() or "no_description"
        tags = [
            f"user_name={profile.user_name}",
            f"platform={profile.platform or 'unknown'}",
            f"followers={profile.user_followers}",
            f"friends={profile.user_friends}",
            f"interests={interests or 'none'}",
            f"description={description}",
            f"neighbor_count={graph_attributes.neighbor_count}",
            f"mutual_neighbor_count={graph_attributes.mutual_neighbor_count}",
            f"received_interaction_count={graph_attributes.received_interaction_count}",
            f"made_interaction_count={graph_attributes.made_interaction_count}",
            f"self_interaction_count={graph_attributes.self_interaction_count}",
            f"is_interaction_source={str(bool(is_interaction_source)).lower()}",
        ]
        return " | ".join(tags)

    def build_candidate_materials(
        self,
        *,
        user_ids: list[str],
        profiles: dict[str, UserProfile] | None = None,
        enriched_profiles: dict[str, EnrichedUserProfile] | None = None,
        source_user_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        profiles = profiles or self.load_profiles()
        if enriched_profiles is None and self._preferred_enriched_profile_path().exists():
            try:
                enriched_profiles = self.load_enriched_profiles()
            except ValueError:
                enriched_profiles = {}
        enriched_profiles = enriched_profiles or {}
        source_user_ids = source_user_ids or set()

        materials: list[dict[str, Any]] = []
        for user_id in user_ids:
            profile = profiles.get(user_id)
            if profile is None:
                continue
            enriched_profile = enriched_profiles.get(user_id)
            materials.append(
                {
                    "user_id": user_id,
                    "user_name": profile.user_name,
                    "semantic_profile": self.summarize_profile_for_semantics(
                        profile,
                        enriched_profile=enriched_profile,
                        is_interaction_source=user_id in source_user_ids,
                    ),
                }
            )
        return materials
