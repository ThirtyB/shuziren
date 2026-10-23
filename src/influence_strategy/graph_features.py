"""图结构特征：结构指标 + 图卷积传播，用于刻画节点的「传播潜力」。

为什么需要它
------------
原有 ``diffusion_score`` 只看节点自身的转发/评论计数，看不到它在关系网络里的
位置：一个粉丝不多、但处在互动密集社群中心的节点，实际带动能力远高于同样粉丝
数的孤立节点。这类信息只有把邻居关系纳入计算才能拿到 —— 也就是图神经网络擅长
的部分。

为什么不是 GraphSAGE
--------------------
GraphSAGE 需要训练，而本项目没有「选人效果」的标注数据，训练目标只能是自监督
（如 link prediction），收益不确定却引入 torch/GNN 框架依赖。这里改用同构的
**SGC（Simplified Graph Convolution）前向传播**：把邻居特征做归一化平均后逐层
聚合，等价于「权重固定的图卷积」。它保留 GNN 最核心的结构信息聚合能力，但无需
训练、无额外依赖，且计算可在毫秒级完成。

输出
----
* ``features``：每行的结构特征（自身 + 1-hop + 2-hop 聚合），供后续打分使用；
* ``potential``：归一化到 [0,1] 的传播潜力综合分，直接并入打分。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

#: 参与 SGC 传播的自身特征列（都先做 log1p 压缩长尾）。
_FEATURE_NAMES = (
    "log_followers",
    "log_friends",
    "log_received_interaction",
    "log_made_interaction",
    "mutual_ratio",
    "engaged_ratio",
    "repost_ratio",
    "not_isolated",
)

#: ``potential`` 各分量的权重。传播潜力先是"被看见"（pagerank），再是"能双向
#: 互动"（mutual），最后才是"处在活跃社群中"（邻域质量）。
_POTENTIAL_WEIGHTS = {
    "pagerank": 0.40,
    "interaction_intensity": 0.30,
    "mutual_ratio": 0.15,
    "neighbor_quality": 0.15,
}


@dataclass(slots=True)
class GraphIndex:
    """离线预计算的图结构产物。"""

    user_ids: list[str]
    features: np.ndarray  # (n, f) 已归一化
    potential: np.ndarray  # (n,) 0~1
    feature_names: list[str]

    def __post_init__(self) -> None:
        if len(self.user_ids) != int(self.features.shape[0]):
            raise ValueError(
                "GraphIndex user_ids 与 features 行数不一致: "
                f"{len(self.user_ids)} != {self.features.shape[0]}"
            )
        if len(self.user_ids) != int(self.potential.shape[0]):
            raise ValueError(
                "GraphIndex user_ids 与 potential 长度不一致: "
                f"{len(self.user_ids)} != {self.potential.shape[0]}"
            )

    @property
    def node_count(self) -> int:
        return len(self.user_ids)


def _log1p_clip(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 0:
        return 0.0
    return float(np.log1p(number))


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        den = float(denominator)
        if den <= 0:
            return 0.0
        return float(numerator) / den
    except (TypeError, ValueError):
        return 0.0


def _minmax(values: np.ndarray) -> np.ndarray:
    """按分位数做稳健归一化，避免极少数超大 V 号把其余节点压成 0。"""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    low = float(np.percentile(values, 1))
    high = float(np.percentile(values, 99))
    if high - low < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _pagerank(transition: Any, node_count: int, damping: float = 0.85, iterations: int = 30) -> np.ndarray:
    """幂迭代 PageRank。

    不用 ``networkx.pagerank``：3.5 万节点上它要走纯 Python 的迭代，而这里
    已经有稀疏矩阵，手写幂迭代只需几十次稀疏乘法，快一个量级。

    ``transition`` 是按行归一化的邻接矩阵（悬挂节点行为全 0）。
    """

    rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    teleport = (1.0 - damping) / node_count
    for _ in range(iterations):
        rank = damping * (transition.T @ rank) + teleport
        total = rank.sum()
        if total > 0:
            rank /= total
    return rank


class GraphFeatureBuilder:
    """把邻接关系转成结构特征与传播潜力分。"""

    def __init__(self, potential_weights: dict[str, float] | None = None) -> None:
        self.weights = dict(potential_weights or _POTENTIAL_WEIGHTS)
        self._index: GraphIndex | None = None
        self._id_to_row: dict[str, int] = {}

    # ---------------- 预计算 ----------------

    def build(self, graph_payload: dict[str, Any]) -> GraphIndex:
        """从 ``*.graph.anon`` 的原始 dict 构建结构索引。

        ``graph_payload`` 形如 ``{user_id: {..., "neighbors": [...]}}``，
        即 ``DataLoader._preferred_enriched_profile_path()`` 的内容。
        """

        from scipy import sparse

        user_ids = [str(key) for key in graph_payload.keys()]
        node_count = len(user_ids)
        if node_count == 0:
            raise ValueError("graph_payload 为空，无法构建图特征")

        row_of = {user_id: row for row, user_id in enumerate(user_ids)}
        raw_features = np.zeros((node_count, len(_FEATURE_NAMES)), dtype=np.float64)

        edge_rows: list[int] = []
        edge_cols: list[int] = []
        edge_weights: list[float] = []

        for user_id in user_ids:
            row = row_of[user_id]
            payload = graph_payload[user_id] or {}
            attributes = payload.get("graph_attributes") or {}
            neighbors = payload.get("neighbors") or []

            follower = payload.get("user_followers", 0)
            friend = payload.get("user_friends", 0)
            received = attributes.get("received_interaction_count", 0)
            made = attributes.get("made_interaction_count", 0)
            mutual = attributes.get("mutual_neighbor_count", 0)
            engaged_to = attributes.get("engaged_to_neighbor_count", 0)
            engaged_by = attributes.get("engaged_by_neighbor_count", 0)
            neighbor_count = attributes.get("neighbor_count", len(neighbors))
            comment = attributes.get("made_comment_count", 0)
            repost = attributes.get("made_repost_count", 0)

            raw_features[row] = (
                _log1p_clip(follower),
                _log1p_clip(friend),
                _log1p_clip(received),
                _log1p_clip(made),
                _ratio(mutual, neighbor_count),
                _ratio(engaged_to + engaged_by, neighbor_count),
                _ratio(repost, comment + repost),
                0.0 if attributes.get("isolated", True) else 1.0,
            )

            for neighbor in neighbors:
                neighbor_id = str(neighbor.get("neighbor_id") or "")
                target = row_of.get(neighbor_id)
                if target is None:
                    continue
                weight = float(neighbor.get("total_interaction_count") or 1.0)
                edge_rows.append(row)
                edge_cols.append(target)
                edge_weights.append(max(weight, 1.0))

        # 无向化：一条「我 → 邻居」的互动边，对双方都意味着结构接近。
        if edge_rows:
            rows = np.asarray(edge_rows + edge_cols, dtype=np.int64)
            cols = np.asarray(edge_cols + edge_rows, dtype=np.int64)
            data = np.asarray(edge_weights + edge_weights, dtype=np.float64)
            adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))
        else:
            adjacency = sparse.csr_matrix((node_count, node_count), dtype=np.float64)

        # 行归一化 → 邻居特征加权平均（SGC 的传播算子）。
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_degree = np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0)
        transition = sparse.diags(inv_degree) @ adjacency

        # 逐列归一化后再传播，否则 followers（几千）会完全压住 ratio（0~1）。
        normalized = np.column_stack(
            [_minmax(raw_features[:, column]) for column in range(raw_features.shape[1])]
        )
        hop1 = np.asarray(transition @ normalized)
        hop2 = np.asarray(transition @ hop1)
        features = np.hstack([normalized, hop1, hop2])
        features = np.column_stack(
            [_minmax(features[:, column]) for column in range(features.shape[1])]
        ).astype(np.float32)

        feature_names = (
            list(_FEATURE_NAMES)
            + [f"hop1_{name}" for name in _FEATURE_NAMES]
            + [f"hop2_{name}" for name in _FEATURE_NAMES]
        )

        # ---- 传播潜力：可解释的加权组合 ----
        pagerank = _pagerank(transition, node_count)
        interaction_intensity = raw_features[:, 2] + raw_features[:, 3]
        mutual_ratio = raw_features[:, 4]
        # 邻域质量：自己 2-hop 邻域的平均互动强度，代表「混在什么圈子里」。
        neighbor_quality = np.asarray(transition @ (transition @ interaction_intensity))

        potential = (
            self.weights["pagerank"] * _minmax(pagerank)
            + self.weights["interaction_intensity"] * _minmax(interaction_intensity)
            + self.weights["mutual_ratio"] * _minmax(mutual_ratio)
            + self.weights["neighbor_quality"] * _minmax(neighbor_quality)
        )
        potential = np.clip(potential, 0.0, 1.0).astype(np.float32)

        self._index = GraphIndex(
            user_ids=user_ids,
            features=features,
            potential=potential,
            feature_names=feature_names,
        )
        self._id_to_row = row_of
        logger.info(
            "图特征构建完成: nodes=%d, edges=%d, features=%d, potential_mean=%.4f",
            node_count,
            len(edge_rows),
            features.shape[1],
            float(potential.mean()),
        )
        return self._index

    # ---------------- 查询 ----------------

    def potentials(self, user_ids: Iterable[str] | None = None) -> dict[str, float]:
        """取指定节点的传播潜力；不传则返回全部。"""

        index = self.index
        if user_ids is None:
            return {
                user_id: float(index.potential[row])
                for row, user_id in enumerate(index.user_ids)
            }
        result: dict[str, float] = {}
        for user_id in user_ids:
            row = self._id_to_row.get(user_id)
            if row is not None:
                result[user_id] = float(index.potential[row])
        return result

    def feature_map(self, user_id: str) -> dict[str, float]:
        """单节点结构特征，便于解释「为什么它潜力高」。"""

        index = self.index
        row = self._id_to_row.get(user_id)
        if row is None:
            return {}
        return {
            name: float(index.features[row, column])
            for column, name in enumerate(index.feature_names)
        }

    @property
    def index(self) -> GraphIndex:
        if self._index is None:
            raise RuntimeError("图特征尚未构建或加载")
        return self._index

    # ---------------- 持久化 ----------------

    def save(self, directory: str | Path) -> Path:
        if self._index is None:
            raise RuntimeError("图特征尚未构建，无法保存")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "graph_index.npz",
            user_ids=np.array(self._index.user_ids, dtype=object),
            features=self._index.features,
            potential=self._index.potential,
            feature_names=np.array(self._index.feature_names, dtype=object),
            weights=np.array(
                [self.weights[key] for key in _POTENTIAL_WEIGHTS], dtype=np.float64
            ),
        )
        logger.info("图特征已保存到 %s", target)
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "GraphFeatureBuilder":
        source = Path(directory)
        path = source / "graph_index.npz"
        if not path.exists():
            raise FileNotFoundError(f"图特征文件不存在: {path}")

        builder = cls()
        with np.load(path, allow_pickle=True) as payload:
            builder._index = GraphIndex(
                user_ids=[str(item) for item in payload["user_ids"].tolist()],
                features=payload["features"].astype(np.float32),
                potential=payload["potential"].astype(np.float32),
                feature_names=[str(item) for item in payload["feature_names"].tolist()],
            )
            stored_weights = payload.get("weights")
            if stored_weights is not None and len(stored_weights) == len(_POTENTIAL_WEIGHTS):
                builder.weights = dict(zip(_POTENTIAL_WEIGHTS.keys(), map(float, stored_weights)))
        builder._id_to_row = {
            user_id: row for row, user_id in enumerate(builder._index.user_ids)
        }
        return builder
