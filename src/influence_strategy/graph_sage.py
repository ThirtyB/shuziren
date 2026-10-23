"""GraphSAGE 图神经网络 —— 真正可训练的传播潜力模型。

与 ``graph_features.GraphFeatureBuilder``（无参数 SGC 传播）接口一致，可直接替换。

模型
----
两层 GraphSAGE（Hamilton et al., 2017），mean 聚合：

    h_v^(1) = ReLU( W_self^(1) x_v + W_neigh^(1) · mean_{u∈N(v)} x_u )
    h_v^(2) =      W_self^(2) h_v^(1) + W_neigh^(2) · mean_{u∈N(v)} h_v^(1)

与 SGC 的关键区别：SGC 的聚合权重是**固定的**（行归一化邻接矩阵），
这里 W_self / W_neigh 是**可学习参数**，靠反向传播从数据里学出来。

训练信号（无人工标注）
----------------------
1. **Link Prediction（自监督）**：把图里的真实边当正样本，随机节点对当负样本，
   用 BCE 训练模型学会"什么样的两个节点会互动"。这一步学到的是结构表示。
2. **活跃度回归（自监督）**：用图自身统计量 ``log1p(received + made interaction)``
   当软标签，接一个线性头回归它。这一步把表示压到一维，作为 ``potential``。

整个过程不需要任何"选人效果"的人工标注，但确实有梯度、有参数更新、有 loss 下降。

产物
----
* ``features``：``[结构特征(24) | GraphSAGE 嵌入(64)]``，供下游使用；
* ``potential``：sigmoid 后的 0~1 传播潜力。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .graph_features import (
    _FEATURE_NAMES,
    GraphIndex,
    _log1p_clip,
    _minmax,
    _ratio,
)

logger = logging.getLogger(__name__)

#: 隐藏层宽度。35k 节点、24 维输入，128 足够，再宽 CPU 上收益递减。
DEFAULT_HIDDEN = 128
#: 输出嵌入维度。
DEFAULT_OUT_DIM = 64
#: 训练轮数。实测 150 轮后 link-prediction loss 基本收敛。
DEFAULT_EPOCHS = 150
#: 每轮采样多少条边做对比。全量边数在几十万量级，采 8 万已足够稳定。
DEFAULT_EDGE_SAMPLES = 80_000


@dataclass(slots=True)
class SageIndex(GraphIndex):
    """在 GraphIndex 基础上增加训练信息。"""

    train_history: list[dict[str, float]] = field(default_factory=list)


class GraphSageBuilder:
    """GraphSAGE 训练、推理与持久化。"""

    backend = "sage"

    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        out_dim: int = DEFAULT_OUT_DIM,
        epochs: int = DEFAULT_EPOCHS,
        edge_samples: int = DEFAULT_EDGE_SAMPLES,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.out_dim = out_dim
        self.epochs = epochs
        self.edge_samples = edge_samples
        self.seed = seed
        self._index: SageIndex | None = None
        self._id_to_row: dict[str, int] = {}

    # ---------------- 构图 ----------------

    def _build_graph(self, graph_payload: dict[str, Any]) -> tuple[list[str], np.ndarray, Any, Any]:
        """返回 (节点顺序, 结构特征矩阵, 稀疏邻接, 原始特征矩阵)。"""

        import torch

        user_ids = [str(key) for key in graph_payload.keys()]
        node_count = len(user_ids)
        if node_count == 0:
            raise ValueError("graph_payload 为空，无法构建图")

        row_of = {user_id: row for row, user_id in enumerate(user_ids)}
        raw = np.zeros((node_count, len(_FEATURE_NAMES)), dtype=np.float64)

        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []

        for user_id in user_ids:
            row = row_of[user_id]
            payload = graph_payload[user_id] or {}
            attributes = payload.get("graph_attributes") or {}
            neighbors = payload.get("neighbors") or []

            neighbor_count = attributes.get("neighbor_count", len(neighbors))
            mutual = attributes.get("mutual_neighbor_count", 0)
            engaged_to = attributes.get("engaged_to_neighbor_count", 0)
            engaged_by = attributes.get("engaged_by_neighbor_count", 0)
            comment = attributes.get("made_comment_count", 0)
            repost = attributes.get("made_repost_count", 0)

            raw[row] = (
                _log1p_clip(payload.get("user_followers", 0)),
                _log1p_clip(payload.get("user_friends", 0)),
                _log1p_clip(attributes.get("received_interaction_count", 0)),
                _log1p_clip(attributes.get("made_interaction_count", 0)),
                _ratio(mutual, neighbor_count),
                _ratio(engaged_to + engaged_by, neighbor_count),
                _ratio(repost, comment + repost),
                0.0 if attributes.get("isolated", True) else 1.0,
            )

            for neighbor in neighbors:
                target = row_of.get(str(neighbor.get("neighbor_id") or ""))
                if target is None:
                    continue
                rows.append(row)
                cols.append(target)
                weights.append(max(float(neighbor.get("total_interaction_count") or 1.0), 1.0))

        # 无向化
        all_rows = np.asarray(rows + cols, dtype=np.int64) if rows else np.zeros(0, dtype=np.int64)
        all_cols = np.asarray(cols + rows, dtype=np.int64) if rows else np.zeros(0, dtype=np.int64)
        all_weights = np.asarray(weights + weights, dtype=np.float64) if rows else np.zeros(0)

        # 行归一化：邻居特征取加权平均（GraphSAGE 的 mean aggregator）。
        degree = np.zeros(node_count, dtype=np.float64)
        np.add.at(degree, all_rows, all_weights)
        inv_degree = np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0)
        norm_weights = all_weights * inv_degree[all_rows] if all_weights.size else all_weights

        adjacency = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack([all_rows, all_cols])),
            torch.from_numpy(norm_weights.astype(np.float32)),
            (node_count, node_count),
        ).coalesce()

        features = np.column_stack(
            [_minmax(raw[:, column]) for column in range(raw.shape[1])]
        ).astype(np.float32)

        self._edges = (all_rows, all_cols)
        return user_ids, features, adjacency, raw

    # ---------------- 训练 ----------------

    def build(
        self,
        graph_payload: dict[str, Any],
        *,
        log_every: int = 30,
    ) -> SageIndex:
        import torch
        import torch.nn.functional as F

        # 防御性恢复梯度：同一进程里若先跑过别的推理代码（可能设置过全局
        # no_grad），这里必须显式打开，否则 loss 没有 grad_fn 无法反向传播。
        torch.set_grad_enabled(True)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        user_ids, structural, adjacency, raw = self._build_graph(graph_payload)
        node_count = len(user_ids)
        logger.info("图就绪: nodes=%d, 边=%d", node_count, len(self._edges[0]))

        x = torch.from_numpy(structural)
        # 活跃度软标签：图自身的互动量统计，不需要人工标注。
        target = torch.from_numpy(
            _minmax(raw[:, 2] + raw[:, 3]).astype(np.float32)
        ).reshape(-1, 1)

        class _Sage(torch.nn.Module):
            def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
                super().__init__()
                self.self1 = torch.nn.Linear(in_dim, hidden)
                self.neigh1 = torch.nn.Linear(in_dim, hidden)
                self.self2 = torch.nn.Linear(hidden, out_dim)
                self.neigh2 = torch.nn.Linear(hidden, out_dim)
                self.head = torch.nn.Linear(out_dim, 1)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, features, adj):
                # 第 1 层：自身 + 邻居均值，ReLU
                agg1 = torch.sparse.mm(adj, features)
                h1 = F.relu(self.self1(features) + self.neigh1(agg1))
                h1 = self.dropout(h1)
                # 第 2 层：线性输出（不激活，保留完整表达）
                agg2 = torch.sparse.mm(adj, h1)
                h2 = self.self2(h1) + self.neigh2(agg2)
                return h2, torch.sigmoid(self.head(h2))

        model = _Sage(structural.shape[1], self.hidden, self.out_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        edge_rows, edge_cols = self._edges
        if len(edge_rows) == 0:
            raise ValueError("图中没有边，GraphSAGE 无法训练")

        history: list[dict[str, float]] = []
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            embedding, potential = model(x, adjacency)

            # ---- 任务 1: link prediction ----
            sample_size = min(self.edge_samples, len(edge_rows))
            pick = np.random.randint(0, len(edge_rows), size=sample_size)
            pos_u = torch.from_numpy(edge_rows[pick])
            pos_v = torch.from_numpy(edge_cols[pick])
            neg_u = torch.randint(0, node_count, (sample_size,))
            neg_v = torch.randint(0, node_count, (sample_size,))

            pos_logits = (embedding[pos_u] * embedding[pos_v]).sum(dim=1)
            neg_logits = (embedding[neg_u] * embedding[neg_v]).sum(dim=1)
            link_loss = (
                F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            )

            # ---- 任务 2: 活跃度回归（自监督软标签）----
            reg_loss = F.mse_loss(potential, target)

            loss = link_loss + 0.5 * reg_loss
            loss.backward()
            optimizer.step()

            if epoch % max(self.epochs // 10, 1) == 0 or epoch == self.epochs - 1:
                record = {
                    "epoch": float(epoch),
                    "loss": float(loss.item()),
                    "link_loss": float(link_loss.item()),
                    "reg_loss": float(reg_loss.item()),
                }
                history.append(record)
                logger.info(
                    "epoch %3d  loss=%.4f  link=%.4f  reg=%.4f",
                    epoch,
                    record["loss"],
                    record["link_loss"],
                    record["reg_loss"],
                )
                if log_every and epoch % log_every == 0:
                    print(
                        f"      sage epoch {epoch:3d}  loss={record['loss']:.4f}  "
                        f"link={record['link_loss']:.4f}  reg={record['reg_loss']:.4f}",
                        flush=True,
                    )

        # ---- 推理：拿最终嵌入 ----
        model.eval()
        with torch.no_grad():
            embedding, potential = model(x, adjacency)
        embedding_np = embedding.numpy().astype(np.float32)
        potential_np = potential.numpy().reshape(-1).astype(np.float32)

        combined = np.hstack([structural, embedding_np]).astype(np.float32)
        feature_names = list(_FEATURE_NAMES) + [f"sage_{index}" for index in range(embedding_np.shape[1])]

        self._index = SageIndex(
            user_ids=user_ids,
            features=combined,
            potential=potential_np,
            feature_names=feature_names,
            train_history=history,
        )
        self._id_to_row = {user_id: row for row, user_id in enumerate(user_ids)}
        logger.info(
            "GraphSAGE 训练完成: nodes=%d, embed_dim=%d, final_loss=%.4f",
            node_count,
            embedding_np.shape[1],
            history[-1]["loss"] if history else float("nan"),
        )
        return self._index

    # ---------------- 查询 ----------------

    def potentials(self, user_ids: Iterable[str] | None = None) -> dict[str, float]:
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
        index = self.index
        row = self._id_to_row.get(user_id)
        if row is None:
            return {}
        return {
            name: float(index.features[row, column])
            for column, name in enumerate(index.feature_names)
        }

    @property
    def index(self) -> SageIndex:
        if self._index is None:
            raise RuntimeError("GraphSAGE 尚未训练或加载")
        return self._index

    # ---------------- 持久化 ----------------

    def save(self, directory: str | Path) -> Path:
        if self._index is None:
            raise RuntimeError("GraphSAGE 尚未训练，无法保存")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "graph_index.npz",
            user_ids=np.array(self._index.user_ids, dtype=object),
            features=self._index.features,
            potential=self._index.potential,
            feature_names=np.array(self._index.feature_names, dtype=object),
            backend=np.array(["sage"], dtype=object),
            train_history=np.array(
                [
                    [item["epoch"], item["loss"], item["link_loss"], item["reg_loss"]]
                    for item in self._index.train_history
                ],
                dtype=np.float64,
            ),
        )
        logger.info("GraphSAGE 产物已保存到 %s", target)
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "GraphSageBuilder":
        source = Path(directory)
        path = source / "graph_index.npz"
        if not path.exists():
            raise FileNotFoundError(f"图特征文件不存在: {path}")

        builder = cls()
        with np.load(path, allow_pickle=True) as payload:
            history: list[dict[str, float]] = []
            if "train_history" in payload and payload["train_history"].size:
                for item in payload["train_history"]:
                    history.append(
                        {
                            "epoch": float(item[0]),
                            "loss": float(item[1]),
                            "link_loss": float(item[2]),
                            "reg_loss": float(item[3]),
                        }
                    )
            builder._index = SageIndex(
                user_ids=[str(item) for item in payload["user_ids"].tolist()],
                features=payload["features"].astype(np.float32),
                potential=payload["potential"].astype(np.float32),
                feature_names=[str(item) for item in payload["feature_names"].tolist()],
                train_history=history,
            )
        builder._id_to_row = {
            user_id: row for row, user_id in enumerate(builder._index.user_ids)
        }
        return builder
