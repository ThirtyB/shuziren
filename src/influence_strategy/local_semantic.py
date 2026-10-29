"""本地语义编码：用 TF-IDF + SVD 建立可离线预计算的话题/节点语义空间。

为什么不用大模型
----------------
话题解析和选人打分原本依赖 LLM 的语义判断（``event_semantic_relevance_score``、
``llm_feature_score``）。改为全本地后，需要一个不联网、可预计算、可复现的语义
表示，且必须在秒级内对 3.5 万个节点完成打分。

方案
----
1. ``jieba`` 分词：中文兴趣标签（「亲子共读」「英语启蒙」）和简介都是短文本，
   按词切分比按字切分更能保留语义单元；
2. ``TfidfVectorizer``：词频-逆文档频率加权，压低「生活」「分享」这类到处出现的
   噪声词，提高「启蒙」「绘本」这类有区分度词的权重；
3. ``TruncatedSVD``：把稀疏 TF-IDF 向量压到 128 维稠密空间（即 LSA），使词面
   不同但共现模式相近的文本互相接近 —— 这正是原来交给 LLM 判断的那部分能力；
4. L2 归一化：此后任意两向量点积即余弦相似度，对全部节点打分退化为一次矩阵
   乘法。

预计算与运行期
--------------
节点侧向量与 sklearn 模型离线算好落盘（见 ``scripts/build_embeddings.py``）；
运行期只需编码 1 条话题，再做一次 ``embeddings @ query``，耗时在毫秒级。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DIM = 128
DEFAULT_MAX_FEATURES = 30_000
#: 兴趣标签在节点文本里重复的次数。兴趣比简介更能代表「这个账号聊什么」，
#: 重复即加权，避免长简介淹没短标签。
INTEREST_REPEAT = 3

_PUNCT = set("，。！？、；：“”‘’（）《》…—～·,.!?;:\"'()[]{}<>/\\|@#$%^&*+=~` ")


def _tokenize(text: str) -> list[str]:
    """jieba 分词 + 噪声过滤。

    只保留长度 >= 2 的词：单字（「的」「了」「我」）几乎没有区分度，还会让
    TF-IDF 的维度爆炸。保留英文/数字词，便于处理「english」这类混合标签。
    """

    import jieba

    tokens: list[str] = []
    for raw in jieba.lcut(str(text or "")):
        token = raw.strip().lower()
        if len(token) < 2:
            continue
        if all(ch in _PUNCT for ch in token):
            continue
        tokens.append(token)
    return tokens


def build_node_text(profile: Any) -> str:
    """把一个节点的档案拼成用于编码的文本。

    兴趣重复 ``INTEREST_REPEAT`` 次做加权；简介只出现一次。这样「亲子共读」
    这类短标签不会被一段长简介稀释掉。
    """

    interests = list(getattr(profile, "user_interests", None) or [])
    description = str(getattr(profile, "user_description", "") or "")
    interest_text = " ".join(interests) * INTEREST_REPEAT
    return f"{interest_text} {description}".strip()


def build_event_text(event: Any) -> str:
    """把一条热点事件拼成查询文本。

    兼容 ``ParsedEvent`` 与原始 hot_event dict：标题、摘要、观点变体、关键词、
    语义标签都拼进去，让查询侧的信息量尽量接近节点侧。
    """

    def _pick(name: str, *aliases: str) -> str:
        value = None
        if isinstance(event, dict):
            for key in (name, *aliases):
                if event.get(key):
                    value = event[key]
                    break
        else:
            for key in (name, *aliases):
                candidate = getattr(event, key, None)
                if candidate:
                    value = candidate
                    break
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        return str(value or "")

    parts = [
        _pick("event_title", "title"),
        # ``ParsedEvent`` 用 ``event_description``，原始 hot_event 用 ``event_summary``。
        _pick("event_summary", "event_description", "summary"),
        _pick("opinion_variants"),
        _pick("extracted_keywords", "keywords"),
        _pick("semantic_tags", "tags"),
        _pick("target", "target_goal"),
    ]
    # 同一段文本重复两次：事件文本整体偏短，重复可提高它在查询向量里的份量。
    text = " ".join(part for part in parts if part.strip())
    return f"{text} {text}".strip()


@dataclass(slots=True)
class SemanticIndex:
    """离线预计算产物：节点 id 顺序 + 归一化后的稠密向量矩阵。"""

    user_ids: list[str]
    embeddings: np.ndarray  # (n_nodes, dim), float32, L2 归一化
    dim: int

    def __post_init__(self) -> None:
        if len(self.user_ids) != int(self.embeddings.shape[0]):
            raise ValueError(
                "SemanticIndex user_ids 与 embeddings 行数不一致: "
                f"{len(self.user_ids)} != {self.embeddings.shape[0]}"
            )

    @property
    def node_count(self) -> int:
        return len(self.user_ids)


class LocalSemanticEncoder:
    """TF-IDF + SVD 语义编码器。

    典型用法::

        encoder = LocalSemanticEncoder()
        index = encoder.fit(profiles)          # 离线，一次
        encoder.save(path)                     # 落盘
        ...
        encoder = LocalSemanticEncoder.load(path)   # 运行期
        scores = encoder.scores(build_event_text(event))   # 毫秒级
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        max_features: int = DEFAULT_MAX_FEATURES,
        min_df: int = 2,
    ) -> None:
        self.dim = dim
        self.max_features = max_features
        self.min_df = min_df
        self._vectorizer: Any = None
        self._svd: Any = None
        self._index: SemanticIndex | None = None
        self._id_to_row: dict[str, int] = {}

    # ---------------- 训练 / 预计算 ----------------

    def fit(self, profiles: dict[str, Any]) -> SemanticIndex:
        """在全部节点档案上训练 TF-IDF + SVD，返回节点语义索引。"""

        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        user_ids = list(profiles.keys())
        if not user_ids:
            raise ValueError("profiles 为空，无法训练语义空间")

        documents = [build_node_text(profiles[user_id]) for user_id in user_ids]
        # 全部文档都为空时（极端数据集），退化为全零向量，避免 sklearn 抛错。
        if not any(doc.strip() for doc in documents):
            logger.warning("所有节点文本均为空，语义空间退化为零向量")
            embeddings = np.zeros((len(user_ids), self.dim), dtype=np.float32)
            self._index = SemanticIndex(user_ids=user_ids, embeddings=embeddings, dim=self.dim)
            self._id_to_row = {uid: i for i, uid in enumerate(user_ids)}
            return self._index

        vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,
            lowercase=False,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(documents)

        # SVD 维度不能超过特征数（小数据集如测试 fixture 只有几十个词）。
        n_components = int(min(self.dim, max(2, tfidf.shape[1] - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(tfidf)
        embeddings = normalize(dense).astype(np.float32)

        self._vectorizer = vectorizer
        self._svd = svd
        self._index = SemanticIndex(
            user_ids=user_ids,
            embeddings=embeddings,
            dim=int(embeddings.shape[1]),
        )
        self._id_to_row = {uid: i for i, uid in enumerate(user_ids)}
        logger.info(
            "语义空间训练完成: nodes=%d, tfidf_features=%d, svd_dim=%d, explained=%.3f",
            len(user_ids),
            tfidf.shape[1],
            self._index.dim,
            float(svd.explained_variance_ratio_.sum()),
        )
        return self._index

    # ---------------- 编码 / 打分 ----------------

    def encode(self, text: str) -> np.ndarray:
        """把一段文本编码为归一化稠密向量。"""

        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("编码器尚未训练或加载，无法 encode")

        from sklearn.preprocessing import normalize

        sparse = self._vectorizer.transform([text])
        dense = self._svd.transform(sparse)
        # 全零向量（查询词全不在词表里）归一化会产生 NaN，这里显式兜底。
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        if float(norm[0, 0]) < 1e-12:
            return np.zeros((1, self._index.dim if self._index else self.dim), dtype=np.float32)[0]
        return normalize(dense).astype(np.float32)[0]

    def scores(self, text: str, user_ids: Sequence[str] | None = None) -> dict[str, float]:
        """给指定节点打语义相似度（余弦，范围 0~1）。

        不传 ``user_ids`` 时对全部节点打分。返回 ``{user_id: score}``。
        """

        if self._index is None:
            raise RuntimeError("语义索引尚未就绪，请先 fit 或 load")

        query = self.encode(text)
        if user_ids is None:
            values = self._index.embeddings @ query
            return {
                user_id: float(values[row])
                for row, user_id in enumerate(self._index.user_ids)
            }

        rows = [self._id_to_row[uid] for uid in user_ids if uid in self._id_to_row]
        if not rows:
            return {}
        values = self._index.embeddings[rows] @ query
        return {
            user_id: float(value)
            for user_id, value in zip(
                [uid for uid in user_ids if uid in self._id_to_row], values
            )
        }

    def scores_dense(self, text: str) -> np.ndarray:
        """对全部节点打分，返回与 ``index.user_ids`` 对齐的 ndarray（避免构造 dict）。"""

        if self._index is None:
            raise RuntimeError("语义索引尚未就绪，请先 fit 或 load")
        return self._index.embeddings @ self.encode(text)

    @property
    def index(self) -> SemanticIndex:
        if self._index is None:
            raise RuntimeError("语义索引尚未就绪")
        return self._index

    # ---------------- 持久化 ----------------

    def save(self, directory: str | Path) -> Path:
        """把 SVD 向量与 sklearn 模型写入目录，返回目录路径。

        * ``semantic_index.npz``：``user_ids`` + ``embeddings``（numpy，加载快）；
        * ``semantic_model.joblib``：``TfidfVectorizer`` + ``TruncatedSVD``+
          id→row 映射（运行期编码新查询需要）。
        """

        import joblib

        if self._index is None or self._vectorizer is None or self._svd is None:
            raise RuntimeError("编码器尚未训练，无法保存")

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "semantic_index.npz",
            user_ids=np.array(self._index.user_ids, dtype=object),
            embeddings=self._index.embeddings,
            dim=np.array([self._index.dim], dtype=np.int64),
        )
        joblib.dump(
            {
                "vectorizer": self._vectorizer,
                "svd": self._svd,
                "dim": self._index.dim,
                "max_features": self.max_features,
                "min_df": self.min_df,
            },
            target / "semantic_model.joblib",
            compress=3,
        )
        logger.info("语义模型已保存到 %s", target)
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "LocalSemanticEncoder":
        """从 ``save`` 产物恢复编码器（含节点向量索引）。"""

        import joblib

        source = Path(directory)
        index_path = source / "semantic_index.npz"
        model_path = source / "semantic_model.joblib"
        if not index_path.exists():
            raise FileNotFoundError(f"语义索引不存在: {index_path}")

        encoder = cls()
        with np.load(index_path, allow_pickle=True) as payload:
            encoder._index = SemanticIndex(
                user_ids=[str(item) for item in payload["user_ids"].tolist()],
                embeddings=payload["embeddings"].astype(np.float32),
                dim=int(payload["dim"][0]),
            )
        encoder._id_to_row = {
            user_id: row for row, user_id in enumerate(encoder._index.user_ids)
        }

        if model_path.exists():
            bundle = joblib.load(model_path)
            encoder._vectorizer = bundle["vectorizer"]
            encoder._svd = bundle["svd"]
            encoder.dim = int(bundle.get("dim", encoder._index.dim))
        else:
            logger.warning(
                "未找到 semantic_model.joblib，只能读取节点向量，无法编码新查询: %s",
                source,
            )
        return encoder


def build_json_report(encoder: LocalSemanticEncoder) -> str:
    """生成一份简短的可读报告，便于人工确认预计算产物是否合理。"""

    index = encoder.index
    norms = np.linalg.norm(index.embeddings, axis=1)
    return json.dumps(
        {
            "node_count": index.node_count,
            "dim": index.dim,
            "norm_min": round(float(norms.min()), 6),
            "norm_max": round(float(norms.max()), 6),
        },
        ensure_ascii=False,
        indent=2,
    )


def iter_text_previews(
    profiles: dict[str, Any], limit: int = 5
) -> Iterable[tuple[str, str]]:
    """调试辅助：抽样查看参与编码的节点文本。"""

    for index, (user_id, profile) in enumerate(profiles.items()):
        if index >= limit:
            break
        yield user_id, build_node_text(profile)


class SemanticClassifier:
    """用预训练语义编码器做「纯语义相似度」分类。

    不再依赖关键词字典的「命中数」判定类别，而是把每个候选类别用一段锚文本
    （anchor）表示，编码成向量并缓存；解析时把待分类文本也编码，与各类锚向量
    逐个算余弦相似度，取最高者。语义空间由同一个本地编码器提供（TF-IDF+SVD 或
    BGE），与选人/打分用的是同一套表示，因此分类与下游匹配口径一致。

    编码器不可用时 :meth:`build` 返回 ``None``，调用方据此退化为默认类别。
    """

    def __init__(self, encoder: Any, anchors: dict[str, str]) -> None:
        self.labels: list[str] = list(anchors.keys())
        vectors = [
            np.asarray(encoder.encode(text), dtype=np.float32)
            for text in anchors.values()
        ]
        if vectors:
            self._matrix = np.stack(vectors, axis=0)  # (n, dim)，已 L2 归一化
        else:
            self._matrix = np.zeros((0, 1), dtype=np.float32)
        self._encoder = encoder

    @classmethod
    def build(cls, encoder: Any, anchors: dict[str, str]) -> "SemanticClassifier | None":
        """构建分类器；``encoder`` 为 ``None`` 或编码失败时返回 ``None``。"""

        if encoder is None:
            return None
        try:
            return cls(encoder, anchors)
        except Exception:  # noqa: BLE001 - 编码失败不应中断事件解析
            return None

    def classify(
        self,
        text: str,
        default_label: str | None = None,
        margin: float = 0.0,
    ) -> tuple[str, float]:
        """返回类别标签与相似度。

        ``default_label`` 为兜底类别（如 ``general``）。给定时，只有当某个**非兜底**
        类别的相似度明显高于兜底类别（超出 ``margin``）才选它，否则回落到兜底类别——
        这样在语义上与「无明确命中则归为通用」等价，且对编码器后端（BGE/SVD）都稳健，
        不依赖绝对相似度的阈值。
        """

        if self._matrix.shape[0] == 0:
            raise ValueError("分类器未包含任何锚类别")
        query = np.asarray(self._encoder.encode(text), dtype=np.float32)
        sims = self._matrix @ query  # 余弦（点积，因均已 L2 归一化）
        if default_label is None or default_label not in self.labels:
            idx = int(np.argmax(sims))
            return self.labels[idx], float(sims[idx])

        default_idx = self.labels.index(default_label)
        default_sim = float(sims[default_idx])
        best_idx = -1
        best_sim = default_sim + margin
        for i, label in enumerate(self.labels):
            if label == default_label:
                continue
            if sims[i] > best_sim:
                best_sim = float(sims[i])
                best_idx = i
        if best_idx < 0:
            return default_label, default_sim
        return self.labels[best_idx], best_sim

    def rank(self, text: str, k: int | None = None) -> list[tuple[str, float]]:
        """返回按相似度降序排列的 ``(label, score)`` 列表，可选取前 ``k`` 个。"""

        if self._matrix.shape[0] == 0:
            raise ValueError("分类器未包含任何锚类别")
        query = np.asarray(self._encoder.encode(text), dtype=np.float32)
        sims = self._matrix @ query
        order = np.argsort(-sims)
        if k is not None:
            order = order[:k]
        return [(self.labels[i], float(sims[i])) for i in order]
