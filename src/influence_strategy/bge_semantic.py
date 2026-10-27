"""BGE 句向量语义编码器 —— 真正的深度模型（Transformer, 12 层）。

与 ``local_semantic.LocalSemanticEncoder``（TF-IDF + SVD 的纯统计方案）接口完全
一致，可在不改动 ``feature_builder`` / ``scorer`` 的前提下直接替换：

    fit(profiles) -> SemanticIndex      离线批量编码全部节点
    encode(text) -> np.ndarray          在线编码 1 条话题
    scores_dense(text) -> np.ndarray    一次矩阵乘法给全部节点打分
    save(dir) / load(dir)               产物落盘与恢复

模型
----
``BAAI/bge-small-zh-v1.5``：12 层 Transformer，hidden=512，约 24M 参数。
取 ``[CLS]`` 位置输出并 L2 归一化（BGE 官方推荐的用法），余弦相似度即点积。

为什么比 SVD 强
---------------
SVD（LSA）只能捕捉**词共现**层面的相似，遇到"亲子共读"与"陪孩子读绘本"这种
零词重叠的表述会失配；BGE 在预训练阶段学到了中文语义空间，能识别两者同义。
这正是原来交给 LLM 判断、而词袋模型做不到的部分。

推理成本
--------
CPU 上单条约 5~10ms，但要对 35,636 个节点编码 —— 一次性约 4~6 分钟（预计算）。
运行期只编码 1 条话题（约 30ms）+ 一次矩阵点积（约 10ms）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .local_semantic import SemanticIndex, build_node_text

logger = logging.getLogger(__name__)

#: BGE 官方建议：检索场景下给 query 加指令前缀，passage 不加。
#: 加前缀能让"话题"与"节点档案"落在更对齐的向量区域。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

#: 节点编码的批大小。CPU 上 64 是比较稳的折中（再大内存涨得快，再小循环开销高）。
DEFAULT_BATCH_SIZE = 64

#: 截断长度。节点文本是「兴趣×3 + 简介」，96 token 足够覆盖，超长部分收益很低。
DEFAULT_MAX_LENGTH = 96

#: ONNX 产物目录后缀（``<model>-onnx/``），由 ``scripts/export_bge_onnx.py`` 生成。
#: 存在时运行期走 onnxruntime + tokenizers，不再 import torch / transformers ——
#: 首次调用从约 9s 降到约 2s。不存在时自动回退 torch 路径，行为不变。
ONNX_SUFFIX = "-onnx"


class BGESemanticEncoder:
    """基于 BGE-small-zh 的语义编码器。"""

    backend = "bge"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        use_query_instruction: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_query_instruction = use_query_instruction
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        # ONNX 推理通道：onnxruntime Session + tokenizers Tokenizer。
        # 存在 ONNX 产物时走这条，完全不 import torch / transformers。
        self._session: Any = None
        self._onnx_tokenizer: Any = None
        self._index: SemanticIndex | None = None
        self._id_to_row: dict[str, int] = {}

    # ---------------- 模型装载 ----------------

    def _onnx_dir(self) -> Path | None:
        """定位 ONNX 产物目录（与 torch 模型目录同级，带 ``-onnx`` 后缀）。

        不存在则返回 ``None``，此时回退到 torch + transformers。这样"还没跑过
        导出脚本"的环境依然能正常工作，不会因为缺 ONNX 而报错。
        """

        if self.model_dir is None:
            return None
        base = Path(self.model_dir)
        candidate = base.with_name(base.name + ONNX_SUFFIX)
        if (candidate / "model.onnx").exists() and (candidate / "tokenizer.json").exists():
            return candidate
        return None

    def _ensure_onnx(self, onnx_dir: Path) -> None:
        """加载 ONNX 图 + 原生分词器（都不依赖 torch / transformers）。"""

        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(onnx_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_length)
        # 必须补齐到**定长**：导出的图在序列维上是静态的（只有 batch 维动态）。
        pad_id = tokenizer.token_to_id("[PAD]")
        tokenizer.enable_padding(
            pad_id=pad_id if pad_id is not None else 0,
            pad_token="[PAD]",
            length=self.max_length,
        )
        session = ort.InferenceSession(
            str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self._onnx_tokenizer = tokenizer
        self._session = session
        logger.info("BGE 模型已加载 (ONNX): %s", onnx_dir)

    def _ensure_model(self) -> None:
        if self._model is not None or self._session is not None:
            return
        if self.model_dir is None or not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"BGE 模型目录不存在: {self.model_dir}。"
                "请先运行 scripts/download_bge.py 或改回 SVD 后端。"
            )

        onnx_dir = self._onnx_dir()
        if onnx_dir is not None:
            self._ensure_onnx(onnx_dir)
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        # 注意：这里**不能**用 torch.set_grad_enabled(False) —— 那是全局开关。
        # 预计算脚本先跑 BGE 再跑 GraphSAGE 时，全局关梯度会让后者的 loss 失去
        # grad_fn，直接报 "element 0 of tensors does not require grad"。
        # 梯度关闭只放在 _encode_batch 的 with 作用域里。
        # CPU 推理：线程数与物理核数对齐，过多线程反而因调度变慢。
        torch.set_num_threads(min(8, torch.get_num_threads()))

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModel.from_pretrained(str(self.model_dir))
        self._model.eval()
        logger.info(
            "BGE 模型已加载: %s (params=%.1fM)",
            self.model_dir,
            sum(p.numel() for p in self._model.parameters()) / 1e6,
        )

    # ---------------- 编码 ----------------

    @staticmethod
    def _prepare(texts: Sequence[str], *, is_query: bool, use_instruction: bool) -> list[str]:
        """给 query 加指令前缀（BGE 官方推荐的检索用法）。"""

        prepared = list(texts)
        if is_query and use_instruction:
            prepared = [QUERY_INSTRUCTION + text for text in prepared]
        return prepared

    def _encode_batch(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        """批量编码，返回 L2 归一化后的 float32 矩阵。"""

        self._ensure_model()
        if self._session is not None:
            return self._encode_onnx(texts, is_query=is_query)
        return self._encode_torch(texts, is_query=is_query)

    def _encode_onnx(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        """ONNX 推理路径：onnxruntime + tokenizers，不触碰 torch。"""

        prepared = self._prepare(
            texts, is_query=is_query, use_instruction=self.use_query_instruction
        )
        encoded = self._onnx_tokenizer.encode_batch(prepared)
        feeds = {
            "input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.array([item.attention_mask for item in encoded], dtype=np.int64),
            "token_type_ids": np.array([item.type_ids for item in encoded], dtype=np.int64),
        }
        hidden = self._session.run(["last_hidden_state"], feeds)[0]
        # BGE 用 [CLS] 作为句向量（不是 mean pooling）。
        embedding = hidden[:, 0].astype(np.float32)
        norms = np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding / np.maximum(norms, 1e-12)

    def _encode_torch(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        """torch / transformers 推理路径（无 ONNX 产物时的回退）。"""

        torch = self._torch
        prepared = self._prepare(
            texts, is_query=is_query, use_instruction=self.use_query_instruction
        )

        batch = self._tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**batch)
            # BGE 用 [CLS] 作为句向量（不是 mean pooling）。
            embedding = output.last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding.cpu().numpy().astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        """编码单条文本（作为 query）。"""

        if self._index is None:
            raise RuntimeError("语义索引尚未就绪")
        return self._encode_batch([text], is_query=True)[0]

    # ---------------- 预计算 ----------------

    def _resolve_dim(self) -> int:
        """取向量维度，兼容 torch 与 ONNX 两种加载路径。

        ``_ensure_model`` 在存在 ONNX 目录时会走 ONNX 分支，此时 ``self._model``
        保持为 None，直接读 ``.config.hidden_size`` 会抛 AttributeError。
        这里按「torch 配置 → 实际编码一次拿输出维度」的顺序兜底。
        """

        if self._model is not None:
            return int(self._model.config.hidden_size)
        return int(self._encode_batch(["无"], is_query=False).shape[1])

    def fit(self, profiles: dict[str, Any], *, log_every: int = 50) -> SemanticIndex:
        """对全部节点档案批量编码，返回语义索引。

        35,636 条在 CPU 上约 4~6 分钟；``log_every`` 控制进度日志的批次间隔。
        """

        user_ids = list(profiles.keys())
        if not user_ids:
            raise ValueError("profiles 为空，无法构建语义索引")

        self._ensure_model()
        dim = self._resolve_dim()
        embeddings = np.zeros((len(user_ids), dim), dtype=np.float32)

        total_batches = (len(user_ids) + self.batch_size - 1) // self.batch_size
        for batch_index, start in enumerate(range(0, len(user_ids), self.batch_size)):
            chunk = user_ids[start : start + self.batch_size]
            texts = [build_node_text(profiles[uid]) for uid in chunk]
            # 空文本编码出来是同一个"空"向量，不影响排序，但会在日志里显得奇怪，
            # 这里替换成一个中性占位，避免 tokenizer 报 padding 相关的告警。
            texts = [text if text.strip() else "无" for text in texts]
            embeddings[start : start + len(chunk)] = self._encode_batch(texts, is_query=False)

            if log_every and (batch_index + 1) % log_every == 0:
                done = min(start + self.batch_size, len(user_ids))
                logger.info("BGE 编码进度: %d/%d (%.1f%%)", done, len(user_ids), 100 * done / len(user_ids))
                print(f"      bge encode {done}/{len(user_ids)} ({100*done/len(user_ids):.1f}%)", flush=True)

        self._index = SemanticIndex(user_ids=user_ids, embeddings=embeddings, dim=dim)
        self._id_to_row = {uid: row for row, uid in enumerate(user_ids)}
        logger.info("BGE 语义索引构建完成: nodes=%d, dim=%d", len(user_ids), dim)
        return self._index

    # ---------------- 打分 ----------------

    def scores(self, text: str, user_ids: Sequence[str] | None = None) -> dict[str, float]:
        if self._index is None:
            raise RuntimeError("语义索引尚未就绪")
        query = self.encode(text)
        if user_ids is None:
            values = self._index.embeddings @ query
            return {
                user_id: float(values[row])
                for row, user_id in enumerate(self._index.user_ids)
            }
        rows = [(uid, self._id_to_row[uid]) for uid in user_ids if uid in self._id_to_row]
        if not rows:
            return {}
        values = self._index.embeddings[[row for _, row in rows]] @ query
        return {uid: float(value) for (uid, _), value in zip(rows, values)}

    def scores_dense(self, text: str) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("语义索引尚未就绪")
        return self._index.embeddings @ self.encode(text)

    @property
    def index(self) -> SemanticIndex:
        if self._index is None:
            raise RuntimeError("语义索引尚未就绪")
        return self._index

    # ---------------- 持久化 ----------------

    def save(self, directory: str | Path) -> Path:
        """保存节点向量。模型本身留在 ``data/models``，这里只记路径。

        只存 73MB 的向量矩阵，不复制 183MB 的模型权重 —— 换后端或重训语义空间
        时不需要重复落盘同一份模型。
        """

        if self._index is None:
            raise RuntimeError("编码器尚未 fit，无法保存")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "semantic_index.npz",
            user_ids=np.array(self._index.user_ids, dtype=object),
            embeddings=self._index.embeddings,
            dim=np.array([self._index.dim], dtype=np.int64),
            backend=np.array(["bge"], dtype=object),
            model_dir=np.array([str(self.model_dir or "")], dtype=object),
            max_length=np.array([self.max_length], dtype=np.int64),
            batch_size=np.array([self.batch_size], dtype=np.int64),
            use_query_instruction=np.array([self.use_query_instruction]),
        )
        logger.info("BGE 语义索引已保存到 %s", target)
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "BGESemanticEncoder":
        source = Path(directory)
        path = source / "semantic_index.npz"
        if not path.exists():
            raise FileNotFoundError(f"语义索引不存在: {path}")

        with np.load(path, allow_pickle=True) as payload:
            model_dir = str(payload["model_dir"][0]) if "model_dir" in payload else ""
            encoder = cls(
                model_dir=model_dir or None,
                batch_size=int(payload["batch_size"][0]) if "batch_size" in payload else DEFAULT_BATCH_SIZE,
                max_length=int(payload["max_length"][0]) if "max_length" in payload else DEFAULT_MAX_LENGTH,
                use_query_instruction=(
                    bool(payload["use_query_instruction"][0])
                    if "use_query_instruction" in payload
                    else True
                ),
            )
            encoder._index = SemanticIndex(
                user_ids=[str(item) for item in payload["user_ids"].tolist()],
                embeddings=payload["embeddings"].astype(np.float32),
                dim=int(payload["dim"][0]),
            )
        encoder._id_to_row = {
            user_id: row for row, user_id in enumerate(encoder._index.user_ids)
        }
        return encoder
