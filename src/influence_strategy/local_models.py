"""本地模型加载入口：语义编码器 + 图结构特征。

负责三件事：

1. **定位产物** —— 默认 ``<workspace>/data/derived/local_models``；
2. **进程内缓存** —— 同一个 workspace 只加载一次，避免每个事件都读盘
   （语义索引 + 图特征合计约 100MB，重复加载会拖慢每秒级请求）；
3. **预热** —— 主动触发一次 jieba 分词与编码，把约 1s 的词典加载成本从
   "第一个真实事件"移到进程启动阶段。

产物不存在时返回 ``None``，调用方据此退化为纯规则打分 —— 这样没跑过预计算的
环境（例如只跑单元测试的最小数据集）仍然能工作，不会因为缺模型直接报错。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

#: 预计算产物的默认相对路径。
MODEL_SUBDIR = Path("data") / "derived" / "local_models"

_CACHE: dict[str, "LocalModels | None"] = {}
_CACHE_LOCK = threading.Lock()


@dataclass(slots=True)
class LocalModels:
    """已加载的本地模型集合。"""

    encoder: Any  # LocalSemanticEncoder
    graph: Any  # GraphFeatureBuilder
    source_dir: Path

    # ---------------- 语义 ----------------

    def semantic_scores(self, event_text: str) -> dict[str, float]:
        """对全部节点打语义相关度，返回 ``{user_id: score}``。"""

        if not event_text.strip():
            return {}
        return self.encoder.scores(event_text)

    # ---------------- 图结构 ----------------

    def graph_potentials(self) -> dict[str, float]:
        """全部节点的传播潜力，返回 ``{user_id: potential}``。"""

        return self.graph.potentials()

    def warmup(self) -> None:
        """预热分词器与向量化路径。

        jieba 首次 ``lcut`` 需要加载词典（约 1s），sklearn 首次 transform 也有
        一次性初始化。这里用一条无意义短句跑通完整链路，让首个真实事件只承担
        常规耗时。
        """

        try:
            self.encoder.encode("预热 语义 编码")
        except Exception as exc:  # 预热失败不应影响主流程
            logger.warning("语义模型预热失败: %s", exc)


def _model_dir(workspace_root: str | Path) -> Path:
    return Path(workspace_root).resolve() / MODEL_SUBDIR


def _detect_backend(npz_path: Path, default: str) -> str:
    """读取产物里的 ``backend`` 标记，决定用哪个实现加载。

    老产物没有这个字段，回退到 ``default``（即 SVD / SGC 的纯统计实现），
    保证升级到深度模型后，旧的工作目录仍然能被正确识别。
    """

    try:
        with np.load(npz_path, allow_pickle=True) as payload:
            if "backend" in payload:
                return str(payload["backend"][0])
    except Exception as exc:
        logger.debug("读取 backend 标记失败，使用默认值 %s: %s", default, exc)
    return default


def _load_semantic_encoder(directory: Path):
    """按产物标记加载 BGE（深度）或 SVD（统计）语义编码器。"""

    backend = _detect_backend(directory / "semantic_index.npz", "svd")
    if backend == "bge":
        from .bge_semantic import BGESemanticEncoder

        return BGESemanticEncoder.load(directory)

    from .local_semantic import LocalSemanticEncoder

    return LocalSemanticEncoder.load(directory)


def _load_graph_builder(directory: Path):
    """按产物标记加载 GraphSAGE（可训练 GNN）或 SGC（无参数传播）。"""

    backend = _detect_backend(directory / "graph_index.npz", "sgc")
    if backend == "sage":
        from .graph_sage import GraphSageBuilder

        return GraphSageBuilder.load(directory)

    from .graph_features import GraphFeatureBuilder

    return GraphFeatureBuilder.load(directory)


def artifacts_exist(directory: str | Path) -> bool:
    """判断预计算产物是否齐全（语义与图特征都要有）。"""

    base = Path(directory)
    return (base / "semantic_index.npz").exists() and (base / "graph_index.npz").exists()


def load_local_models(
    workspace_root: str | Path,
    *,
    warmup: bool = True,
) -> LocalModels | None:
    """加载（并缓存）本地模型；产物缺失时返回 ``None``。

    缓存键用解析后的绝对路径，避免相对路径 / 大小写差异导致重复加载。
    """

    key = str(Path(workspace_root).resolve())
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]

    directory = _model_dir(key)
    if not artifacts_exist(directory):
        logger.info("未找到本地模型产物，退化为纯规则打分: %s", directory)
        with _CACHE_LOCK:
            _CACHE[key] = None
        return None

    try:
        encoder = _load_semantic_encoder(directory)
        graph = _load_graph_builder(directory)
    except Exception as exc:
        logger.warning("本地模型加载失败，退化为纯规则打分: %s", exc)
        with _CACHE_LOCK:
            _CACHE[key] = None
        return None

    models = LocalModels(encoder=encoder, graph=graph, source_dir=directory)
    if warmup:
        models.warmup()

    with _CACHE_LOCK:
        _CACHE[key] = models
    logger.info(
        "本地模型已加载: semantic=%d nodes / graph=%d nodes (%s)",
        encoder.index.node_count,
        graph.index.node_count,
        directory,
    )
    return models


def clear_cache() -> None:
    """清空进程内缓存（测试用）。"""

    with _CACHE_LOCK:
        _CACHE.clear()


def describe(models: LocalModels | None) -> dict[str, Any]:
    """返回一份可序列化的模型摘要，供接口/日志展示。"""

    if models is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "source_dir": str(models.source_dir),
        # backend 标明当前跑的是深度模型还是统计回退，便于日志与接口排查。
        "semantic_backend": str(getattr(models.encoder, "backend", "svd")),
        "graph_backend": str(getattr(models.graph, "backend", "sgc")),
        "semantic_nodes": models.encoder.index.node_count,
        "semantic_dim": models.encoder.index.dim,
        "graph_nodes": models.graph.index.node_count,
        "graph_features": int(models.graph.index.features.shape[1]),
    }


def iter_missing(models: LocalModels, user_ids: Iterable[str]) -> list[str]:
    """调试辅助：列出缺少本地模型向量的节点（数据与模型不同步时排查用）。"""

    known = set(models.encoder.index.user_ids)
    return [uid for uid in user_ids if uid not in known]
