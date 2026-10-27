"""文案弹药库（Content Arsenal）。

把"发帖正文"从**事件内部按序号轮转**，升级成**跨事件可检索的素材库**。

设计要点
--------
1. **数据是资产**：弹药以 JSONL 放在 ``data/derived/content_arsenal/``，
   可以整体替换而不改代码。当前只覆盖 L1（发帖内容文案），
   L2 动作骨架 / L3 策略表述暂不入库，等这层跑通后再扩。

2. **选择是确定性的**：同一 ``(event_type, seed_key)`` 永远产出同一批文案。
   种子用 blake2b，不依赖 Python 内置的 ``hash()``（它跨进程不稳定），
   也不用真随机——保证结果可复现、测试可写。

3. **候选不足时跨类借用**：冷门 event_type 凑不够时从全库补齐，
   宁可相关性略降，也不返回空文案。

4. **语义重排是可选的**：传入 ``scorer`` 才启用（例如接 BGE）。
   不传就退化为"标签检索 + 确定性轮转"，零额外耗时。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_ARSENAL_DIR = Path("data") / "derived" / "content_arsenal"
TOPIC_AMMO_FILENAME = "topic_ammo.jsonl"
#: 弹药文本的预计算向量。离线生成、运行期只读；缺失时回退到实时编码。
AMMO_VECTOR_FILENAME = "ammo_vectors.npz"

def _stable_seed(*parts: str) -> int:
    """基于 blake2b 的稳定种子，保证跨进程 / 跨平台可复现。"""

    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


def _arsenal_fingerprint(texts: Iterable[str]) -> str:
    """弹药库内容指纹。

    预计算向量与弹药文本一一对应，弹药库一旦增删改，存盘向量就整体失效
    （顺序错位会取到别人的文案），所以落盘记指纹、加载时校验。
    """

    return hashlib.blake2b("\n".join(texts).encode("utf-8"), digest_size=16).hexdigest()


class AmmoVectorIndex:
    """弹药文本的预计算向量索引。

    动机：BGE 在 CPU 上编码一条短文本约 20ms，一个 event_type 下动辄几百条，
    实时算要 8 秒左右。弹药库是静态资产，离线算一次存盘后，运行期只需编码
    1 条事件文本再做矩阵乘法（毫秒级）。
    """

    def __init__(self, texts: list[str], embeddings: Any, fingerprint: str) -> None:
        self._texts = list(texts)
        self._embeddings = embeddings
        self.fingerprint = fingerprint

    @property
    def texts(self) -> list[str]:
        return self._texts

    # ---------- 落盘 / 加载 ----------

    def save(self, directory: str | Path) -> Path:
        import numpy as np

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / AMMO_VECTOR_FILENAME
        np.savez_compressed(
            path,
            texts=np.array(self._texts, dtype=object),
            embeddings=self._embeddings,
            fingerprint=np.array([self.fingerprint], dtype=object),
        )
        return path

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_fingerprint: str | None = None,
    ) -> "AmmoVectorIndex | None":
        """加载预计算向量。

        文件缺失、损坏，或指纹与当前弹药库不一致时返回 ``None``，由调用方
        回退到实时编码 —— 宁可慢，也不能取错文案。
        """

        path = Path(directory)
        path = path / AMMO_VECTOR_FILENAME if path.is_dir() else path
        if not path.exists():
            return None
        try:
            import numpy as np

            with np.load(path, allow_pickle=True) as payload:
                texts = [str(item) for item in payload["texts"]]
                embeddings = payload["embeddings"]
                fingerprint = str(payload["fingerprint"][0])
        except Exception:
            return None

        if expected_fingerprint and fingerprint != expected_fingerprint:
            return None
        return cls(texts, embeddings, fingerprint)

    # ---------- 打分 ----------

    def scores(self, query_vector: Any) -> list[float]:
        """与事件向量的余弦相似度（两侧均已 L2 归一化，点积即余弦）。"""

        return (self._embeddings @ query_vector).tolist()


class ContentArsenal:
    """按 event_type 索引的文案弹药库。"""

    def __init__(
        self,
        items: list[dict[str, Any]],
        *,
        source_dir: Path | None = None,
    ) -> None:
        self._items = list(items)
        # 弹药库所在目录：预计算向量与它放在一起，便于一起替换/失效。
        self.source_dir = source_dir
        self._by_event_type: dict[str, list[dict[str, Any]]] = {}
        for item in self._items:
            for event_type in item.get("event_types") or ():
                self._by_event_type.setdefault(event_type, []).append(item)

    # ---------- 加载 ----------

    @classmethod
    def load(cls, root: str | Path | None = None) -> "ContentArsenal":
        """从 JSONL 加载。文件缺失时返回空库，而不是抛异常。

        空库不会让调用方崩溃——``select`` 只会返回空列表，
        由上层决定是否退回原有实现。
        """

        path = Path(root) if root is not None else DEFAULT_ARSENAL_DIR
        if path.is_dir():
            # ``root`` 既可能是工作区根目录（此时弹药库在它下面的 DEFAULT_ARSENAL_DIR
            # 里），也可能直接就是弹药库目录本身——两种都兼容。
            nested = path / DEFAULT_ARSENAL_DIR / TOPIC_AMMO_FILENAME
            path = nested if nested.exists() else path / TOPIC_AMMO_FILENAME
        if not path.exists():
            return cls([])

        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("text"):
                    items.append(payload)
        return cls(items, source_dir=path.parent)

    # ---------- 元信息 ----------

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[dict[str, Any]]:
        """弹药条目副本。"""

        return list(self._items)

    def event_types(self) -> list[str]:
        return sorted(self._by_event_type)

    def counts_by_event_type(self) -> dict[str, int]:
        return {key: len(value) for key, value in sorted(self._by_event_type.items())}

    def stats(self) -> dict[str, Any]:
        lengths = [len(item["text"]) for item in self._items]
        return {
            "total": len(self._items),
            "event_types": len(self._by_event_type),
            "sources": dict(Counter(str(item.get("source") or "-") for item in self._items)),
            "char_len_min": min(lengths) if lengths else 0,
            "char_len_max": max(lengths) if lengths else 0,
            "char_len_avg": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        }

    # ---------- 选择 ----------

    def select(
        self,
        *,
        count: int,
        event_type: str | None = None,
        seed_key: str = "",
        exclude_texts: Iterable[str] | None = None,
        scorer: Callable[[str], float] | None = None,
    ) -> list[dict[str, Any]]:
        """选出 ``count`` 条互不重复的弹药。

        参数
        ----
        count
            需要的条数。库里不够时返回实际能给的条数，不抛异常。
        event_type
            优先从该类型的弹药里选；不足时跨类借用。
        seed_key
            确定性种子的一部分。同一批节点里给不同节点传不同 seed_key
            （例如节点序号），就能拿到不同文案，且每次运行结果一致。
        exclude_texts
            已经用过的文案文本，避免同一批次内重复。
        scorer
            可选的语义打分函数，签名为 ``(text) -> float``，分数越高越相关。
            不传则只做标签检索 + 确定性轮转。
        """

        if count <= 0 or not self._items:
            return []

        excluded = set(exclude_texts or ())
        chosen: list[dict[str, Any]] = []
        seen: set[str] = set(excluded)

        # 第一遍：同类优先
        chosen.extend(
            self._take(
                pool=self._by_event_type.get(event_type) or [],
                count=count,
                seed_parts=(event_type or "-", seed_key),
                seen=seen,
                scorer=scorer,
            )
        )

        # 第二遍：还不够就从全库借（冷门类目不至于返回空）
        if len(chosen) < count:
            chosen.extend(
                self._take(
                    pool=self._items,
                    count=count - len(chosen),
                    seed_parts=("-fallback-", event_type or "-", seed_key),
                    seen=seen,
                    scorer=scorer,
                )
            )
        return chosen

    def select_texts(self, **kwargs: Any) -> list[str]:
        """``select`` 的便捷封装，只返回文案文本。"""

        return [item["text"] for item in self.select(**kwargs)]

    # ---------- 内部 ----------

    def _take(
        self,
        *,
        pool: list[dict[str, Any]],
        count: int,
        seed_parts: tuple[str, ...],
        seen: set[str],
        scorer: Callable[[str], float] | None,
    ) -> list[dict[str, Any]]:
        if count <= 0 or not pool:
            return []

        if scorer is not None:
            # 有语义信号时直接按相关度取，不再轮转——轮转会破坏相关性排序。
            # 多样性由调用方一次取 count 条（内部已去重）+ exclude_texts 保证。
            candidates = sorted(pool, key=lambda item: scorer(item["text"]), reverse=True)
        else:
            # 没有语义信号时做确定性轮转，避免每次都命中库里靠前的那几条。
            candidates = list(pool)
            random.Random(_stable_seed(*seed_parts)).shuffle(candidates)

        picked: list[dict[str, Any]] = []
        for item in candidates:
            text = item["text"]
            if text in seen:
                continue
            seen.add(text)
            picked.append(item)
            if len(picked) >= count:
                break
        return picked
