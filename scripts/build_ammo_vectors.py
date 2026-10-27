"""预计算弹药库文本的语义向量并落盘。

弹药库是静态资产，把它离线编码一次存盘后，运行期只需编码 1 条事件文本
再做矩阵乘法（毫秒级），不必每次运行都重编整个弹药库。

产物
----
``data/derived/content_arsenal/ammo_vectors.npz``，含 ``texts`` /
``embeddings`` / ``fingerprint`` 三项。

``fingerprint`` 是弹药库内容的哈希——弹药库一旦增删改，存盘向量自动失效，
运行期会回退到实时编码，不会取错文案。换弹药库后重跑本脚本即可。

用法
----
    python scripts/build_ammo_vectors.py [--workspace-root .]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.content_arsenal import (  # noqa: E402
    AmmoVectorIndex,
    ContentArsenal,
    _arsenal_fingerprint,
)
from influence_strategy.local_models import load_local_models  # noqa: E402


def _encode(encoder: object, texts: list[str]) -> object:
    """批量编码，兼容 BGE（``_encode_batch``）与 SVD（只有单条 ``encode``）。"""

    if hasattr(encoder, "_encode_batch"):
        return encoder._encode_batch(texts, is_query=False)
    import numpy as np

    return np.stack([encoder.encode(text) for text in texts])


def main() -> int:
    parser = argparse.ArgumentParser(description="预计算弹药库文本的语义向量")
    parser.add_argument("--workspace-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    root = Path(args.workspace_root).resolve()

    arsenal = ContentArsenal.load(root)
    if not len(arsenal):
        print(f"弹药库为空或缺失: {root / 'data' / 'derived' / 'content_arsenal'}")
        return 1

    texts = [item["text"] for item in arsenal.items]
    print(f"弹药库: {len(texts)} 条，目录 {arsenal.source_dir}")

    local_models = load_local_models(root)
    if local_models is None:
        print(
            "未找到本地语义模型产物，先运行：\n"
            "  python scripts/build_local_models.py"
        )
        return 1

    started = time.perf_counter()
    embeddings = _encode(local_models.encoder, texts)
    elapsed = time.perf_counter() - started
    print(f"编码完成: {getattr(embeddings, 'shape', '?')}，耗时 {elapsed:.1f}s")

    index = AmmoVectorIndex(texts, embeddings, _arsenal_fingerprint(texts))
    path = index.save(arsenal.source_dir)
    print(f"已写入: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
