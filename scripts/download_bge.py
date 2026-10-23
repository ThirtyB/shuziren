"""下载 BGE 中文句向量模型到项目目录。

用法::

    python scripts/download_bge.py
    python scripts/download_bge.py --repo BAAI/bge-base-zh-v1.5

为什么单独一个脚本
------------------
模型权重 183MB，不适合入库；``build_local_models.py`` 与运行期都只按路径读取。
换机器部署时先跑本脚本（一次），再跑预计算。

网络说明
--------
默认走 ``hf-mirror.com``（国内可达）。``hf-xet`` 的 CAS 服务在镜像站会返回 401，
所以脚本强制 ``HF_HUB_DISABLE_XET=1`` 走普通 HTTP 下载。若身处可直连
HuggingFace 的网络，加 ``--endpoint https://huggingface.co`` 即可。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "BAAI/bge-small-zh-v1.5"
DEFAULT_TARGET = PROJECT_ROOT / "data" / "models" / "bge-small-zh-v1.5"


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 BGE 中文句向量模型")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # 镜像站不支持 Xet 协议，必须退回普通 HTTP。
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    target = Path(args.target)
    if (target / "config.json").exists():
        print(f"模型已存在，跳过下载: {target}")
        return 0

    print(f"repo     : {args.repo}")
    print(f"endpoint : {args.endpoint}")
    print(f"target   : {target}")

    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(target),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "*.model"],
    )
    elapsed = time.perf_counter() - started
    size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"下载完成: {elapsed:.1f}s, {size_mb:.1f} MB")
    print(f"接下来运行: python scripts/build_local_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
