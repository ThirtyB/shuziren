"""预计算本地语义 / 图结构模型。

用法::

    # 默认：深度模型（BGE 句向量 + GraphSAGE）
    python scripts/build_local_models.py

    # 回退到零依赖的统计方案（TF-IDF+SVD / 无参数 SGC）
    python scripts/build_local_models.py --semantic-backend svd --graph-backend sgc

    # 只重建语义层
    python scripts/build_local_models.py --skip-graph

产物（默认 ``data/derived/local_models/``）::

    semantic_index.npz    节点语义向量 + backend 标记
    graph_index.npz       图结构特征 + 传播潜力 + backend 标记
    build_report.json     耗时、backend、训练曲线等统计

运行期由 ``local_models._detect_backend`` 读 ``backend`` 标记，自动选择对应的
实现加载，因此两种后端的产物可以互相替换而无需改调用方代码。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.data_loader import DataLoader  # noqa: E402

#: BGE 模型默认位置（由 scripts/download_bge.py 下载）。
DEFAULT_MODEL_DIR = Path("data") / "models" / "bge-small-zh-v1.5"


def _resolve_paths(workspace_root: Path) -> tuple[Path, Path]:
    """返回 (profile 文件, 图文件) 路径。优先用带平台字段的派生数据。"""

    loader = DataLoader(workspace_root)
    profile_path = loader._preferred_profile_path()
    graph_path = loader._preferred_enriched_profile_path()
    return profile_path, graph_path


def _build_semantic(backend: str, profiles, args, output_dir: Path):
    """按后端构建语义编码器，返回 (encoder, index, extra)。"""

    if backend == "bge":
        from influence_strategy.bge_semantic import BGESemanticEncoder

        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(args.workspace_root).resolve() / model_dir
        if not model_dir.exists():
            raise FileNotFoundError(
                f"BGE 模型目录不存在: {model_dir}\n"
                "先运行 scripts/download_bge.py 下载，或加 --semantic-backend svd 回退。"
            )
        print(f"      using BGE: {model_dir}")
        encoder = BGESemanticEncoder(model_dir=model_dir, batch_size=args.batch_size)
        index = encoder.fit(profiles)
        return encoder, index, {"model_dir": str(model_dir)}

    from influence_strategy.local_semantic import LocalSemanticEncoder

    print("      using TF-IDF + SVD")
    encoder = LocalSemanticEncoder(dim=args.dim, max_features=args.max_features)
    index = encoder.fit(profiles)
    return encoder, index, {"dim": index.dim}


def _build_graph(backend: str, graph_payload, args):
    """按后端构建图模型，返回 (builder, index, extra)。"""

    if backend == "sage":
        from influence_strategy.graph_sage import GraphSageBuilder

        print(f"      using GraphSAGE: epochs={args.epochs}, hidden={args.hidden}")
        builder = GraphSageBuilder(
            hidden=args.hidden,
            out_dim=args.out_dim,
            epochs=args.epochs,
            edge_samples=args.edge_samples,
        )
        index = builder.build(graph_payload)
        return builder, index, {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "out_dim": args.out_dim,
            "final_loss": index.train_history[-1]["loss"] if index.train_history else None,
            "train_history": index.train_history,
        }

    from influence_strategy.graph_features import GraphFeatureBuilder

    print("      using SGC (no-parameter propagation)")
    builder = GraphFeatureBuilder()
    index = builder.build(graph_payload)
    return builder, index, {}


def main() -> int:
    parser = argparse.ArgumentParser(description="预计算本地语义与图结构模型")
    parser.add_argument("--workspace-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default=None, help="产物目录，默认 <workspace>/data/derived/local_models")
    parser.add_argument(
        "--semantic-backend",
        choices=["bge", "svd"],
        default="bge",
        help="语义层实现：bge=深度 Transformer，svd=TF-IDF+SVD",
    )
    parser.add_argument(
        "--graph-backend",
        choices=["sage", "sgc"],
        default="sage",
        help="图结构层实现：sage=可训练 GraphSAGE，sgc=无参数传播",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="BGE 模型目录")
    parser.add_argument("--dim", type=int, default=128, help="SVD 语义维度（仅 svd 后端）")
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=64, help="BGE 编码批大小")
    parser.add_argument("--epochs", type=int, default=150, help="GraphSAGE 训练轮数")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--edge-samples", type=int, default=80_000)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace_root / "data" / "derived" / "local_models"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path, graph_path = _resolve_paths(workspace_root)
    print(f"[1/3] workspace: {workspace_root}")
    print(f"      profile : {profile_path}")
    print(f"      graph   : {graph_path}")
    print(f"      output  : {output_dir}")
    if not profile_path.exists():
        print(f"ERROR: profile 文件不存在: {profile_path}")
        return 2
    if not graph_path.exists() and not args.skip_graph:
        print(f"ERROR: 图文件不存在: {graph_path}")
        return 2

    report: dict[str, object] = {
        "workspace_root": str(workspace_root),
        "profile_path": str(profile_path),
        "graph_path": str(graph_path),
        "output_dir": str(output_dir),
        "semantic_backend": args.semantic_backend if not args.skip_semantic else None,
        "graph_backend": args.graph_backend if not args.skip_graph else None,
    }

    # ---- 语义模型 ----
    if not args.skip_semantic:
        started = time.perf_counter()
        print("[2/3] 加载 profile 并构建语义索引 ...")
        loader = DataLoader(workspace_root)
        profiles = loader.load_profiles()
        print(f"      profiles={len(profiles)}  ({time.perf_counter() - started:.1f}s)")

        encoder, index, extra = _build_semantic(args.semantic_backend, profiles, args, output_dir)
        encoder.save(output_dir)
        elapsed = time.perf_counter() - started
        report["semantic"] = {
            "backend": args.semantic_backend,
            "node_count": index.node_count,
            "dim": index.dim,
            "seconds": round(elapsed, 2),
            **extra,
        }
        print(
            f"      semantic done: backend={args.semantic_backend} "
            f"nodes={index.node_count} dim={index.dim} {elapsed:.1f}s"
        )
    else:
        print("[2/3] skip semantic")

    # ---- 图结构模型 ----
    if not args.skip_graph:
        started = time.perf_counter()
        print("[3/3] 构建图结构模型 ...")
        with graph_path.open("r", encoding="utf-8") as handle:
            graph_payload = json.load(handle)
        print(f"      graph nodes={len(graph_payload)}  ({time.perf_counter() - started:.1f}s)")

        builder, index, extra = _build_graph(args.graph_backend, graph_payload, args)
        builder.save(output_dir)
        elapsed = time.perf_counter() - started
        report["graph"] = {
            "backend": args.graph_backend,
            "node_count": index.node_count,
            "feature_dim": int(index.features.shape[1]),
            "potential_mean": round(float(index.potential.mean()), 4),
            "potential_p95": round(float(np.percentile(index.potential, 95)), 4),
            "seconds": round(elapsed, 2),
            **extra,
        }
        print(
            f"      graph done: backend={args.graph_backend} nodes={index.node_count} "
            f"features={index.features.shape[1]} {elapsed:.1f}s"
        )
    else:
        print("[3/3] skip graph")

    report_path = output_dir / "build_report.json"
    # 训练曲线很长，单独存一份，报告里只留摘要。
    history = (report.get("graph") or {}).pop("train_history", None) if isinstance(report.get("graph"), dict) else None
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if history:
        (output_dir / "sage_train_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\n报告: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
