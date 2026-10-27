"""把 BGE 模型导出为 ONNX，供运行期用 onnxruntime 推理。

为什么需要这一步
----------------
运行期 BGE 只做一件事：把 1 条话题文本编码成 512 维向量。但为此要
``import torch``(2.0s) + ``import transformers``(3.0s)，再加模型加载 4s ——
首次调用约 9 秒，而真正的推理只要 0.02 秒。

导出成 ONNX 后，运行期只需要：

    import onnxruntime   # 0.36s
    import tokenizers    # 0.01s

两者合计 0.37s，比 torch + transformers 的 4.96s 省掉约 4.6s，且不再需要
torch / transformers 参与运行（它们只在跑本脚本时才需要）。

产物
----
    data/models/bge-small-zh-v1.5-onnx/model.onnx       图
    data/models/bge-small-zh-v1.5-onnx/tokenizer.json   分词器（tokenizers 原生格式）

用法
----
    python scripts/export_bge_onnx.py            # 导出 + 校验
    python scripts/export_bge_onnx.py --check    # 只校验已有产物
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_SRC = PROJECT_ROOT / "data" / "models" / "bge-small-zh-v1.5"
DEFAULT_DST = PROJECT_ROOT / "data" / "models" / "bge-small-zh-v1.5-onnx"

#: 导出用的序列长度。与 BGESemanticEncoder 一致；ONNX 用静态 shape 更稳。
MAX_LENGTH = 96


def export(src: Path, dst: Path) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not src.exists():
        raise SystemExit(f"BGE 模型目录不存在: {src}\n请先运行 scripts/download_bge.py")

    print(f"源模型: {src}")
    model = AutoModel.from_pretrained(str(src))
    model.eval()

    dst.mkdir(parents=True, exist_ok=True)

    # 构造一个静态 shape 的样例输入（全 1 = padding token，值不重要，只要 shape 对）。
    batch = 1
    dummy = {
        "input_ids": torch.ones((batch, MAX_LENGTH), dtype=torch.long),
        "attention_mask": torch.ones((batch, MAX_LENGTH), dtype=torch.long),
        "token_type_ids": torch.zeros((batch, MAX_LENGTH), dtype=torch.long),
    }

    onnx_path = dst / "model.onnx"
    print(f"导出到: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            # batch 维保持动态，序列长度固定为 MAX_LENGTH（简单且够用：
            # 运行期只有 1 条话题，长度靠 tokenizer 截断到 96）。
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "token_type_ids": {0: "batch"},
                "last_hidden_state": {0: "batch"},
            },
            opset_version=14,
        )

    # 分词器转成 tokenizers 原生格式，运行期不必 import transformers。
    tokenizer = AutoTokenizer.from_pretrained(str(src))
    tokenizer_json = src / "tokenizer.json"
    if tokenizer_json.exists():
        shutil.copyfile(tokenizer_json, dst / "tokenizer.json")
    else:
        tokenizer.backend_tokenizer.save(str(dst / "tokenizer.json"))
    print(f"分词器: {dst / 'tokenizer.json'}")
    print("导出完成")


def check(src: Path, dst: Path) -> int:
    """对比 ONNX 与 torch 的输出是否一致。"""

    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModel, AutoTokenizer

    onnx_path = dst / "model.onnx"
    if not onnx_path.exists():
        print(f"ONNX 产物不存在: {onnx_path}")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(str(src))
    model = AutoModel.from_pretrained(str(src))
    model.eval()

    texts = [
        "亲子共读与英语启蒙话题讨论",
        "加拿大铁路工人罢工导致货运受阻",
        "高端AI芯片供应与出口限制",
    ]

    # 必须 padding 到**定长** MAX_LENGTH：ONNX 导出用的是静态序列维度，
    # 只在 batch 维上动态。padding=True 会按批次内的最长句补齐（这里是 17），
    # 与模型期望的 96 不符，onnxruntime 会直接报维度错误。
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        torch_cls = model(**encoded).last_hidden_state[:, 0].numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {name: encoded[name].numpy() for name in ("input_ids", "attention_mask", "token_type_ids")}
    onnx_cls = session.run(["last_hidden_state"], feeds)[0][:, 0]

    print()
    print("逐条余弦相似度（torch vs onnx，应 > 0.9999）:")
    worst = 1.0
    for index, text in enumerate(texts):
        a = torch_cls[index] / np.linalg.norm(torch_cls[index])
        b = onnx_cls[index] / np.linalg.norm(onnx_cls[index])
        cos = float(a @ b)
        worst = min(worst, cos)
        print(f"  {cos:.6f}  {text}")
    max_abs = float(np.abs(torch_cls - onnx_cls).max())
    print(f"最大绝对误差: {max_abs:.2e}")

    ok = worst > 0.9999
    print("一致性:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 BGE 为 ONNX")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--check", action="store_true", help="只校验已有产物")
    args = parser.parse_args()

    started = time.perf_counter()
    if not args.check:
        export(args.src, args.dst)
    code = check(args.src, args.dst)
    print(f"总耗时: {time.perf_counter() - started:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
