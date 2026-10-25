"""枚举当前 LLM 接口**实际支持**的模型（用于挑选 A/B 实验的低延迟对照组）。

为什么需要它：项目配置里只有 ``LLM_MODEL=qwen3.7-plus``（见 ``.env``），
``llm_client.py`` 也从不在代码里硬编码模型清单。为了避免"猜模型名"，
这里直接问接口本身 —— OpenAI 兼容网关都提供 ``GET /models``。

用法：
    python tests/discover_llm_models.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.llm_client import OpenAICompatibleLLMClient  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "selector_llm_profile"


# llm_client.py:72-82 —— 当前解析顺序里出现过的全部模型相关 key，作为候选来源
def _configured_candidates() -> dict[str, str]:
    keys: dict[str, str] = {}
    root = PROJECT_ROOT
    for path in (root / ".env", root / "src" / "influence_strategy" / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k.endswith("MODEL") or k == "MODEL_NAME":
                keys[k] = v
    return keys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT)
    if client is None:
        print("无 LLM 凭据，无法枚举模型。")
        sys.exit(2)

    print(f"base_url      : {client.base_url}")
    print(f"provider      : {client.provider}")
    print(f"current model : {client.model}")
    print(f"timeout       : {client.timeout}")
    print()

    print("=== .env 中配置过的 MODEL 相关键 ===")
    for k, v in _configured_candidates().items():
        print(f"  {k} = {v or '<空>'}")
    print()

    url = f"{client.base_url}/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {client.api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"GET {url}")
        print(f"  失败：{type(exc).__name__}: {exc}")
        print()
        print("→ 该接口不支持 /models 枚举。请勿猜模型名。")
        sys.exit(1)

    models = body.get("data") or body.get("models") or []
    print(f"=== 接口返回 {len(models)} 个模型 ===")
    rows = []
    for m in models:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("model") or str(m)
            owned = m.get("owned_by") or m.get("owner") or ""
        else:
            mid, owned = str(m), ""
        rows.append({"id": mid, "owned_by": owned})
        print(f"  {mid}" + (f"   (owned_by={owned})" if owned else ""))

    (OUT_DIR / "available_models.json").write_text(
        json.dumps(
            {
                "base_url": client.base_url,
                "provider": client.provider,
                "current_model": client.model,
                "models": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"→ 已写入 {OUT_DIR / 'available_models.json'}")


if __name__ == "__main__":
    main()
