"""构建文案弹药库（当前只覆盖 L1：发帖内容文案）。

两步
----
1. ``migrate`` —— 把事件库里的 ``opinion_variants`` 迁移成弹药，
   并用 ``RuleBasedEventParser`` 给每条打上 event_type 标签。
2. ``expand``  —— 用 LLM 批量生成，把欠料类目补齐。
   **离线跑一次后固化进 JSONL，运行时不再调 LLM。**

用法
----
    python scripts/build_content_arsenal.py              # 只迁移
    python scripts/build_content_arsenal.py --expand     # 迁移 + LLM 扩充
    python scripts/build_content_arsenal.py --stats      # 只看统计（不写盘）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from influence_strategy.content_arsenal import (  # noqa: E402
    DEFAULT_ARSENAL_DIR,
    TOPIC_AMMO_FILENAME,
    ContentArsenal,
)
from influence_strategy.event_parser import RuleBasedEventParser  # noqa: E402

SOURCE_EVENTS = ROOT / "eval" / "hot_event_opinion_variants_200.json"
OUTPUT_FILE = ROOT / DEFAULT_ARSENAL_DIR / TOPIC_AMMO_FILENAME

#: 每个 event_type 期望的弹药条数，不足的部分由 --expand 补齐。
#:
#: 迁移后的实际分布（2256 条 / 15 类）差异很大：cyber_data_governance 有 390 条，
#: 而 social_unrest 只有 30 条。定在 150 是为了让 LLM 扩充真正补到**冷门类目**
#: 上（增加它们的多样性），而不是给已经很充裕的类目凑数。
#: 想只补齐最低水位就把它调回 60；想扩得更多就继续上调。
TARGET_PER_TYPE = 150

#: LLM 单次请求生成的条数上限，避免单条响应过长被截断。
MAX_PER_CALL = 20

#: 单类目最多尝试的轮数，防止 LLM 一直返回重复内容时死循环。
MAX_ROUNDS_PER_TYPE = 6

#: 文案长度区间，与现存素材（18~69 字）保持一致。
MIN_LEN, MAX_LEN = 18, 70

EXPAND_SYSTEM_PROMPT = (
    "你是社交媒体内容策略助手，负责为时事热点事件生成观点型短帖文案。"
    "只输出 JSON，格式为 {\"items\": [\"文案1\", \"文案2\"]}，不要输出任何其他内容。"
)


def _expand_user_prompt(event_type: str, samples: list[str], need: int) -> str:
    sample_text = "\n".join(f"- {item}" for item in samples[:5]) or "-（暂无示例）"
    return (
        f"事件类别：{event_type}\n\n"
        f"该类别已有的文案示例（请贴近这种风格）：\n{sample_text}\n\n"
        f"请再生成 {need} 条同类文案，要求：\n"
        f"1. 每条 {MIN_LEN}~{MAX_LEN} 字，观点型、口语化，像真实用户发的短帖；\n"
        "2. 不写话题标签、不写 @、不要引号和序号；\n"
        "3. 不编造具体数字、时间、机构名等未经证实的事实；\n"
        "4. 彼此切入角度不同，不要同义重复；\n"
        "5. 只输出 JSON。"
    )


def _parse_items(raw: Any) -> list[str]:
    """从 LLM 返回里取出文案列表，兼容几种常见包裹方式。"""

    if isinstance(raw, dict):
        for key in ("items", "texts", "contents", "data", "result"):
            value = raw.get(key)
            if isinstance(value, list):
                raw = value
                break
        else:
            # 兜底：取第一个 list 值
            for value in raw.values():
                if isinstance(value, list):
                    raw = value
                    break
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
        else:
            continue
        if text:
            result.append(text)
    return result


def migrate() -> list[dict[str, Any]]:
    """把事件库里的 opinion_variants 迁移成带 event_type 标签的弹药。"""

    events = json.loads(SOURCE_EVENTS.read_text(encoding="utf-8"))
    parser = RuleBasedEventParser()
    items: list[dict[str, Any]] = []

    for hot in events:
        parsed = parser.parse(
            {
                "event_id": str(hot.get("event_id") or ""),
                "event_title": str(hot.get("event_title") or ""),
                "event_description": str(hot.get("event_summary") or ""),
                "target_goal": str(hot.get("target") or ""),
            }
        )
        event_type = parsed.event_type
        for variant in hot.get("opinion_variants") or []:
            text = str(variant).strip()
            if not text:
                continue
            items.append(
                {
                    "id": f"T{len(items) + 1:05d}",
                    "text": text,
                    "event_types": [event_type],
                    "domain": str(hot.get("domain") or ""),
                    "source_event_id": str(hot.get("event_id") or ""),
                    "source": SOURCE_EVENTS.name,
                    "char_len": len(text),
                }
            )
    return items


def expand(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """用 LLM 把每个 event_type 补齐到 TARGET_PER_TYPE 条。

    需要 LLM 配置（.env 或 LLM_API_KEY 环境变量），参考 env.example。
    """

    from influence_strategy.llm_client import OpenAICompatibleLLMClient

    client = OpenAICompatibleLLMClient.from_env_files(ROOT)
    if client is None:
        raise SystemExit(
            "未检测到 LLM 配置：请在项目根目录创建 .env（参考 env.example），"
            "或设置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 环境变量。"
        )
    print(f"LLM: {client.describe()}")

    counts: Counter[str] = Counter(
        event_type for item in items for event_type in item["event_types"]
    )
    by_type: dict[str, list[str]] = defaultdict(list)
    for item in items:
        for event_type in item["event_types"]:
            by_type[event_type].append(item["text"])

    seen = {item["text"] for item in items}
    added = 0

    for event_type in sorted(by_type):
        need = TARGET_PER_TYPE - counts[event_type]
        if need <= 0:
            continue
        samples = by_type[event_type]
        rounds = 0
        while need > 0 and rounds < MAX_ROUNDS_PER_TYPE:
            rounds += 1
            batch = min(need, MAX_PER_CALL)
            raw = client.generate_json(
                system_prompt=EXPAND_SYSTEM_PROMPT,
                user_prompt=_expand_user_prompt(event_type, samples, batch),
            )
            candidates = _parse_items(raw)
            if not candidates:
                print(f"  [{event_type}] 第 {rounds} 轮未返回有效内容，跳过")
                break
            for text in candidates:
                if text in seen:
                    continue
                if not (MIN_LEN <= len(text) <= MAX_LEN):
                    continue
                seen.add(text)
                items.append(
                    {
                        "id": f"T{len(items) + 1:05d}",
                        "text": text,
                        "event_types": [event_type],
                        "domain": "",
                        "source_event_id": "",
                        "source": "llm_generated",
                        "char_len": len(text),
                    }
                )
                added += 1
                need -= 1
                if need <= 0:
                    break
        print(f"  [{event_type}] 补齐 {TARGET_PER_TYPE - counts[event_type] - need} 条")

    return items, added


def write_items(items: list[dict[str, Any]]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_stats(items: list[dict[str, Any]]) -> None:
    arsenal = ContentArsenal(items)
    stats = arsenal.stats()
    print(f"\n总条数 {stats['total']} | 类目 {stats['event_types']}")
    print(f"长度 min/avg/max = {stats['char_len_min']}/{stats['char_len_avg']}/{stats['char_len_max']}")
    print("来源:", stats["sources"])
    print("\n各 event_type 条数：")
    counts = arsenal.counts_by_event_type()
    for event_type, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = "" if count >= TARGET_PER_TYPE else f"  (差 {TARGET_PER_TYPE - count} 条)"
        print(f"  {event_type:<26} {count:>4}{flag}")


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="构建文案弹药库")
    arg_parser.add_argument("--expand", action="store_true", help="用 LLM 扩充欠料类目")
    arg_parser.add_argument("--stats", action="store_true", help="只打印现有弹药库统计")
    args = arg_parser.parse_args()

    if args.stats:
        if not OUTPUT_FILE.exists():
            raise SystemExit(f"弹药库尚未生成：{OUTPUT_FILE}")
        print_stats(ContentArsenal.load(ROOT / DEFAULT_ARSENAL_DIR).items)
        return

    print("迁移 opinion_variants ...")
    items = migrate()
    print(f"  得到 {len(items)} 条")

    if args.expand:
        print("LLM 扩充 ...")
        items, added = expand(items)
        print(f"  新增 {added} 条")

    write_items(items)
    print(f"已写入 {OUTPUT_FILE}")
    print_stats(items)


if __name__ == "__main__":
    main()
