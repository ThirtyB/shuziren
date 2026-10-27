"""文案弹药库（ContentArsenal）测试。

只验证“弹药库本身能用”，不涉及它如何接入流水线——接入是后续步骤。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from influence_strategy.content_arsenal import ContentArsenal

ROOT = Path(__file__).resolve().parents[1]
ARSENAL_DIR = ROOT / "data" / "derived" / "content_arsenal"


def _item(index: int, text: str, event_type: str) -> dict:
    return {
        "id": f"T{index:03d}",
        "text": text,
        "event_types": [event_type],
        "char_len": len(text),
    }


class ContentArsenalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.arsenal = ContentArsenal(
            [
                _item(1, "红海航运紧张推高保险成本", "industry_supply_chain"),
                _item(2, "绕航时间变长影响全球物流", "industry_supply_chain"),
                _item(3, "多国护航协作能否降温", "industry_supply_chain"),
                _item(4, "港口拥堵让交货周期拉长", "industry_supply_chain"),
                _item(5, "运价指数连续上行", "industry_supply_chain"),
                _item(6, "供应链转移带来成本重构", "industry_supply_chain"),
                _item(7, "加息预期升温市场波动加大", "macro_economy"),
            ]
        )

    def test_selects_from_matching_event_type(self) -> None:
        allowed = {
            item["text"]
            for item in self.arsenal.items
            if "industry_supply_chain" in item["event_types"]
        }
        texts = self.arsenal.select_texts(count=2, event_type="industry_supply_chain", seed_key="n1")
        self.assertEqual(len(texts), 2)
        for text in texts:
            self.assertIn(text, allowed)

    def test_is_deterministic(self) -> None:
        """同一 seed 必须永远得到同一结果，否则结果不可复现。"""

        first = self.arsenal.select_texts(count=3, event_type="industry_supply_chain", seed_key="n1")
        second = self.arsenal.select_texts(count=3, event_type="industry_supply_chain", seed_key="n1")
        self.assertEqual(first, second)

    def test_different_seed_gives_different_texts(self) -> None:
        """不同节点（不同 seed）应拿到不同文案，避免整批发一样的。

        比较小批量下的完整序列而不是单条——3 条里取 1 条有正常碰撞概率。
        """

        first = self.arsenal.select_texts(count=3, event_type="industry_supply_chain", seed_key="n1")
        second = self.arsenal.select_texts(count=3, event_type="industry_supply_chain", seed_key="n2")
        self.assertNotEqual(first, second)

    def test_no_duplicates_within_one_batch(self) -> None:
        """同一批次内 N 个节点拿到的文案互不重复。"""

        used: list[str] = []
        for index in range(3):
            texts = self.arsenal.select_texts(
                count=1,
                event_type="industry_supply_chain",
                seed_key=f"node_{index}",
                exclude_texts=used,
            )
            used.extend(texts)
        self.assertEqual(len(used), len(set(used)))

    def test_falls_back_when_type_has_too_few(self) -> None:
        """类目弹药不够时跨类借用，而不是返回不足额。"""

        texts = self.arsenal.select_texts(count=4, event_type="macro_economy", seed_key="n1")
        self.assertEqual(len(texts), 4)

    def test_returns_empty_on_empty_arsenal(self) -> None:
        self.assertEqual(ContentArsenal([]).select_texts(count=3, seed_key="n1"), [])

    def test_scorer_narrows_candidates(self) -> None:
        """传入 scorer 时优先给相关度高的文案。"""

        texts = self.arsenal.select_texts(
            count=1,
            event_type="industry_supply_chain",
            seed_key="n1",
            scorer=lambda text: 1.0 if "保险" in text else 0.0,
        )
        self.assertEqual(texts, ["红海航运紧张推高保险成本"])

    def test_stats(self) -> None:
        stats = self.arsenal.stats()
        self.assertEqual(stats["total"], 7)
        self.assertEqual(stats["event_types"], 2)


class BuiltArsenalSmokeTest(unittest.TestCase):
    """对真实弹药库做冒烟检查；库还没生成时跳过，不算失败。"""

    def test_built_arsenal_is_usable(self) -> None:
        if not (ARSENAL_DIR / "topic_ammo.jsonl").exists():
            self.skipTest("弹药库尚未生成，先跑 scripts/build_content_arsenal.py")

        arsenal = ContentArsenal.load(ARSENAL_DIR)
        self.assertGreater(len(arsenal), 0)
        for event_type in arsenal.event_types():
            texts = arsenal.select_texts(count=3, event_type=event_type, seed_key="smoke")
            self.assertEqual(len(texts), 3, f"{event_type} 选不出 3 条")
            self.assertEqual(len(set(texts)), 3, f"{event_type} 选出了重复文案")


if __name__ == "__main__":
    unittest.main()
