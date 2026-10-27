"""多人协同（单平台）身份分配的验证：人数配额 + 时期占比 + 端到端互斥。"""

import unittest
from pathlib import Path

from influence_strategy.pipeline import StrategyPipeline
from influence_strategy.selector import Selector
from influence_strategy.strategy_generator import StrategyGenerator

ROOT = Path(__file__).resolve().parents[1]

ALL_ROLES = {
    "core_publish_node",
    "amplification_node",
    "interaction_response_node",
    "support_node",
}


class RoleQuotaTest(unittest.TestCase):
    """人数配额：首发 20% / 扩散 40% / 互动 20%，支持拿剩余。"""

    def test_ten_nodes_split_by_ratio(self) -> None:
        quotas = Selector()._role_quotas(10)
        self.assertEqual(quotas["core_publish_node"], 2)
        self.assertEqual(quotas["amplification_node"], 4)
        self.assertEqual(quotas["interaction_response_node"], 2)
        self.assertEqual(quotas["support_node"], 2)

    def test_launch_and_amplification_always_at_least_one(self) -> None:
        for total in (2, 3, 4):
            quotas = Selector()._role_quotas(total)
            self.assertGreaterEqual(quotas["core_publish_node"], 1)
            self.assertGreaterEqual(quotas["amplification_node"], 1)

    def test_quotas_never_exceed_total(self) -> None:
        for total in (2, 3, 5, 10):
            self.assertLessEqual(sum(Selector()._role_quotas(total).values()), total)


class TimePlanTest(unittest.TestCase):
    """四个时期按 0.1 / 0.2 / 0.4 / 0.3 首尾相接铺满窗口。"""

    def test_ten_hour_window(self) -> None:
        plan = StrategyGenerator()._build_time_plan(10)
        self.assertEqual(plan["stage_1_launch"], "T0 - T+1h")
        self.assertEqual(plan["stage_2_engage"], "T+1h - T+3h")
        self.assertEqual(plan["stage_3_amplify"], "T+3h - T+7h")
        self.assertEqual(plan["stage_2_support"], "T+7h - T+10h")

    def test_24_hour_window(self) -> None:
        plan = StrategyGenerator()._build_time_plan(24)
        self.assertEqual(plan["stage_1_launch"], "T0 - T+2h")
        self.assertEqual(plan["stage_2_engage"], "T+2h - T+7h")
        self.assertEqual(plan["stage_3_amplify"], "T+7h - T+17h")
        self.assertEqual(plan["stage_2_support"], "T+17h - T+24h")


class RoleAssignmentIntegrationTest(unittest.TestCase):
    """端到端：真实数据跑一遍，确认身份互斥且配额生效。"""

    def test_roles_are_exclusive_and_quota_honored(self) -> None:
        artifacts = StrategyPipeline().run_with_artifacts(
            workspace_root=ROOT,
            event_input="围绕人工智能治理和数据安全监管做一次传播活动，提升讨论度并控制风险。",
            recall_size=200,
            recall_mode="platform_aware",
            dispatch_platform="facebook_simulated",
        )
        nodes = artifacts.selection_result.selected_nodes
        self.assertTrue(nodes)

        roles = [node.selected_role for node in nodes]
        # 只会出现四种身份
        self.assertTrue(set(roles) <= ALL_ROLES)
        # 同一个人不会拿到两个身份
        self.assertEqual(len({node.user_id for node in nodes}), len(nodes))
        # 首发与扩散至少各 1 人
        self.assertGreaterEqual(roles.count("core_publish_node"), 1)
        self.assertGreaterEqual(roles.count("amplification_node"), 1)


if __name__ == "__main__":
    unittest.main()
