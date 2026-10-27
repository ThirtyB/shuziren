"""多人协同动作编排的验证：四种动作、四个时期、依赖必须指向已有动作。"""

import unittest
from pathlib import Path

from influence_strategy.eval_hot_events import _build_staged_node_outputs
from influence_strategy.pipeline import StrategyPipeline

ROOT = Path(__file__).resolve().parents[1]
EVENT = "围绕人工智能治理和数据安全监管做一次传播活动，提升讨论度并控制风险。"
PLATFORM = "facebook_simulated"

#: 允许的动作只有这四种
ALLOWED_ACTIONS = {"publish_post", "comment_post", "like_post", "repost_post"}
#: 这三种必须依赖已有动作
DEPENDENT_ACTIONS = {"comment_post", "like_post", "repost_post"}
STAGES = {"首发期", "互动期", "扩散期", "支持期"}


class StagedActionPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifacts = StrategyPipeline().run_with_artifacts(
            workspace_root=ROOT,
            event_input=EVENT,
            recall_size=200,
            recall_mode="platform_aware",
            dispatch_platform=PLATFORM,
        )
        cls.nodes = artifacts.strategy_result.selected_nodes
        outputs, _, _ = _build_staged_node_outputs(
            strategy_result=artifacts.strategy_result,
            selected_nodes=cls.nodes,
            platform=PLATFORM,
            workspace_root=ROOT,
        )
        cls.actions = [
            action for payload in outputs.values() for action in payload["动作清单"]
        ]
        cls.action_ids = {action["动作编号"] for action in cls.actions}

    def test_only_four_action_types(self) -> None:
        self.assertTrue({a["动作类型"] for a in self.actions} <= ALLOWED_ACTIONS)

    def test_dependent_actions_reference_existing_action(self) -> None:
        for action in self.actions:
            if action["动作类型"] not in DEPENDENT_ACTIONS:
                continue
            self.assertTrue(action["依赖动作"], action["动作编号"])
            for dep in action["依赖动作"]:
                self.assertIn(dep, self.action_ids)

    def test_every_stage_has_actions(self) -> None:
        self.assertEqual({a["动作分组"] for a in self.actions}, STAGES)

    def test_launch_nodes_publish_once_each(self) -> None:
        publish = [a for a in self.actions if a["动作分组"] == "首发期"]
        launch_ids = {
            node.user_id for node in self.nodes if node.selected_role == "core_publish_node"
        }
        self.assertEqual(len(publish), len(launch_ids))
        self.assertTrue(all(a["动作类型"] == "publish_post" for a in publish))

    def test_publish_actions_never_empty(self) -> None:
        """发帖内容必须有文案（弹药库取不到时会退回规则正文）。"""

        for action in self.actions:
            if action["动作类型"] == "publish_post":
                self.assertTrue(action["执行参数"].get("正文草案"), action["动作编号"])


if __name__ == "__main__":
    unittest.main()
