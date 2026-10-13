from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_eval
from influence_strategy.eval_hot_events import MULTIPLATFORM_OUTPUT_FORMAT_VERSION


PROJECT_ROOT = Path(run_eval.__file__).resolve().parent

#: 多平台最终业务 JSON 中绝不允许出现的旧版单平台业务字段/标识。
LEGACY_STRATEGY_MARKERS = (
    "五维调度策略",
    "数字人发帖内容",
    "动作清单",
    "weibo_only",
    "weibo_simulated",
)


def _assert_no_legacy_strategy_payload(testcase: unittest.TestCase, payload: object) -> None:
    """递归扫描多平台 JSON，确认其中不含任何旧版单平台业务字段或标识。"""

    found: list[str] = []

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                if key_text in LEGACY_STRATEGY_MARKERS:
                    found.append(f"{path}.{key_text}")
                _walk(value, f"{path}.{key_text}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                _walk(item, f"{path}[{index}]")
        elif isinstance(node, str):
            for marker in LEGACY_STRATEGY_MARKERS:
                if marker in node:
                    found.append(f"{path} 值包含 {marker!r}")

    _walk(payload, "$")
    testcase.assertEqual(found, [], f"多平台输出混入旧版单平台业务内容：{found}")


def _write_minimal_workspace(root: Path, *, node_count: int = 4) -> None:
    """在临时工作目录写入最小可用数据集。

    仅用于「默认输出路径」这类必须把 ``--workspace-root`` 指向临时目录的测试：
    项目自带数据集有 81MB，不能整体复制，但 DataLoader 仍需要
    ``data/raw`` 与 ``data/derived`` 下的文件才能跑完整个流程。
    """

    raw_dir = root / "data" / "raw"
    derived_dir = root / "data" / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, dict] = {}
    enriched: dict[str, dict] = {}
    #: 三个模拟平台轮流分配给最小数据集的数字人，保证每个平台都有候选。
    simulated_platforms = (
        "facebook_simulated",
        "twitter_simulated",
        "instagram_simulated",
    )
    for index in range(1, node_count + 1):
        user_id = str(index)
        profile = {
            "user_id": index,
            "user_name": f"reading_node_{index}",
            "user_followers": 2400 - index * 200,
            "user_friends": 80 + index,
            "user_interests": ["reading", "parenting", "english", "education"],
            "user_description": (
                "Focus on parent-child reading and early English learning practice."
            ),
            "platform": simulated_platforms[(index - 1) % len(simulated_platforms)],
        }
        profiles[user_id] = profile
        enriched[user_id] = {
            **profile,
            "graph_attributes": {
                "neighbor_count": node_count - 1,
                "mutual_neighbor_count": 1,
                "received_interaction_count": 6,
                "received_comment_count": 4,
                "received_repost_count": 2,
                "made_interaction_count": 3,
                "made_comment_count": 2,
                "made_repost_count": 1,
                "isolated": False,
            },
            "neighbors": [],
        }

    (raw_dir / "abc_reading_product_info.json").write_text(
        json.dumps(
            {
                "product_name": "abc_reading",
                "domain": "reading",
                "ads": "reading context",
                "influencer_ids": ["1"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (raw_dir / "abc_reading_profile.graph.anon").write_text(
        json.dumps(profiles, ensure_ascii=False), encoding="utf-8"
    )
    (raw_dir / "abc_reading_interaction.graph.anon").write_text(
        json.dumps(
            {
                user_id: [{"interact_id": 1, "interact_type": "comment"}]
                for user_id in profiles
                if user_id != "1"
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
        json.dumps(enriched, ensure_ascii=False), encoding="utf-8"
    )


def _write_minimal_hot_event_input(root: Path) -> Path:
    """写入一个 ids 与真实测试集一致的最小 hot_event 输入。"""

    input_path = root / "eval" / "hot_event_minimal.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        json.dumps(
            {
                "event_id": "hot_event_003",
                "event_title": "亲子共读与英语启蒙话题讨论",
                "event_summary": "围绕亲子共读和英语启蒙，引导家长分享陪读经验与分级阅读书单。",
                "target": "扩大亲子阅读话题触达并引导理性讨论",
                "domain": "reading",
                "opinion_variants": [
                    "陪孩子读绘本时如何兼顾英语启蒙",
                    "分级阅读书单应该怎么选",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return input_path


def _hot_event_input_path() -> Path:
    expanded = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
    if expanded.exists():
        return expanded
    return PROJECT_ROOT / "eval" / "hot_event_opinion_variants.json"


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """以给定 argv 调用 run_eval.main()，返回 (退出码, stdout, stderr)。"""

    stdout = io.StringIO()
    stderr = io.StringIO()
    with patch("sys.argv", ["run_eval.py", *argv]), redirect_stdout(
        stdout
    ), redirect_stderr(stderr):
        try:
            exit_code = int(run_eval.main())
        except SystemExit as exc:
            code = exc.code
            if code is None:
                exit_code = 0
            elif isinstance(code, int):
                exit_code = code
            else:
                exit_code = 1
    return exit_code, stdout.getvalue(), stderr.getvalue()


class RunEvalCliTest(unittest.TestCase):
    def test_arg_parser_accepts_image_inputs(self) -> None:
        parser = run_eval.build_arg_parser()
        args = parser.parse_args(["--image", "eval/image/4.png", "--event-limit", "1"])

        self.assertEqual(args.image, Path("eval/image/4.png"))
        self.assertIsNone(args.image_dir)
        self.assertEqual(args.event_limit, 1)

    def test_main_routes_image_mode_through_image_loader_and_dict_evaluation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "eval" / "image" / "4.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            input_path = root / "eval" / "hot_event_opinion_variants.json"
            input_path.write_text("[]", encoding="utf-8")

            with patch.object(
                run_eval,
                "load_hot_events",
                return_value=[{"event_id": "hot_event_005"}],
            ) as load_hot_events_mock, patch.object(
                run_eval,
                "load_image_events",
                return_value=[
                    {
                        "event_id": "hot_event_005",
                        "event_title": "Image event",
                        "source_type": "image",
                        "source_image": str(image),
                    }
                ],
            ) as load_image_events_mock, patch.object(
                run_eval,
                "run_hot_event_dict_evaluations",
                return_value=[
                    (
                        root / "eval" / "output" / "hot_event_005_strategy_output.json",
                        {"事件名称": "Image event"},
                    )
                ],
            ) as run_dict_mock, patch(
                "sys.argv",
                [
                    "run_eval.py",
                    "--workspace-root",
                    str(root),
                    "--input",
                    str(input_path),
                    "--image",
                    str(image),
                    "--disable-llm",
                ],
            ):
                exit_code = run_eval.main()

            self.assertEqual(exit_code, 0)
            load_hot_events_mock.assert_called_once_with(input_path.resolve())
            load_image_events_mock.assert_called_once()
            run_dict_mock.assert_called_once()
            self.assertFalse(run_dict_mock.call_args.kwargs["use_llm"])


class PlatformArgumentParsingTest(unittest.TestCase):
    """--platforms / --platform-counts 的解析与校验。"""

    def test_three_platforms_with_distinct_counts(self) -> None:
        specs = run_eval.build_platform_specs(
            "facebook,twitter,instagram",
            "facebook=3,twitter=5,instagram=2",
        )
        self.assertEqual(
            [(spec.platform_key, spec.requested_count) for spec in specs],
            [("facebook", 3), ("twitter", 5), ("instagram", 2)],
        )

    def test_two_platforms_only(self) -> None:
        specs = run_eval.build_platform_specs("facebook,twitter", "facebook=3,twitter=5")
        self.assertEqual(
            [(spec.platform_key, spec.requested_count) for spec in specs],
            [("facebook", 3), ("twitter", 5)],
        )

    def test_platform_keys_and_counts_are_normalized(self) -> None:
        specs = run_eval.build_platform_specs(
            " FaceBook , TWITTER ", "FACEBOOK=2 , twitter = 4 "
        )
        self.assertEqual(
            [(spec.platform_key, spec.requested_count) for spec in specs],
            [("facebook", 2), ("twitter", 4)],
        )

    def test_unsupported_platform_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs("facebook,myspace", "facebook=3,myspace=1")
        message = str(ctx.exception)
        self.assertIn("myspace", message)
        for key in ("facebook", "twitter", "instagram"):
            self.assertIn(key, message)

    def test_missing_platform_count_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs("facebook,twitter", "facebook=3")
        self.assertIn("twitter", str(ctx.exception))

    def test_missing_counts_flag_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs("facebook", None)
        self.assertIn("--platform-counts", str(ctx.exception))

    def test_extra_platform_count_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs("facebook", "facebook=3,twitter=5")
        message = str(ctx.exception)
        self.assertIn("twitter", message)
        self.assertIn("未声明", message)

    def test_duplicated_platform_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs("facebook,facebook", "facebook=3")
        self.assertIn("重复", str(ctx.exception))

    def test_duplicated_count_config_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs(
                "facebook,twitter", "facebook=3,facebook=5,twitter=1"
            )
        self.assertIn("重复", str(ctx.exception))

    def test_zero_or_negative_count_is_rejected(self) -> None:
        for value in ("0", "-2"):
            with self.subTest(value=value):
                with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
                    run_eval.build_platform_specs("facebook", f"facebook={value}")
                self.assertIn("正整数", str(ctx.exception))

    def test_non_integer_count_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs("facebook", "facebook=abc")
        self.assertIn("正整数", str(ctx.exception))

    def test_malformed_count_item_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs("facebook", "facebook:3")
        self.assertIn("platform=count", str(ctx.exception))

    def test_empty_platforms_value_is_rejected(self) -> None:
        for value in ("", "   ", ",,"):
            with self.subTest(value=value):
                with self.assertRaises(run_eval.PlatformArgumentError):
                    run_eval.build_platform_specs(value, "facebook=3")

    def test_counts_without_platforms_is_rejected(self) -> None:
        with self.assertRaises(run_eval.PlatformArgumentError) as ctx:
            run_eval.build_platform_specs(None, "facebook=3")
        self.assertIn("--platforms", str(ctx.exception))

    def test_no_platform_flags_returns_empty_specs(self) -> None:
        self.assertEqual(run_eval.build_platform_specs(None, None), [])


class DispatchPlatformCliTest(unittest.TestCase):
    """--dispatch-platform：多人分发指定平台参数的解析与校验。"""

    def test_dispatch_platform_defaults_to_none(self) -> None:
        parser = run_eval.build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.dispatch_platform)

    def test_dispatch_platform_is_parsed(self) -> None:
        parser = run_eval.build_arg_parser()
        args = parser.parse_args(["--dispatch-platform", "twitter_simulated"])
        self.assertEqual(args.dispatch_platform, "twitter_simulated")

    def test_invalid_dispatch_platform_is_rejected(self) -> None:
        exit_code, _, stderr = _run_cli(["--dispatch-platform", "myspace"])
        self.assertNotEqual(exit_code, 0)
        self.assertIn("myspace", stderr)

    def test_dispatch_platform_conflicts_with_multi_platform(self) -> None:
        exit_code, _, stderr = _run_cli(
            [
                "--dispatch-platform",
                "twitter",
                "--platforms",
                "facebook",
                "--platform-counts",
                "facebook=1",
            ]
        )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("--dispatch-platform", stderr)

    def test_dispatch_platform_conflicts_with_single_target(self) -> None:
        exit_code, _, stderr = _run_cli(
            ["--dispatch-platform", "twitter", "--target-user-id", "1"]
        )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("--dispatch-platform", stderr)


class MultiPlatformCliTest(unittest.TestCase):
    """多平台命令行路由、冲突与旧模式兼容。"""

    def test_arg_parser_exposes_platform_flags(self) -> None:
        args = run_eval.build_arg_parser().parse_args(
            [
                "--platforms",
                "facebook,twitter,instagram",
                "--platform-counts",
                "facebook=3,twitter=5,instagram=2",
            ]
        )
        self.assertEqual(args.platforms, "facebook,twitter,instagram")
        self.assertEqual(args.platform_counts, "facebook=3,twitter=5,instagram=2")
        self.assertIsNone(args.target_user_id)

    def test_platform_flags_default_to_none(self) -> None:
        args = run_eval.build_arg_parser().parse_args([])
        self.assertIsNone(args.platforms)
        self.assertIsNone(args.platform_counts)

    def _base_argv(self, root: Path) -> list[str]:
        return [
            "--workspace-root",
            str(root),
            "--input",
            str(root / "eval" / "hot_event_opinion_variants.json"),
            "--event-id",
            "hot_event_003",
            "--disable-llm",
            "--trace-dir",
            str(root / "trace"),
            "--output-dir",
            str(root / "output"),
        ]

    def test_platforms_conflict_with_target_user_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, stderr = _run_cli(
                self._base_argv(root)
                + [
                    "--platforms",
                    "facebook",
                    "--platform-counts",
                    "facebook=3",
                    "--target-user-id",
                    "110133",
                ]
            )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("--target-user-id", stderr)
        self.assertIn("互斥", stderr)

    def test_invalid_platform_exits_non_zero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, stderr = _run_cli(
                self._base_argv(root)
                + ["--platforms", "facebook,myspace", "--platform-counts", "facebook=3,myspace=1"]
            )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("myspace", stderr)

    def test_counts_without_platforms_exits_non_zero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, stderr = _run_cli(
                self._base_argv(root) + ["--platform-counts", "facebook=3"]
            )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("--platforms", stderr)

    def test_missing_platform_count_exits_non_zero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, stderr = _run_cli(
                self._base_argv(root)
                + ["--platforms", "facebook,twitter", "--platform-counts", "facebook=3"]
            )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("twitter", stderr)

    def test_platforms_conflict_with_image_input(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, stderr = _run_cli(
                self._base_argv(root)
                + [
                    "--platforms",
                    "facebook",
                    "--platform-counts",
                    "facebook=3",
                    "--image",
                    str(root / "eval" / "image" / "4.png"),
                ]
            )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("--image", stderr)

    def test_multiplatform_routes_to_multiplatform_runner(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(
                run_eval, "run_hot_event_multiplatform_evaluations", return_value=[]
            ) as mp_mock, patch.object(
                run_eval, "run_hot_event_evaluations", return_value=[]
            ) as legacy_mock:
                exit_code, _, _ = _run_cli(
                    self._base_argv(root)
                    + [
                        "--platforms",
                        "facebook,twitter",
                        "--platform-counts",
                        "facebook=3,twitter=5",
                    ]
                )

        self.assertEqual(exit_code, 0)
        mp_mock.assert_called_once()
        legacy_mock.assert_not_called()
        kwargs = mp_mock.call_args.kwargs
        self.assertEqual(
            [(spec.platform_key, spec.requested_count) for spec in kwargs["platform_specs"]],
            [("facebook", 3), ("twitter", 5)],
        )
        self.assertEqual(kwargs["event_id"], "hot_event_003")
        self.assertNotIn("target_user_id", kwargs)

    def test_automatic_mode_unchanged_without_platform_flags(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(
                run_eval, "run_hot_event_evaluations", return_value=[]
            ) as legacy_mock, patch.object(
                run_eval, "run_hot_event_multiplatform_evaluations", return_value=[]
            ) as mp_mock:
                exit_code, stdout, _ = _run_cli(self._base_argv(root))

        self.assertEqual(exit_code, 0)
        legacy_mock.assert_called_once()
        mp_mock.assert_not_called()
        self.assertEqual(legacy_mock.call_args.kwargs["target_user_id"], None)
        summary = json.loads(stdout)
        self.assertEqual(summary["outputs"], [])
        self.assertNotIn("多平台", stdout)

    def test_targeted_mode_unchanged_without_platform_flags(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(
                run_eval, "run_hot_event_evaluations", return_value=[]
            ) as legacy_mock, patch.object(
                run_eval, "run_hot_event_multiplatform_evaluations", return_value=[]
            ) as mp_mock:
                exit_code, _, _ = _run_cli(
                    self._base_argv(root) + ["--target-user-id", "110133"]
                )

        self.assertEqual(exit_code, 0)
        legacy_mock.assert_called_once()
        mp_mock.assert_not_called()
        self.assertEqual(legacy_mock.call_args.kwargs["target_user_id"], "110133")

    def test_console_summary_reports_multi_platform_counts(self) -> None:
        fake_payload = {
            "事件名称": "fake event",
            "输出格式版本": "action_schema_v5_five_dimensions_minimal",
            "五维调度策略": {"目标对象": {"选取数字人id组": ["1"]}},
            "多平台分发策略": {
                "dispatch_mode": "cross_platform",
                "event_id": "hot_event_003",
                "platforms": [
                    {
                        "platform_key": "facebook",
                        "requested_count": 3,
                        "selected_count": 2,
                        "shortfall_count": 1,
                        "selected_nodes": [{"user_id": "1"}, {"user_id": "2"}],
                    }
                ],
                "unique_user_count": 2,
                "overlapping_user_ids": ["1"],
            },
        }

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = (
                root / "output" / "hot_event_003_multiplatform_strategy_output.json"
            )
            with patch.object(
                run_eval,
                "run_hot_event_multiplatform_evaluations",
                return_value=[(output_path, fake_payload)],
            ):
                exit_code, stdout, _ = _run_cli(
                    self._base_argv(root)
                    + ["--platforms", "facebook", "--platform-counts", "facebook=3"]
                )

        self.assertEqual(exit_code, 0)
        item = json.loads(stdout)["outputs"][0]
        self.assertEqual(item["event_id"], "hot_event_003")
        self.assertEqual(item["selected_digital_human_ids"], ["1"])
        self.assertEqual(item["multi_platform"]["dispatch_mode"], "cross_platform")
        platform_summary = item["multi_platform"]["platforms"][0]
        self.assertEqual(platform_summary["requested_count"], 3)
        self.assertEqual(platform_summary["selected_count"], 2)
        self.assertEqual(platform_summary["shortfall_count"], 1)
        self.assertEqual(platform_summary["selected_user_ids"], ["1", "2"])


class OutputFileNameTest(unittest.TestCase):
    """输出文件名区分事件 ID 与多平台任务。"""

    def test_legacy_output_name(self) -> None:
        self.assertEqual(
            run_eval._event_id_from_output_path(
                Path("eval/output/hot_event_003_strategy_output.json")
            ),
            "hot_event_003",
        )

    def test_multiplatform_output_name(self) -> None:
        for name in (
            "hot_event_003_multiplatform_facebook-twitter-instagram_strategy_output.json",
            "hot_event_003_multiplatform_facebook_strategy_output.json",
            "hot_event_003_multiplatform_strategy_output.json",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    run_eval._event_id_from_output_path(Path(name)), "hot_event_003"
                )

    def test_unexpected_name_falls_back_to_stem(self) -> None:
        self.assertEqual(
            run_eval._event_id_from_output_path(Path("hot_event_003.json")),
            "hot_event_003",
        )


class MultiPlatformSmokeTest(unittest.TestCase):
    """端到端冒烟：真实数据 + 真实流水线 + 每次运行独占目录。

    ``--output-dir`` / ``--trace-dir`` 现在是**分类根目录**，实际结果落在
    ``<根>/<模式>/run_NNN/`` 内；三种模式不会再写入同一个事件目录，
    因此不再需要把上一轮 trace 复制留档。
    """

    def _base_argv(self, root: Path, input_path: Path, trace_root: Path) -> list[str]:
        return [
            "--workspace-root",
            str(PROJECT_ROOT),
            "--input",
            str(input_path),
            "--event-id",
            "hot_event_003",
            "--profile-limit",
            "40",
            "--disable-llm",
            "--output-dir",
            str(root / "output"),
            "--trace-dir",
            str(trace_root),
        ]

    def _read_json(self, path: Path) -> dict:
        self.assertTrue(path.exists(), f"缺少文件：{path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_multiplatform_smoke_writes_json_and_keeps_legacy_output(self) -> None:
        input_path = _hot_event_input_path()
        self.assertTrue(input_path.exists(), f"缺少 hot_event 测试集：{input_path}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "output"
            trace_root = root / "trace"

            # 第一次运行：未指定 --platforms / --target-user-id → multi_user
            legacy_code, _, _ = _run_cli(
                self._base_argv(root, input_path, trace_root)
            )
            self.assertEqual(legacy_code, 0)
            legacy_run_dir = output_root / "multi_user" / "run_001"
            legacy_path = legacy_run_dir / "hot_event_003_strategy_output.json"
            self.assertTrue(legacy_path.exists(), "旧 automatic 模式输出文件缺失")
            self.assertTrue(
                (legacy_run_dir / "run_summary.json").exists(),
                "运行摘要必须落在本次运行目录内",
            )
            first_legacy_payload = json.loads(
                legacy_path.read_text(encoding="utf-8")
            )
            # 多人分发必须保持旧版结构，且不写多平台字段
            self.assertEqual(
                first_legacy_payload["输出格式版本"],
                "action_schema_v5_five_dimensions_minimal",
            )
            self.assertIn("五维调度策略", first_legacy_payload)
            self.assertNotIn("多平台分发策略", first_legacy_payload)
            legacy_trace_dir = (
                trace_root / "multi_user" / "run_001" / "trace" / "hot_event_003"
            )
            legacy_trace = self._read_json(legacy_trace_dir / "07_final_output.json")
            self.assertEqual(legacy_trace["run_label"], "automatic")
            # 每次运行独占目录，因此同一 trace 目录内不会再出现历史归档
            self.assertFalse((legacy_trace_dir / "runs").exists())

            # 第二次运行：启用 --platforms → multi_platform，编号从各自的分类目录独立开始
            mp_code, mp_stdout, _ = _run_cli(
                self._base_argv(root, input_path, trace_root)
                + [
                    "--platforms",
                    "facebook,twitter,instagram",
                    "--platform-counts",
                    "facebook=3,twitter=5,instagram=2",
                ]
            )
            self.assertEqual(mp_code, 0)

            mp_run_dir = output_root / "multi_platform" / "run_001"
            mp_path = (
                mp_run_dir
                / "hot_event_003_multiplatform_facebook-twitter-instagram_strategy_output.json"
            )
            self.assertTrue(mp_path.exists(), "多平台输出文件缺失")
            mp_trace_dir = (
                trace_root / "multi_platform" / "run_001" / "trace" / "hot_event_003"
            )
            mp_trace = self._read_json(mp_trace_dir / "07_final_output.json")
            self.assertEqual(
                mp_trace["run_label"], "multiplatform_facebook-twitter-instagram"
            )
            # 不同模式落在不同分类目录里，互不覆盖
            self.assertTrue(legacy_path.exists(), "多平台运行覆盖了旧 automatic 输出")
            self.assertTrue(
                (legacy_trace_dir / "07_final_output.json").exists(),
                "多平台运行覆盖了旧 automatic trace",
            )
            # 每个分类目录根部只允许出现 run_NNN，不得散落结果文件
            self.assertEqual(
                sorted(path.name for path in (output_root / "multi_user").iterdir()),
                ["run_001"],
            )

            # 第三次运行：同一模式再跑一次 → 编号递增到 run_002
            mp_two_code, mp_two_stdout, _ = _run_cli(
                self._base_argv(root, input_path, trace_root)
                + [
                    "--platforms",
                    "facebook,twitter",
                    "--platform-counts",
                    "facebook=3,twitter=5",
                ]
            )
            self.assertEqual(mp_two_code, 0)
            mp_two_run_dir = output_root / "multi_platform" / "run_002"
            mp_two_path = (
                mp_two_run_dir
                / "hot_event_003_multiplatform_facebook-twitter_strategy_output.json"
            )
            self.assertTrue(mp_two_path.exists(), "两个平台的多平台输出文件缺失")
            self.assertTrue(mp_path.exists(), "两个平台的多平台运行覆盖了三个平台的结果")
            self.assertEqual(
                sorted(
                    path.name for path in (output_root / "multi_platform").iterdir()
                ),
                ["run_001", "run_002"],
                "同一分类目录下必须按 run_NNN 独立编号递增",
            )
            self.assertEqual(
                mp_two_stdout and json.loads(mp_two_stdout)["run"]["run_id"], "run_002"
            )
            mp_two_payload = self._read_json(mp_two_path)
            self.assertEqual(
                [
                    platform["platform_key"]
                    for platform in mp_two_payload["多平台分发策略"]["platforms"]
                ],
                ["facebook", "twitter"],
            )
            # 旧结果未被覆盖，仍保持旧版单平台输出结构
            self.assertTrue(legacy_path.exists(), "多平台运行覆盖了旧输出文件")
            legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
            self.assertNotIn("多平台分发策略", legacy_payload)
            self.assertEqual(
                set(legacy_payload.keys()),
                {"事件名称", "五维调度策略", "输出格式版本"},
            )
            self.assertEqual(
                legacy_payload["输出格式版本"],
                "action_schema_v5_five_dimensions_minimal",
            )
            for dimension in ("目标对象", "时间", "频率", "平台", "内容"):
                self.assertIn(dimension, legacy_payload["五维调度策略"])
            self.assertIn(
                "数字人发帖内容", legacy_payload["五维调度策略"]["内容"]
            )
            self.assertIn("动作清单", legacy_payload["五维调度策略"]["内容"])

            payload = json.loads(mp_path.read_text(encoding="utf-8"))
            # 多平台输出由 build_multiplatform_output 独立构造：
            # 只有事件信息、多平台版本标识与跨平台调度结果，不含旧版五维调度策略。
            self.assertEqual(
                payload["输出格式版本"], MULTIPLATFORM_OUTPUT_FORMAT_VERSION
            )
            self.assertEqual(
                set(payload.keys()),
                {"事件名称", "多平台分发策略", "输出格式版本"},
            )
            self.assertNotIn("五维调度策略", payload)
            _assert_no_legacy_strategy_payload(self, payload)

            block = payload["多平台分发策略"]
            self.assertEqual(block["dispatch_mode"], "cross_platform")
            self.assertEqual(block["event_id"], "hot_event_003")
            self.assertIn("尚未在任何真实平台发布", block["内容草案说明"])
            self.assertIn("旧版单平台模拟分发", block["结果范围说明"])
            self.assertTrue(block["warnings"])
            self.assertTrue(block["policy_disclaimer"])
            self.assertTrue(block["fit_limitations"])
            self.assertGreaterEqual(block["unique_user_count"], 1)

            platforms = block["platforms"]
            self.assertEqual(
                [platform["platform_key"] for platform in platforms],
                ["facebook", "twitter", "instagram"],
            )
            self.assertEqual(
                [platform["requested_count"] for platform in platforms], [3, 5, 2]
            )
            # 总人数统计不因输出重构而丢失
            self.assertEqual(
                block["total_requested_count"],
                sum(platform["requested_count"] for platform in platforms),
            )
            self.assertEqual(
                block["total_selected_count"],
                sum(platform["selected_count"] for platform in platforms),
            )
            tier_totals = block["total_selected_safety_tiers"]
            self.assertEqual(
                sum(tier_totals.values()), block["total_selected_count"]
            )

            for platform in platforms:
                self.assertEqual(
                    platform["shortfall_count"],
                    max(0, platform["requested_count"] - platform["selected_count"]),
                    platform["platform_key"],
                )
                self.assertLessEqual(
                    platform["selected_count"], platform["requested_count"]
                )
                self.assertTrue(platform["warnings"])
                self.assertTrue(platform["platform_roles"])
                self.assertTrue(platform["content_plan"]["sections"])
                self.assertTrue(platform["content_plan"]["content_modes"])
                self.assertTrue(platform["content_plan"]["guardrails"])

                # 真实数据上的时间计划必须落在活动窗口内
                time_plan = platform["time_plan"]
                self.assertTrue(time_plan["stages"])
                self.assertEqual(
                    time_plan["campaign_window_hours"],
                    block["campaign_window_hours"],
                    platform["platform_key"],
                )
                self.assertTrue(time_plan["stage_schedule_note"])
                for stage in time_plan["stages"]:
                    ctx = f"{platform['platform_key']} {stage['stage_key']}"
                    self.assertGreaterEqual(stage["start_hour"], 0, ctx)
                    self.assertGreaterEqual(
                        stage["end_hour"], stage["start_hour"], ctx
                    )
                    self.assertLessEqual(
                        stage["end_hour"],
                        time_plan["campaign_window_hours"],
                        ctx,
                    )
                    self.assertEqual(
                        stage["window"],
                        f"T+{stage['start_hour']}h - T+{stage['end_hour']}h",
                        ctx,
                    )

                user_ids = [node["user_id"] for node in platform["selected_nodes"]]
                self.assertEqual(
                    len(user_ids), len(set(user_ids)), platform["platform_key"]
                )

                # 合格 / 待人工复核 / 不合格三层必须可机读，且与逐节点标记一致
                tiers = platform["selected_safety_tiers"]
                self.assertEqual(
                    sum(tiers.values()),
                    platform["selected_count"],
                    platform["platform_key"],
                )
                self.assertEqual(
                    tiers["safe"],
                    sum(
                        1
                        for node in platform["selected_nodes"]
                        if node["safety_tier"] == "safe"
                    ),
                    platform["platform_key"],
                )
                self.assertEqual(
                    platform["safe_shortfall_count"],
                    max(0, platform["requested_count"] - tiers["safe"]),
                    platform["platform_key"],
                )

                for node in platform["selected_nodes"]:
                    self.assertTrue(node["platform_role"])
                    self.assertIn(
                        node["safety_tier"],
                        {"safe", "review_required", "not_eligible"},
                    )
                    self.assertIn("final_score", node)
                    self.assertIn("platform_fit_score", node)
                    self.assertIn("platform_dispatch_score", node)
                    self.assertTrue(node["fit_basis"])
                    self.assertIn("undeclared_inputs", node)
                    # 不合格节点绝不能被标记成合格/安全
                    if node["safety_tier"] == "not_eligible":
                        self.assertFalse(node["eligible"], node["user_id"])
                    if node["risk_level"] == "high":
                        self.assertFalse(node["eligible"], node["user_id"])

            by_key = {platform["platform_key"]: platform for platform in platforms}
            facebook_sections = [
                section["section_key"]
                for section in by_key["facebook"]["content_plan"]["sections"]
            ]
            twitter_sections = [
                section["section_key"]
                for section in by_key["twitter"]["content_plan"]["sections"]
            ]
            instagram_sections = [
                section["section_key"]
                for section in by_key["instagram"]["content_plan"]["sections"]
            ]
            self.assertEqual(
                {"post_theme", "core_viewpoint", "discussion_question", "risk_note"}
                - set(facebook_sections),
                set(),
            )
            self.assertEqual(
                {
                    "thread_opening",
                    "thread_body",
                    "evidence_note",
                    "reply_prompt",
                }
                - set(twitter_sections),
                set(),
            )
            self.assertEqual(
                {
                    "carousel_theme",
                    "carousel_frames",
                    "caption_requirements",
                    "stories_interaction",
                }
                - set(instagram_sections),
                set(),
            )
            self.assertEqual(
                len(
                    {
                        tuple(facebook_sections),
                        tuple(twitter_sections),
                        tuple(instagram_sections),
                    }
                ),
                3,
                "三个平台的内容模板必须真正不同",
            )

            # 每个数字人只归属唯一模拟平台：跨平台重复选人必须为空
            self.assertEqual(
                block["overlapping_user_ids"],
                [],
                "同一数字人不应被分配到多个平台",
            )
            # 平台适配分随平台权重变化，而不是 final_score 的副本
            fit_pairs = [
                (node["final_score"], node["platform_fit_score"])
                for platform in platforms
                for node in platform["selected_nodes"]
            ]
            self.assertTrue(fit_pairs)
            self.assertTrue(
                any(abs(final - fit) > 1e-9 for final, fit in fit_pairs),
                f"平台适配分不应等于 final_score 的副本：{fit_pairs}",
            )

            # 三次运行分属三个独立目录，每一轮的 trace 都完整可回溯
            latest_trace_dir = (
                trace_root / "multi_platform" / "run_002" / "trace" / "hot_event_003"
            )
            latest_trace = self._read_json(latest_trace_dir / "07_final_output.json")
            self.assertEqual(latest_trace["run_label"], "multiplatform_facebook-twitter")
            self.assertIn("多平台分发策略", latest_trace["payload"])
            # trace 里留存的最终产物也必须已经是「无旧版策略」的多平台 JSON，
            # 否则 trace 与最终业务结果会不一致。
            self.assertNotIn("五维调度策略", latest_trace["payload"])
            _assert_no_legacy_strategy_payload(self, latest_trace["payload"])

            run_trace_labels = {
                "multi_user/run_001": self._read_json(
                    trace_root
                    / "multi_user"
                    / "run_001"
                    / "trace"
                    / "hot_event_003"
                    / "07_final_output.json"
                )["run_label"],
                "multi_platform/run_001": mp_trace["run_label"],
                "multi_platform/run_002": latest_trace["run_label"],
            }
            self.assertEqual(
                run_trace_labels,
                {
                    "multi_user/run_001": "automatic",
                    "multi_platform/run_001": (
                        "multiplatform_facebook-twitter-instagram"
                    ),
                    "multi_platform/run_002": "multiplatform_facebook-twitter",
                },
            )
            # 每次运行的产物都必须落在自己的 run_NNN 目录内，不得散落在分类根目录
            for category, run_id in (
                ("multi_user", "run_001"),
                ("multi_platform", "run_001"),
                ("multi_platform", "run_002"),
            ):
                run_dir = output_root / category / run_id
                self.assertTrue(run_dir.is_dir(), run_dir)
                self.assertTrue(
                    list(run_dir.glob("*_strategy_output.json")),
                    f"{category}/{run_id} 缺少策略 JSON",
                )
                self.assertTrue(
                    (run_dir / "run_summary.json").exists(),
                    f"{category}/{run_id} 缺少运行摘要",
                )
                # 显式给出 --trace-dir 时，该轮的 trace 独占对应的 run_NNN，
                # 并且运行目录内不会再留一份重复 trace。
                self.assertTrue(
                    (
                        trace_root / category / run_id / "trace" / "hot_event_003"
                    ).is_dir(),
                    f"{category}/{run_id} 的 trace 不在本轮 run_NNN 内",
                )
                self.assertFalse(
                    (run_dir / "trace").exists(),
                    f"{category}/{run_id} 出现了重复 trace 副本",
                )

            summary_item = json.loads(mp_stdout)["outputs"][0]
            self.assertEqual(summary_item["event_id"], "hot_event_003")
            self.assertIn("multi_platform", summary_item)
            self.assertGreater(summary_item["multi_platform"]["unique_user_count"], 0)
            self.assertEqual(
                summary_item["multi_platform"]["total_selected_safety_tiers"]["safe"]
                + summary_item["multi_platform"]["total_selected_safety_tiers"][
                    "review_required"
                ]
                + summary_item["multi_platform"]["total_selected_safety_tiers"][
                    "not_eligible"
                ],
                3 + 5 + 2,
            )
            self.assertEqual(
                summary_item["multi_platform"]["total_safe_shortfall_count"],
                sum(
                    platform_summary["safe_shortfall_count"]
                    for platform_summary in summary_item["multi_platform"]["platforms"]
                ),
            )

    def test_short_campaign_window_keeps_time_plan_legal(self) -> None:
        """真实入口 + 1 小时活动窗口：JSON 正常生成且时间计划不越界。"""

        input_path = _hot_event_input_path()
        self.assertTrue(input_path.exists(), f"缺少 hot_event 测试集：{input_path}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, _ = _run_cli(
                self._base_argv(root, input_path, root / "trace_short_window")
                + [
                    "--campaign-window-hours",
                    "1",
                    "--platforms",
                    "facebook,twitter,instagram",
                    "--platform-counts",
                    "facebook=3,twitter=5,instagram=2",
                ]
            )
            self.assertEqual(exit_code, 0)

            mp_path = (
                root
                / "output"
                / "multi_platform"
                / "run_001"
                / "hot_event_003_multiplatform_facebook-twitter-instagram_strategy_output.json"
            )
            self.assertTrue(mp_path.exists(), "1 小时窗口下未生成多平台 JSON")
            payload = json.loads(mp_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload.keys()),
                {"事件名称", "多平台分发策略", "输出格式版本"},
            )
            _assert_no_legacy_strategy_payload(self, payload)
            block = payload["多平台分发策略"]
            self.assertEqual(block["campaign_window_hours"], 1)

            for platform in block["platforms"]:
                plan = platform["time_plan"]
                self.assertEqual(plan["campaign_window_hours"], 1)
                self.assertTrue(plan["stage_schedule_note"])
                for stage in plan["stages"]:
                    ctx = f"{platform['platform_key']} {stage['stage_key']}"
                    self.assertGreaterEqual(stage["start_hour"], 0, ctx)
                    self.assertGreaterEqual(
                        stage["end_hour"], stage["start_hour"], ctx
                    )
                    self.assertLessEqual(stage["end_hour"], 1, ctx)
                    self.assertNotEqual(stage["window"], "T+1h - T+2h", ctx)

    def test_multi_user_mode_keeps_legacy_payload_only(self) -> None:
        """回归：多人分发（未指定平台参数）仍是旧版五维调度策略，不混入多平台结果。"""

        input_path = _hot_event_input_path()
        self.assertTrue(input_path.exists(), f"缺少 hot_event 测试集：{input_path}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, _ = _run_cli(
                self._base_argv(root, input_path, root / "trace_multi_user")
            )
            self.assertEqual(exit_code, 0)

            output_path = (
                root / "output" / "multi_user" / "run_001"
                / "hot_event_003_strategy_output.json"
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload.keys()),
                {"事件名称", "五维调度策略", "输出格式版本"},
            )
            self.assertEqual(
                payload["输出格式版本"], "action_schema_v5_five_dimensions_minimal"
            )
            self.assertNotIn("多平台分发策略", payload)

            strategy = payload["五维调度策略"]
            for dimension in ("目标对象", "时间", "频率", "平台", "内容"):
                self.assertIn(dimension, strategy)
            # 旧版多人分发的选人、帖子内容与动作清单必须原样保留
            self.assertTrue(strategy["目标对象"]["选取数字人id组"])
            self.assertTrue(strategy["内容"]["数字人发帖内容"])
            self.assertTrue(strategy["内容"]["动作清单"])
            self.assertNotIn("定向分发诊断", payload)

    def test_multi_user_dispatch_platform_restricts_selection(self) -> None:
        """多人分发指定平台后，选中的数字人必须全部来自该平台。"""

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_minimal_workspace(root)
            input_path = _write_minimal_hot_event_input(root)
            exit_code, _, _ = _run_cli(
                [
                    "--workspace-root",
                    str(root),
                    "--input",
                    str(input_path),
                    "--event-id",
                    "hot_event_003",
                    "--dispatch-platform",
                    "facebook_simulated",
                    "--max-selected-nodes",
                    "5",
                    "--disable-llm",
                ]
            )
            self.assertEqual(exit_code, 0)

            output_path = (
                root / "eval" / "output" / "multi_user" / "run_001"
                / "hot_event_003_strategy_output.json"
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            strategy = payload["五维调度策略"]
            self.assertTrue(strategy["目标对象"]["选取数字人id组"])
            self.assertEqual(strategy["平台"]["平台模式"], "facebook_only")
            for item in strategy["平台"]["数字人平台分发"]:
                self.assertEqual(item["分发平台"], "facebook_simulated")

    def test_targeted_mode_still_works_without_platform_flags(self) -> None:
        input_path = _hot_event_input_path()
        self.assertTrue(input_path.exists(), f"缺少 hot_event 测试集：{input_path}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            exit_code, _, _ = _run_cli(
                self._base_argv(root, input_path, root / "trace_targeted")
                + ["--target-user-id", "82848"]
            )
            self.assertEqual(exit_code, 0)

            output_path = (
                root / "output" / "single_target" / "run_001"
                / "hot_event_003_strategy_output.json"
            )
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertNotIn("多平台分发策略", payload)
            # 回归：定向分发同样保留旧版五维调度策略结构
            self.assertEqual(
                set(payload.keys()),
                {"事件名称", "定向分发诊断", "五维调度策略", "输出格式版本"},
            )
            self.assertIn("五维调度策略", payload)
            self.assertIn(
                "数字人发帖内容", payload["五维调度策略"]["内容"]
            )
            self.assertIn("动作清单", payload["五维调度策略"]["内容"])
            diagnostic = payload.get("定向分发诊断", {})
            self.assertEqual(diagnostic.get("selection_mode"), "targeted")
            self.assertEqual(diagnostic.get("target_user_id"), "82848")
            # 单人分发结果必须能看到被选中的人属于哪个模拟平台
            self.assertIn(
                diagnostic.get("target_platform"),
                ("facebook", "instagram", "twitter"),
            )

    def test_default_run_keeps_everything_inside_one_run_dir(self) -> None:
        """不指定 --output-dir/--trace-dir 时：默认落到 eval/output/<模式>/run_NNN/，
        且 trace 与摘要都在同一个运行目录内。

        这里必须把 ``--workspace-root`` 指向临时目录（否则测不出默认路径），
        因此用最小数据集让流水线真实跑完。
        """

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_minimal_workspace(root)
            input_path = _write_minimal_hot_event_input(root)
            exit_code, stdout, _ = _run_cli(
                [
                    "--workspace-root",
                    str(root),
                    "--input",
                    str(input_path),
                    "--event-id",
                    "hot_event_003",
                    "--profile-limit",
                    "40",
                    "--disable-llm",
                ]
            )
            self.assertEqual(exit_code, 0)

            summary = json.loads(stdout)
            run_info = summary["run"]
            self.assertEqual(run_info["mode"], "multi_user")
            self.assertEqual(run_info["run_id"], "run_001")
            run_dir = root / "eval" / "output" / "multi_user" / "run_001"
            self.assertEqual(Path(run_info["run_dir"]).resolve(), run_dir.resolve())
            # trace 默认在运行目录内部
            self.assertEqual(
                Path(run_info["trace_dir"]).resolve(),
                (run_dir / "trace").resolve(),
            )
            self.assertTrue(run_dir.is_dir(), run_dir)
            self.assertTrue((run_dir / "run_summary.json").exists())
            self.assertIn(
                "hot_event_003_strategy_output.json",
                [path.name for path in run_dir.glob("*.json")],
            )
            self.assertTrue(
                (run_dir / "trace" / "hot_event_003" / "07_final_output.json").exists()
            )
            # 只跑了 multi_user，因此 eval/output 下只应有该分类目录，
            # 且分类目录根部不得出现散落的结果文件
            self.assertEqual(
                sorted(path.name for path in (root / "eval" / "output").iterdir()),
                ["multi_user"],
            )
            self.assertEqual(
                sorted(path.name for path in (root / "eval" / "output" / "multi_user").iterdir()),
                ["run_001"],
            )

    def test_runs_are_classified_by_mode_and_numbered_independently(self) -> None:
        """三种模式必须落到各自的分类目录，且编号互不干扰。"""

        cases = [
            ([], "multi_user", 1),
            ([], "multi_user", 2),
            (["--target-user-id", "82848"], "single_target", 1),
            (
                ["--platforms", "facebook", "--platform-counts", "facebook=1"],
                "multi_platform",
                1,
            ),
        ]

        input_path = _hot_event_input_path()
        self.assertTrue(input_path.exists(), f"缺少 hot_event 测试集：{input_path}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "output"
            trace_root = root / "trace"

            for extra_args, expected_category, expected_number in cases:
                with self.subTest(category=expected_category, number=expected_number):
                    exit_code, stdout, _ = _run_cli(
                        self._base_argv(root, input_path, trace_root) + extra_args
                    )
                    self.assertEqual(exit_code, 0)
                    summary = json.loads(stdout)
                    run_info = summary["run"]
                    self.assertEqual(run_info["mode"], expected_category)
                    run_id = f"run_{expected_number:03d}"
                    self.assertEqual(run_info["run_id"], run_id)
                    expected_run_dir = (
                        output_root / expected_category / run_id
                    ).resolve()
                    self.assertEqual(Path(run_info["run_dir"]).resolve(), expected_run_dir)
                    # trace 默认落在本次运行目录内部
                    expected_trace_dir = (
                        trace_root / expected_category / run_id / "trace"
                    ).resolve()
                    self.assertEqual(
                        Path(run_info["trace_dir"]).resolve(), expected_trace_dir
                    )
                    for item in summary["outputs"]:
                        # 结果 JSON 与摘要都在同一个运行目录内
                        self.assertEqual(
                            Path(item["json_output_path"]).resolve().parent,
                            expected_run_dir,
                        )
                        self.assertEqual(
                            Path(item["trace_output_dir"]).resolve().parent,
                            expected_trace_dir,
                        )


if __name__ == "__main__":
    unittest.main()
