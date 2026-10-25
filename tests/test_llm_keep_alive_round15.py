"""第十五轮：Ollama 模型常驻（``keep_alive``）与启动预热的回归测试。

背景（本机实测）：Ollama 默认只把模型留在内存 5 分钟，空闲超时后卸载，
下一次请求要多花 3.5~6.5s 重新加载。本轮的做法是**请求级 keep_alive +
服务启动预热**，不改任何选人逻辑。必须保证：

* 不配置时请求体与过去**完全一致**（云端端点看不到这个 Ollama 扩展参数）；
* 配置了也**只对 Ollama 端点**发送；
* 预热默认关闭，且失败不静默吞异常。
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from influence_strategy import llm_client as llm_client_module
from influence_strategy.llm_client import (
    OpenAICompatibleLLMClient,
    _is_ollama_endpoint,
    _resolve_keep_alive,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _CapturingUrlOpen:
    """替换 ``urllib.request.urlopen``，记下请求体并返回一个合法响应。"""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, request, **kwargs):
        self.payloads.append(json.loads((request.data or b"{}").decode("utf-8")))
        return _FakeResponse(
            {"choices": [{"message": {"content": '{"ok":true}'}}], "usage": {}}
        )


class KeepAlivePayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._real_urlopen = llm_client_module.urllib.request.urlopen
        self.capture = _CapturingUrlOpen()
        llm_client_module.urllib.request.urlopen = self.capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        llm_client_module.urllib.request.urlopen = self._real_urlopen  # type: ignore[assignment]

    def _client(self, *, keep_alive=None, base_url="http://localhost:11434/v1"):
        return OpenAICompatibleLLMClient(
            api_key="ollama",
            base_url=base_url,
            model="qwen3:8b",
            provider="openai",
            keep_alive=keep_alive,
        )

    def test_no_keep_alive_by_default(self) -> None:
        """默认不配置 → 请求体里不出现 keep_alive（与改动前完全一致）。"""
        self._client().generate_json(system_prompt="s", user_prompt="u")
        self.assertNotIn("keep_alive", self.capture.payloads[-1])

    def test_keep_alive_sent_to_ollama_endpoint(self) -> None:
        self._client(keep_alive=-1).generate_json(system_prompt="s", user_prompt="u")
        self.assertEqual(self.capture.payloads[-1]["keep_alive"], -1)

    def test_keep_alive_not_sent_to_cloud_endpoint(self) -> None:
        """云端 OpenAI 兼容端点不认识 keep_alive，不能给它带上。"""
        self._client(
            keep_alive=-1,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).generate_json(system_prompt="s", user_prompt="u")
        self.assertNotIn("keep_alive", self.capture.payloads[-1])

    def test_keep_alive_also_applies_to_image_requests(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "x.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            self._client(keep_alive=1800).generate_json_with_image(
                system_prompt="s", user_prompt="u", image_path=image
            )
        self.assertEqual(self.capture.payloads[-1]["keep_alive"], 1800)


class KeepAliveResolutionTest(unittest.TestCase):
    def test_resolves_int_string_and_duration(self) -> None:
        self.assertEqual(_resolve_keep_alive({"LLM_KEEP_ALIVE": "-1"}, ("LLM_KEEP_ALIVE",)), -1)
        self.assertEqual(_resolve_keep_alive({"LLM_KEEP_ALIVE": "1800"}, ("LLM_KEEP_ALIVE",)), 1800)
        self.assertEqual(_resolve_keep_alive({"LLM_KEEP_ALIVE": "30m"}, ("LLM_KEEP_ALIVE",)), "30m")

    def test_missing_key_returns_none(self) -> None:
        self.assertIsNone(_resolve_keep_alive({}, ("LLM_KEEP_ALIVE",)))
        self.assertIsNone(_resolve_keep_alive({"LLM_KEEP_ALIVE": "  "}, ("LLM_KEEP_ALIVE",)))

    def test_endpoint_detection(self) -> None:
        self.assertTrue(_is_ollama_endpoint("http://localhost:11434/v1", "openai"))
        self.assertTrue(_is_ollama_endpoint("http://127.0.0.1:8000", "ollama"))
        self.assertFalse(_is_ollama_endpoint("https://api.openai.com/v1", "openai"))


class FromEnvFilesTest(unittest.TestCase):
    def test_reads_keep_alive_from_env_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "LLM_API_KEY=ollama\n"
                "LLM_BASE_URL=http://localhost:11434/v1\n"
                "LLM_MODEL=qwen3:8b\n"
                "LLM_KEEP_ALIVE = -1\n",
                encoding="utf-8",
            )
            client = OpenAICompatibleLLMClient.from_env_files(root)
        self.assertIsNotNone(client)
        self.assertEqual(client.keep_alive, -1)  # type: ignore[union-attr]

    def test_absent_keep_alive_stays_none(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "LLM_API_KEY=ollama\nLLM_BASE_URL=http://localhost:11434/v1\nLLM_MODEL=qwen3:8b\n",
                encoding="utf-8",
            )
            client = OpenAICompatibleLLMClient.from_env_files(root)
        self.assertIsNotNone(client)
        self.assertIsNone(client.keep_alive)  # type: ignore[union-attr]


class WarmupTest(unittest.TestCase):
    """启动预热：默认关闭，且不做静默吞异常。"""

    def setUp(self) -> None:
        from influence_strategy import web_app as web_app_module

        self.web_app = web_app_module
        self._saved = {
            k: os.environ.get(k)
            for k in ("LLM_WARMUP_ON_STARTUP", "LLM_KEEP_ALIVE")
        }

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _without_env_file(self) -> None:
        """把项目 .env 隔离掉：项目当前在 .env 里开了预热，会干扰"默认值"断言。"""
        self._real_read_env_values = self.web_app.read_env_values
        self.web_app.read_env_values = lambda _root: {}  # type: ignore[assignment]

    def _restore_env_file(self) -> None:
        if hasattr(self, "_real_read_env_values"):
            self.web_app.read_env_values = self._real_read_env_values  # type: ignore[assignment]

    def test_env_flag_reads_env_file_when_process_env_absent(self) -> None:
        """进程环境没有时，要能读到项目 .env 里的开关（否则 .env 的配置白写）。"""
        os.environ.pop("LLM_WARMUP_ON_STARTUP", None)
        self.web_app.read_env_values = lambda _root: {"LLM_WARMUP_ON_STARTUP": "true"}  # type: ignore[assignment]
        try:
            self.assertIs(self.web_app._env_flag("LLM_WARMUP_ON_STARTUP", default=False), True)
        finally:
            self._restore_env_file()

    def test_process_env_overrides_env_file(self) -> None:
        self.web_app.read_env_values = lambda _root: {"LLM_WARMUP_ON_STARTUP": "true"}  # type: ignore[assignment]
        os.environ["LLM_WARMUP_ON_STARTUP"] = "false"
        try:
            self.assertIs(self.web_app._env_flag("LLM_WARMUP_ON_STARTUP", default=False), False)
        finally:
            self._restore_env_file()

    def test_env_flag_defaults(self) -> None:
        self._without_env_file()
        try:
            os.environ.pop("LLM_WARMUP_ON_STARTUP", None)
            self.assertIs(self.web_app._env_flag("LLM_WARMUP_ON_STARTUP", default=False), False)
            os.environ["LLM_WARMUP_ON_STARTUP"] = "false"
            self.assertIs(self.web_app._env_flag("LLM_WARMUP_ON_STARTUP", default=False), False)
            os.environ["LLM_WARMUP_ON_STARTUP"] = "true"
            self.assertIs(self.web_app._env_flag("LLM_WARMUP_ON_STARTUP", default=False), True)
        finally:
            self._restore_env_file()

    def test_warmup_disabled_by_default(self) -> None:
        """默认关闭 → 不会在服务启动时偷偷发出请求。"""
        self._without_env_file()
        os.environ.pop("LLM_WARMUP_ON_STARTUP", None)
        calls: list[str] = []

        class _Boom:
            @staticmethod
            def from_env_files(_root):
                calls.append("from_env_files")
                raise AssertionError("默认关闭时不应构造客户端")

        original = self.web_app.OpenAICompatibleLLMClient
        self.web_app.OpenAICompatibleLLMClient = _Boom  # type: ignore[assignment]
        try:
            self.web_app._warmup_llm_model()
        finally:
            self.web_app.OpenAICompatibleLLMClient = original  # type: ignore[assignment]
            self._restore_env_file()
        self.assertEqual(calls, [])

    def test_warmup_skipped_without_keep_alive(self) -> None:
        """没配 keep_alive 时预热没有意义（5 分钟后照样卸载）。"""
        os.environ["LLM_WARMUP_ON_STARTUP"] = "true"
        calls: list[str] = []

        class _Client:
            keep_alive = None

            @staticmethod
            def from_env_files(_root):
                return _Client()

            def generate_json(self, **_kwargs):
                calls.append("generate_json")

        original = self.web_app.OpenAICompatibleLLMClient
        self.web_app.OpenAICompatibleLLMClient = _Client  # type: ignore[assignment]
        try:
            self.web_app._warmup_llm_model()
        finally:
            self.web_app.OpenAICompatibleLLMClient = original  # type: ignore[assignment]
        self.assertEqual(calls, [])

    def test_warmup_runs_when_enabled(self) -> None:
        os.environ["LLM_WARMUP_ON_STARTUP"] = "true"
        calls: list[dict] = []

        class _Client:
            keep_alive = -1

            @staticmethod
            def from_env_files(_root):
                return _Client()

            def generate_json(self, **kwargs):
                calls.append(kwargs)

        original = self.web_app.OpenAICompatibleLLMClient
        self.web_app.OpenAICompatibleLLMClient = _Client  # type: ignore[assignment]
        try:
            self.web_app._warmup_llm_model()
        finally:
            self.web_app.OpenAICompatibleLLMClient = original  # type: ignore[assignment]
        self.assertEqual(len(calls), 1, "启用后应只发一次极小预热请求")

    def test_warmup_does_not_swallow_errors(self) -> None:
        """预热失败必须抛出去（由启动线程打印），不能静默通过。"""
        os.environ["LLM_WARMUP_ON_STARTUP"] = "true"

        class _Client:
            keep_alive = -1

            @staticmethod
            def from_env_files(_root):
                return _Client()

            def generate_json(self, **_kwargs):
                raise RuntimeError("ollama unreachable")

        original = self.web_app.OpenAICompatibleLLMClient
        self.web_app.OpenAICompatibleLLMClient = _Client  # type: ignore[assignment]
        try:
            with self.assertRaises(RuntimeError):
                self.web_app._warmup_llm_model()
        finally:
            self.web_app.OpenAICompatibleLLMClient = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
