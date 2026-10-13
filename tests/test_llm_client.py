from __future__ import annotations

import base64
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.llm_client import OpenAICompatibleLLMClient


class LLMClientTest(unittest.TestCase):
    def test_from_env_files_supports_generic_llm_keys_for_deepseek(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=test-key",
                        "LLM_BASE_URL=https://api.deepseek.com",
                        "LLM_MODEL=deepseek-v4-flash",
                    ]
                ),
                encoding="utf-8",
            )

            client = OpenAICompatibleLLMClient.from_env_files(root)

            self.assertIsNotNone(client)
            assert client is not None
            self.assertEqual(client.provider, "deepseek")
            self.assertEqual(client.model, "deepseek-v4-flash")
            self.assertEqual(client.base_url, "https://api.deepseek.com")
            self.assertEqual(
                client.describe(),
                {
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com",
                },
            )


class VisionLLMClientTest(unittest.TestCase):
    def test_from_vision_env_files_prefers_vision_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=text-key",
                        "LLM_BASE_URL=https://text.example/v1",
                        "LLM_MODEL=text-model",
                        "VISION_LLM_API_KEY=vision-key",
                        "VISION_LLM_BASE_URL=https://vision.example/v1",
                        "VISION_LLM_MODEL=qwen3.7-plus",
                    ]
                ),
                encoding="utf-8",
            )

            client = OpenAICompatibleLLMClient.from_vision_env_files(root)

            self.assertIsNotNone(client)
            assert client is not None
            self.assertEqual(client.api_key, "vision-key")
            self.assertEqual(client.base_url, "https://vision.example/v1")
            self.assertEqual(client.model, "qwen3.7-plus")
            self.assertEqual(client.provider, "dashscope")

    def test_from_vision_env_files_falls_back_to_text_llm_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "DASHSCOPE_API_KEY=dashscope-key",
                        "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "MODEL_NAME=qwen3.7-plus",
                    ]
                ),
                encoding="utf-8",
            )

            client = OpenAICompatibleLLMClient.from_vision_env_files(root)

            self.assertIsNotNone(client)
            assert client is not None
            self.assertEqual(client.api_key, "dashscope-key")
            self.assertEqual(client.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual(client.model, "qwen3.7-plus")

    def test_generate_json_with_image_sends_openai_compatible_image_content(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            image_path.write_bytes(b"fake-png-bytes")
            captured_payloads: list[dict] = []

            class CapturingClient(OpenAICompatibleLLMClient):
                def _post_chat_completions(self, payload: dict) -> dict:
                    captured_payloads.append(payload)
                    return {"choices": [{"message": {"content": '{"ok": true}'}}]}

            client = CapturingClient(
                api_key="key",
                base_url="https://example.test/v1",
                model="vision-model",
            )

            result = client.generate_json_with_image(
                system_prompt="system",
                user_prompt="describe",
                image_path=image_path,
            )

            self.assertEqual(result, {"ok": True})
            user_content = captured_payloads[0]["messages"][1]["content"]
            self.assertEqual(user_content[0]["type"], "text")
            self.assertEqual(user_content[0]["text"], "describe")
            self.assertEqual(user_content[1]["type"], "image_url")
            expected_b64 = base64.b64encode(b"fake-png-bytes").decode("ascii")
            self.assertEqual(
                user_content[1]["image_url"]["url"],
                f"data:image/png;base64,{expected_b64}",
            )


if __name__ == "__main__":
    unittest.main()
