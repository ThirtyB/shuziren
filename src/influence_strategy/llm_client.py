from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LLMClientError(RuntimeError):
    pass


class LLMHTTPError(LLMClientError):
    """后端返回非 2xx 状态码。携带 status_code 以便区分"该不该重试"。"""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider: str = "openai_compatible",
        timeout: int = 60,
        reasoning_effort: str | None = None,
        keep_alive: int | str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.timeout = timeout
        # 推理模型（如 qwen3）默认会先生成一段思维链；对本地小模型这既拖慢
        # 速度，又会让 response_format=json_object 退化成空对象 "{}"。
        # 由 .env 的 LLM_REASONING_EFFORT 控制，留空则不发送该参数。
        self.reasoning_effort = reasoning_effort
        # 第十五轮：模型常驻时长（Ollama 的 ``keep_alive``）。
        # Ollama 默认只让模型在内存里待 5 分钟，空闲超时后卸载 —— 实测这会让
        # 下一次请求多出 3.5~6.5s 的模型加载（第十五轮 A 组：连续 3.5s →
        # 空闲 5.5 分钟后 10.77s）。默认 None 表示**不发送**该参数，
        # 以免给云端 OpenAI 兼容端点带去它们不认识的字段。
        self.keep_alive = keep_alive

    @classmethod
    def from_env_files(cls, workspace_root: str | Path) -> "OpenAICompatibleLLMClient | None":
        env_values: dict[str, str] = {}
        root = Path(workspace_root)
        for path in (
            root / ".env",
            root / "src" / "influence_strategy" / ".env",
        ):
            env_values.update(_read_env_file(path))

        provider = (
            env_values.get("LLM_PROVIDER")
            or os.environ.get("LLM_PROVIDER")
            or ""
        ).strip()

        api_key = _first_non_empty(
            env_values.get("LLM_API_KEY"),
            os.environ.get("LLM_API_KEY"),
            env_values.get("DEEPSEEK_API_KEY"),
            os.environ.get("DEEPSEEK_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
            env_values.get("DASHSCOPE_API_KEY"),
            os.environ.get("DASHSCOPE_API_KEY"),
        )
        base_url = _first_non_empty(
            env_values.get("LLM_BASE_URL"),
            os.environ.get("LLM_BASE_URL"),
            env_values.get("DEEPSEEK_BASE_URL"),
            os.environ.get("DEEPSEEK_BASE_URL"),
            env_values.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            env_values.get("OPENAI_API_BASE"),
            os.environ.get("OPENAI_API_BASE"),
            env_values.get("DASHSCOPE_BASE_URL"),
            os.environ.get("DASHSCOPE_BASE_URL"),
            "https://api.openai.com/v1",
        )
        model = _first_non_empty(
            env_values.get("LLM_MODEL"),
            os.environ.get("LLM_MODEL"),
            env_values.get("DEEPSEEK_MODEL"),
            os.environ.get("DEEPSEEK_MODEL"),
            env_values.get("OPENAI_MODEL"),
            os.environ.get("OPENAI_MODEL"),
            env_values.get("MODEL_NAME"),
            os.environ.get("MODEL_NAME"),
            "gpt-4o-mini",
        )

        if not api_key:
            return None

        reasoning_effort = _first_non_empty(
            env_values.get("LLM_REASONING_EFFORT"),
            os.environ.get("LLM_REASONING_EFFORT"),
        ) or None

        resolved_provider = provider or _infer_provider(base_url=base_url, model=model)
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=resolved_provider,
            timeout=_resolve_timeout(env_values, ("LLM_TIMEOUT",), default=90),
            reasoning_effort=reasoning_effort,
            keep_alive=_resolve_keep_alive(env_values, ("LLM_KEEP_ALIVE",)),
        )

    @classmethod
    def from_vision_env_files(cls, workspace_root: str | Path) -> "OpenAICompatibleLLMClient | None":
        env_values: dict[str, str] = {}
        root = Path(workspace_root)
        for path in (
            root / ".env",
            root / "src" / "influence_strategy" / ".env",
        ):
            env_values.update(_read_env_file(path))

        api_key = _first_non_empty(
            env_values.get("VISION_LLM_API_KEY"),
            os.environ.get("VISION_LLM_API_KEY"),
            env_values.get("LLM_API_KEY"),
            os.environ.get("LLM_API_KEY"),
            env_values.get("DASHSCOPE_API_KEY"),
            os.environ.get("DASHSCOPE_API_KEY"),
            env_values.get("OPENAI_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        )
        base_url = _first_non_empty(
            env_values.get("VISION_LLM_BASE_URL"),
            os.environ.get("VISION_LLM_BASE_URL"),
            env_values.get("LLM_BASE_URL"),
            os.environ.get("LLM_BASE_URL"),
            env_values.get("DASHSCOPE_BASE_URL"),
            os.environ.get("DASHSCOPE_BASE_URL"),
            env_values.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            env_values.get("OPENAI_API_BASE"),
            os.environ.get("OPENAI_API_BASE"),
            "https://api.openai.com/v1",
        )
        model = _first_non_empty(
            env_values.get("VISION_LLM_MODEL"),
            os.environ.get("VISION_LLM_MODEL"),
            env_values.get("LLM_MODEL"),
            os.environ.get("LLM_MODEL"),
            env_values.get("MODEL_NAME"),
            os.environ.get("MODEL_NAME"),
            env_values.get("DASHSCOPE_MODEL"),
            os.environ.get("DASHSCOPE_MODEL"),
            env_values.get("OPENAI_MODEL"),
            os.environ.get("OPENAI_MODEL"),
            "gpt-4o-mini",
        )
        provider = _first_non_empty(
            env_values.get("VISION_LLM_PROVIDER"),
            os.environ.get("VISION_LLM_PROVIDER"),
            env_values.get("LLM_PROVIDER"),
            os.environ.get("LLM_PROVIDER"),
        )

        if not api_key:
            return None

        reasoning_effort = _first_non_empty(
            env_values.get("VISION_LLM_REASONING_EFFORT"),
            os.environ.get("VISION_LLM_REASONING_EFFORT"),
            env_values.get("LLM_REASONING_EFFORT"),
            os.environ.get("LLM_REASONING_EFFORT"),
        ) or None

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider or _infer_provider(base_url=base_url, model=model),
            timeout=_resolve_timeout(env_values, ("VISION_LLM_TIMEOUT", "LLM_TIMEOUT"), default=90),
            reasoning_effort=reasoning_effort,
            keep_alive=_resolve_keep_alive(
                env_values, ("VISION_LLM_KEEP_ALIVE", "LLM_KEEP_ALIVE")
            ),
        )

    def describe(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        self._apply_keep_alive(payload)
        try:
            return self._post_chat_completions(payload)
        except LLMHTTPError as exc:
            # 只有"后端不认 response_format"这类 4xx 才值得去掉它重试。
            # 超时 / 连接失败再发一次只会让等待时间翻倍（大提示词场景尤其明显）。
            if not 400 <= exc.status_code < 500:
                raise
            payload.pop("response_format", None)
            return self._post_chat_completions(payload)

    def generate_json_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        image = Path(image_path)
        resolved_mime_type = mime_type or mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        image_b64 = base64.b64encode(image.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{resolved_mime_type};base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        self._apply_keep_alive(payload)
        try:
            return _normalize_json_result(self._post_chat_completions(payload))
        except LLMHTTPError as exc:
            if not 400 <= exc.status_code < 500:
                raise
            payload.pop("response_format", None)
            return _normalize_json_result(self._post_chat_completions(payload))

    def _apply_keep_alive(self, payload: dict[str, Any]) -> None:
        """把 Ollama 的 ``keep_alive`` 写进请求体（仅对 Ollama 端点）。

        它是 Ollama 的扩展参数：云端 OpenAI 兼容端点不认识它，最坏情况是
        400，而 400 会触发下面"去掉 response_format 重试"的逻辑，白白多一次
        往返。所以只对本机 Ollama 端点发送。
        """
        if self.keep_alive is None:
            return
        if not _is_ollama_endpoint(self.base_url, self.provider):
            return
        payload["keep_alive"] = self.keep_alive

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise LLMHTTPError(
                f"LLM request failed with HTTP {exc.code}: {body[:300]}",
                status_code=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMClientError(f"LLM request failed: {exc}") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM response missing choices[0].message.content") from exc

        return _parse_json_content(str(content))


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise LLMClientError("LLM response is not valid JSON.")
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON response must be an object.")
    return parsed


def _normalize_json_result(result: dict[str, Any]) -> dict[str, Any]:
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return result
    return _parse_json_content(str(content))


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _resolve_timeout(env_values: dict[str, str], keys: tuple[str, ...], default: int) -> int:
    """从 .env 或进程环境读取请求超时（秒）；非法或缺失时回退到默认值。"""
    for key in keys:
        raw = env_values.get(key)
        if raw is None:
            raw = os.environ.get(key)
        if raw is None or not str(raw).strip():
            continue
        try:
            value = int(str(raw).strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return default


def read_env_values(workspace_root: str | Path) -> dict[str, str]:
    """按项目约定读取 .env（供需要读同一份配置的其它模块复用）。

    只返回 .env 里的值，不掺进程环境变量——谁优先由调用方决定。
    """
    env_values: dict[str, str] = {}
    root = Path(workspace_root)
    for path in (
        root / ".env",
        root / "src" / "influence_strategy" / ".env",
    ):
        env_values.update(_read_env_file(path))
    return env_values


def _resolve_keep_alive(env_values: dict[str, str], keys: tuple[str, ...]) -> int | str | None:
    """读取模型常驻时长；留空表示**不发送** ``keep_alive``。

    Ollama 接受整数秒、``"10m"`` 这类时长字符串；``-1`` 表示加载后永不卸载。
    默认不发送，是为了让云端 OpenAI 兼容端点看到和过去完全一样的请求体。
    """
    for key in keys:
        raw = env_values.get(key)
        if raw is None:
            raw = os.environ.get(key)
        if raw is None or not str(raw).strip():
            continue
        text = str(raw).strip()
        try:
            return int(text)
        except ValueError:
            return text
    return None


def _is_ollama_endpoint(base_url: str, provider: str) -> bool:
    """判断当前客户端是否指向本机 Ollama（``keep_alive`` 只对它有意义）。"""
    lowered = f"{base_url} {provider}".lower()
    return "ollama" in lowered or ":11434" in lowered


def _infer_provider(*, base_url: str, model: str) -> str:
    combined = f"{base_url} {model}".lower()
    if "deepseek" in combined:
        return "deepseek"
    if "dashscope" in combined or "qwen" in combined:
        return "dashscope"
    if "openai" in combined or "gpt" in combined:
        return "openai"
    return "openai_compatible"
