"""Selector ③ LLM 单次调用的耗时拆分实验（只读，不改正式代码）。

**为什么要插桩而不是改代码**：正式代码里 ③ 的 5 个阶段全部发生在
``Selector._llm_selection_plan`` 一个函数体内（``selector.py:506-596``），
且 prompt 构造是作为实参内联写在 ``client.generate_json(...)`` 调用里的
（``selector.py:542-549``），JSON 解析则藏在 ``llm_client._post_chat_completions``
内部（``llm_client.py:249``）。本脚本通过**运行时 monkeypatch** 在接缝处插入
计时器，不改任何源文件。

接缝与对应关系：
  selector.build_selector_system_prompt / build_selector_user_prompt  → ① prompt 构造
  OpenAICompatibleLLMClient.generate_json                             → ② 总入口
    ├─ urllib.request.urlopen 拆成 open / read 两段                   → ② 网络/模型等待
    └─ llm_client._parse_json_content                                 → ③ JSON 解析
  Selector._reorder_pool_with_llm                                     → ④ 后处理
  Selector._llm_selection_plan / Selector.select 外层计时              → ⑤ 总耗时

为避免其它 LLM 调用污染计时，EventParser / FeatureBuilder 一律 use_llm=False
（② 线上 100% 失败，开了只会白白等 timeout），**只有 Selector 这一步 use_llm=True**，
因此一次运行只产生 1 次 ③ 调用。

用法：
    python tests/profile_selector_llm_call.py
    python tests/profile_selector_llm_call.py --no-recall
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import influence_strategy.llm_client as llm_mod  # noqa: E402
import influence_strategy.selector as sel_mod  # noqa: E402
from influence_strategy.event_parser import RuleBasedEventParser  # noqa: E402
from influence_strategy.eval_hot_events import (  # noqa: E402
    hot_event_to_pipeline_payload,
    load_hot_events,
)
from influence_strategy.feature_builder import FeatureBuilder  # noqa: E402
from influence_strategy.llm_client import (  # noqa: E402
    OpenAICompatibleLLMClient,
    _parse_json_content,
)
from influence_strategy.prompts import (  # noqa: E402
    build_selector_system_prompt as _real_sys_prompt,
    build_selector_user_prompt as _real_user_prompt,
)
from influence_strategy.scorer import Scorer  # noqa: E402
from influence_strategy.selector import Selector  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "selector_llm_profile"
DISPATCH_PLATFORM = "facebook_simulated"
EVENT_ID = "hot_event_001"

DETAIL: dict[str, object] = {}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 响应体包装：把 open（连接+响应头）与 read（读 body）分开计时。
# 非流式 completions 接口的"模型生成等待"绝大部分落在 read 上。
# --------------------------------------------------------------------------- #
class _TimingResponse:
    def __init__(self, raw, call: dict) -> None:
        self._raw = raw
        self._call = call
        self._body: bytes | None = None

    def read(self, *args, **kwargs):
        if self._body is None:
            t0 = perf_counter()
            self._body = self._raw.read(*args, **kwargs)
            self._call["read_sec"] = perf_counter() - t0
            self._call["body_chars"] = len(
                self._body.decode("utf-8", errors="replace")
            )
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        closer = getattr(self._raw, "__exit__", None)
        if closer is not None:
            return closer(*exc)
        close = getattr(self._raw, "close", None)
        if close is not None:
            close()
        return False

    def __getattr__(self, item):
        return getattr(self._raw, item)


class Profiler:
    """安装 / 卸载全部计时钩子。

    注意：钩子必须是**模块级函数**，不能是本类的 bound method。
    ``setattr(Selector, "_llm_selection_plan", bound_method)`` 之后，
    ``selector._llm_selection_plan(...)`` 不会再补传 ``self``
    （bound method 已经绑过实例了），会直接抛 missing positional argument。
    """

    def __init__(self) -> None:
        self.http_calls: list[dict] = []
        self.json_parse_sec: list[float] = []
        self.prompt_calls: list[dict] = []
        self.generate_json: dict = {}
        self.plan: dict = {}
        self.reorder: dict = {}
        self._orig: list[tuple] = []

    def install(self) -> None:
        global _PROF
        _PROF = self
        # ① prompt 构造（selector 模块命名空间内的绑定才是真正的调用点）
        self._patch(sel_mod, "build_selector_system_prompt", _h_sys_prompt)
        self._patch(sel_mod, "build_selector_user_prompt", _h_user_prompt)
        # ② 网络总入口（方法 → 走描述符协议，自动补 self）
        self._patch(OpenAICompatibleLLMClient, "generate_json", _h_generate_json)
        # ② HTTP open / read 分离 + ③ JSON 解析（模块级函数）
        self._patch(urllib.request, "urlopen", _h_urlopen)
        self._patch(llm_mod, "_parse_json_content", _h_json_parse)
        # ④ / ⑤
        self._patch(Selector, "_llm_selection_plan", _h_plan)
        self._patch(Selector, "_reorder_pool_with_llm", _h_reorder)

    def _patch(self, obj, name: str, new) -> None:
        self._orig.append((obj, name, getattr(obj, name)))
        setattr(obj, name, new)

    def uninstall(self) -> None:
        for obj, name, orig in self._orig:
            setattr(obj, name, orig)
        self._orig.clear()


# 真实函数句柄（必须在 install 之前保存）
_real_generate_json = OpenAICompatibleLLMClient.generate_json
_real_urlopen = urllib.request.urlopen
_real_json_parse = _parse_json_content
_real_plan = Selector._llm_selection_plan
_real_reorder = Selector._reorder_pool_with_llm

_PROF: "Profiler" = None  # type: ignore[assignment]


# ---------------- 钩子（全部是模块级函数） ----------------
def _h_sys_prompt(*a, **kw):
    t0 = perf_counter()
    text = _real_sys_prompt(*a, **kw)
    _PROF.prompt_calls.append(
        {"kind": "system", "sec": perf_counter() - t0, "chars": len(text)}
    )
    return text


def _h_user_prompt(*a, **kw):
    t0 = perf_counter()
    text = _real_user_prompt(*a, **kw)
    _PROF.prompt_calls.append(
        {"kind": "user", "sec": perf_counter() - t0, "chars": len(text)}
    )
    return text


def _h_generate_json(client, *, system_prompt, user_prompt):
    DETAIL["model"] = getattr(client, "model", None)
    DETAIL["http_timeout_sec"] = getattr(client, "timeout", None)
    DETAIL["base_url"] = getattr(client, "base_url", None)
    DETAIL["provider"] = getattr(client, "provider", None)
    t0 = perf_counter()
    try:
        result = _real_generate_json(
            client, system_prompt=system_prompt, user_prompt=user_prompt
        )
        err = None
    except Exception as exc:  # noqa: BLE001  实验要记录失败本身
        result, err = None, f"{type(exc).__name__}: {exc}"
    _PROF.generate_json = {"sec": perf_counter() - t0, "success": err is None, "error": err}
    if err is not None:
        raise RuntimeError(err)
    return result


def _h_urlopen(request, **kwargs):
    call: dict = {"timeout_arg": kwargs.get("timeout")}
    payload: dict = {}
    try:
        raw_bytes = request.data or b""
        payload = json.loads(raw_bytes.decode("utf-8"))
        call["request_chars"] = len(raw_bytes.decode("utf-8"))
        call["messages_chars"] = sum(
            len(str(m.get("content", ""))) for m in (payload.get("messages") or [])
        )
    except Exception:  # noqa: BLE001
        pass
    call["model"] = payload.get("model")
    call["temperature"] = payload.get("temperature")
    call["stream_raw"] = payload.get("stream")
    call["stream"] = bool(payload.get("stream"))
    call["has_response_format"] = "response_format" in payload

    t_open = perf_counter()
    try:
        raw = _real_urlopen(request, **kwargs)
    except Exception as exc:  # noqa: BLE001  异常继续传播，只记录
        call["open_sec"] = perf_counter() - t_open
        call["error"] = f"{type(exc).__name__}: {exc}"
        _PROF.http_calls.append(call)
        raise
    call["open_sec"] = perf_counter() - t_open

    _PROF.http_calls.append(call)
    # call 是同一个 dict 引用，read 完成后字段会自动回填
    return _TimingResponse(raw, call)


def _h_json_parse(content, *a, **kw):
    DETAIL["content_chars"] = len(content)
    t0 = perf_counter()
    try:
        return _real_json_parse(content, *a, **kw)
    finally:
        _PROF.json_parse_sec.append(perf_counter() - t0)


def _h_plan(selector, *a, **kw):
    t0 = perf_counter()
    result = _real_plan(selector, *a, **kw)
    _PROF.plan = {"sec": perf_counter() - t0, "returned_plan": result is not None}
    return result


def _h_reorder(selector, pool, llm_selection_plan):
    t0 = perf_counter()
    result = _real_reorder(selector, pool, llm_selection_plan)
    _PROF.reorder = {
        "sec": perf_counter() - t0,
        "pool_size": len(pool),
        "applied": bool(llm_selection_plan),
    }
    return result


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", default=EVENT_ID)
    ap.add_argument("--platform", default=DISPATCH_PLATFORM)
    ap.add_argument(
        "--no-recall",
        action="store_true",
        help="关闭召回走全量 35,636（默认走与生产一致的 recall_size=200）",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 准备 ScoreResult（LLM 全关，避免污染计时） ----------
    _log("[prep] 解析事件 / 建特征 / 打分（use_llm=False）...")
    events = load_hot_events(
        PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
    )
    hot = next(e for e in events if str(e.get("event_id")) == args.event_id)
    payload = hot_event_to_pipeline_payload(
        hot,
        max_selected_nodes=5,
        risk_level=None,
        campaign_window_hours=24,
        max_frequency_per_day=3,
        allowed_platforms=None,
    )
    event = RuleBasedEventParser().parse(
        payload, workspace_root=PROJECT_ROOT, use_llm=False
    )

    recall_size = None if args.no_recall else 200
    recall_mode = None if args.no_recall else "platform_aware"
    feature_result = FeatureBuilder().build_from_loader(
        workspace_root=PROJECT_ROOT,
        event=event,
        use_llm=False,
        recall_size=recall_size,
        recall_mode=recall_mode,
        dispatch_platform=args.platform,
    )
    score_result = Scorer().score(feature_result)
    rerank_top_k = score_result.event.dispatch_preferences.rerank_top_k
    safe_pool_size = sum(
        1
        for n in score_result.node_scores
        if getattr(n, "eligible", False)
        and not getattr(n, "manual_review_required", False)
    )
    _log(
        f"[prep] node_count={feature_result.summary.node_count} "
        f"total={feature_result.summary.total_candidate_count} "
        f"safe_pool={safe_pool_size} rerank_top_k={rerank_top_k}"
    )

    client = OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT)
    if client is None:
        print("无 LLM 凭据（api_key 缺失），无法进行 ③ 调用实验。")
        sys.exit(2)

    # ---------- 插桩并跑一次 ③ ----------
    prof = Profiler()
    prof.install()
    _log("[run] 开始单次 Selector.select(use_llm=True) ...")
    t_all = perf_counter()
    selection = None
    exc_info = None
    try:
        selection = Selector().select(
            score_result,
            workspace_root=PROJECT_ROOT,
            use_llm=True,
            llm_client=client,
            target_user_id=None,
            dispatch_platform=args.platform,
        )
    except Exception as exc:  # noqa: BLE001
        exc_info = f"{type(exc).__name__}: {exc}"
        print(f"[run] select() 抛出异常：{exc_info}")
    finally:
        total_sec = perf_counter() - t_all
        prof.uninstall()

    _log(f"[run] 完成，总耗时 {total_sec:.2f}s")

    selected_ids = (
        [n.user_id for n in selection.selected_nodes] if selection is not None else []
    )

    result = {
        "event_id": args.event_id,
        "platform": args.platform,
        "recall": {
            "enabled": not args.no_recall,
            "size": recall_size,
            "mode": recall_mode,
        },
        "shortlist": {
            "rerank_top_k": rerank_top_k,
            "safe_pool_size": safe_pool_size,
        },
        "total_sec": round(total_sec, 4),
        "select_exception": exc_info,
        "selected_ids": selected_ids,
        "detail": DETAIL,
        "prof": {
            "prompt_calls": prof.prompt_calls,
            "generate_json": prof.generate_json,
            "http_calls": prof.http_calls,
            "json_parse_sec": prof.json_parse_sec,
            "plan": prof.plan,
            "reorder": prof.reorder,
        },
    }
    (OUT_DIR / "raw.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_report(result)


def _write_report(r: dict) -> None:
    p = r["prof"]
    gj = p.get("generate_json") or {}
    http = p.get("http_calls") or []
    prompt_sec = sum(c["sec"] for c in p["prompt_calls"])
    sys_chars = sum(c["chars"] for c in p["prompt_calls"] if c["kind"] == "system")
    user_chars = sum(c["chars"] for c in p["prompt_calls"] if c["kind"] == "user")
    open_sec = sum(c.get("open_sec", 0.0) for c in http)
    read_sec = sum(c.get("read_sec", 0.0) for c in http)
    json_sec = sum(p.get("json_parse_sec") or [])
    reord = p.get("reorder") or {}
    reorder_sec = reord.get("sec", 0.0)
    plan = p.get("plan") or {}
    plan_sec = plan.get("sec", 0.0)
    total = r["total_sec"]

    def pct(x):
        return f"{x / total * 100:.1f}%" if total else "-"

    lines = []
    out = lines.append
    out("# Selector ③ LLM 单次调用 · 耗时拆分")
    out("")
    out(
        f"事件 `{r['event_id']}` ｜ 平台 `{r['platform']}` ｜ "
        f"召回 {'开启 N=200' if r['recall']['enabled'] else '关闭（全量）'} ｜ "
        "EventParser/FeatureBuilder 均 use_llm=False，**本次只有 1 次 ③ 调用**"
    )
    out("")
    out("## 1. 耗时拆分")
    out("")
    out("| 阶段 | 耗时(s) | 占总耗时 |")
    out("| --- | ---: | ---: |")
    out(f"| ① 构造 system + user prompt | {prompt_sec:.4f} | {pct(prompt_sec)} |")
    out(f"| ② client.generate_json() 合计 | {gj.get('sec', 0):.4f} | {pct(gj.get('sec', 0))} |")
    out(f"| &nbsp;&nbsp;├─ HTTP 连接 + 响应头 (urlopen) | {open_sec:.4f} | {pct(open_sec)} |")
    out(f"| &nbsp;&nbsp;└─ 响应体读取 (read) | {read_sec:.4f} | {pct(read_sec)} |")
    out(f"| ③ JSON 解析 (_parse_json_content) | {json_sec:.6f} | {pct(json_sec)} |")
    out(f"| ④ _reorder_pool_with_llm 后处理 | {reorder_sec:.6f} | {pct(reorder_sec)} |")
    out(f"| **⑤ 总耗时** | **{total:.4f}** | **100.0%** |")
    out("")
    accounted = prompt_sec + gj.get("sec", 0) + json_sec + reorder_sec
    other = max(total - accounted, 0.0)
    out(f"- 已归属合计：{accounted:.4f}s")
    out(f"- 其余（③ 的 plan 归一化 + 选人主循环 + 对象构造）：{other:.4f}s ({pct(other)})")
    out(f"- `_llm_selection_plan` 整体：{plan_sec:.4f}s；其余 select() 部分：{max(total - plan_sec, 0):.4f}s")
    out("")

    out("## 2. 调用元数据")
    out("")
    out("| 项 | 值 |")
    out("| --- | --- |")
    if gj:
        out(f"| 调用成功/失败 | {'✅ 成功' if gj.get('success') else '❌ 失败'} |")
        if gj.get("error"):
            out(f"| 错误信息 | `{str(gj.get('error'))[:300]}` |")
    if r.get("select_exception"):
        out(f"| select() 异常 | `{str(r['select_exception'])[:300]}` |")
    out(f"| model | `{r['detail'].get('model')}` |")
    out(f"| provider | `{r['detail'].get('provider')}` |")
    out(f"| base_url | `{r['detail'].get('base_url')}` |")
    out(f"| temperature | {http[0].get('temperature') if http else '-'} |")
    out(
        f"| 是否流式 | {'是' if (http and http[0].get('stream')) else '否'} "
        f"(payload.stream={http[0].get('stream_raw') if http else '-'}) |"
    )
    out(f"| HTTP timeout | {r['detail'].get('http_timeout_sec')}s (urlopen 实参: {http[0].get('timeout_arg') if http else '-'}) |")
    out(f"| HTTP 实际调用次数 | {len(http)} |")
    for i, c in enumerate(http, 1):
        out(f"| &nbsp;&nbsp;第{i}次 has_response_format | {c.get('has_response_format')} |")
    if len(http) > 1:
        out("| ⚠️ 多次调用 | `generate_json` 在 response_format 报错后会去掉该字段重发一次（llm_client.py:182-186） |")
    out(f"| rerank_top_k（shortlist 上限） | {r['shortlist']['rerank_top_k']} |")
    out(f"| safe_pool 人数 | {r['shortlist']['safe_pool_size']} |")
    out("")

    out("## 3. 字符数")
    out("")
    out("| 项 | 字符数 |")
    out("| --- | ---: |")
    out(f"| system_prompt | {sys_chars} |")
    out(f"| user_prompt | {user_chars} |")
    out(f"| **prompt 合计** | **{sys_chars + user_chars}** |")
    out(f"| HTTP 请求体 | {http[0].get('request_chars') if http else '-'} |")
    out(f"| 其中 messages content | {http[0].get('messages_chars') if http else '-'} |")
    out(f"| HTTP 响应体 | {http[0].get('body_chars') if http else '-'} |")
    out(f"| response content (choices[0].message.content) | {r['detail'].get('content_chars', '-')} |")
    out("")

    if r.get("selected_ids"):
        out("## 4. 最终选人（供一致性核对）")
        out("")
        out("```")
        out(str(r["selected_ids"]))
        out("```")
        out("")

    report = "\n".join(lines)
    (OUT_DIR / "profile_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
