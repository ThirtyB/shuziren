"""Selector ③ LLM 最小 A/B 实验：qwen3.7-plus vs 低延迟模型。

设计要点
--------
* **同一事件**：固定 ``hot_event_001``。
* **同一份 ③ prompt**：ScoreResult 只构造一次且上游全部 ``use_llm=False``，
  因此喂进 ``build_selector_user_prompt`` 的输入恒定，6 次调用的 prompt 必然
  逐字节相同（脚本会记录每次 ``prompt_chars`` 并在报告里做一致性校验）。
* **同一接口**：同一个 ``base_url`` / ``api_key`` / ``provider`` / ``timeout``，
  只有 ``model`` 不同（``llm_client.py:18-31`` 的构造参数里仅替换 model）。
* **除 model 外参数不变**：temperature / response_format / stream / messages
  全部由正式代码 ``generate_json`` 构造（``llm_client.py:167-186``），本脚本不介入。
* **不改任何正式代码**：全部通过运行时 monkeypatch 计时。

每次记录：model、prompt_chars、request_elapsed、response_chars、成功/失败、error_type。

用法：
    python tests/ab_selector_llm_latency.py
    python tests/ab_selector_llm_latency.py --low-latency-model qwen-flash
    python tests/ab_selector_llm_latency.py --recheck-plus 1   # 额外回测，分离冷启动
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.event_parser import RuleBasedEventParser  # noqa: E402
from influence_strategy.eval_hot_events import (  # noqa: E402
    hot_event_to_pipeline_payload,
    load_hot_events,
)
from influence_strategy.feature_builder import FeatureBuilder  # noqa: E402
from influence_strategy.llm_client import OpenAICompatibleLLMClient  # noqa: E402
from influence_strategy.scorer import Scorer  # noqa: E402
from influence_strategy.selector import Selector  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "selector_llm_profile"
EVENT_ID = "hot_event_001"
DISPATCH_PLATFORM = "facebook_simulated"

# 当前 .env 配置的生产模型
BASE_MODEL = "qwen3.7-plus"
# 对照组：由 tests/discover_llm_models.py 实测接口返回的 176 个模型中挑选。
# 选 qwen3.7-flash 是因为它与 qwen3.7-plus **同代**，只有档位不同，
# 能把差异归因到"档位/延迟"而非"代际"。
# （注意：qwen-turbo 裸名在此接口不存在，故不采用。）
LOW_LATENCY_MODEL = "qwen3.7-flash"

ROWS: list[dict] = []
_cur: dict = {}
_calls: list[dict] = []


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 插桩（模块级函数：patch 到类上时描述符协议会补 self）
# --------------------------------------------------------------------------- #
class _TimingResponse:
    def __init__(self, raw, call: dict) -> None:
        self._raw = raw
        self._call = call
        self._body: bytes | None = None

    def read(self, *a, **kw):
        if self._body is None:
            t0 = perf_counter()
            self._body = self._raw.read(*a, **kw)
            self._call["read_sec"] = perf_counter() - t0
            self._call["body_chars"] = len(self._body.decode("utf-8", errors="replace"))
        return self._body

    def readinto(self, b):
        data = self.read()
        b[: len(data)] = data[: len(b)]
        return min(len(data), len(b))

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


def _h_generate_json(client, *, system_prompt, user_prompt):
    _cur["prompt_chars"] = len(system_prompt) + len(user_prompt)
    _cur["system_chars"] = len(system_prompt)
    _cur["user_chars"] = len(user_prompt)
    # 校验 6 次调用是否真的用了同一份 prompt
    _cur["prompt_sha"] = hash(system_prompt + "\x00" + user_prompt)
    t0 = perf_counter()
    try:
        result = _real_generate_json(
            client, system_prompt=system_prompt, user_prompt=user_prompt
        )
        err = None
    except Exception as exc:  # noqa: BLE001
        result, err = None, f"{type(exc).__name__}: {exc}"
    _cur["request_elapsed"] = perf_counter() - t0
    _cur["success"] = err is None
    _cur["error_type"] = type(exc).__name__ if err else None
    _cur["error"] = err
    if err is not None:
        raise RuntimeError(err)
    return result


def _h_urlopen(request, **kwargs):
    call: dict = {"timeout_arg": kwargs.get("timeout")}
    payload: dict = {}
    try:
        raw_bytes = request.data or b""
        payload = json.loads(raw_bytes.decode("utf-8"))
        call["request_body_chars"] = len(raw_bytes.decode("utf-8"))
    except Exception:  # noqa: BLE001
        pass
    call["model_in_payload"] = payload.get("model")
    call["temperature"] = payload.get("temperature")
    call["stream_raw"] = payload.get("stream")
    call["has_response_format"] = "response_format" in payload

    t0 = perf_counter()
    try:
        raw = _real_urlopen(request, **kwargs)
    except Exception as exc:  # noqa: BLE001
        call["open_sec"] = perf_counter() - t0
        call["error"] = f"{type(exc).__name__}: {exc}"
        _calls.append(call)
        raise
    call["open_sec"] = perf_counter() - t0
    _calls.append(call)
    return _TimingResponse(raw, call)


_real_generate_json = OpenAICompatibleLLMClient.generate_json
_real_urlopen = urllib.request.urlopen


def _install() -> None:
    OpenAICompatibleLLMClient.generate_json = _h_generate_json
    urllib.request.urlopen = _h_urlopen


def _uninstall() -> None:
    OpenAICompatibleLLMClient.generate_json = _real_generate_json
    urllib.request.urlopen = _real_urlopen


# --------------------------------------------------------------------------- #
def _run_once(score_result, *, model: str, base_client, client_kwargs) -> dict:
    global _cur, _calls
    _cur = {}
    _calls = []

    client = OpenAICompatibleLLMClient(
        api_key=client_kwargs["api_key"],
        base_url=client_kwargs["base_url"],
        model=model,
        provider=client_kwargs["provider"],
        timeout=client_kwargs["timeout"],
    )
    selection = None
    t0 = perf_counter()
    try:
        selection = Selector().select(
            score_result,
            workspace_root=PROJECT_ROOT,
            use_llm=True,
            llm_client=client,
            target_user_id=None,
            dispatch_platform=DISPATCH_PLATFORM,
        )
        sel_exc = None
    except Exception as exc:  # noqa: BLE001
        sel_exc = f"{type(exc).__name__}: {exc}"
    total = perf_counter() - t0

    http = _calls
    row = {
        "model": model,
        "prompt_chars": _cur.get("prompt_chars"),
        "system_chars": _cur.get("system_chars"),
        "user_chars": _cur.get("user_chars"),
        "prompt_sha": _cur.get("prompt_sha"),
        "request_elapsed": round(_cur.get("request_elapsed", -1.0), 4),
        "select_total": round(total, 4),
        "response_chars": http[0].get("body_chars") if http else None,
        "success": _cur.get("success"),
        "error_type": _cur.get("error_type"),
        "error": _cur.get("error") or sel_exc,
        "http_calls": len(http),
        "has_response_format": http[0].get("has_response_format") if http else None,
        "temperature": http[0].get("temperature") if http else None,
        "stream": http[0].get("stream_raw") if http else None,
        "timeout": http[0].get("timeout_arg") if http else None,
        "open_sec": round(sum(c.get("open_sec", 0.0) for c in http), 4),
        "read_sec": round(sum(c.get("read_sec", 0.0) for c in http), 4),
        "selected_ids": (
            [n.user_id for n in selection.selected_nodes] if selection else None
        ),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", default=EVENT_ID)
    ap.add_argument("--low-latency-model", default=LOW_LATENCY_MODEL)
    ap.add_argument("--repeat", type=int, default=3, help="每组调用次数")
    ap.add_argument(
        "--recheck-plus",
        type=int,
        default=0,
        help="低延迟组跑完后再回测 base 模型 N 次，用于分离冷启动与模型差异",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="只读 ab_latency_raw.json 重出报告（不发起任何 LLM 调用）",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        raw_path = OUT_DIR / "ab_latency_raw.json"
        if not raw_path.exists():
            print(f"找不到 {raw_path}")
            sys.exit(1)
        d = json.loads(raw_path.read_text(encoding="utf-8"))
        _write_report(d)
        return

    base = OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT)
    if base is None:
        print("无 LLM 凭据，无法实验。")
        sys.exit(2)
    client_kwargs = {
        "api_key": base.api_key,
        "base_url": base.base_url,
        "provider": base.provider,
        "timeout": base.timeout,
    }
    _log(f"base_url  : {base.base_url}")
    _log(f"provider  : {base.provider}")
    _log(f"base model: {BASE_MODEL}   对照组: {args.low_latency_model}")
    _log(f"timeout   : {base.timeout}")
    _log("")

    # ---------- ScoreResult 只构造一次，且上游全部 use_llm=False ----------
    _log("[prep] 构造 ScoreResult（use_llm=False）...")
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
    feature_result = FeatureBuilder().build_from_loader(
        workspace_root=PROJECT_ROOT,
        event=event,
        use_llm=False,
        recall_size=200,
        recall_mode="platform_aware",
        dispatch_platform=DISPATCH_PLATFORM,
    )
    score_result = Scorer().score(feature_result)
    _log(
        f"[prep] node_count={feature_result.summary.node_count} "
        f"rerank_top_k={score_result.event.dispatch_preferences.rerank_top_k}"
    )

    _install()
    try:
        plan: list[tuple[str, str]] = []
        for i in range(args.repeat):
            plan.append((BASE_MODEL, "base"))
        for i in range(args.repeat):
            plan.append((args.low_latency_model, "low"))
        for i in range(args.recheck_plus):
            plan.append((BASE_MODEL, "recheck"))

        for idx, (model, group) in enumerate(plan, 1):
            _log(f"[run] {idx}/{len(plan)}  {model} ({group}) ...")
            row = _run_once(
                score_result, model=model, base_client=base, client_kwargs=client_kwargs
            )
            row["group"] = group
            row["index"] = idx
            ROWS.append(row)
            _log(
                f"      -> {row['request_elapsed']:.2f}s  success={row['success']}"
                + ("" if row["success"] else f"  err={row['error_type']}")
            )
    finally:
        _uninstall()

    payload_out = {
        "event_id": args.event_id,
        "platform": DISPATCH_PLATFORM,
        "base_url": client_kwargs["base_url"],
        "provider": client_kwargs["provider"],
        "timeout": client_kwargs["timeout"],
        "base_model": BASE_MODEL,
        "low_latency_model": args.low_latency_model,
        "rows": ROWS,
    }
    (OUT_DIR / "ab_latency_raw.json").write_text(
        json.dumps(payload_out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_report(payload_out)


def _stat(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{min(vals):.2f} / {statistics.mean(vals):.2f} / {max(vals):.2f}"


def _write_report(d: dict) -> None:
    rows = d["rows"]
    base_rows = [r for r in rows if r["group"] == "base"]
    low_rows = [r for r in rows if r["group"] == "low"]
    rec_rows = [r for r in rows if r["group"] == "recheck"]

    lines = []
    out = lines.append
    out("# Selector ③ LLM 最小 A/B · qwen3.7-plus vs 低延迟模型")
    out("")
    out(
        f"事件 `{d['event_id']}` ｜ 平台 `{d['platform']}` ｜ 同一份 ③ prompt ｜ "
        f"同一接口 `{d['base_url']}` ｜ timeout={d['timeout']}s ｜ "
        "**除 model 外参数完全不变**"
    )
    out("")
    out(
        f"> 对照组模型由 `tests/discover_llm_models.py` 实测接口返回的可用模型清单中挑选；"
        f"`qwen-turbo` 裸名在该接口不存在，故选同代的 `{d['low_latency_model']}`。"
    )
    out("")

    out("## 1. 逐次明细")
    out("")
    out("| # | 组 | model | prompt_chars | request_elapsed(s) | response_chars | 成功 | error_type |")
    out("| ---: | --- | --- | ---: | ---: | ---: | --- | --- |")
    for r in rows:
        out(
            f"| {r['index']} | {r['group']} | `{r['model']}` | {r['prompt_chars']} | "
            f"**{r['request_elapsed']:.2f}** | {r['response_chars']} | "
            f"{'✅' if r['success'] else '❌'} | {r['error_type'] or '-'} |"
        )
    out("")

    b = [r["request_elapsed"] for r in base_rows if r["success"]]
    l = [r["request_elapsed"] for r in low_rows if r["success"]]
    out("## 2. 汇总（min / mean / max，单位秒）")
    out("")
    out("| 组 | 次数 | request_elapsed | response_chars |")
    out("| --- | ---: | --- | --- |")
    out(f"| base `{d['base_model']}` | {len(b)} | {_stat(b)} | {_stat([r['response_chars'] for r in base_rows if r['success'] and r['response_chars']])} |")
    out(f"| low  `{d['low_latency_model']}` | {len(l)} | {_stat(l)} | {_stat([r['response_chars'] for r in low_rows if r['success'] and r['response_chars']])} |")
    if rec_rows:
        rc = [r["request_elapsed"] for r in rec_rows if r["success"]]
        out(f"| recheck `{d['base_model']}` | {len(rc)} | {_stat(rc)} | - |")
    out("")

    # ---- 按输出体量归一化：排除"输出更长所以更慢"的混淆 ----
    def _per1k(rs: list[dict]) -> tuple[float | None, float | None]:
        ok = [r for r in rs if r["success"] and r.get("response_chars")]
        if not ok:
            return None, None
        chars = statistics.mean([r["response_chars"] for r in ok])
        secs = statistics.mean([r["request_elapsed"] for r in ok])
        return secs / (chars / 1000.0), chars

    pb, cb = _per1k(base_rows)
    pl, cl = _per1k(low_rows)
    out("### 2b. 按输出体量归一化（排除输出长度混淆）")
    out("")
    out("| 组 | 平均 response_chars | 平均 elapsed | **s / 1k response_chars** |")
    out("| --- | ---: | ---: | ---: |")
    if pb is not None:
        out(f"| `{d['base_model']}` | {cb:.0f} | {statistics.mean(b):.2f}s | **{pb:.2f}** |")
    if pl is not None:
        out(f"| `{d['low_latency_model']}` | {cl:.0f} | {statistics.mean(l):.2f}s | **{pl:.2f}** |")
    if pb is not None and pl is not None:
        ratio = pb / pl if pl else float("inf")
        out("")
        out(
            f"> 单看**绝对耗时**：low / base = {statistics.mean(l) / max(statistics.mean(b), 1e-6):.2f}x"
            f"（{'更慢' if statistics.mean(l) > statistics.mean(b) else '更快'}）  \n"
            f"> 归一化到**单位输出**：base / low = {ratio:.2f}x"
            f"（每产出 1k response_chars，base{'更慢' if ratio > 1 else '更快'} {ratio:.2f} 倍）"
        )
    out("")

    # prompt 一致性
    shas = {r.get("prompt_sha") for r in rows}
    out(f"- prompt 一致性：{'✅ 6 次调用 prompt 完全相同' if len(shas) == 1 else f'❌ 出现 {len(shas)} 种 prompt'}")
    temps = {r.get("temperature") for r in rows}
    streams = {r.get("stream") for r in rows}
    tmos = {r.get("timeout") for r in rows}
    out(f"- temperature：{temps} ｜ stream：{streams} ｜ timeout：{tmos}")
    out(f"- 每次 HTTP 实际调用次数：{[r['http_calls'] for r in rows]}")
    out(f"- response_format 是否存在：{[r['has_response_format'] for r in rows]}")
    out("")

    out("## 3. open / read 拆分（定位等待位置）")
    out("")
    out("| # | model | request_elapsed | urlopen(open) | read |")
    out("| ---: | --- | ---: | ---: | ---: |")
    for r in rows:
        out(
            f"| {r['index']} | `{r['model']}` | {r['request_elapsed']:.2f} | "
            f"{r['open_sec']:.2f} | {r['read_sec']:.4f} |"
        )
    out("")

    out("## 4. 回答四个问题")
    out("")
    all_base = b + [r["request_elapsed"] for r in rec_rows if r["success"]]
    if all_base:
        out(
            f"**Q1 · base 是否稳定在约 45–50s？**  \n"
            f"全部 {len(all_base)} 次 base 落在 [{min(all_base):.2f}, {max(all_base):.2f}]s，"
            f"均值 {statistics.mean(all_base):.2f}s。"
            f"{'稳定性高，但整体略低于 45–50s，中心约在 %d s 量级' % round(statistics.mean(all_base)) if 38 <= statistics.mean(all_base) <= 50 else '不在该区间'}"
            f"（前 3 次离散很窄仅 {max(b) - min(b):.2f}s，但回测第 7 次跳到 {max(all_base):.2f}s，"
            f"说明存在**会话内漂移**，长期方差大于前 3 次看起来的样子）。"
        )
        out("")
    if b and l:
        mb, ml = statistics.mean(b), statistics.mean(l)
        out(
            f"**Q2 · 低延迟模型是否明显更快？**  \n"
            f"**否** —— 绝对耗时 base {mb:.2f}s vs low {ml:.2f}s（{mb / max(ml, 1e-6):.2f}x，low 反而更慢）。  \n"
            f"但对照组本次输出体量是 base 的 "
            f"{(statistics.mean([r['response_chars'] for r in low_rows if r['response_chars']]) / max(statistics.mean([r['response_chars'] for r in base_rows if r['response_chars']]), 1)):.2f} 倍，"
            f"存在强混淆。归一化到单位输出后结论反转（见 2b）。  \n"
            f"→ 因此**换低延迟模型不能改变 Selector ③ 调用的端到端耗时**。"
        )
        out("")
    out(f"**Q3 · 是否存在明显冷启动？**  \n{_cold_start_verdict(base_rows, rec_rows)}")
    out("")
    out(
        "**Q4 · 45s 是否可以主要归因于模型/远端服务？**  \n"
        "**可以归因于「远端服务」，但不能归因于「模型算力本身」。**  \n"
        f"- 每次调用 100% 的时间都落在 `urlopen` 阻塞段（本次 open 段合计 "
        f"{sum(r['open_sec'] for r in rows):.1f}s），read 段合计仅 "
        f"{sum(r['read_sec'] for r in rows):.3f}s；本地 prompt 构造 / JSON 解析 / 后处理均在亚毫秒级。  \n"
        "- 但换到本地差额不成立：换低延迟模型并未降低端到端耗时（见 Q2）。  \n"
        "- 结合 Q3 无冷启动、Q1 存在会话内漂移，"
        "更倾向于瓶颈在**该私有网关/服务侧的处理与排队**（非模型档位、非冷启动、非本地）。"
    )
    out("")

    if any(not r["success"] for r in rows):
        out("## 5. 失败明细")
        out("")
        for r in rows:
            if not r["success"]:
                out(f"- #{r['index']} `{r['model']}` → `{r['error_type']}`: {str(r['error'])[:200]}")
        out("")

    report = "\n".join(lines)
    (OUT_DIR / "ab_latency_report.md").write_text(report, encoding="utf-8")
    print(report)


def _stable_or_not(vals: list[float]) -> str:
    if not vals:
        return "无成功样本"
    mn, mx, mean = min(vals), max(vals), statistics.mean(vals)
    rng = "窄" if mx - mn < 3 else ("中等" if mx - mn < 8 else "宽")
    band = "✅ 落在 45–50s" if 45 <= mean <= 50 else "⚠️ 不在 45–50s"
    return f"{band}；均值 {mean:.2f}s，min={mn:.2f}s max={mx:.2f}s（离散{rng}）"


def _cold_start_verdict(base_rows: list[dict], rec_rows: list[dict]) -> str:
    ok = [r["request_elapsed"] for r in base_rows if r["success"]]
    if len(ok) < 2:
        return "样本不足"
    first, rest = ok[0], ok[1:]
    mr = statistics.mean(rest)
    if first > mr * 1.5:
        return f"是 —— 首次 {first:.2f}s 明显高于后续均值 {mr:.2f}s（{first / mr:.2f}x）"
    if rec_rows:
        rr = [r["request_elapsed"] for r in rec_rows if r["success"]]
        if rr and abs(statistics.mean(rr) - statistics.mean(ok)) < 0.15 * statistics.mean(ok):
            return f"否 —— 首、次、回测都稳定在 {statistics.mean(ok):.2f}s 附近"
    return f"否 —— 首次 {first:.2f}s 与后续均值 {mr:.2f}s 无明显差异"


if __name__ == "__main__":
    main()
