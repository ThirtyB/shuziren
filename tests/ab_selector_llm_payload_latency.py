"""Selector ③ LLM 延迟的「prompt/输出体量 vs 网关固定开销」归因实验。

**为什么不走 Selector**：我们想隔离"LLM 调用本身"的耗时与 prompt/输出体量的
关系，因此 3 组都直接调 ``client.generate_json(...)``，**不**经过
``Selector._llm_selection_plan``，避免把主循环的耗时混进来。

3 组（每组 3 次，连真实 LLM）：
  A · 真实 ③ prompt（~11,894 字符）—— 与生产完全相同
  B · 极小 prompt（~150 字符）   —— 同样要求 json_object，**返回最小 JSON**
  C · 中等 prompt（~3,000 字符） —— 同样要求 json_object，**返回最小 JSON**

所有组同 endpoint、同模型 qwen3.7-plus、同 temperature=0.4、同 timeout=90、
非流式、``response_format={"type":"json_object"}``，**只 prompt 内容不同**。

全部通过运行时 monkeypatch 计时；不动任何源文件。

用法：
    python tests/ab_selector_llm_payload_latency.py
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
from influence_strategy.prompts import (  # noqa: E402
    build_selector_system_prompt,
    build_selector_user_prompt,
)
from influence_strategy.scorer import Scorer  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "selector_llm_profile"
EVENT_ID = "hot_event_001"
DISPATCH_PLATFORM = "facebook_simulated"
MODEL = "qwen3.7-plus"
REPEAT = 3


# ---- 计时（每组重置） ----
_CUR: dict = {}
_CALLS: list[dict] = []


class _TimingResponse:
    def __init__(self, raw, call: dict) -> None:
        self._raw, self._call, self._body = raw, call, None

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
        c = getattr(self._raw, "close", None)
        if c:
            c()
        return False

    def __getattr__(self, item):
        return getattr(self._raw, item)


def _h_generate_json(client, *, system_prompt, user_prompt):
    _CUR["sys_chars"] = len(system_prompt)
    _CUR["usr_chars"] = len(user_prompt)
    _CUR["prompt_chars"] = len(system_prompt) + len(user_prompt)
    _CUR["prompt_sha"] = hash(system_prompt + "\x00" + user_prompt)
    t0 = perf_counter()
    try:
        result = _real_generate_json(
            client, system_prompt=system_prompt, user_prompt=user_prompt
        )
        err = None
    except Exception as exc:  # noqa: BLE001
        result, err = None, f"{type(exc).__name__}: {exc}"
    _CUR["elapsed"] = perf_counter() - t0
    _CUR["success"] = err is None
    _CUR["error_type"] = type(exc).__name__ if err else None
    _CUR["error"] = err
    if err is not None:
        raise RuntimeError(err)
    return result


def _h_urlopen(request, **kwargs):
    call: dict = {"timeout_arg": kwargs.get("timeout")}
    try:
        raw_bytes = request.data or b""
        payload = json.loads(raw_bytes.decode("utf-8"))
        call["request_body_chars"] = len(raw_bytes.decode("utf-8"))
        call["temperature"] = payload.get("temperature")
        call["stream_raw"] = payload.get("stream")
        call["has_response_format"] = "response_format" in payload
    except Exception:  # noqa: BLE001
        pass

    t0 = perf_counter()
    try:
        raw = _real_urlopen(request, **kwargs)
    except Exception as exc:  # noqa: BLE001
        call["open_sec"] = perf_counter() - t0
        call["error"] = f"{type(exc).__name__}: {exc}"
        _CALLS.append(call)
        raise
    call["open_sec"] = perf_counter() - t0
    _CALLS.append(call)
    return _TimingResponse(raw, call)


_real_generate_json = OpenAICompatibleLLMClient.generate_json
_real_urlopen = urllib.request.urlopen


def _install() -> None:
    OpenAICompatibleLLMClient.generate_json = _h_generate_json
    urllib.request.urlopen = _h_urlopen


def _uninstall() -> None:
    OpenAICompatibleLLMClient.generate_json = _real_generate_json
    urllib.request.urlopen = _real_urlopen


# ---- prompt 构造 ----
def _real_prompt(score_result, max_selected: int) -> tuple[str, str]:
    """A 组：与 Selector._llm_selection_plan 构造的 prompt **逐字节相同**。"""
    safe_pool = [
        n
        for n in score_result.node_scores
        if getattr(n, "eligible", False)
        and not getattr(n, "manual_review_required", False)
    ]
    shortlist = safe_pool[: score_result.event.dispatch_preferences.rerank_top_k]
    shortlist_payload = [
        {
            "user_id": n.user_id,
            "user_name": n.user_name,
            "role_hint": n.role_hint,
            "final_score": n.final_score,
            "llm_feature_score": n.llm_feature_score,
            "topic_match_score": n.topic_match_score,
            "risk_level": n.risk_level,
            "semantic_tags": n.semantic_tags,
            "matched_keywords": n.matched_keywords,
            "selection_reasons": n.selection_reasons[:6],
        }
        for n in shortlist
    ]
    return build_selector_system_prompt(), build_selector_user_prompt(
        event=score_result.event, max_selected=max_selected, shortlist=shortlist_payload
    )


def _tiny_prompt() -> tuple[str, str]:
    """B 组：极小 system + 极小 user；都要求**最小** JSON 输出。"""
    sys_p = (
        "你是节点选择助手。从给定候选中选 1 个。"
        "**只**返回最小 JSON："
        '{"selected_order":[{"user_id":"","recommended_role":"core_publish_node"}],'
        '"fallback_order":[],"global_notes":[]}'
    )
    usr_p = json.dumps(
        {
            "shortlist": [
                {
                    "user_id": "u1",
                    "user_name": "n1",
                    "role_hint": "core_broadcast",
                    "final_score": 0.5,
                    "llm_feature_score": 0.0,
                    "topic_match_score": 0.0,
                    "risk_level": "low",
                    "semantic_tags": [],
                    "matched_keywords": [],
                    "selection_reasons": [],
                }
            ]
        },
        ensure_ascii=False,
    )
    return sys_p, usr_p


def _medium_prompt(target_chars: int = 3000) -> tuple[str, str]:
    """C 组：与 B 同 system，按 user 体量增到 ~目标字符数。"""
    sys_p = (
        "你是节点选择助手。从给定候选中选 1 个。"
        "**只**返回最小 JSON："
        '{"selected_order":[{"user_id":"","recommended_role":"core_publish_node"}],'
        '"fallback_order":[],"global_notes":[]}'
    )
    one = {
        "user_id": "u",
        "user_name": "name",
        "role_hint": "core_broadcast",
        "final_score": 0.5,
        "llm_feature_score": 0.0,
        "topic_match_score": 0.0,
        "risk_level": "low",
        "semantic_tags": [],
        "matched_keywords": [],
        "selection_reasons": ["x" * 80],
    }
    # 估算每条约 100~120 字符，先按需扩字段保证达到目标
    n = 25
    while True:
        shortlist = []
        for i in range(n):
            d = dict(one)
            d["user_id"] = f"u_{i:03d}"
            d["user_name"] = f"user_{i:03d}"
            shortlist.append(d)
        usr_p = json.dumps({"shortlist": shortlist}, ensure_ascii=False)
        if len(usr_p) >= target_chars or n > 200:
            break
        n += 5
        one["selection_reasons"] = ["x" * 200]
    return sys_p, usr_p


# ---- 单次调用 ----
def _call_once(client, *, sys_p: str, usr_p: str) -> dict:
    global _CUR, _CALLS
    _CUR = {}
    _CALLS = []
    try:
        client.generate_json(system_prompt=sys_p, user_prompt=usr_p)
        succ = True
    except Exception as exc:  # noqa: BLE001
        succ = False
        _CUR["error"] = f"{type(exc).__name__}: {exc}"
        _CUR["error_type"] = type(exc).__name__
    http = _CALLS
    return {
        "elapsed": round(_CUR.get("elapsed", -1.0), 4),
        "success": _CUR.get("success", succ),
        "error_type": _CUR.get("error_type"),
        "error": _CUR.get("error"),
        "sys_chars": _CUR.get("sys_chars"),
        "usr_chars": _CUR.get("usr_chars"),
        "prompt_chars": _CUR.get("prompt_chars"),
        "prompt_sha": _CUR.get("prompt_sha"),
        "request_body_chars": http[0].get("request_body_chars") if http else None,
        "body_chars": http[0].get("body_chars") if http else None,
        "open_sec": round(sum(c.get("open_sec", 0.0) for c in http), 4),
        "read_sec": round(sum(c.get("read_sec", 0.0) for c in http), 4),
        "http_calls": len(http),
        "has_response_format": http[0].get("has_response_format") if http else None,
        "temperature": http[0].get("temperature") if http else None,
        "stream": http[0].get("stream_raw") if http else None,
        "timeout": http[0].get("timeout_arg") if http else None,
    }


# ---- 主流程 ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", default=EVENT_ID)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    ap.add_argument(
        "--report-only", action="store_true", help="只读 raw.json 重出报告（不发 API）"
    )
    ap.add_argument("--medium-chars", type=int, default=3000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "payload_latency_raw.json"

    if args.report_only:
        if not raw_path.exists():
            print(f"missing {raw_path}")
            sys.exit(1)
        _write_report(json.loads(raw_path.read_text(encoding="utf-8")))
        return

    base = OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT)
    if base is None:
        print("无 LLM 凭据。")
        sys.exit(2)

    _log(f"base_url : {base.base_url}")
    _log(f"provider : {base.provider}")
    _log(f"model    : {MODEL}")
    _log(f"timeout  : {base.timeout}")
    _log("")

    # 准备 A 组真实 prompt（只构造一次，后续 3 次共用）
    _log("[prep] 构造 ScoreResult（use_llm=False）...")
    events = load_hot_events(PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json")
    hot = next(e for e in events if str(e.get("event_id")) == args.event_id)
    payload = hot_event_to_pipeline_payload(
        hot, max_selected_nodes=5, risk_level=None,
        campaign_window_hours=24, max_frequency_per_day=3, allowed_platforms=None,
    )
    event = RuleBasedEventParser().parse(
        payload, workspace_root=PROJECT_ROOT, use_llm=False
    )
    fr = FeatureBuilder().build_from_loader(
        workspace_root=PROJECT_ROOT, event=event, use_llm=False,
        recall_size=200, recall_mode="platform_aware",
        dispatch_platform=DISPATCH_PLATFORM,
    )
    score_result = Scorer().score(fr)

    a_sys, a_usr = _real_prompt(score_result, max_selected=5)
    b_sys, b_usr = _tiny_prompt()
    c_sys, c_usr = _medium_prompt(args.medium_chars)
    _log(
        f"[prep] A prompt={len(a_sys)+len(a_usr)} chars  "
        f"B prompt={len(b_sys)+len(b_usr)} chars  "
        f"C prompt={len(c_sys)+len(c_usr)} chars"
    )
    _log("")

    client = OpenAICompatibleLLMClient(
        api_key=base.api_key, base_url=base.base_url, model=MODEL,
        provider=base.provider, timeout=base.timeout,
    )

    groups = [
        ("A_real", a_sys, a_usr, "真实 ③ prompt"),
        ("B_tiny", b_sys, b_usr, "极小 prompt + 最小 JSON"),
        ("C_medium", c_sys, c_usr, f"~{args.medium_chars} 字符 prompt + 最小 JSON"),
    ]
    rows: list[dict] = []
    _install()
    try:
        for tag, sp, up, desc in groups:
            for i in range(args.repeat):
                idx = len(rows) + 1
                _log(f"[run] {idx}/{len(groups)*args.repeat}  {tag} ({desc}) ...")
                row = _call_once(client, sys_p=sp, usr_p=up)
                row.update({"group": tag, "desc": desc, "index": idx})
                rows.append(row)
                _log(
                    f"      -> {row['elapsed']:.2f}s  prompt={row['prompt_chars']}  "
                    f"body={row['body_chars']}  success={row['success']}"
                    + (f"  err={row['error_type']}" if not row["success"] else "")
                )
    finally:
        _uninstall()

    out = {
        "event_id": args.event_id,
        "platform": DISPATCH_PLATFORM,
        "base_url": base.base_url,
        "provider": base.provider,
        "model": MODEL,
        "timeout": base.timeout,
        "groups": {tag: desc for tag, _, _, desc in groups},
        "rows": rows,
    }
    raw_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_report(out)


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _stat(v: list[float]) -> str:
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.2f}"
    return f"{min(v):.2f} / {statistics.mean(v):.2f} / {max(v):.2f}"


def _write_report(d: dict) -> None:
    rows = d["rows"]
    by_g: dict[str, list[dict]] = {}
    for r in rows:
        by_g.setdefault(r["group"], []).append(r)

    L = []
    A = L.append
    A("# Selector ③ LLM 延迟归因 · prompt/输出体量 vs 网关固定开销")
    A("")
    A(
        f"事件 `{d['event_id']}` ｜ 模型 `{d['model']}` ｜ endpoint `{d['base_url']}` ｜ "
        f"temperature=0.4 ｜ timeout={d['timeout']}s ｜ 非流式 ｜ response_format=json_object ｜ "
        "**只 prompt 内容不同**，其它参数与生产完全一致。"
    )
    A("")

    A("## 1. 三组 prompt 设计")
    A("")
    A("| 组 | 描述 | system 字符 | user 字符 | prompt 字符 |")
    A("| --- | --- | ---: | ---: | ---: |")
    for r in by_g.get("A_real", [])[:1]:
        A(f"| A | 真实 ③ prompt | {r['sys_chars']} | {r['usr_chars']} | {r['prompt_chars']} |")
    for r in by_g.get("B_tiny", [])[:1]:
        A(f"| B | 极小 prompt + 最小 JSON | {r['sys_chars']} | {r['usr_chars']} | {r['prompt_chars']} |")
    for r in by_g.get("C_medium", [])[:1]:
        A(f"| C | ~3000 字符 + 最小 JSON | {r['sys_chars']} | {r['usr_chars']} | {r['prompt_chars']} |")
    A("")

    A("## 2. 逐次明细")
    A("")
    A("| # | 组 | prompt_chars | request_body | response_chars | open | read | total | 成功 |")
    A("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: |")
    for r in rows:
        A(
            f"| {r['index']} | {r['group']} | {r['prompt_chars']} | "
            f"{r['request_body_chars']} | {r['body_chars']} | "
            f"{r['open_sec']:.2f} | {r['read_sec']:.4f} | **{r['elapsed']:.2f}** | "
            f"{'✅' if r['success'] else '❌'} |"
        )
    A("")

    A("## 3. 汇总（min / mean / max，单位秒）")
    A("")
    A("| 组 | 次数 | total | open | read | response_chars |")
    A("| --- | ---: | --- | --- | --- | ---: |")
    for tag in ("A_real", "B_tiny", "C_medium"):
        rs = by_g.get(tag, [])
        ok = [r for r in rs if r["success"]]
        t = [r["elapsed"] for r in ok]
        o = [r["open_sec"] for r in ok]
        rd = [r["read_sec"] for r in ok]
        b = [r["body_chars"] for r in ok if r["body_chars"]]
        A(
            f"| {tag} | {len(ok)} | {_stat(t)} | {_stat(o)} | "
            f"{_stat(rd)} | {_stat(b)} |"
        )
    A("")

    # 关键归因
    b_ok = [r for r in by_g.get("B_tiny", []) if r["success"]]
    c_ok = [r for r in by_g.get("C_medium", []) if r["success"]]
    a_ok = [r for r in by_g.get("A_real", []) if r["success"]]

    A("## 4. 关键归因（4 个问题）")
    A("")
    if b_ok:
        mb = statistics.mean([r["elapsed"] for r in b_ok])
        verdict = "**是**" if mb > 5 else ("**接近边界**" if mb > 2 else "**否**")
        A(f"**Q1 · 极小 prompt 是否仍 > 5s？** {verdict}（B 组均值 **{mb:.2f}s**，最小 {min(r['elapsed'] for r in b_ok):.2f}s）")
    else:
        A("**Q1 · 极小 prompt 是否仍 > 5s？** B 组无成功样本")
    A("")

    if a_ok and b_ok:
        ma = statistics.mean([r["elapsed"] for r in a_ok])
        diff = ma - mb
        pa = a_ok[0]["prompt_chars"]
        pb = b_ok[0]["prompt_chars"]
        A(
            f"**Q2 · 真实 prompt 比极小 prompt 多花多少？**  \n"
            f"mean A = {ma:.2f}s，mean B = {mb:.2f}s，**差 {diff:+.2f}s**  \n"
            f"prompt 体量差：{pa} − {pb} = {pa - pb} 字符（{pa / pb:.1f}×）  \n"
            f"response_chars：A 组 {a_ok[0]['body_chars']} / B 组 {b_ok[0]['body_chars']}"
        )
    else:
        A("**Q2 · 真实 prompt 比极小 prompt 多花多少？** 数据不足")
    A("")

    if b_ok and c_ok:
        mb = statistics.mean([r["elapsed"] for r in b_ok])
        mc = statistics.mean([r["elapsed"] for r in c_ok])
        pb = b_ok[0]["prompt_chars"]
        pc = c_ok[0]["prompt_chars"]
        bb = b_ok[0]["body_chars"]
        bc = c_ok[0]["body_chars"]
        slope = (mc - mb) / max(pc - pb, 1)
        A(
            f"**Q3 · 延迟是否随 prompt 规模明显增长？**  \n"
            f"在 B（~{pb} 字符）→ C（~{pc} 字符）上，输出体量相近（{bb}/{bc} 字符），"
            f"均值 {mb:.2f}s → {mc:.2f}s（Δ {mc - mb:+.2f}s），斜率约 **{slope:.4f} s/字符**（{(slope * 1000):.3f} s/1k 字符）"
        )
    else:
        A("**Q3 · 延迟是否随 prompt 规模明显增长？** 数据不足")
    A("")

    if a_ok and b_ok:
        ma = statistics.mean([r["elapsed"] for r in a_ok])
        mb = statistics.mean([r["elapsed"] for r in b_ok])
        ba = a_ok[0]["body_chars"]
        bb = b_ok[0]["body_chars"]
        delta_prompt = a_ok[0]["prompt_chars"] - b_ok[0]["prompt_chars"]
        delta_body = ba - bb
        A(
            f"**Q4 · 45s 中能否判断存在明显固定网关延迟？**  \n"
            f"- B 组 prompt {b_ok[0]['prompt_chars']} 字符 / 响应 {bb} 字符 / 耗时 {mb:.2f}s ——"
            f"**这是「极小任务 + 最小输出」的下限，可视作该网关的固定开销**。  \n"
            f"- A 组 prompt {a_ok[0]['prompt_chars']} 字符 / 响应 {ba} 字符 / 耗时 {ma:.2f}s。  \n"
            f"- A − B = **{ma - mb:+.2f}s** 来自："
            f"prompt 多 {delta_prompt} 字符 + 输出多 {delta_body} 字符。  \n"
            f"- 若 B > 5s：**判定存在明显固定网关延迟**（与 prompt/输出规模无关，发生在该私有网关/服务侧）。"
        )
    else:
        A("**Q4 · 固定网关延迟？** 数据不足")
    A("")

    # 关键控制
    shas_by_g = {g: {r["prompt_sha"] for r in rs} for g, rs in by_g.items()}
    A("### 控制变量一致性")
    A("")
    for g, shas in shas_by_g.items():
        A(f"- {g}：{len(shas)} 种 prompt（组内 3 次调用应完全相同）")
    temps = {r.get("temperature") for r in rows if r.get("temperature") is not None}
    streams = {r.get("stream") for r in rows}
    tmos = {r.get("timeout") for r in rows if r.get("timeout") is not None}
    A(f"- temperature：{temps} ｜ stream：{streams} ｜ timeout：{tmos}")
    rf = {r.get("has_response_format") for r in rows if r.get("has_response_format") is not None}
    A(f"- response_format：{rf}")
    A("")

    if any(not r["success"] for r in rows):
        A("### 失败明细")
        A("")
        for r in rows:
            if not r["success"]:
                A(f"- #{r['index']} {r['group']} → `{r['error_type']}`: {str(r['error'])[:200]}")
        A("")

    (OUT_DIR / "payload_latency_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
