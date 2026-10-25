"""Selector ③ `enable_thinking` A/B 最终确认实验。

目标
----
验证给同一个请求加一个 `enable_thinking: false` 字段，能否把 Selector ③ 的
LLM 调用从约 40s 稳定降到约 10s，同时选人行为不发生明显改变。

设计
----
- 固定 hot_event_001；上游 EventParser / FeatureBuilder / Scorer 全部
  ``use_llm=False``，保证 ScoreResult 与 ③ prompt 逐字节可复现（用 prompt_sha 校验）。
- 同一个 ``OpenAICompatibleLLMClient``（qwen3.7-plus / 同 endpoint /
  temperature=0.4 / timeout=90 / response_format=json_object / 非流式）。
- **A 组**：原样走 ``generate_json``（thinking 默认开启），5 次。
- **B 组**：**完全相同的请求**，只在 HTTP 层注入 ``enable_thinking: false``，5 次。
- A/B **交错执行**（A1 B1 A2 B2 ...），避免会话内漂移整体偏向某一组。

注入方式（不碰正式代码）
------------------------
在 ``urllib.request.urlopen`` 的钩子中，把已序列化好的 request body 解出 JSON，
塞入 ``enable_thinking: false`` 后重新写回 ``request.data``。
``llm_client`` 没有显式设置 Content-length，urllib 会在发送时按 ``request.data``
重新计算，因此不需要额外维护头（脚本里仍做了防御性清理）。

用法
----
    python tests/ab_selector_llm_thinking.py                  # A/B 各 5 次
    python tests/ab_selector_llm_thinking.py --repeat 1       # 冒烟
    python tests/ab_selector_llm_thinking.py --report-only    # 离线重出报告
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request
from collections import Counter
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
REPEAT = 5

# 当前这次调用是否要注入 enable_thinking:false
_INJECT: bool = False
_CUR: dict = {}
_CALLS: list[dict] = []


# --------------------------------------------------------------------------- #
# HTTP 层钩子：捕获 open/read/body，并可注入 enable_thinking
# --------------------------------------------------------------------------- #
class _TimingResponse:
    def __init__(self, raw, call: dict) -> None:
        self._raw, self._call, self._body = raw, call, None

    def read(self, *a, **kw):
        if self._body is None:
            t0 = perf_counter()
            self._body = self._raw.read(*a, **kw)
            self._call["read_sec"] = perf_counter() - t0
            text = self._body.decode("utf-8", errors="replace")
            self._call["body_chars"] = len(text)
            self._call["raw_text"] = text
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


def _h_urlopen(request, **kwargs):
    call: dict = {"timeout_arg": kwargs.get("timeout")}
    payload: dict = {}
    try:
        raw = request.data or b""
        payload = json.loads(raw.decode("utf-8"))

        if _INJECT:
            payload["enable_thinking"] = False
            new = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request.data = new
            # 防御：清掉可能已存在的 Content-length，让 handler 用新长度重算
            try:
                request.remove_header("Content-length")
            except Exception:  # noqa: BLE001
                pass
            request.unredirected_hdrs.pop("Content-length", None)
            request.unredirected_hdrs.pop("Content-Length", None)
            raw = new

        call["request_body_chars"] = len(raw)
        call["enable_thinking_in_body"] = payload.get("enable_thinking", "<absent>")
        call["model"] = payload.get("model")
        call["temperature"] = payload.get("temperature")
        call["stream"] = payload.get("stream")
        call["has_response_format"] = "response_format" in payload
    except Exception as exc:  # noqa: BLE001
        call["hook_error"] = f"{type(exc).__name__}: {exc}"

    t0 = perf_counter()
    try:
        raw_resp = _real_urlopen(request, **kwargs)
    except Exception as exc:  # noqa: BLE001
        call["open_sec"] = perf_counter() - t0
        call["error"] = f"{type(exc).__name__}: {exc}"
        _CALLS.append(call)
        raise
    call["open_sec"] = perf_counter() - t0
    _CALLS.append(call)
    return _TimingResponse(raw_resp, call)


def _h_generate_json(client, *, system_prompt, user_prompt):
    _CUR["sys_chars"] = len(system_prompt)
    _CUR["usr_chars"] = len(user_prompt)
    _CUR["prompt_chars"] = len(system_prompt) + len(user_prompt)
    _CUR["prompt_sha"] = hashlib.sha256(
        (system_prompt + "\x00" + user_prompt).encode("utf-8")
    ).hexdigest()[:16]
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
    _CUR["error"] = err
    if err is not None:
        raise RuntimeError(err)
    return result


_real_urlopen = urllib.request.urlopen
_real_generate_json = OpenAICompatibleLLMClient.generate_json


def _install() -> None:
    urllib.request.urlopen = _h_urlopen
    OpenAICompatibleLLMClient.generate_json = _h_generate_json


def _uninstall() -> None:
    urllib.request.urlopen = _real_urlopen
    OpenAICompatibleLLMClient.generate_json = _real_generate_json


# --------------------------------------------------------------------------- #
# 单次调用
# --------------------------------------------------------------------------- #
def _ids_roles(parsed: dict) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    roles: list[str] = []
    so = parsed.get("selected_order") if isinstance(parsed, dict) else None
    if isinstance(so, list):
        for it in so:
            if isinstance(it, dict):
                u = str(it.get("user_id", "")).strip()
                r = str(it.get("recommended_role", "")).strip()
                if u:
                    ids.append(u)
                    roles.append(r)
    return ids, roles


def _call_once(client, *, sys_p: str, usr_p: str, inject: bool) -> dict:
    global _CUR, _CALLS, _INJECT
    _CUR = {}
    _CALLS = []
    _INJECT = inject
    parsed: dict | None = None
    try:
        parsed = client.generate_json(system_prompt=sys_p, user_prompt=usr_p)
    except Exception:  # noqa: BLE001
        pass
    finally:
        _INJECT = False

    calls = _CALLS
    # 取最后一次**拿到了响应体**的 HTTP 记录（generate_json 失败重发时会有多次）
    last = next((c for c in reversed(calls) if c.get("raw_text")), (calls[-1] if calls else {}))

    usage: dict = {}
    msg: dict = {}
    if last.get("raw_text"):
        try:
            d = json.loads(last["raw_text"])
            usage = d.get("usage") or {}
            msg = ((d.get("choices") or [{}])[0] or {}).get("message") or {}
        except Exception:  # noqa: BLE001
            pass
    details = usage.get("completion_tokens_details") or {}
    reasoning_content = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""

    ids, roles = _ids_roles(parsed or {})

    return {
        "elapsed": round(_CUR.get("elapsed", -1.0), 4),
        "success": bool(_CUR.get("success", False)) and parsed is not None,
        "error": _CUR.get("error"),
        "prompt_sha": _CUR.get("prompt_sha"),
        "prompt_chars": _CUR.get("prompt_chars"),
        "sys_chars": _CUR.get("sys_chars"),
        "usr_chars": _CUR.get("usr_chars"),
        "request_body_chars": last.get("request_body_chars"),
        "enable_thinking_in_body": last.get("enable_thinking_in_body"),
        "body_chars": last.get("body_chars"),
        "open_sec": round(sum(c.get("open_sec", 0.0) or 0.0 for c in calls), 4),
        "read_sec": round(sum(c.get("read_sec", 0.0) or 0.0 for c in calls), 4),
        "http_calls": len(calls),
        "model": last.get("model"),
        "temperature": last.get("temperature"),
        "stream": last.get("stream"),
        "has_response_format": last.get("has_response_format"),
        "timeout": last.get("timeout_arg"),
        "message_keys": sorted(msg.keys()),
        "reasoning_content_chars": len(reasoning_content),
        "content_chars": len(content),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "text_tokens": details.get("text_tokens"),
        "selected_ids": ids,
        "selected_roles": roles,
    }


# --------------------------------------------------------------------------- #
# 准备 ScoreResult + ③ prompt
# --------------------------------------------------------------------------- #
def _prepare(event_id: str) -> tuple[str, str, dict]:
    events = load_hot_events(PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json")
    hot = next(e for e in events if str(e.get("event_id")) == event_id)
    payload = hot_event_to_pipeline_payload(
        hot, max_selected_nodes=5, risk_level=None,
        campaign_window_hours=24, max_frequency_per_day=3, allowed_platforms=None,
    )
    event = RuleBasedEventParser().parse(payload, workspace_root=PROJECT_ROOT, use_llm=False)
    fr = FeatureBuilder().build_from_loader(
        workspace_root=PROJECT_ROOT, event=event, use_llm=False,
        recall_size=200, recall_mode="platform_aware",
        dispatch_platform=DISPATCH_PLATFORM,
    )
    score_result = Scorer().score(fr)

    safe_pool = [
        n for n in score_result.node_scores
        if getattr(n, "eligible", False) and not getattr(n, "manual_review_required", False)
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
    max_selected = score_result.event.constraints.max_selected_nodes
    sys_p = build_selector_system_prompt()
    usr_p = build_selector_user_prompt(
        event=score_result.event, max_selected=max_selected, shortlist=shortlist_payload
    )
    meta = {
        "rerank_top_k": len(shortlist_payload),
        "max_selected": max_selected,
        "safe_pool": len(safe_pool),
    }
    return sys_p, usr_p, meta


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", default=EVENT_ID)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    ap.add_argument("--report-only", action="store_true", help="只读 raw.json 重出报告")
    ap.add_argument(
        "--gap",
        type=float,
        default=2.0,
        help="相邻两次 LLM 调用之间的间隔秒数（降低被网关断连的概率）",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "thinking_ab_raw.json"

    if args.report_only:
        if not raw_path.exists():
            print(f"missing {raw_path}")
            sys.exit(1)
        _write_report(json.loads(raw_path.read_text(encoding="utf-8")))
        return

    client = OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT)
    if client is None:
        print("无 LLM 凭据。")
        sys.exit(2)

    _log(f"base_url : {client.base_url}")
    _log(f"provider : {client.provider}")
    _log(f"model    : {client.model}")
    _log(f"timeout  : {client.timeout}")
    _log("")

    _log("[prep] 构造 ScoreResult（上游全部 use_llm=False）...")
    sys_p, usr_p, meta = _prepare(args.event_id)
    _log(
        f"[prep] prompt={len(sys_p) + len(usr_p)} chars  "
        f"rerank_top_k={meta['rerank_top_k']} max_selected={meta['max_selected']} "
        f"safe_pool={meta['safe_pool']}"
    )
    _log("")

    plan: list[tuple[str, bool]] = []
    for i in range(args.repeat):
        plan.append(("A_thinking_on", False))
        plan.append(("B_thinking_off", True))

    rows: list[dict] = []
    _install()
    try:
        for tag, inject in plan:
            idx = len(rows) + 1
            _log(f"[run] {idx}/{len(plan)}  {tag} ...")
            row = _call_once(client, sys_p=sys_p, usr_p=usr_p, inject=inject)
            row["group"] = tag
            row["index"] = idx
            rows.append(row)
            _log(
                f"      -> {row['elapsed']:.2f}s  body={row['body_chars']}  "
                f"reasoning={row['reasoning_content_chars']}  "
                f"content={row['content_chars']}  "
                f"http={row['http_calls']}  success={row['success']}"
                + (f"  err={row['error']}" if not row["success"] else "")
            )
            if idx < len(plan):
                time.sleep(args.gap)
    finally:
        _uninstall()

    out = {
        "event_id": args.event_id,
        "platform": DISPATCH_PLATFORM,
        "base_url": client.base_url,
        "provider": client.provider,
        "model": client.model,
        "timeout": client.timeout,
        "temperature": 0.4,
        "prep": meta,
        "rows": rows,
    }
    raw_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_report(out)


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _stats_line(name: str, vals: list[float]) -> str:
    if not vals:
        return f"| {name} | 0 | - | - | - | - |"
    return (
        f"| {name} | {len(vals)} | {statistics.mean(vals):.2f} | "
        f"{statistics.median(vals):.2f} | {min(vals):.2f} | {max(vals):.2f} |"
    )


def _write_report(d: dict) -> None:
    rows = d["rows"]
    by_g: dict[str, list[dict]] = {}
    for r in rows:
        by_g.setdefault(r["group"], []).append(r)
    A = by_g.get("A_thinking_on", [])
    B = by_g.get("B_thinking_off", [])
    a_ok = [r for r in A if r["success"]]
    b_ok = [r for r in B if r["success"]]

    at = [r["elapsed"] for r in a_ok]
    bt = [r["elapsed"] for r in b_ok]

    L = []
    w = L.append
    w("# Selector ③ `enable_thinking` A/B 最终确认")
    w("")
    w(
        f"事件 `{d['event_id']}` ｜ 模型 `{d['model']}` ｜ endpoint `{d['base_url']}` ｜ "
        f"temperature={d['temperature']} ｜ timeout={d['timeout']}s ｜ 非流式 ｜ "
        f"response_format=json_object ｜ rerank_top_k={d['prep']['rerank_top_k']} "
        f"max_selected={d['prep']['max_selected']}"
    )
    w("")
    w(
        "A 组 = 原样请求（thinking 默认开启）；B 组 = **完全相同的请求**，"
        "仅在 HTTP 层注入 `enable_thinking: false`。A/B **交错执行**以对冲会话内漂移。"
    )
    w("")

    w("## ① A/B 耗时统计（秒）")
    w("")
    w("| 组 | 次数 | mean | median | min | max |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    w(_stats_line("A · thinking 开", at))
    w(_stats_line("B · `enable_thinking:false`", bt))
    w("")

    if at and bt:
        ma, mb = statistics.mean(at), statistics.mean(bt)
        speedup = ma / max(mb, 1e-6)
        w("## ② 加速比与节省")
        w("")
        w(f"- A mean **{ma:.2f}s** → B mean **{mb:.2f}s**")
        w(f"- **加速比 {speedup:.2f}x**，**节省 {(ma - mb):.2f}s（{(ma - mb) / ma * 100:.1f}%）**")
        w("")

    w("## ③ token 对比（mean）")
    w("")
    w("| 组 | prompt_tokens | completion_tokens | reasoning_tokens | text_tokens |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for tag, name, rs in (("A", "A · thinking 开", a_ok), ("B", "B · thinking 关", b_ok)):
        if not rs:
            w(f"| {name} | - | - | - | - |")
            continue
        def mv(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return statistics.mean(v) if v else None
        w(
            f"| {name} | {_fmt(mv('prompt_tokens'))} | {_fmt(mv('completion_tokens'))} | "
            f"**{_fmt(mv('reasoning_tokens'))}** | {_fmt(mv('text_tokens'))} |"
        )
    w("")

    # 选人一致性
    w("## ④ A/B 选人一致性")
    w("")
    a_sets = [set(r["selected_ids"]) for r in a_ok]
    b_sets = [set(r["selected_ids"]) for r in b_ok]
    if a_sets and b_sets:
        ref = a_sets[0]
        # A 内部：两两交集均值（自身噪声基线）
        a_pairs = [
            len(a_sets[i] & a_sets[j])
            for i in range(len(a_sets)) for j in range(len(a_sets)) if i < j
        ]
        a_self_exact = sum(1 for s in a_sets if s == ref) / len(a_sets)
        # B 内部
        b_pairs = [
            len(b_sets[i] & b_sets[j])
            for i in range(len(b_sets)) for j in range(len(b_sets)) if i < j
        ]
        b_self_exact = sum(1 for s in b_sets if s == b_sets[0]) / len(b_sets)
        # 跨组：所有 (A_i, B_j) 对
        cross = [len(x & y) for x in a_sets for y in b_sets]
        cross_exact = sum(1 for s in b_sets if s == ref) / len(b_sets)
        n = len(ref)

        w(f"- A 参考集（A 第 1 次）: `{sorted(ref)}`  （{n} 人）")
        w("")
        w("| 对比 | 平均交集 | 与参考集完全一致率 |")
        w("| --- | ---: | ---: |")
        w(
            f"| A 内部（自身噪声基线） | "
            f"{statistics.mean(a_pairs):.2f}/{n} | {a_self_exact * 100:.0f}% |"
        )
        w(f"| B 内部 | {statistics.mean(b_pairs):.2f}/{n} | {b_self_exact * 100:.0f}% |")
        w(
            f"| **跨组 A×B（{len(cross)} 组配对）** | "
            f"**{statistics.mean(cross):.2f}/{n}** | **{cross_exact * 100:.0f}%** |"
        )
        w("")
        w("逐次 selected_ids：")
        w("")
        for r in rows:
            if r["success"]:
                w(f"  - {r['group']} #{r['index']}: `{r['selected_ids']}`  roles=`{r['selected_roles']}`")
        w("")

        # ---- 逐位置分析：定位差异到底发生在哪一位 ----
        n_pos = max(len(r["selected_ids"]) for r in a_ok + b_ok)
        w("### 逐位置众数对比（定位差异发生在第几位）")
        w("")
        w("| 位置 | A 众数（出现/总数） | B 众数（出现/总数） | 一致 |")
        w("| ---: | --- | --- | :-: |")
        same_pos = 0
        for k in range(n_pos):
            a_col = [r["selected_ids"][k] for r in a_ok if len(r["selected_ids"]) > k]
            b_col = [r["selected_ids"][k] for r in b_ok if len(r["selected_ids"]) > k]
            if not a_col or not b_col:
                continue
            ma_ = Counter(a_col).most_common(1)[0]
            mb_ = Counter(b_col).most_common(1)[0]
            hit = ma_[0] == mb_[0]
            same_pos += int(hit)
            w(
                f"| {k + 1} | `{ma_[0]}`（{ma_[1]}/{len(a_col)}） | "
                f"`{mb_[0]}`（{mb_[1]}/{len(b_col)}） | {'✅' if hit else '⚠️'} |"
            )
        w("")
        w(f"- 众数完全相同的位置：**{same_pos}/{n_pos}**")
        w("")

        # ---- top-K 集合一致率 ----
        w("### top-K 集合一致率（忽略顺序，只看集合）")
        w("")
        w("| K | A 内部（自身基线） | 跨组 A×B |")
        w("| ---: | ---: | ---: |")
        for k in range(1, n_pos + 1):
            a_top = [tuple(sorted(r["selected_ids"][:k])) for r in a_ok if len(r["selected_ids"]) >= k]
            b_top = [tuple(sorted(r["selected_ids"][:k])) for r in b_ok if len(r["selected_ids"]) >= k]
            a_den = len(a_top) * (len(a_top) - 1) // 2
            if a_den == 0:
                continue
            a_p = sum(
                1 for i in range(len(a_top)) for j in range(i + 1, len(a_top)) if a_top[i] == a_top[j]
            )
            cross_n = sum(1 for x in a_top for y in b_top if x == y)
            cross_den = len(a_top) * len(b_top)
            w(f"| {k} | {a_p / a_den * 100:.0f}% | **{cross_n / cross_den * 100:.0f}%** |")
        w("")

    w("## ⑤ A 自身抖动（5 次）")
    w("")
    if at:
        sd = statistics.pstdev(at) if len(at) > 1 else 0.0
        w(f"- 范围 **{min(at):.2f} ~ {max(at):.2f}s**，极差 **{max(at) - min(at):.2f}s**，标准差 {sd:.2f}s")
        w(f"- 逐次：{', '.join(f'{v:.2f}' for v in at)}")
    else:
        w("- 无成功样本")
    w("")

    w("## ⑥ B 自身抖动（5 次）")
    w("")
    if bt:
        sd = statistics.pstdev(bt) if len(bt) > 1 else 0.0
        w(f"- 范围 **{min(bt):.2f} ~ {max(bt):.2f}s**，极差 **{max(bt) - min(bt):.2f}s**，标准差 {sd:.2f}s")
        w(f"- 逐次：{', '.join(f'{v:.2f}' for v in bt)}")
    else:
        w("- 无成功样本")
    w("")

    w("## ⑦ prompt_sha 一致性")
    w("")
    shas = {r["prompt_sha"] for r in rows if r.get("prompt_sha")}
    w(f"- 全部 {len(rows)} 次调用共 **{len(shas)}** 种 prompt_sha：`{sorted(shas)}`")
    w(f"- {'✅ A/B 用的是逐字节相同的同一份 ③ prompt' if len(shas) == 1 else '❌ prompt 不一致，实验无效'}")
    w("")

    w("## ⑧ 失败 / 重试")
    w("")
    failed = [r for r in rows if not r["success"]]
    retried = [r for r in rows if r["http_calls"] > 1]
    w(f"- 失败次数：**{len(failed)}** / {len(rows)}")
    for r in failed:
        w(f"  - {r['group']} #{r['index']}: `{r['error']}`")
    w(
        f"- 发生重试（单次调用内 HTTP > 1 次，即 response_format 失败回退）的次数："
        f"**{len(retried)}**"
    )
    for r in retried:
        w(f"  - {r['group']} #{r['index']}: http_calls={r['http_calls']}")
    if not failed and not retried:
        w("- 无失败、无重试。")
    w("")

    w("## 逐次明细")
    w("")
    w("| # | 组 | elapsed | body_chars | reasoning_chars | content_chars | reasoning_tok | text_tok | http | 成功 |")
    w("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |")
    for r in rows:
        w(
            f"| {r['index']} | {r['group']} | **{r['elapsed']:.2f}** | {r['body_chars']} | "
            f"{r['reasoning_content_chars']} | {r['content_chars']} | "
            f"{r['reasoning_tokens']} | {r['text_tokens']} | {r['http_calls']} | "
            f"{'✅' if r['success'] else '❌'} |"
        )
    w("")

    # ---- 特别检查 ----
    w("## 特别检查")
    w("")
    if bt:
        in_range = all(8.0 <= v <= 12.0 for v in bt)
        w(
            f"- **B 是否稳定在 8–12s？** "
            f"{'✅ 是' if in_range else '⚠️ 否'}（B 范围 {min(bt):.2f} ~ {max(bt):.2f}s，"
            f"mean {statistics.mean(bt):.2f}s）"
        )
        rz = [r["reasoning_content_chars"] for r in b_ok]
        rz_ok = all(v == 0 for v in rz)
        w(
            f"- **B 的 reasoning_content 是否稳定为 0？** "
            f"{'✅ 是（全部为 0）' if rz_ok else f'❌ 否（{rz}）'}"
        )
        rt = [r["reasoning_tokens"] for r in b_ok if r.get("reasoning_tokens") is not None]
        if rt:
            w(f"  - B 的 reasoning_tokens：{rt}")
    if a_sets and b_sets:
        n = len(a_sets[0])
        a_self = statistics.mean(
            [len(a_sets[i] & a_sets[j]) for i in range(len(a_sets)) for j in range(len(a_sets)) if i < j]
        )
        cross = statistics.mean([len(x & y) for x in a_sets for y in b_sets])
        w(
            f"- **B 的选人是否与 A 基本一致？** 跨组平均交集 **{cross:.2f}/{n}**，"
            f"A 自身噪声基线 **{a_self:.2f}/{n}**  \n"
            f"  {'✅ 跨组一致性不劣于 A 自身抖动，人员差异主要来自 temperature 随机性，不应归因于 enable_thinking:false' if cross >= a_self - 0.3 else '⚠️ 跨组一致性低于 A 自身基线，需要进一步判断'}"
        )
        top_a = [r["selected_ids"][:3] for r in a_ok]
        top_b = [r["selected_ids"][:3] for r in b_ok]
        same_top3 = sum(1 for t in top_b if t == top_a[0]) / len(top_b) if top_b else 0
        w(f"  - B 的 top-3 与 A 第 1 次 top-3 完全相同率：**{same_top3 * 100:.0f}%**")
    w("")

    (OUT_DIR / "thinking_ab_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
