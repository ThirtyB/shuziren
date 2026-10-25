"""Selector ③ 紧凑输出 C 方案实验。

A = 当前生产 ③ prompt（与 Selector 实际使用的一致）。
C = **完整保留输入字段**（llm_feature_score / topic_match_score / semantic_tags /
   matched_keywords / risk_level / selection_reasons）—— 模型仍可参照这些信息做决策，
   但**严格禁止**把它们复读到输出里；输出只允许：
     selected_order: [{user_id, recommended_role, reason(≤20字)}]
     fallback_order: [user_id]

设计要点
--------
- **A/C 的唯一变量是 system prompt**：user prompt 与 A 完全一致（含 6 个被 C 显式
  列为"输入可参考、输出禁止"的字段），所以任何差异都归因于"输出 schema 约束"。
- 同样 3 次真实 LLM 调用，对比 (A) 生产 ③ prompt 与 (C) 输入全保 + 输出压缩。
- 全部用运行时 monkeypatch，不动任何源文件。

用法：
    python tests/ab_selector_llm_compact.py
    python tests/ab_selector_llm_compact.py --report-only
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

# 每次调用的运行时数据
_CUR: dict = {}
_CALLS: list[dict] = []


# --------------------------------------------------------------------------- #
# 计时钩子
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# prompt 构造
# --------------------------------------------------------------------------- #
# 这 6 个字段 = A 在 shortlist 里给 LLM 的全部"非身份"信号
_INPUT_FIELDS = (
    "llm_feature_score",
    "topic_match_score",
    "semantic_tags",
    "matched_keywords",
    "risk_level",
    "selection_reasons",
)


def _build_shortlist(score_result) -> tuple[list[dict], int]:
    safe_pool = [
        n
        for n in score_result.node_scores
        if getattr(n, "eligible", False)
        and not getattr(n, "manual_review_required", False)
    ]
    shortlist = safe_pool[: score_result.event.dispatch_preferences.rerank_top_k]
    payload = [
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
    return payload, score_result.event.constraints.max_selected_nodes


def _prompt_A(score_result, shortlist_payload, max_selected) -> tuple[str, str]:
    """A 组：与生产完全一致。"""
    return build_selector_system_prompt(), build_selector_user_prompt(
        event=score_result.event, max_selected=max_selected, shortlist=shortlist_payload
    )


def _prompt_C(score_result, shortlist_payload, max_selected) -> tuple[str, str]:
    """C 组：user prompt 与 A 完全一致，system prompt 显式禁止复读 6 个输入字段。

    输出 schema：
      selected_order: [{"user_id","recommended_role","reason"}]  （reason ≤ 20 字）
      fallback_order: ["user_id", ...]
    无 global_notes / reasoning_map / 其它字段。
    """
    sys_p = (
        "你是节点选择助手。从给定候选中选择一组主选 + fallback。\n"
        "**输入与输出必须严格分离**：\n"
        f"- 输入字段（shortlist 提供的全部信息，含 "
        f"{', '.join(_INPUT_FIELDS)} 等）你**可以**用来决策，\n"
        f"  但**禁止**把它们复读、复制、改名复写到输出 JSON 中。\n"
        "- 输出只允许以下 schema，禁止任何其它字段：\n\n"
        "{\n"
        '  "selected_order": [\n'
        '    {"user_id": "<id>", "recommended_role": "<role>", "reason": "<≤20 字>"}\n'
        "  ],\n"
        '  "fallback_order": ["<id>"]\n'
        "}\n\n"
        f"规则：\n"
        f"1. selected_order 最多 {max_selected} 项；每项只含 3 个字段：\n"
        f"   - user_id: shortlist 中存在的 ID\n"
        f"   - recommended_role: 只能是 core_publish_node / interaction_response_node / "
        f"amplification_node / support_node 之一\n"
        f"   - reason: ≤ 20 字（中文/英文均可），只写最关键的 1 个理由，**禁止**写长句\n"
        f"2. fallback_order: user_id 字符串数组，**只**放 user_id，**不放**任何分析\n"
        f"3. 禁止输出：global_notes、reasoning、analysis、explanation、llm_feature_score、"
        f"topic_match_score、semantic_tags、matched_keywords、risk_level、selection_reasons、"
        f"final_score、role_hint 任何字段或它们的同名变体\n"
        f"4. 禁止 Markdown 代码块包裹；禁止解释前言"
    )
    # user prompt 维持与 A 完全一致（这正是 C 的"输入全保"承诺）
    usr_p = build_selector_user_prompt(
        event=score_result.event, max_selected=max_selected, shortlist=shortlist_payload
    )
    return sys_p, usr_p


# --------------------------------------------------------------------------- #
# 单次调用 + 拆解
# --------------------------------------------------------------------------- #
def _bloat_breakdown(parsed: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in (parsed or {}).items():
        try:
            out[k] = len(json.dumps(v, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            out[k] = -1
    return out


def _normalize_ids_roles(selected_order) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    roles: list[str] = []
    if not isinstance(selected_order, list):
        return ids, roles
    for it in selected_order:
        if isinstance(it, dict):
            u = str(it.get("user_id", "")).strip()
            r = str(it.get("recommended_role", "")).strip()
            if u:
                ids.append(u)
                roles.append(r)
    return ids, roles


def _call_once(client, *, sys_p: str, usr_p: str) -> dict:
    global _CUR, _CALLS
    _CUR = {}
    _CALLS = []
    parsed: dict | None = None
    try:
        parsed = client.generate_json(system_prompt=sys_p, user_prompt=usr_p)
    except Exception as exc:  # noqa: BLE001
        pass

    http = _CALLS
    elapsed = _CUR.get("elapsed", -1.0)
    succ = bool(_CUR.get("success", False)) and parsed is not None
    # generate_json 失败时会去掉 response_format 重发一次，
    # 所以要取**最后一次**成功的 HTTP 记录，而不是 http[0]（可能是失败的第一次）
    last = http[-1] if http else {}

    so = parsed.get("selected_order") if isinstance(parsed, dict) else None
    so_count = len(so) if isinstance(so, list) else 0
    fo = parsed.get("fallback_order") if isinstance(parsed, dict) else None
    fo_count = len(fo) if isinstance(fo, list) else 0

    bloat = _bloat_breakdown(parsed) if isinstance(parsed, dict) else {}

    ids, roles = _normalize_ids_roles(so)
    reason_lens = [
        len(str(it.get("reason", ""))) if isinstance(it, dict) else 0
        for it in (so or [])
    ]

    return {
        "elapsed": round(elapsed, 4),
        "success": succ,
        "error_type": _CUR.get("error_type"),
        "error": _CUR.get("error"),
        "sys_chars": _CUR.get("sys_chars"),
        "usr_chars": _CUR.get("usr_chars"),
        "prompt_chars": _CUR.get("prompt_chars"),
        "prompt_sha": _CUR.get("prompt_sha"),
        "request_body_chars": last.get("request_body_chars"),
        "body_chars": last.get("body_chars"),
        "open_sec": round(sum(c.get("open_sec", 0.0) or 0.0 for c in http), 4),
        "read_sec": round(sum(c.get("read_sec", 0.0) or 0.0 for c in http), 4),
        "http_calls": len(http),
        "has_response_format": last.get("has_response_format"),
        "temperature": last.get("temperature"),
        "stream": last.get("stream_raw"),
        "timeout": last.get("timeout_arg"),
        "parse_ok": succ,
        "selected_order_count": so_count,
        "selected_ids": ids,
        "selected_roles": roles,
        "reason_lens": reason_lens,
        "fallback_count": fo_count,
        "bloat": bloat,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", default=EVENT_ID)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    ap.add_argument(
        "--report-only", action="store_true", help="只读 raw.json 重出报告（不发 API）"
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "compact_raw.json"

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
    shortlist_payload, max_selected = _build_shortlist(score_result)
    _log(
        f"[prep] rerank_top_k={len(shortlist_payload)} max_selected={max_selected}"
    )

    a_sys, a_usr = _prompt_A(score_result, shortlist_payload, max_selected)
    c_sys, c_usr = _prompt_C(score_result, shortlist_payload, max_selected)
    _log(
        f"[prep] A prompt={len(a_sys) + len(a_usr)} chars  "
        f"C prompt={len(c_sys) + len(c_usr)} chars  "
        f"A user==C user? {a_usr == c_usr}"
    )
    _log("")

    client = OpenAICompatibleLLMClient(
        api_key=base.api_key, base_url=base.base_url, model=MODEL,
        provider=base.provider, timeout=base.timeout,
    )

    plan: list[tuple[str, str, str]] = []
    for _ in range(args.repeat):
        plan.append(("A_prod", a_sys, a_usr))
    for _ in range(args.repeat):
        plan.append(("C_compact", c_sys, c_usr))

    rows: list[dict] = []
    _install()
    try:
        for tag, sp, up in plan:
            idx = len(rows) + 1
            _log(f"[run] {idx}/{len(plan)}  {tag} ...")
            row = _call_once(client, sys_p=sp, usr_p=up)
            row.update({"group": tag, "index": idx})
            rows.append(row)
            _log(
                f"      -> {row['elapsed']:.2f}s  body={row['body_chars']}  "
                f"so={row['selected_order_count']}  success={row['success']}"
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
        "rerank_top_k": len(shortlist_payload),
        "max_selected": max_selected,
        "a_user_eq_c_user": a_usr == c_usr,
        "rows": rows,
    }
    raw_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_report(out)


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _stat(v) -> str:
    vals = [x for x in v if x is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{min(vals):.2f} / {statistics.mean(vals):.2f} / {max(vals):.2f}"


def _write_report(d: dict) -> None:
    rows = d["rows"]
    by_g: dict[str, list[dict]] = {}
    for r in rows:
        by_g.setdefault(r["group"], []).append(r)
    A = by_g.get("A_prod", [])
    C = by_g.get("C_compact", [])

    L = []
    w = L.append
    w("# Selector ③ 紧凑输出 C 方案 · A(生产) vs C(输入全保 + 输出压缩)")
    w("")
    w(
        f"事件 `{d['event_id']}` ｜ 模型 `{d['model']}` ｜ endpoint `{d['base_url']}` ｜ "
        f"temperature=0.4 ｜ timeout={d['timeout']}s ｜ 非流式 ｜ response_format=json_object ｜ "
        f"rerank_top_k={d['rerank_top_k']} max_selected={d['max_selected']}"
    )
    w("")
    w(
        f"**A/C user prompt 逐字节相同：{d.get('a_user_eq_c_user')}**  \n"
        f"**唯一变量是 system prompt**：C 显式禁止 LLM 复读 "
        f"{', '.join(_INPUT_FIELDS)} 等输入字段到输出，输出只允许 "
        f"`selected_order` (user_id+recommended_role+reason≤20字) 与 `fallback_order`。"
    )
    w("")

    w("## 1. prompt 设计")
    w("")
    w("| 组 | system 字符 | user 字符 | prompt 字符 |")
    w("| --- | ---: | ---: | ---: |")
    for r in A[:1]:
        w(f"| A · 生产 ③ | {r['sys_chars']} | {r['usr_chars']} | {r['prompt_chars']} |")
    for r in C[:1]:
        w(f"| C · 输入全保 + 输出压缩 | {r['sys_chars']} | {r['usr_chars']} | {r['prompt_chars']} |")
    w("")

    w("## 2. 逐次明细")
    w("")
    w("| # | 组 | elapsed | body_chars | request_body | 解析 | selected_order | fallback |")
    w("| ---: | --- | ---: | ---: | ---: | :-: | ---: | ---: |")
    for r in rows:
        w(
            f"| {r['index']} | {r['group']} | **{r['elapsed']:.2f}** | {r['body_chars']} | "
            f"{r['request_body_chars']} | {'✅' if r['parse_ok'] else '❌'} | "
            f"{r['selected_order_count']} | {r['fallback_count']} |"
        )
    w("")

    w("## 3. 汇总（min / mean / max，单位秒）")
    w("")
    w("| 组 | 次数 | elapsed | body_chars |")
    w("| --- | ---: | --- | ---: |")
    for tag, name in (("A_prod", "A · 生产"), ("C_compact", "C · 输入全保 + 输出压缩")):
        rs = by_g.get(tag, [])
        ok = [r for r in rs if r["success"]]
        t = [r["elapsed"] for r in ok]
        b = [r["body_chars"] for r in ok if r["body_chars"]]
        w(
            f"| {name} | {len(ok)} | {_stat(t)} | "
            f"{_stat(b) if b else '-'} |"
        )
    w("")

    # 选人一致性
    a_ref = next((r for r in A if r["success"]), None)
    c_ok = [r for r in C if r["success"]]
    a_ok = [r for r in A if r["success"]]
    w("## 4. 选人一致性")
    w("")
    if a_ref:
        a_ids = a_ref["selected_ids"]
        a_roles = a_ref["selected_roles"]
        w(f"**A 参考 (call #{a_ref['index']})**: {a_ids}")
        w(f"  roles: {a_roles}")
        w("")

        # A 组内部一致性（噪声参考）
        if len(a_ok) > 1:
            w("**A 组自身噪声参考**（同一 prompt、不同次调用）：")
            for r in a_ok:
                same = r["selected_ids"] == a_ids
                same_set = set(r["selected_ids"]) == set(a_ids)
                w(
                    f"  - A#{r['index']}: ids={r['selected_ids']}  "
                    f"{'✅ 与参考完全一致' if same else ('(set 一致)' if same_set else '⚠️ set 也不同')}"
                )
            w("")

        if c_ok:
            w("**C vs A 参考**：")
            for r in c_ok:
                same_order = r["selected_ids"] == a_ids
                same_set = set(r["selected_ids"]) == set(a_ids)
                role_match = r["selected_roles"] == a_roles
                w(
                    f"  - C#{r['index']}: ids={r['selected_ids']}  "
                    f"roles={r['selected_roles']}  "
                    f"{'✅ 完全一致' if same_order and role_match else ('⚠️ 顺序/角色不同' + ('(set 一致)' if same_set else '(set 也不一致)'))}"
                )
            w("")

        # C 自身稳定性
        if len(c_ok) > 1:
            w("**C 组自身稳定性**（同一 prompt、不同次调用）：")
            for r in c_ok:
                same = r["selected_ids"] == c_ok[0]["selected_ids"]
                w(
                    f"  - C#{r['index']}: ids={r['selected_ids']}  "
                    f"{'✅' if same else '⚠️'} 与 C#{c_ok[0]['index']} {'一致' if same else '不同'}"
                )
            w("")

        # reason 字段长度
        if c_ok:
            all_lens = [n for r in c_ok for n in r.get("reason_lens", [])]
            if all_lens:
                w(
                    f"**C reason 字段长度**：min={min(all_lens)} max={max(all_lens)} "
                    f"mean={statistics.mean(all_lens):.1f}（约束为 ≤ 20）"
                )
                w("")
    else:
        w("A 组无成功样本")
    w("")

    # C 输出体量拆解
    w("## 5. C 组输出字段拆解（顶层紧凑 dump）")
    w("")
    for r in C:
        if not r["success"]:
            continue
        w(f"**C call #{r['index']}** (elapsed={r['elapsed']:.2f}s, body={r['body_chars']}):")
        for k, v in r["bloat"].items():
            w(f"  - `{k}`: {v} 字符（json.dumps 估算）")
        w("")

    # 节省
    if a_ok and c_ok:
        ma = statistics.mean([r["elapsed"] for r in a_ok])
        mc = statistics.mean([r["elapsed"] for r in c_ok])
        saved = ma - mc
        bca = statistics.mean([r["body_chars"] for r in a_ok if r["body_chars"]])
        bcc = statistics.mean([r["body_chars"] for r in c_ok if r["body_chars"]])
        a_missing = sum(1 for r in a_ok if not r["body_chars"])
        c_missing = sum(1 for r in c_ok if not r["body_chars"])
        if a_missing or c_missing:
            w(
                f"> 注：A 有 {a_missing} 次、C 有 {c_missing} 次未捕获到 body 长度"
                f"（响应体在重试分支上未回填），均值只统计到样本。"
            )
            w("")
        w("## 6. 节省估算")
        w("")
        w(
            f"- A 均值 **{ma:.2f}s**、C 均值 **{mc:.2f}s**，"
            f"**实际节省 {saved:+.2f}s**（{saved / ma * 100:.1f}%）"
        )
        w(
            f"- A 输出 {bca:.0f} 字符 → C 输出 {bcc:.0f} 字符，"
            f"绝对值省 {bca - bcc:.0f} 字符，"
            f"相对仅压缩到原来的 {bcc / bca * 100:.1f}%"
            f"（即只压掉 {(1 - bcc / bca) * 100:.1f}%）"
        )
        w("")

    # 核心判定
    w("## 7. 判定")
    w("")
    if a_ok and c_ok:
        ma = statistics.mean([r["elapsed"] for r in a_ok])
        mc = statistics.mean([r["elapsed"] for r in c_ok])

        # 一致性：以 A 参考为基准
        a_ref = a_ok[0]
        a_ids = a_ref["selected_ids"]
        a_roles = a_ref["selected_roles"]
        a_set = set(a_ids)
        c_sets = [set(r["selected_ids"]) for r in c_ok]
        # 与 A 参考 set 相同的比例
        set_match_rate = sum(1 for s in c_sets if s == a_set) / len(c_sets)
        # 平均交集大小
        avg_inter = statistics.mean([len(s & a_set) for s in c_sets])
        # 角色匹配（按 A 顺序对齐，仅对共同存在的位置）
        role_hits = 0
        role_total = 0
        for r in c_ok:
            c_id2role = dict(zip(r["selected_ids"], r["selected_roles"]))
            for uid, role in zip(a_ids, a_roles):
                if uid in c_id2role:
                    role_total += 1
                    if c_id2role[uid] == role:
                        role_hits += 1
        role_rate = role_hits / role_total if role_total else 0.0

        # A 自身噪声（同样口径）
        a_sets = [set(r["selected_ids"]) for r in a_ok]
        a_self_match = sum(1 for s in a_sets if s == a_set) / len(a_sets)
        a_avg_inter = statistics.mean([len(s & a_set) for s in a_sets])

        w(
            f"- **耗时：{ma:.2f}s → {mc:.2f}s，{'✅ 降到 20s 左右' if mc <= 25 else ('⚠️ 仍高于 25s' if mc <= 35 else '❌ 未达 20s 目标')}"
            f"（{mc:.1f}s，节省 {ma - mc:.1f}s / {(ma - mc) / ma * 100:.1f}%）**"
        )
        w(
            f"- **选人集合一致性（vs A 参考）**：C 与 A set 完全相同率 "
            f"{set_match_rate * 100:.0f}%；平均交集 {avg_inter:.1f}/{len(a_ids)} 人"
        )
        w(
            f"- **噪声基线（A 自身不同次调用）**：A 与 A 参考 set 相同率 "
            f"{a_self_match * 100:.0f}%；平均交集 {a_avg_inter:.1f}/{len(a_ids)} 人"
        )
        w(f"- **角色一致性**：A 与 C 共同入选节点中，角色一致率 {role_rate * 100:.0f}%")
        w("")
        consistent = set_match_rate >= a_self_match and avg_inter >= a_avg_inter - 0.5
        fast = mc <= 25
        if consistent and fast:
            w(
                "> **结论：C 方案成立** —— 在选人一致性不劣于 A 自身噪声的前提下，"
                "把耗时降到 20s 左右。"
            )
        elif fast and not consistent:
            w(
                "> **结论：C 省时达标，但选人一致性下降** —— 说明 output schema 收紧"
                "并非纯收益，需要权衡。"
            )
        elif consistent and not fast:
            w("> **结论：C 保持一致性，但未达成 20s 目标**")
        else:
            w("> **结论：C 在两个维度都未达标**")
    else:
        w("数据不足")
    w("")

    # ---- 根因：隐藏思维链（来自独立探针，若已跑过则并入报告） ----
    rp = OUT_DIR / "reasoning_probe.json"
    dp = OUT_DIR / "disable_thinking_probe.json"
    if rp.exists() or dp.exists():
        w("## 8. 根因：C 为什么到不了 20s")
        w("")
        if rp.exists():
            try:
                rj = json.loads(rp.read_text(encoding="utf-8"))
                for c in rj.get("calls", []):
                    fs = c.get("field_sizes") or {}
                    w(
                        f"- 生产 ③ 单次调用探针：elapsed **{rj['elapsed']:.2f}s**，"
                        f"body {c['body_chars']} 字符，"
                        f"message 字段 = {c.get('message_keys')}"
                    )
                    for k, v in fs.items():
                        if k == "role":
                            continue
                        w(f"  - `{k}`: **{v} 字符**")
                    usage = c.get("usage") or {}
                    det = (usage.get("completion_tokens_details") or {})
                    if det:
                        w(f"  - usage.completion_tokens_details = {det}")
                    w("")
            except Exception as exc:  # noqa: BLE001
                w(f"（探针解析失败：{exc}）")
        w(
            "> **`qwen3.7-plus` 是推理模型**：响应 message 里除了 `content`（我们真正解析的\n"
            "> JSON，仅 ~916 字符）还有一个 **`reasoning_content` 隐藏思维链（~4,234 字符 /\n"
            "> ~2,057 reasoning tokens）**。它从不被 `_parse_json_content` 读取，但**完全计入\n"
            "> 端到端生成时间**。可见 JSON 只占生成量的 ~18%，所以压缩输出 schema 的\n"
            "> C 方案天然收益有限。"
        )
        w("")
        if dp.exists():
            try:
                dj = json.loads(dp.read_text(encoding="utf-8"))
                w("**关掉 thinking 的对照（同一 ③ prompt、同模型、同 temperature）**：")
                w("")
                w("| 变体 | elapsed | body_chars | reasoning_chars | content_chars | message 字段 |")
                w("| --- | ---: | ---: | ---: | ---: | --- |")
                for c in dj.get("calls", []):
                    if c.get("error"):
                        w(f"| {c['name']} | 失败 | - | - | - | {str(c['error'])[:60]} |")
                        continue
                    w(
                        f"| {c['name']} | **{c['elapsed']:.2f}s** | {c['body_chars']} | "
                        f"{c['reasoning_chars']} | {c['content_chars']} | {c['message_keys']} |"
                    )
                w("")
                ok_calls = [c for c in dj.get("calls", []) if not c.get("error")]
                for c in ok_calls:
                    if c.get("reasoning_chars") == 0 and c["name"] != "thinking_default":
                        w(
                            f"> **`enable_thinking: false` → {c['elapsed']:.2f}s**"
                            f"（reasoning_content 归零，body {c['body_chars']} 字符），"
                            f"选人 {c['selected_ids']}"
                        )
                w("")
            except Exception as exc:  # noqa: BLE001
                w(f"（探针解析失败：{exc}）")
        w("")

    (OUT_DIR / "compact_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
