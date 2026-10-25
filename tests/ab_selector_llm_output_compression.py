"""Selector ③ LLM 输出压缩 A/B 实验。

**目标**：验证"严格限制 ③ JSON 输出字段"能否把 ~40s 降到 ~10s，且选人结果仍可用。

**设计**：
- 同一事件 hot_event_001、同一个 ScoreResult、同一个短名单 (shortlist)、
  同一接口、同一 qwen3.7-plus、温度 0.4、timeout 90、非流式、json_object。
- 全部用运行时 monkeypatch，不动任何源文件。
- 同样 3 次真实 LLM 调用，对比 (A) 当前生产 ③ prompt 与 (B) 最小 JSON 约束。

**与上次 A/B 不同的关键点**：上次是「prompt 体量 vs 固定延迟」归因；本次是
**「让 ③ 输出 JSON 真的变短」**——把 prompt 改成"严格 JSON 模板 + 禁止 reasoning /
解释 / 重复字段"，看端到端能否从 40s+ 降到 ~10s。

用法：
    python tests/ab_selector_llm_output_compression.py
    python tests/ab_selector_llm_output_compression.py --report-only
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
def _build_shortlist(score_result) -> tuple[list[dict], int]:
    """与 ``Selector._llm_selection_plan`` 完全一致的 shortlist（逐字段）。"""
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
    """A 组：与生产完全一致 —— 直接用 ``build_selector_*_prompt``。"""
    sys_p = build_selector_system_prompt()
    usr_p = build_selector_user_prompt(
        event=score_result.event, max_selected=max_selected, shortlist=shortlist_payload
    )
    return sys_p, usr_p


def _prompt_B(score_result, shortlist_payload, max_selected) -> tuple[str, str]:
    """B 组：最小 JSON 约束。"""
    sys_p = (
        "你是节点选择助手。按以下**严格** JSON 模板返回，禁止任何其它文字、"
        "Markdown、解释、reasoning、reason、explanation、注释、重复候选信息：\n\n"
        '{\n'
        '  "selected_order": [{"user_id": "<id>", "recommended_role": "<role>"}],\n'
        '  "fallback_order": ["<id>"],\n'
        '  "global_notes": ""\n'
        '}\n\n'
        f"规则：\n"
        f"1. selected_order 最多 {max_selected} 项，每项只含两个字段：\n"
        f"   - user_id: 必须是 shortlist 中存在的 ID\n"
        f"   - recommended_role: 只能是 core_publish_node / interaction_response_node / "
        f"amplification_node / support_node 之一\n"
        f"2. fallback_order 是 user_id 字符串数组（数量不限，但只放 user_id）\n"
        f"3. global_notes 必须是空字符串 \"\"（不放任何分析）\n"
        f"4. 禁止 reasoning / reason / explanation / 分析 等任何额外字段\n"
        f"5. 禁止重复 shortlist 候选池里已有的 user_name / final_score 等字段"
    )
    # 简化 user：保留必要上下文（event + 紧凑版 shortlist）
    compact_shortlist = [
        {
            "user_id": n["user_id"],
            "role_hint": n.get("role_hint", ""),
            "final_score": round(float(n.get("final_score", 0.0)), 4),
        }
        for n in shortlist_payload
    ]
    usr_p = json.dumps(
        {
            "event_id": score_result.event.event_id,
            "event_title": score_result.event.event_title,
            "target_goal": score_result.event.target_goal,
            "event_type": score_result.event.event_type,
            "max_selected": max_selected,
            "shortlist": compact_shortlist,
        },
        ensure_ascii=False,
    )
    return sys_p, usr_p


# --------------------------------------------------------------------------- #
# 单次调用 + 解析
# --------------------------------------------------------------------------- #
_ROLE_REVERSE = {
    "core_publish_node": "core_broadcast",
    "interaction_response_node": "interaction_response",
    "amplification_node": "amplification",
    "support_node": "general_support",
}


def _bloat_breakdown(parsed: dict) -> dict[str, int]:
    """对解析后的 dict 逐字段估算原始体量（json.dumps 长度），定位膨胀字段。"""
    out = {}
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
    """单次 client.generate_json 调用 + 拆解 + 解析。"""
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

    # 解析后的字段拆解
    so = parsed.get("selected_order") if isinstance(parsed, dict) else None
    so_count = len(so) if isinstance(so, list) else 0
    fo = parsed.get("fallback_order") if isinstance(parsed, dict) else None
    fo_count = len(fo) if isinstance(fo, list) else 0
    gn = parsed.get("global_notes") if isinstance(parsed, dict) else None
    gn_chars = len(str(gn)) if gn is not None else 0

    bloat = _bloat_breakdown(parsed) if isinstance(parsed, dict) else {}

    # 选人归一化
    ids, roles = _normalize_ids_roles(so)

    return {
        "elapsed": round(elapsed, 4),
        "success": succ,
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
        "parse_ok": succ,
        "selected_order_count": so_count,
        "selected_ids": ids,
        "selected_roles": roles,
        "fallback_count": fo_count,
        "global_notes_chars": gn_chars,
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
    raw_path = OUT_DIR / "output_compression_raw.json"

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

    # 准备 ScoreResult + shortlist
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
    b_sys, b_usr = _prompt_B(score_result, shortlist_payload, max_selected)
    _log(
        f"[prep] A prompt={len(a_sys) + len(a_usr)} chars  "
        f"B prompt={len(b_sys) + len(b_usr)} chars"
    )
    _log("")

    client = OpenAICompatibleLLMClient(
        api_key=base.api_key, base_url=base.base_url, model=MODEL,
        provider=base.provider, timeout=base.timeout,
    )

    plan: list[tuple[str, str, str]] = []
    for i in range(args.repeat):
        plan.append(("A_prod", a_sys, a_usr))
    for i in range(args.repeat):
        plan.append(("B_min", b_sys, b_usr))

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
    A = by_g.get("A_prod", [])
    B = by_g.get("B_min", [])

    L = []
    w = L.append
    w("# Selector ③ 输出压缩 A/B · A(生产 prompt) vs B(最小 JSON 约束)")
    w("")
    w(
        f"事件 `{d['event_id']}` ｜ 模型 `{d['model']}` ｜ endpoint `{d['base_url']}` ｜ "
        f"temperature=0.4 ｜ timeout={d['timeout']}s ｜ 非流式 ｜ response_format=json_object ｜ "
        f"rerank_top_k={d['rerank_top_k']} max_selected={d['max_selected']}"
    )
    w("")

    w("## 1. prompt 设计")
    w("")
    w("| 组 | system 字符 | user 字符 | prompt 字符 |")
    w("| --- | ---: | ---: | ---: |")
    for r in A[:1]:
        w(f"| A · 生产 ③ prompt | {r['sys_chars']} | {r['usr_chars']} | {r['prompt_chars']} |")
    for r in B[:1]:
        w(f"| B · 最小 JSON 约束 | {r['sys_chars']} | {r['usr_chars']} | {r['prompt_chars']} |")
    w("")

    w("## 2. 逐次明细")
    w("")
    w("| # | 组 | elapsed | body_chars | request_body | 解析 | selected_order 条数 | global_notes 字符 |")
    w("| ---: | --- | ---: | ---: | ---: | :-: | ---: | ---: |")
    for r in rows:
        w(
            f"| {r['index']} | {r['group']} | **{r['elapsed']:.2f}** | {r['body_chars']} | "
            f"{r['request_body_chars']} | {'✅' if r['parse_ok'] else '❌'} | "
            f"{r['selected_order_count']} | {r['global_notes_chars']} |"
        )
    w("")

    w("## 3. 汇总（min / mean / max，单位秒）")
    w("")
    w("| 组 | 次数 | elapsed | body_chars |")
    w("| --- | ---: | --- | ---: |")
    for tag, name in (("A_prod", "A · 生产"), ("B_min", "B · 最小 JSON")):
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
    b_ok = [r for r in B if r["success"]]
    w("## 4. 选人一致性")
    w("")
    if a_ref:
        a_ids = a_ref["selected_ids"]
        a_roles = a_ref["selected_roles"]
        w(f"**A 参考 (call #{a_ref['index']})**: {a_ids}")
        w(f"  roles: {a_roles}")
        w("")
        if b_ok:
            for r in b_ok:
                same_order = r["selected_ids"] == a_ids
                same_set = set(r["selected_ids"]) == set(a_ids)
                role_match = r["selected_roles"] == a_roles
                w(
                    f"- B call #{r['index']}: ids={r['selected_ids']} "
                    f"roles={r['selected_roles']}  "
                    f"**{'✅ 完全一致' if same_order and role_match else ('⚠️ 顺序或角色不同 ' + ('(set 一致)' if same_set else '(set 也不一致)'))}**"
                )
        else:
            w("B 组无成功样本")
    else:
        w("A 组无成功样本，无法对比")
    w("")

    # B 输出体量拆解
    w("## 5. B 组输出字段拆解（定位膨胀源）")
    w("")
    for r in B:
        if not r["success"]:
            continue
        w(f"**B call #{r['index']}** (elapsed={r['elapsed']:.2f}s, body={r['body_chars']}):")
        for k, v in r["bloat"].items():
            w(f"  - `{k}`: {v} 字符（json.dumps 估算）")
        w("")

    # 节省预估
    a_ok = [r for r in A if r["success"]]
    if a_ok and b_ok:
        ma = statistics.mean([r["elapsed"] for r in a_ok])
        mb = statistics.mean([r["elapsed"] for r in b_ok])
        saved = ma - mb
        bca = statistics.mean([r["body_chars"] for r in a_ok])
        bcb = statistics.mean([r["body_chars"] for r in b_ok])
        # 用上次测得的 ~7.3 s/1k response 字符 作为参考斜率
        per1k = 7.3
        expected_by_shrinking = (bca - bcb) / 1000.0 * per1k
        w("## 6. 节省估算")
        w("")
        w(
            f"- A 均值 **{ma:.2f}s**、B 均值 **{mb:.2f}s**，"
            f"**实际节省 {saved:+.2f}s**（{saved / ma * 100:.1f}%）"
        )
        w(
            f"- A 输出 {bca:.0f} 字符 → B 输出 {bcb:.0f} 字符，"
            f"省 {bca - bcb:.0f} 字符（{bcb / bca * 100:.1f}%）"
        )
        w(
            f"- 用上次测得的 ~{per1k}s / 1k response 字符 斜率估算："
            f"若输出完全按 prompt 收缩，理论上可省 ≈ {expected_by_shrinking:+.2f}s"
        )
        w("")

    (OUT_DIR / "output_compression_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
