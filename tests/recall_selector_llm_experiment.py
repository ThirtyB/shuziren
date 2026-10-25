"""固定同一 event，比较 ③ 开启时：全量 vs 分层召回 N=200。

    A：35,636 全量          → Scorer → Selector(③ 开启)
    B：platform_aware N=200 → Scorer → Selector(③ 开启)

③ 有 temperature=0.4 且无 seed，本身会抖动。所以除了 live 3×3，
额外做**录制/重放**：把 A 那次 ③ 的真实响应喂给 B，得到 ``B_replay``。
这样 A vs B_replay 共用同一个 LLM 响应，任何差异都只能来自召回本身。

本脚本不修改任何项目代码。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from influence_strategy.eval_hot_events import (  # noqa: E402
    hot_event_to_pipeline_payload,
    load_hot_events,
)
from influence_strategy.event_parser import RuleBasedEventParser  # noqa: E402
from influence_strategy.feature_builder import FeatureBuilder  # noqa: E402
from influence_strategy.llm_client import OpenAICompatibleLLMClient  # noqa: E402
from influence_strategy.platform_profiles import normalize_platform_key  # noqa: E402
from influence_strategy.scorer import Scorer  # noqa: E402
from influence_strategy.selector import Selector  # noqa: E402

EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
OUT_ROOT = PROJECT_ROOT / "outputs" / "recall_selector_llm"

DISPATCH_PLATFORM = "facebook_simulated"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class RecordingClient:
    """透传真实 client，并录制每次调用的 prompt 与响应。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[dict] = []

    def describe(self):
        return self._inner.describe()

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        t0 = time.monotonic()
        try:
            resp = self._inner.generate_json(
                system_prompt=system_prompt, user_prompt=user_prompt
            )
        except BaseException as exc:  # noqa: BLE001
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": round(time.monotonic() - t0, 3),
                }
            )
            raise
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": resp,
                "elapsed_sec": round(time.monotonic() - t0, 3),
            }
        )
        return resp


class ReplayClient:
    """重放一个已录制的响应（不再发远端请求）。"""

    def __init__(self, response: dict, source: str) -> None:
        self._response = response
        self.source = source
        self.calls = 0

    def describe(self):
        return {"provider": "replay", "model": self.source, "base_url": "-"}

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls += 1
        return json.loads(json.dumps(self._response))


def stratified_recall(node_features: list, *, size: int, platform: str) -> list:
    """platform_aware 分层召回（与 recall_experiment.py 同款）。"""
    target = normalize_platform_key(platform)
    pool = [f for f in node_features if normalize_platform_key(f.platform) == target]
    cap = max(2, size // 6)
    picked: list = []
    used: set[str] = set()

    def _generic(f) -> bool:
        return (
            f.keyword_hit_count == 0
            and f.topic_match_score == 0.0
            and f.influence_score >= 0.65
        )

    def add_bucket(features: list) -> None:
        for f in features:
            if len(picked) >= size or f.user_id in used:
                continue
            if _generic(f) and len([p for p in picked if _generic(p)]) >= cap:
                continue
            picked.append(f)
            used.add(f.user_id)

    add_bucket(
        sorted(
            pool,
            key=lambda i: (
                i.keyword_hit_count,
                i.topic_match_score,
                i.stability_score,
                i.activity_score,
            ),
            reverse=True,
        )
    )
    add_bucket(
        sorted(
            [i for i in pool if i.is_interaction_source or i.role_hint == "interaction_response"],
            key=lambda i: (
                i.activity_score,
                i.diffusion_score,
                i.stability_score,
                i.influence_score,
            ),
            reverse=True,
        )
    )
    add_bucket(
        sorted(
            pool,
            key=lambda i: (
                i.diffusion_score,
                i.activity_score,
                i.stability_score,
                i.influence_score,
            ),
            reverse=True,
        )
    )
    add_bucket(list(pool))
    return picked[:size]


def snapshot(selection) -> list[dict]:
    return [
        {
            "user_id": n.user_id,
            "selected_role": n.selected_role,
            "dispatch_stage": n.dispatch_stage,
            "platform": n.platform,
            "risk_level": n.risk_level,
            "final_score": round(n.final_score, 6),
        }
        for n in selection.selected_nodes
    ]


def ids(selection) -> list[str]:
    return [n.user_id for n in selection.selected_nodes]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", default="hot_event_001")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--size", type=int, default=200)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    events = load_hot_events(EVENTS_PATH)
    hot_event = next(e for e in events if str(e.get("event_id")) == args.event_id)
    payload = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=5,
        risk_level=None,
        campaign_window_hours=24,
        max_frequency_per_day=3,
        allowed_platforms=None,
    )

    # 上游冻结：① 规则解析、② 关闭（当前线上 ② 恒失败，llm_feature_score 恒 0）
    event = RuleBasedEventParser().parse(payload, workspace_root=PROJECT_ROOT, use_llm=False)
    fr_full = FeatureBuilder().build_from_loader(
        workspace_root=PROJECT_ROOT, event=event, profile_limit=None, use_llm=False
    )
    recalled = stratified_recall(
        fr_full.node_features, size=args.size, platform=DISPATCH_PLATFORM
    )
    fr_200 = fr_full.model_copy(update={"node_features": recalled})

    real = OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT)
    scorer = Scorer()
    selector = Selector()

    score_full = scorer.score(fr_full)
    score_200 = scorer.score(fr_200)

    records = []
    for rep in range(1, args.reps + 1):
        _log(f"[run] rep {rep}: A(live)")
        rc_a = RecordingClient(real)
        sel_a = selector.select(
            score_full,
            workspace_root=PROJECT_ROOT,
            use_llm=True,
            llm_client=rc_a,
            dispatch_platform=DISPATCH_PLATFORM,
        )
        _log(f"[run] rep {rep}: B(live)")
        rc_b = RecordingClient(real)
        sel_b = selector.select(
            score_200,
            workspace_root=PROJECT_ROOT,
            use_llm=True,
            llm_client=rc_b,
            dispatch_platform=DISPATCH_PLATFORM,
        )

        prompt_a = rc_a.calls[0]["user_prompt"] if rc_a.calls else ""
        prompt_b = rc_b.calls[0]["user_prompt"] if rc_b.calls else ""
        prompt_identical = prompt_a == prompt_b

        rec: dict = {
            "rep": rep,
            "prompt_identical": prompt_identical,
            "prompt_len_a": len(prompt_a),
            "prompt_len_b": len(prompt_b),
            "a_used": bool(rc_a.calls and "response" in rc_a.calls[0]),
            "b_used": bool(rc_b.calls and "response" in rc_b.calls[0]),
            "a_error": (rc_a.calls[0].get("error") if rc_a.calls else None),
            "b_error": (rc_b.calls[0].get("error") if rc_b.calls else None),
            "a_elapsed": (rc_a.calls[0].get("elapsed_sec") if rc_a.calls else None),
            "b_elapsed": (rc_b.calls[0].get("elapsed_sec") if rc_b.calls else None),
            "A": snapshot(sel_a),
            "B": snapshot(sel_b),
        }

        # ---- 重放配对：同一份 ③ 响应 ----
        if rec["a_used"] and rec["b_used"]:
            sel_b_ra = selector.select(
                score_200,
                workspace_root=PROJECT_ROOT,
                use_llm=True,
                llm_client=ReplayClient(rc_a.calls[0]["response"], "from_A"),
                dispatch_platform=DISPATCH_PLATFORM,
            )
            sel_a_rb = selector.select(
                score_full,
                workspace_root=PROJECT_ROOT,
                use_llm=True,
                llm_client=ReplayClient(rc_b.calls[0]["response"], "from_B"),
                dispatch_platform=DISPATCH_PLATFORM,
            )
            rec["B_replay_A"] = snapshot(sel_b_ra)
            rec["A_replay_B"] = snapshot(sel_a_rb)
            rec["A_vs_B_replayA_identical"] = snapshot(sel_a) == snapshot(sel_b_ra)
            rec["B_vs_A_replayB_identical"] = snapshot(sel_b) == snapshot(sel_a_rb)

        rec["A_vs_B_live_identical"] = snapshot(sel_a) == snapshot(sel_b)
        records.append(rec)
        _log(
            f"[run] rep {rep}: live_identical={rec['A_vs_B_live_identical']} "
            f"replay_identical={rec.get('A_vs_B_replayA_identical')} "
            f"prompt_identical={prompt_identical}"
        )
        (OUT_ROOT / f"rep{rep}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (OUT_ROOT / "raw.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- 结论 ----------------
    def fmt_ids(items: list[dict]) -> str:
        return "[" + ", ".join(i["user_id"] for i in items) + "]"

    print(f"# ③ 开启：全量 vs 召回 N={args.size}（{args.event_id}，{args.reps} 次）")
    print()
    print("| rep | A 选中 | B 选中 | live 全等 | ③prompt 全等 | 重放配对全等 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in records:
        same_ids = [i["user_id"] for i in r["A"]] == [i["user_id"] for i in r["B"]]
        print(
            f"| {r['rep']} | `{fmt_ids(r['A'])}` | `{fmt_ids(r['B'])}` | "
            f"{'✅' if r['A_vs_B_live_identical'] else ('ids同' if same_ids else '❌')} | "
            f"{'✅' if r['prompt_identical'] else '❌'} | "
            f"{'✅' if r.get('A_vs_B_replayA_identical') else '❌'} |"
        )
    print()

    print("## A 自身 3 次是否一致（③ 抖动基线）")
    a_sets = [tuple(i["user_id"] for i in r["A"]) for r in records]
    b_sets = [tuple(i["user_id"] for i in r["B"]) for r in records]
    print(f"- A 三次互比：{'✅ 全一致' if len(set(a_sets)) == 1 else '❌ 有抖动 ' + str(set(a_sets))}")
    print(f"- B 三次互比：{'✅ 全一致' if len(set(b_sets)) == 1 else '❌ 有抖动 ' + str(set(b_sets))}")
    print()

    print("## 逐项差异（字段级）")
    any_diff = False
    for r in records:
        for label in ("B", "B_replay_A"):
            if label not in r:
                continue
            for i, (x, y) in enumerate(zip(r["A"], r[label])):
                if x == y:
                    continue
                any_diff = True
                diffs = [k for k in x if x[k] != y.get(k)]
                print(f"- rep{r['rep']} A[{i}] vs {label}[{i}] `{x['user_id']}`: 差异字段 {diffs}")
    if not any_diff:
        print("- 无：A 与 B / B_replay_A 的 selected_ids、selected_role、dispatch_stage、platform、risk_level、final_score **逐项全等**")
    print()
    print(f"原始数据：`{OUT_ROOT}/raw.json`")


if __name__ == "__main__":
    main()
