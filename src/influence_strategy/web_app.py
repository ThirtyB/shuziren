from __future__ import annotations

import copy
import json
import os
import random
import re
import shutil
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .eval_hot_events import (
    EVAL_MODE_MULTI_PLATFORM,
    EVAL_MODE_MULTI_USER,
    EVAL_MODE_SINGLE_TARGET,
    _resolve_run_dirs,
    _run_hot_event_evaluation,
    _run_hot_event_multiplatform_evaluation,
    load_hot_events,
    select_hot_events,
)
from .image_event_loader import SUPPORTED_IMAGE_SUFFIXES, load_image_events
from .llm_client import OpenAICompatibleLLMClient, read_env_values
from .platform_dispatcher import PlatformSpec, to_json_dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
WEB_RUNS_ROOT = PROJECT_ROOT / "outputs" / "web_runs"
BENCHMARK_RUNS_ROOT = WEB_RUNS_ROOT / "benchmark_runs"
FRONTEND_DIST = PROJECT_ROOT / "web" / "frontend" / "dist"
MAX_WORKERS = 1
BENCHMARK_DEFAULT_ITERATIONS = 5

_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
# 压测任务与普通 run 完全隔离：单独一份内存缓存 + 独立落盘目录，方便前端做
# "测试历史"列表，不与主控制台的运行历史混在一起。
_BENCHMARKS: dict[str, dict[str, Any]] = {}
_BENCHMARK_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


class RunOptions(BaseModel):
    profile_limit: int | None = None
    max_selected_nodes: int = Field(default=5, ge=1, le=20)
    risk_level: Literal["low", "medium", "high"] | None = None
    campaign_window_hours: int = Field(default=24, ge=1, le=168)
    max_frequency_per_day: int = Field(default=3, ge=1, le=24)
    allowed_platforms: list[str] = Field(
        default_factory=lambda: [
            "facebook_simulated",
            "instagram_simulated",
            "twitter_simulated",
        ]
    )

    # 三种分发场景由 dispatch_mode 显式区分：
    #   "single_target"   —— 单人定向分发（需指定 target_user_id，系统自动取该数字人平台）
    #   "multi_user"      —— 多数字人协同分发（需指定 dispatch_platform 从该平台选人）
    #   "cross_platform"  —— 跨平台并行分发（需指定 platform_specs 多平台同时选人）
    # 各模式的互斥校验集中在 ``_run_events_job`` 中按模式分流。
    dispatch_mode: Literal["single_target", "multi_user", "cross_platform"] = "multi_user"
    # 单人定向分发：指定唯一数字人 user_id；多人分发下无效。
    target_user_id: str | None = None
    # 多人分发指定平台：只从该平台已有的数字人中选人；空表示不限平台。
    dispatch_platform: str | None = None
    # 跨平台并行分发：每个平台独立的人数配置；至少 1 个平台。
    platform_specs: list[PlatformSpec] = Field(default_factory=list)

    use_llm: bool = True
    # 第十轮：事件解析阶段默认**不调用** LLM（与 pipeline 生产默认一致，
    # 见 DEFAULT_USE_LLM_EVENT_PARSER），selector 仍用 LLM。
    # 前端不传或传 null 时都走规则解析；显式传 true 可做对照实验。
    # 事件解析不调用 LLM 时仍然接受任意新事件（走规则解析）。
    use_llm_event_parser: bool | None = False
    # 第七轮：选人阶段的 LLM 独立开关，语义与 use_llm_event_parser 一致
    # （None = 跟随 use_llm；True/False = 显式覆盖）。
    use_llm_selector: bool | None = None
    # 选人阶段大模型重排前的规则初选候选上限，默认 15；<=0 表示关闭初选。
    selector_candidate_limit: int | None = Field(default=15, ge=0)
    # 第十四轮：**页面路径**的节点正文 LLM 开关，默认关闭。
    # 关闭后不再为每个数字人生成社交媒体发帖正文，节点正文走已有的规则兜底
    # （字段照旧输出，只是内容来源是规则而不是 LLM），单事件可省约 6~7s。
    # 需要复现"正文也用 LLM"的旧行为时显式传 true 即可；CLI / run_eval /
    # ablation 等其它入口不受这个开关影响，仍保持原行为。
    use_llm_content: bool = False
    event_limit: int | None = Field(default=None, ge=1)
    event_id: str | None = None


class TextRunRequest(BaseModel):
    event: dict[str, Any] | None = None
    event_text: str | None = None
    options: RunOptions = Field(default_factory=RunOptions)


class EvalRunRequest(BaseModel):
    options: RunOptions = Field(default_factory=lambda: RunOptions(event_limit=200))


class BenchmarkRequest(BaseModel):
    """一键 N 次压测的请求体（默认 5 轮）。

    与 ``EvalRunRequest`` 的差别：压测**异步**执行，``POST`` 立即返回 queued 状态
    与 ``benchmark_id``，前端凭它轮询进度；每轮自动随机抽一条互不相同的
    hot_event，只有在显式传 ``event_id`` 时才锁定同一条事件跑 N 轮。
    """

    options: RunOptions = Field(default_factory=RunOptions)
    iterations: int = Field(default=BENCHMARK_DEFAULT_ITERATIONS, ge=1, le=10)
    event_id: str | None = None
    label: str | None = None


class RunStatus(BaseModel):
    run_id: str
    mode: Literal["text", "image", "eval"]
    # 运行级别分发场景（顶层 Tab 语义）：由发起 RunOptions.dispatch_mode 决定。
    dispatch_mode: Literal["single_target", "multi_user", "cross_platform"] = "multi_user"
    state: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    total: int
    completed: int = 0
    failed: int = 0
    current_event_id: str | None = None
    output_dir: str
    trace_dir: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


def _env_flag(name: str, default: bool) -> bool:
    """读布尔开关：进程环境变量优先，其次项目根目录的 .env。

    项目的 LLM 配置本来就在 .env 里（不在进程环境里），只查 os.environ
    会让 .env 里的开关永远读不到。
    """
    raw = os.environ.get(name)
    if raw is None:
        raw = read_env_values(PROJECT_ROOT).get(name)
    raw = (raw or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _warmup_llm_model() -> None:
    """第十五轮：服务启动时把本地模型加载进内存，消除"首次点击必冷"。

    实测（本机 qwen3:8b / Ollama 0.40.10）：模型未加载时一次请求要多花
    3.5~6.5s 的加载时间；且 Ollama 默认只驻留 5 分钟，空闲超时后会再次卸载
    （A 组实测：连续 3.5s → 空闲 5.5 分钟后 10.77s）。

    做法只是"发一次极小请求让模型加载"，配合 ``LLM_KEEP_ALIVE`` 让它常驻，
    **不改动任何选人逻辑**。两个前置条件同时满足才执行：
    配置了 ``LLM_KEEP_ALIVE``（否则加载完 5 分钟后照样卸载，白白占用显存），
    且 ``LLM_WARMUP_ON_STARTUP`` 未显式关闭（默认关闭，需在本机 .env 开启）。

    失败时打印到 stderr 并继续（服务本身可用，后续真实请求会照常报错），
    不做静默吞异常。
    """
    if not _env_flag("LLM_WARMUP_ON_STARTUP", default=False):
        print("[warmup] LLM_WARMUP_ON_STARTUP 未开启，跳过模型预热", file=sys.stderr)
        return
    client = OpenAICompatibleLLMClient.from_env_files(PROJECT_ROOT)
    if client is None:
        print("[warmup] 未配置 LLM，跳过模型预热", file=sys.stderr)
        return
    if client.keep_alive is None:
        print(
            "[warmup] 未配置 LLM_KEEP_ALIVE，跳过预热（加载后 5 分钟仍会卸载）",
            file=sys.stderr,
        )
        return
    started = time.monotonic()
    client.generate_json(
        system_prompt="只输出 JSON 对象，不要输出其它文字。",
        user_prompt='只输出：{"ok":true}',
    )
    print(
        f"[warmup] 模型已加载并常驻（keep_alive={client.keep_alive}），"
        f"耗时 {time.monotonic() - started:.2f}s",
        file=sys.stderr,
    )


def _build_lifespan():
    """服务启动钩子：把模型预热放到后台线程，不阻塞启动与 /api/health。"""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # pragma: no cover - 启动副作用
        threading.Thread(target=_warmup_llm_model, name="llm-warmup", daemon=True).start()
        yield

    return lifespan


def create_app() -> FastAPI:
    app = FastAPI(
        title="Influence Strategy Web Console",
        version="0.1.0",
        lifespan=_build_lifespan(),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "workspace_root": str(PROJECT_ROOT),
            "default_events_path": str(DEFAULT_EVENTS_PATH),
            "web_runs_root": str(WEB_RUNS_ROOT),
            "llm_default_enabled": True,
        }

    @app.get("/api/event-template")
    def event_template() -> dict[str, Any]:
        # event_id 由后端 ``_normalize_hot_event`` 在缺失时自动生成
        # （``web_event_<timestamp>``），所以模板里不再预置 ``event_id``，
        # 避免前端把它误渲染为"必填 ID"输入框。
        return {
            "domain": "general",
            "event_title": "请输入事件标题",
            "event_summary": "请输入事件概要",
            "target": "引导公众理解事件影响并进行理性讨论。",
            "is_synthetic": False,
            "opinion_variants": [
                "请输入可用于发帖内容生成的观点变体。",
            ],
        }

    @app.get("/api/events")
    def list_system_events(
        limit: int = 50,
        offset: int = 0,
        domain: str | None = None,
        q: str | None = None,
    ) -> dict[str, Any]:
        """系统事件库（``DEFAULT_EVENTS_PATH``）的只读索引。

        用于前端的"系统事件"挑选下拉/搜索；返回的字段经过裁剪（不含
        opinion_variants 全文以减小 payload），单条全字段由
        ``/api/events/{event_id}`` 提供。
        """
        if limit < 1 or limit > 500:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
        if offset < 0:
            raise HTTPException(status_code=400, detail="offset must be >= 0")
        events = load_hot_events(DEFAULT_EVENTS_PATH)
        if domain:
            events = [e for e in events if str(e.get("domain", "")).lower() == domain.lower()]
        if q:
            needle = q.lower().strip()
            events = [
                e
                for e in events
                if needle in str(e.get("event_title", "")).lower()
                or needle in str(e.get("event_summary", "")).lower()
                or needle in str(e.get("event_id", "")).lower()
            ]
        total = len(events)
        page = events[offset : offset + limit]
        domains = sorted(
            {str(e.get("domain", "")) for e in load_hot_events(DEFAULT_EVENTS_PATH) if e.get("domain")}
        )
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "domains": domains,
            "items": [
                {
                    "event_id": e.get("event_id"),
                    "domain": e.get("domain"),
                    "event_title": e.get("event_title"),
                    "event_summary": e.get("event_summary"),
                    "is_synthetic": bool(e.get("is_synthetic", False)),
                    "opinion_variants_count": len(e.get("opinion_variants") or []),
                }
                for e in page
            ],
        }

    @app.get("/api/events/{event_id}")
    def get_system_event(event_id: str) -> dict[str, Any]:
        """单条系统事件全字段，供前端"系统事件"模式直接喂给后端。"""
        events = load_hot_events(DEFAULT_EVENTS_PATH)
        for event in events:
            if str(event.get("event_id")) == event_id:
                return event
        raise HTTPException(status_code=404, detail=f"system event not found: {event_id}")

    @app.post("/api/runs/text", response_model=RunStatus)
    def run_text_event(request: TextRunRequest) -> dict[str, Any]:
        event = _event_from_text_request(request)
        _validate_dispatch_options(request.options)
        status = _create_run(
            mode="text",
            total=1,
            dispatch_mode=request.options.dispatch_mode,
        )
        _submit_job(
            status["run_id"],
            lambda: _run_events_job(
                run_id=status["run_id"],
                events=[event],
                options=request.options,
            ),
        )
        return _read_status(status["run_id"])

    @app.post("/api/runs/image", response_model=RunStatus)
    def run_image_event(
        image: UploadFile = File(...),
        options: str = Form("{}"),
    ) -> dict[str, Any]:
        parsed_options = _parse_options(options)
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {suffix or image.filename}")
        _validate_dispatch_options(parsed_options)

        status = _create_run(
            mode="image",
            total=1,
            dispatch_mode=parsed_options.dispatch_mode,
        )
        run_dir = _run_dir(status["run_id"])
        upload_dir = run_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        image_path = upload_dir / f"input{suffix}"
        with image_path.open("wb") as handle:
            shutil.copyfileobj(image.file, handle)

        _submit_job(
            status["run_id"],
            lambda: _run_image_job(
                run_id=status["run_id"],
                image_path=image_path,
                options=parsed_options,
            ),
        )
        return _read_status(status["run_id"])

    @app.post("/api/eval-runs", response_model=RunStatus)
    def run_eval_events(request: EvalRunRequest | None = None) -> dict[str, Any]:
        options = (request.options if request is not None else RunOptions(event_limit=200))
        if options.event_limit is None:
            options.event_limit = 200
        _validate_dispatch_options(options)
        events = select_hot_events(
            load_hot_events(DEFAULT_EVENTS_PATH),
            event_id=options.event_id,
            event_limit=options.event_limit,
        )
        status = _create_run(
            mode="eval",
            total=len(events),
            dispatch_mode=options.dispatch_mode,
        )
        _submit_job(
            status["run_id"],
            lambda: _run_events_job(
                run_id=status["run_id"],
                events=events,
                options=options,
            ),
        )
        return _read_status(status["run_id"])

    @app.post("/api/benchmarks")
    def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
        """一键 N 次压测（默认 5）：**异步**提交，立即返回 queued 状态。

        设计要点：

        * 不阻塞请求 —— 后台线程串行跑 N 轮，前端拿 ``benchmark_id`` 后轮询
          ``GET /api/benchmarks/{benchmark_id}``；因此用户可以在上一次测试还在
          "测试中"的时候继续点新测试，新任务会排队（线程池 max_workers=1）。
        * 事件随机且互不相同 —— 任务开始时就从 200 条 hot_event 里不放回地抽
          ``iterations`` 条，保证同一组压测里每个测试点跑的都是不同事件；这样
          阶段耗时不会被同一条事件的画像 / LLM 缓存"预热"。
        * LLM 开关只跑当前 ``use_llm`` 的那一种；想对比开/关差异，切换顶栏
          开关后再点一次，两次结果会各自作为一条测试历史保留。
        """
        options = request.options.model_copy(deep=True)
        # 压测面板不接 run 事件，因此强制 event_limit=1。
        options.event_limit = 1
        if request.event_id:
            options.event_id = request.event_id
        _validate_dispatch_options(options)

        iterations = int(request.iterations)
        label = (request.label or "").strip() or None
        benchmark_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        benchmark_dir = BENCHMARK_RUNS_ROOT / benchmark_id
        (benchmark_dir / "output").mkdir(parents=True, exist_ok=True)
        (benchmark_dir / "trace").mkdir(parents=True, exist_ok=True)

        now = _now()
        payload: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "label": label,
            "state": "queued",
            "iterations": iterations,
            "completed": 0,
            "failed": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "options": options.model_dump(mode="json"),
            "events": [],
            "runs": [],
            "summary": None,
            "output_dir": str(benchmark_dir),
            "created_at": now,
            "updated_at": now,
            "error": None,
        }
        _write_benchmark(benchmark_id, payload)

        _EXECUTOR.submit(
            _benchmark_job_wrapper,
            benchmark_id,
            options,
            iterations,
            request.event_id,
        )
        return dict(payload)

    @app.get("/api/benchmarks")
    def list_benchmarks() -> dict[str, Any]:
        """测试历史列表（压测专用，与 ``/api/runs`` 完全隔离）。

        内存缓存优先（刚提交还在排队 / 运行中的任务不在磁盘上完整），磁盘兜底
        （服务重启后仍能看到历史）。按创建时间倒序。
        """
        BENCHMARK_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        merged: dict[str, dict[str, Any]] = {}
        for path in BENCHMARK_RUNS_ROOT.glob("*/benchmark.json"):
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = str(stored.get("benchmark_id") or path.parent.name)
            merged[key] = _normalize_benchmark_payload(stored)
        with _BENCHMARK_LOCK:
            merged.update({key: dict(value) for key, value in _BENCHMARKS.items()})

        items = [_benchmark_history_item(payload) for payload in merged.values()]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"benchmarks": items[:50]}

    @app.get("/api/benchmarks/{benchmark_id}")
    def get_benchmark(benchmark_id: str) -> dict[str, Any]:
        """单条测试详情（含每轮 stage_times / summary），供前端轮询。"""
        return _read_benchmark(benchmark_id)

    @app.delete("/api/benchmarks/{benchmark_id}")
    def delete_benchmark(benchmark_id: str) -> dict[str, Any]:
        """删除一条测试历史（磁盘目录 + 内存缓存）。

        "测试中"的任务拒绝删除：后台线程仍在往该目录写 ``benchmark.json``。
        """
        payload = _read_benchmark(benchmark_id)
        if str(payload.get("state")) in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail=f"测试 {benchmark_id} 仍在进行（state={payload.get('state')}），请等待结束后再删除。",
            )
        benchmark_dir = _benchmark_dir(benchmark_id)
        if benchmark_dir.exists():
            shutil.rmtree(benchmark_dir)
        with _BENCHMARK_LOCK:
            _BENCHMARKS.pop(benchmark_id, None)
        return {"deleted": True, "benchmark_id": benchmark_id}

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        WEB_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        statuses = []
        for status_path in sorted(WEB_RUNS_ROOT.glob("*/status.json"), reverse=True):
            try:
                statuses.append(_load_json(status_path))
            except (OSError, json.JSONDecodeError):
                continue
        return {"runs": statuses[:50]}

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict[str, Any]:
        """删除指定 run 的全部产物（status.json / output / trace）。

        用于前端"运行历史"面板的删除入口。已结束的 run 可被任意删除；
        正在运行的 run 拒绝删除（线程已持有路径下的 status.json，
        强制删会导致后续写失败）。
        """
        run_dir = _run_dir(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        status_path = run_dir / "status.json"
        if status_path.exists():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if str(payload.get("state", "")).lower() in {"queued", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail=f"Run {run_id} 仍在运行（state={payload.get('state')}），请等待结束后再删除。",
                )
        shutil.rmtree(run_dir)
        return {"deleted": True, "run_id": run_id}

    @app.get("/api/runs/{run_id}", response_model=RunStatus)
    def get_run_status(run_id: str) -> dict[str, Any]:
        return _read_status(run_id)

    @app.get("/api/runs/{run_id}/results")
    def get_run_results(run_id: str) -> dict[str, Any]:
        status = _read_status(run_id)
        return {"run_id": run_id, "results": status.get("results", [])}

    @app.get("/api/runs/{run_id}/results/{event_id}")
    def get_result_detail(run_id: str, event_id: str) -> dict[str, Any]:
        output_path = _resolve_result_output_path(run_id, event_id)
        if output_path is None:
            raise HTTPException(status_code=404, detail=f"Result not found: {event_id}")
        return _load_json(output_path)

    @app.get("/api/runs/{run_id}/traces/{event_id}")
    def get_trace_detail(run_id: str, event_id: str) -> dict[str, Any]:
        trace_dir = _run_dir(run_id) / "trace" / event_id
        if not trace_dir.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {event_id}")
        files = []
        for path in sorted(trace_dir.glob("*.json")):
            try:
                payload = _load_json(path)
            except json.JSONDecodeError:
                payload = path.read_text(encoding="utf-8")
            files.append({"name": path.name, "payload": payload})
        return {"run_id": run_id, "event_id": event_id, "files": files}

    @app.get("/api/runs/{run_id}/files/{event_id}")
    def download_result(run_id: str, event_id: str) -> FileResponse:
        output_path = _resolve_result_output_path(run_id, event_id)
        if output_path is None:
            raise HTTPException(status_code=404, detail=f"Result not found: {event_id}")
        return FileResponse(output_path, filename=output_path.name, media_type="application/json")

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


app = create_app()


def _event_from_text_request(request: TextRunRequest) -> dict[str, Any]:
    if request.event is not None and request.event_text:
        raise HTTPException(status_code=400, detail="Use either event JSON or event text, not both.")
    if request.event is None and not request.event_text:
        raise HTTPException(status_code=400, detail="Event input is required.")
    if request.event is not None:
        return _normalize_hot_event(dict(request.event))
    text = str(request.event_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Event text is empty.")
    return _normalize_hot_event(
        {
            "event_id": _safe_event_id(f"web_event_{_timestamp_id()}"),
            "domain": "general",
            "event_title": text[:48] or "新事件",
            "event_summary": text,
            "target": "引导公众理解事件影响并进行理性讨论。",
            "is_synthetic": False,
            "opinion_variants": [text],
        }
    )


def _normalize_hot_event(event: dict[str, Any]) -> dict[str, Any]:
    event_id = _safe_event_id(str(event.get("event_id") or f"web_event_{_timestamp_id()}"))
    title = str(event.get("event_title") or event.get("event_summary") or event_id).strip()
    summary = str(event.get("event_summary") or title).strip()
    variants = event.get("opinion_variants") or [summary or title]
    if isinstance(variants, str):
        variants = [item.strip() for item in variants.splitlines() if item.strip()]
    if not isinstance(variants, list) or not variants:
        variants = [summary or title]
    normalized = dict(event)
    normalized.update(
        {
            "event_id": event_id,
            "domain": str(event.get("domain") or "general").strip() or "general",
            "event_title": title,
            "event_summary": summary,
            "target": str(event.get("target") or "引导公众理解事件影响并进行理性讨论。").strip(),
            "is_synthetic": bool(event.get("is_synthetic", False)),
            "opinion_variants": [str(item).strip() for item in variants if str(item).strip()],
        }
    )
    if not normalized["event_title"] and not normalized["event_summary"]:
        raise HTTPException(status_code=400, detail="event_title or event_summary is required.")
    return normalized


def _pick_random_hot_events(
    hot_events: list[dict[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    """从 200 条 hot_event 集合里**不放回**地随机抽 ``count`` 条用于压测。

    用于 ``/api/benchmarks``：一次测试里每个测试点跑的都是不同事件，避免同一
    事件的画像 / LLM 缓存被"预热"后拉低后续轮次耗时，从而更准确地反映 LLM
    开关 / 分发配置差异。

    - 当集合为空时返回 ``[]``，由上层抛 404；
    - ``count`` 超过集合大小时返回全部（不放回，因此不会有重复）。
    """
    if not hot_events:
        return []
    size = max(0, min(count, len(hot_events)))
    if size == 0:
        return []
    return random.sample(hot_events, size)


def _parse_options(raw_options: str) -> RunOptions:
    try:
        payload = json.loads(raw_options or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Options must be a JSON object.")
    return RunOptions.model_validate(payload)


def _create_run(
    *,
    mode: Literal["text", "image", "eval"],
    total: int,
    dispatch_mode: Literal["single_target", "multi_user", "cross_platform"] = "multi_user",
) -> dict[str, Any]:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = _run_dir(run_id)
    output_dir = run_dir / "output"
    trace_dir = run_dir / "trace"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    now = _now()
    status = RunStatus(
        run_id=run_id,
        mode=mode,
        dispatch_mode=dispatch_mode,
        state="queued",
        created_at=now,
        updated_at=now,
        total=total,
        output_dir=str(output_dir),
        trace_dir=str(trace_dir),
    ).model_dump(mode="json")
    _write_status(run_id, status)
    return status


def _submit_job(run_id: str, job: Any) -> None:
    _EXECUTOR.submit(_run_job_wrapper, run_id, job)


def _run_job_wrapper(run_id: str, job: Any) -> None:
    try:
        _update_status(run_id, state="running")
        job()
        status = _read_status(run_id)
        final_state = "failed" if status.get("completed", 0) == 0 and status.get("failed", 0) else "completed"
        _update_status(run_id, state=final_state, current_event_id=None)
    except Exception as exc:  # pragma: no cover - defensive boundary for background jobs
        _append_error(run_id, event_id=None, message=f"{type(exc).__name__}: {exc}")
        _update_status(run_id, state="failed", current_event_id=None)


def _run_image_job(*, run_id: str, image_path: Path, options: RunOptions) -> None:
    _update_status(run_id, current_event_id="image_recognition")
    reference_events = load_hot_events(DEFAULT_EVENTS_PATH)
    events = load_image_events(
        image=image_path,
        image_dir=None,
        reference_events=reference_events,
        workspace_root=PROJECT_ROOT,
        event_limit=1,
        event_id=options.event_id,
        use_llm=options.use_llm,
    )
    _update_status(run_id, total=len(events))
    _run_events_job(run_id=run_id, events=events, options=options)


def _validate_dispatch_options(options: RunOptions) -> None:
    """校验三种 dispatch_mode 的专属参数互斥关系。

    * single_target 必须指定 ``target_user_id``；
    * multi_user     推荐显式指定 ``dispatch_platform``（不指定会用默认值）；
    * cross_platform 必须提供至少一个 ``platform_specs``。
    """

    if options.dispatch_mode == "single_target":
        if not options.target_user_id:
            raise HTTPException(
                status_code=400,
                detail="单人定向分发必须指定 target_user_id；多人协同或跨平台模式请切换 dispatch_mode。",
            )
        if options.platform_specs:
            raise HTTPException(
                status_code=400,
                detail="单人定向分发不需要 platform_specs；多人协同或跨平台模式请切换 dispatch_mode。",
            )
        if options.dispatch_platform:
            raise HTTPException(
                status_code=400,
                detail="单人定向分发会自动使用目标数字人所属平台，不需要 dispatch_platform。",
            )
    elif options.dispatch_mode == "cross_platform":
        if not options.platform_specs:
            raise HTTPException(
                status_code=400,
                detail="跨平台并行分发必须提供 platform_specs（每个平台至少 1 人）。",
            )
        if options.target_user_id:
            raise HTTPException(
                status_code=400,
                detail="跨平台并行分发不支持 target_user_id；该参数仅用于单人定向。",
            )
    else:  # multi_user
        if options.target_user_id:
            raise HTTPException(
                status_code=400,
                detail="多人协同分发不支持 target_user_id；如需定向请切换 dispatch_mode=single_target。",
            )
        if options.platform_specs:
            raise HTTPException(
                status_code=400,
                detail="多人协同分发不需要 platform_specs；如需多平台请切换 dispatch_mode=cross_platform。",
            )


def _run_events_job(*, run_id: str, events: list[dict[str, Any]], options: RunOptions) -> None:
    """按 ``options.dispatch_mode`` 把同一事件分别路由到不同的 evaluation helper。

    选择原则：

    * cross_platform → ``_run_hot_event_multiplatform_evaluation``，不复用旧 mode。
      后端会为该运行单独创建 ``multi_platform/run_NNN/`` 输出目录；
    * single_target → 复用原有 ``_run_hot_event_evaluation``，但显式传
      ``target_user_id``；platform 由 Selector 自动取目标数字人所属平台；
    * multi_user → 复用 ``_run_hot_event_evaluation``，传 ``dispatch_platform``。
    """

    _validate_dispatch_options(options)

    output_dir = _run_dir(run_id) / "output"
    trace_dir = _run_dir(run_id) / "trace"

    if options.dispatch_mode == "cross_platform":
        run_dirs = _resolve_run_dirs(
            workspace_root=PROJECT_ROOT,
            mode=EVAL_MODE_MULTI_PLATFORM,
            output_dir=output_dir,
            trace_dir=trace_dir,
        )
        effective_output_dir = run_dirs.output_dir
        effective_trace_dir = run_dirs.trace_dir
    else:
        effective_output_dir = output_dir
        effective_trace_dir = trace_dir

    for event in events:
        normalized_event = _normalize_hot_event(event)
        event_id = str(normalized_event["event_id"])
        _update_status(run_id, current_event_id=event_id)
        try:
            if options.dispatch_mode == "cross_platform":
                output_path, payload, _stage_times = _run_hot_event_multiplatform_evaluation(
                    workspace_root=PROJECT_ROOT,
                    output_dir=effective_output_dir,
                    hot_event=normalized_event,
                    platform_specs=list(options.platform_specs),
                    profile_limit=options.profile_limit,
                    max_selected_nodes=options.max_selected_nodes,
                    risk_level=options.risk_level,
                    campaign_window_hours=options.campaign_window_hours,
                    max_frequency_per_day=options.max_frequency_per_day,
                    allowed_platforms=options.allowed_platforms,
                    use_llm=options.use_llm,
                    use_llm_event_parser=options.use_llm_event_parser,
                    use_llm_selector=options.use_llm_selector,
                    selector_candidate_limit=options.selector_candidate_limit,
                    # 第十四轮：页面测试控制台不再为每个数字人生成发帖正文。
                    use_llm_content=options.use_llm_content,
                    trace_dir=effective_trace_dir,
                )
            else:
                output_path, payload, _stage_times = _run_hot_event_evaluation(
                    workspace_root=PROJECT_ROOT,
                    output_dir=effective_output_dir,
                    hot_event=normalized_event,
                    profile_limit=options.profile_limit,
                    max_selected_nodes=options.max_selected_nodes,
                    risk_level=options.risk_level,
                    campaign_window_hours=options.campaign_window_hours,
                    max_frequency_per_day=options.max_frequency_per_day,
                    allowed_platforms=options.allowed_platforms,
                    use_llm=options.use_llm,
                    use_llm_event_parser=options.use_llm_event_parser,
                    use_llm_selector=options.use_llm_selector,
                    selector_candidate_limit=options.selector_candidate_limit,
                    # 第十四轮：页面测试控制台不再为每个数字人生成发帖正文。
                    use_llm_content=options.use_llm_content,
                    trace_dir=effective_trace_dir,
                    target_user_id=options.target_user_id,
                    dispatch_platform=options.dispatch_platform
                    if options.dispatch_mode == "multi_user"
                    else None,
                )
            _append_result(
                run_id,
                _result_summary(
                    output_path=output_path,
                    payload=payload,
                    dispatch_mode=options.dispatch_mode,
                    stage_times=_stage_times,
                ),
            )
        except HTTPException:
            raise
        except Exception as exc:
            _append_error(run_id, event_id=event_id, message=f"{type(exc).__name__}: {exc}")


def _result_summary(
    *,
    output_path: Path,
    payload: dict[str, Any],
    dispatch_mode: Literal["single_target", "multi_user", "cross_platform"] = "multi_user",
    stage_times: dict[str, float] | None = None,
) -> dict[str, Any]:
    """把每条事件的最终 payload 压缩成结果列表项；不同 dispatch_mode 走不同分支。

    ``stage_times`` 是该事件评估产物中各阶段耗时（秒），可选；
    5 次压测面板会直接消费这层数据做阶段平均用时计算与可视化。
    """

    # 多平台分发的输出文件名是 ``{event_id}_multiplatform_{tag}_strategy_output.json``，
    # 先剥掉 ``_multiplatform`` 后缀，再去掉 ``_strategy_output.json``，得到 event_id。
    name = output_path.name
    suffix = "_strategy_output.json"
    stem = name[: -len(suffix)] if name.endswith(suffix) else output_path.stem
    if "_multiplatform" in stem:
        event_id = stem.split("_multiplatform", 1)[0]
    else:
        event_id = stem

    summary: dict[str, Any] = {
        "event_id": event_id,
        "event_name": payload.get("事件名称", event_id),
        "dispatch_mode": dispatch_mode,
        "json_output_path": str(output_path),
        # 阶段耗时：键名稳定，缺省为空 dict；前端用 0 做兜底展示。
        "stage_times": dict(stage_times) if stage_times else {},
    }

    # single_target / multi_user / cross_platform 现在共用同一套五维调度策略结构，
    # 因此读取逻辑也共用一份；跨平台模式只是额外把逐平台的人数与选取 id 汇总出来，
    # 供结果列表按平台展示。
    strategy = payload.get("五维调度策略", {})
    target_object = strategy.get("目标对象", {}) if isinstance(strategy, dict) else {}
    content = strategy.get("内容", {}) if isinstance(strategy, dict) else {}
    selected_ids = (
        target_object.get("选取数字人id组", []) if isinstance(target_object, dict) else []
    )
    summary["selected_digital_human_ids"] = selected_ids
    summary["content_llm"] = (
        content.get("内容生成诊断", {}) if isinstance(content, dict) else {}
    )

    if dispatch_mode == "cross_platform":
        summary["per_platform"] = _cross_platform_per_platform(
            strategy if isinstance(strategy, dict) else {}
        )
        return summary

    if dispatch_mode == "single_target":
        diagnostic = payload.get("定向分发诊断")
        if isinstance(diagnostic, dict):
            summary["target_dispatch"] = {
                "target_user_id": diagnostic.get("target_user_id"),
                "target_platform": diagnostic.get("target_platform"),
                "target_role": diagnostic.get("target_role"),
                "warnings": diagnostic.get("warnings") or [],
            }
    return summary


def _cross_platform_per_platform(strategy: dict[str, Any]) -> list[dict[str, Any]]:
    """从五维调度策略的「平台」维度取出逐平台人数与选取 id。

    跨平台模式与多人分发共用五维结构，各平台独立的选人结果集中在
    ``五维调度策略.平台.各平台方案``；结果列表只需要其中的人数与 id，
    因此在这里做一次窄化映射，而不是把整块平台方案透传给前端。
    """

    platform_dimension = strategy.get("平台")
    schemes = (
        platform_dimension.get("各平台方案")
        if isinstance(platform_dimension, dict)
        else None
    )
    if not isinstance(schemes, list):
        return []

    per_platform: list[dict[str, Any]] = []
    for item in schemes:
        if not isinstance(item, dict):
            continue
        ids = item.get("选取数字人id组") or []
        per_platform.append(
            {
                "platform_key": item.get("平台"),
                "requested_count": item.get("目标人数"),
                "selected_count": item.get("选取人数"),
                "shortfall_count": item.get("人数缺口"),
                "safe_shortfall_count": item.get("合格缺口"),
                "selected_user_ids": [
                    str(user_id) for user_id in ids if user_id is not None
                ],
            }
        )
    return per_platform


def _append_result(run_id: str, result: dict[str, Any]) -> None:
    with _LOCK:
        status = _read_status_unlocked(run_id)
        status.setdefault("results", []).append(result)
        status["completed"] = int(status.get("completed", 0)) + 1
        status["updated_at"] = _now()
        _write_status_unlocked(run_id, status)


def _append_error(run_id: str, *, event_id: str | None, message: str) -> None:
    with _LOCK:
        status = _read_status_unlocked(run_id)
        status.setdefault("errors", []).append({"event_id": event_id, "message": message})
        status["failed"] = int(status.get("failed", 0)) + 1
        status["updated_at"] = _now()
        _write_status_unlocked(run_id, status)


def _update_status(run_id: str, **updates: Any) -> None:
    with _LOCK:
        status = _read_status_unlocked(run_id)
        status.update(updates)
        status["updated_at"] = _now()
        _write_status_unlocked(run_id, status)


def _read_status(run_id: str) -> dict[str, Any]:
    with _LOCK:
        return _read_status_unlocked(run_id)


def _read_status_unlocked(run_id: str) -> dict[str, Any]:
    if run_id in _RUNS:
        return dict(_RUNS[run_id])
    status_path = _run_dir(run_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    status = _load_json(status_path)
    _RUNS[run_id] = status
    return dict(status)


def _write_status(run_id: str, status: dict[str, Any]) -> None:
    with _LOCK:
        _write_status_unlocked(run_id, status)


def _write_status_unlocked(run_id: str, status: dict[str, Any]) -> None:
    _RUNS[run_id] = dict(status)
    status_path = _run_dir(run_id) / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_dir(run_id: str) -> Path:
    return WEB_RUNS_ROOT / _safe_run_id(run_id)


def _resolve_result_output_path(run_id: str, event_id: str) -> Path | None:
    """定位某个事件的结果 JSON 文件。

    必须按 status.json 里记录的 ``json_output_path`` 优先解析，因为不同
    dispatch_mode 的产物命名并不统一：

    * multi_user / single_target → ``{event_id}_strategy_output.json``
    * cross_platform            → ``{event_id}_multiplatform_{platform_tag}_strategy_output.json``

    旧实现直接拼 ``{event_id}_strategy_output.json``，导致跨平台模式虽然在
    output 目录里生成了正确文件，detail/download 接口却 404，前端显示"无结果"。
    """

    # 1) 优先用 status.json 中记录的真实产物路径。
    try:
        status = _read_status(run_id)
    except HTTPException:
        status = {}
    for item in status.get("results") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("event_id")) != event_id:
            continue
        raw_path = item.get("json_output_path")
        if not raw_path:
            continue
        candidate = Path(str(raw_path))
        for path in (candidate, PROJECT_ROOT / candidate) if not candidate.is_absolute() else (candidate,):
            if path.exists():
                return path

    # 2) 回退到旧命名（multi_user / single_target）。
    output_dir = _run_dir(run_id) / "output"
    legacy = output_dir / f"{event_id}_strategy_output.json"
    if legacy.exists():
        return legacy

    # 3) 回退到 glob：兼容跨平台等多后缀命名，取最新一份。
    safe_event_id = _safe_event_id(event_id)
    matches = sorted(output_dir.glob(f"{safe_event_id}_*strategy_output.json"))
    if matches:
        return matches[-1]
    return None


def _safe_run_id(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", run_id)


def _safe_event_id(event_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", event_id.strip())
    return cleaned or f"web_event_{_timestamp_id()}"


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_benchmark(
    *, iterations: int, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """汇总 5 次压测结果：各阶段平均 / 总用时 / 每次迭代总用时。

    输入 ``results`` 是 ``run_benchmark`` 累积的迭代列表；只统计 ``ok=True`` 的轮。
    输出结构是前端 5 次压测面板直接消费的形状（含 ``stage_averages`` /
    ``stage_min_max`` / ``per_iteration_total`` 等）。
    """
    ok_results = [item for item in results if item.get("ok")]
    # 只统计"已明确失败"的轮次；未跑完的轮次不算失败（异步任务会边跑边更新，
    # 中间状态下的 iterations - len(ok_results) 会把"还没轮到"误判为失败）。
    failed_iterations = sum(1 for item in results if not item.get("ok"))

    # 各阶段汇总：只在所有成功轮里都出现过的阶段才计算。
    stage_samples: dict[str, list[float]] = {}
    per_iteration_total: list[float] = []
    per_iteration_iteration: list[int] = []
    for item in ok_results:
        iteration_index = int(item.get("iteration", 0))
        stage_times = item.get("stage_times") or {}
        for stage, duration in stage_times.items():
            stage_samples.setdefault(stage, []).append(float(duration))
        event_total = float(item.get("elapsed_seconds", 0.0))
        per_iteration_total.append(event_total)
        per_iteration_iteration.append(iteration_index)

    stage_averages: dict[str, float] = {}
    stage_min_max: dict[str, dict[str, float]] = {}
    for stage, samples in stage_samples.items():
        stage_averages[stage] = statistics.fmean(samples)
        stage_min_max[stage] = {
            "min": min(samples),
            "max": max(samples),
            "samples": samples,
        }

    # 总用时：从所有轮次的 elapsed_seconds 求和（这是端到端的真实等待时间）。
    total_seconds = sum(per_iteration_total)
    return {
        "iterations": iterations,
        "successful_iterations": len(ok_results),
        "failed_iterations": failed_iterations,
        "total_seconds": total_seconds,
        "average_seconds": (
            statistics.fmean(per_iteration_total) if per_iteration_total else 0.0
        ),
        "min_seconds": min(per_iteration_total) if per_iteration_total else 0.0,
        "max_seconds": max(per_iteration_total) if per_iteration_total else 0.0,
        "stage_averages": stage_averages,
        "stage_min_max": stage_min_max,
        "per_iteration": [
            {"iteration": iteration, "seconds": seconds}
            for iteration, seconds in zip(per_iteration_iteration, per_iteration_total)
        ],
    }


# --- 压测（benchmark）任务：内存缓存 + 独立落盘目录 + 后台线程 ---


def _benchmark_dir(benchmark_id: str) -> Path:
    return BENCHMARK_RUNS_ROOT / _safe_run_id(benchmark_id)


def _benchmark_path(benchmark_id: str) -> Path:
    return _benchmark_dir(benchmark_id) / "benchmark.json"


def _read_benchmark(benchmark_id: str) -> dict[str, Any]:
    """读一条压测状态：内存优先，磁盘兜底（服务重启后依然能看到历史）。"""
    with _BENCHMARK_LOCK:
        cached = _BENCHMARKS.get(benchmark_id)
        if cached is not None:
            return dict(cached)
    path = _benchmark_path(benchmark_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Benchmark 状态文件损坏: {benchmark_id}"
        ) from exc
    payload = _normalize_benchmark_payload(payload)
    with _BENCHMARK_LOCK:
        _BENCHMARKS[benchmark_id] = payload
    return dict(payload)


def _normalize_benchmark_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """兼容早期"同步压测"落盘的 ``benchmark.json``（缺 state / completed / events）。

    旧版是跑完才写盘的，所以缺 ``state`` 的文件一定是已完成的历史记录；
    事件字段旧版叫 ``event``（单条），这里统一补成 ``events`` 列表。
    """
    normalized = dict(payload)
    normalized.setdefault("state", "completed")
    normalized.setdefault("completed", normalized.get("successful_iterations") or 0)
    normalized.setdefault("failed", normalized.get("failed_iterations") or 0)
    normalized.setdefault("error", None)
    normalized.setdefault("updated_at", normalized.get("created_at"))

    events = normalized.get("events")
    if not events:
        single = normalized.get("event")
        if isinstance(single, dict) and single.get("event_id"):
            normalized["events"] = [{"iteration": 1, **single}]
        else:
            normalized["events"] = []
    runs = normalized.get("runs")
    normalized["runs"] = runs if isinstance(runs, list) else []
    return normalized


def _write_benchmark(benchmark_id: str, payload: dict[str, Any]) -> None:
    """覆盖写内存缓存与 ``benchmark.json``。

    这里必须 deepcopy：``events`` / ``runs`` 是后台任务逐轮 append 的同一个
    list，浅拷贝会让内存缓存"提前"看到后续轮次的事件（``completed`` 与实际
    事件数对不上）。
    """
    snapshot = copy.deepcopy(payload)
    with _BENCHMARK_LOCK:
        _BENCHMARKS[benchmark_id] = snapshot
    path = _benchmark_path(benchmark_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_benchmark(benchmark_id: str, **updates: Any) -> dict[str, Any]:
    """读-改-写：更新内存里那条压测并落盘，返回更新后的快照。

    每条压测只有一个后台线程在写，因此这里不做跨键加锁的复杂处理；
    锁只用来保护 ``_BENCHMARKS`` 字典本身。
    """
    with _BENCHMARK_LOCK:
        payload = dict(_BENCHMARKS.get(benchmark_id) or {})
        payload.update(updates)
        payload["updated_at"] = _now()
        snapshot = dict(payload)
        _BENCHMARKS[benchmark_id] = snapshot
    path = _benchmark_path(benchmark_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def _benchmark_history_item(payload: dict[str, Any]) -> dict[str, Any]:
    """把完整压测 payload 裁剪成"测试历史列表"需要的轻量条目。"""
    summary = payload.get("summary") or {}
    options = payload.get("options") or {}
    events = payload.get("events") or []
    return {
        "benchmark_id": payload.get("benchmark_id"),
        "label": payload.get("label"),
        "state": payload.get("state") or "running",
        "iterations": payload.get("iterations"),
        "completed": payload.get("completed") or 0,
        "failed": payload.get("failed") or 0,
        "successful_iterations": payload.get("successful_iterations") or 0,
        "failed_iterations": payload.get("failed_iterations") or 0,
        "use_llm": bool(options.get("use_llm")),
        "dispatch_mode": options.get("dispatch_mode"),
        "event_count": len(events),
        "events": events,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "total_seconds": summary.get("total_seconds"),
        "average_seconds": summary.get("average_seconds"),
        "output_dir": payload.get("output_dir"),
        "error": payload.get("error"),
    }


def _benchmark_job_wrapper(
    benchmark_id: str,
    options: RunOptions,
    iterations: int,
    pinned_event_id: str | None,
) -> None:
    """后台线程入口：把压测从 queued 推进到 running → completed/failed。"""
    try:
        _update_benchmark(benchmark_id, state="running")
        _run_benchmark_job(
            benchmark_id=benchmark_id,
            options=options,
            iterations=iterations,
            pinned_event_id=pinned_event_id,
        )
        payload = _read_benchmark(benchmark_id)
        succeeded = int(payload.get("successful_iterations") or 0)
        _update_benchmark(
            benchmark_id,
            state="completed" if succeeded > 0 else "failed",
            error=None if succeeded > 0 else "所有轮次都失败了",
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for background jobs
        _update_benchmark(
            benchmark_id, state="failed", error=f"{type(exc).__name__}: {exc}"
        )


def _run_benchmark_job(
    *,
    benchmark_id: str,
    options: RunOptions,
    iterations: int,
    pinned_event_id: str | None,
) -> None:
    """串行跑 N 轮压测，每轮换一条事件；每轮结束都刷新一次状态供前端轮询。

    事件选择策略：

    * 用户显式传了 ``event_id`` → 锁定该事件跑 N 轮（用于对比稳定性）；
    * 否则 → 一开始就**不放回**地随机抽 N 条，保证一次测试内每个测试点的事件
      都不一样；只有当 N 超过事件总数时才退化为允许重复。
    """
    benchmark_dir = _benchmark_dir(benchmark_id)
    output_root = benchmark_dir / "output"
    trace_root = benchmark_dir / "trace"

    hot_events = load_hot_events(DEFAULT_EVENTS_PATH)
    if pinned_event_id:
        picked = select_hot_events(hot_events, event_id=pinned_event_id, event_limit=1)
        if not picked:
            raise RuntimeError(f"指定的事件 ID 没有匹配到 hot_event：{pinned_event_id}")
        selected = [picked[0]] * iterations
    else:
        selected = _pick_random_hot_events(hot_events, count=iterations)
        if not selected:
            raise RuntimeError("hot_event 集合为空，无法执行压测。")
        while len(selected) < iterations:  # 轮数 > 事件总数时才允许重复
            selected.append(random.choice(selected))

    iteration_results: list[dict[str, Any]] = []
    event_entries: list[dict[str, Any]] = []

    for index, raw_event in enumerate(selected, start=1):
        normalized_event = _normalize_hot_event(raw_event)
        event_entry = {
            "iteration": index,
            "event_id": str(normalized_event["event_id"]),
            "event_title": normalized_event.get("event_title"),
            "domain": normalized_event.get("domain"),
        }
        event_entries.append(event_entry)
        # 抽定就公告：让前端轮询立刻看到这一轮用的是哪条事件（显示为"测试中"）。
        _update_benchmark(benchmark_id, events=list(event_entries))

        iter_output_dir = output_root / f"iter_{index:02d}"
        iter_trace_dir = trace_root / f"iter_{index:02d}"
        iter_output_dir.mkdir(parents=True, exist_ok=True)
        iter_trace_dir.mkdir(parents=True, exist_ok=True)

        iteration_start = time.perf_counter()
        try:
            if options.dispatch_mode == "cross_platform":
                output_path, payload, stage_times = _run_hot_event_multiplatform_evaluation(
                    workspace_root=PROJECT_ROOT,
                    output_dir=iter_output_dir,
                    hot_event=normalized_event,
                    platform_specs=list(options.platform_specs),
                    profile_limit=options.profile_limit,
                    max_selected_nodes=options.max_selected_nodes,
                    risk_level=options.risk_level,
                    campaign_window_hours=options.campaign_window_hours,
                    max_frequency_per_day=options.max_frequency_per_day,
                    allowed_platforms=options.allowed_platforms,
                    use_llm=options.use_llm,
                    use_llm_event_parser=options.use_llm_event_parser,
                    use_llm_selector=options.use_llm_selector,
                    selector_candidate_limit=options.selector_candidate_limit,
                    # 第十四轮：页面压测同样不生成发帖正文，避免每轮多一次 LLM。
                    use_llm_content=options.use_llm_content,
                    trace_dir=iter_trace_dir,
                )
            else:
                output_path, payload, stage_times = _run_hot_event_evaluation(
                    workspace_root=PROJECT_ROOT,
                    output_dir=iter_output_dir,
                    hot_event=normalized_event,
                    profile_limit=options.profile_limit,
                    max_selected_nodes=options.max_selected_nodes,
                    risk_level=options.risk_level,
                    campaign_window_hours=options.campaign_window_hours,
                    max_frequency_per_day=options.max_frequency_per_day,
                    allowed_platforms=options.allowed_platforms,
                    use_llm=options.use_llm,
                    use_llm_event_parser=options.use_llm_event_parser,
                    use_llm_selector=options.use_llm_selector,
                    selector_candidate_limit=options.selector_candidate_limit,
                    # 第十四轮：页面压测同样不生成发帖正文，避免每轮多一次 LLM。
                    use_llm_content=options.use_llm_content,
                    trace_dir=iter_trace_dir,
                    target_user_id=options.target_user_id,
                    dispatch_platform=options.dispatch_platform
                    if options.dispatch_mode == "multi_user"
                    else None,
                )
            elapsed = time.perf_counter() - iteration_start
            iteration_results.append(
                {
                    "iteration": index,
                    "run_id": f"{benchmark_id}_iter{index:02d}",
                    "ok": True,
                    "elapsed_seconds": elapsed,
                    "stage_times": stage_times,
                    "event": event_entry,
                    "summary": _result_summary(
                        output_path=output_path,
                        payload=payload,
                        dispatch_mode=options.dispatch_mode,
                        stage_times=stage_times,
                    ),
                }
            )
        except Exception as exc:  # 单轮失败不影响其它轮
            iteration_results.append(
                {
                    "iteration": index,
                    "run_id": f"{benchmark_id}_iter{index:02d}",
                    "ok": False,
                    "elapsed_seconds": time.perf_counter() - iteration_start,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stage_times": {},
                    "event": event_entry,
                    "summary": None,
                }
            )

        # 每轮结束立即刷新：前端轮询能看到进度条与已完成的柱状数据逐步长出来。
        succeeded = sum(1 for item in iteration_results if item["ok"])
        failed = len(iteration_results) - succeeded
        _update_benchmark(
            benchmark_id,
            completed=succeeded,
            failed=failed,
            successful_iterations=succeeded,
            failed_iterations=failed,
            events=event_entries,
            runs=iteration_results,
            summary=_summarize_benchmark(iterations=iterations, results=iteration_results),
        )

