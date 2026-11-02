from __future__ import annotations

import copy
import json
import random
import re
import shutil
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from .ocr_recognizer import run_ocr
from .platform_dispatcher import PlatformSpec, to_json_dict
from .data_loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "eval" / "hot_event_opinion_variants_200.json"
WEB_RUNS_ROOT = PROJECT_ROOT / "outputs" / "web_runs"

# 数字人原始画像缓存：按 user_id 取回底层画像库（数据库）中的原始记录，避免每次
# 悬停 / 点击都重新加载整个画像文件。进程内只加载一次（存原始节点，含全部字段）。
_NODE_RAW_CACHE: dict[str, Any] | None = None
_NODE_RAW_CACHE_ROOT: Path | None = None
TEST_RUNS_ROOT = WEB_RUNS_ROOT / "test_runs"
FRONTEND_DIST = PROJECT_ROOT / "web" / "frontend" / "dist"
MAX_WORKERS = 1

_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
# 总体测试：独立内存缓存 + 独立落盘目录，与 /api/runs 运行历史隔离。
_TEST_RUNS: dict[str, dict[str, Any]] = {}
_TEST_RUN_LOCK = threading.Lock()
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
    # 单人定向分发：指定唯一数字人 user_id（思路一=主发布节点 / 思路二=被 @ 的路人）；多人分发下无效。
    target_user_id: str | None = None
    # 多人分发指定平台：只从该平台已有的数字人中选人；空表示不限平台。
    dispatch_platform: str | None = None
    # 单人定向的两种打法（仅 single_target 下生效）：
    #   "self_operate" —— 思路一：指定数字人自运营闭环（默认）；
    #   "attention"    —— 思路二：多数字人协同，全程 @ 指定路人（target_user_id）使其注意到事件。
    single_target_mode: Literal["self_operate", "attention"] = "self_operate"
    # 思路二中 @ 路人时引用的"旧帖编号"（直接假设路人有一条旧帖，仅用编号点名，不读取旧帖内容）。
    target_old_post_id: str | None = None
    # 白库水军参与：放大参与人数（仅点赞/转发，无文本），不计入 max_selected_nodes 上限。
    water_army_enabled: bool = False
    water_army_count: int = Field(default=0, ge=0, le=300)
    # 跨平台并行分发：每个平台独立的人数配置；至少 1 个平台。
    platform_specs: list[PlatformSpec] = Field(default_factory=list)

    use_llm: bool = True
    event_limit: int | None = Field(default=None, ge=1)
    event_id: str | None = None


class TextRunRequest(BaseModel):
    event: dict[str, Any] | None = None
    event_text: str | None = None
    options: RunOptions = Field(default_factory=RunOptions)


class EvalRunRequest(BaseModel):
    options: RunOptions = Field(default_factory=lambda: RunOptions(event_limit=200))


class SubTaskSpec(BaseModel):
    """总体测试中的一个子任务：可指定单人 / 多人 / 多平台，以及是否含图、跑几条随机事件。"""

    kind: Literal["single_target", "multi_user", "cross_platform"]
    # 单人定向的两种打法（仅 single_target 生效）。
    single_target_mode: Literal["self_operate", "attention"] = "self_operate"
    # 单人定向必须手动指定目标数字人；多人 / 跨平台忽略。
    target_user_id: str | None = None
    # 数字人注意（attention）模式下引用的"原帖编号"，仅用于 @ 点名，不读取内容。
    target_old_post_id: str | None = None
    # 多人 / 跨平台的人数与平台配置（复用 PlatformSpec）；单人模式忽略。
    platform_specs: list[PlatformSpec] = Field(default_factory=list)
    # 是否含图：true 时系统从 eval/image 随机抽一张图，OCR 成事件再分发。
    with_image: bool = False
    # 本子任务跑几条随机事件（每条事件独立计时）。
    event_count: int = Field(default=1, ge=1, le=50)
    # 时间窗口：该子任务的投放窗口（小时）。
    campaign_window_hours: int = Field(default=24, ge=1, le=168)
    # 白库特殊支持节点（水军）：仅点赞 / 转发以放大参与人数，不计入选人上限。
    water_army_enabled: bool = False
    water_army_count: int = Field(default=0, ge=0, le=300)
    # 是否启用 LLM（由顶栏开关统一控制，按子任务透传）。
    use_llm: bool = False


class TestRunRequest(BaseModel):
    """总体测试请求：一组有序子任务，每个子任务独立运行、各自产出计时，最后汇总。"""

    subtasks: list[SubTaskSpec] = Field(default_factory=list, min_length=1)
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
    # 端到端耗时（秒）：从收到请求（``created_at``）到本次运行产出结果为止。
    # 进行中为 None，只有在终态（completed / failed）时才写入。
    elapsed_seconds: float | None = None


def create_app() -> FastAPI:
    app = FastAPI(title="Influence Strategy Web Console", version="0.1.0")
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

    @app.on_event("startup")
    def _warmup_local_models() -> None:
        # 预加载本地模型（BGE 语义编码器 + GraphSAGE 图特征），把约 0.8s 的冷启动
        # 成本从「首个运行请求」挪到「服务启动」时，避免用户第一次点运行要等模型载入。
        # 模型产物缺失/损坏时静默跳过，不影响服务启动与其余功能。
        try:
            from .local_models import load_local_models

            load_local_models(PROJECT_ROOT, warmup=True)
        except Exception:  # noqa: BLE001 - 预加载失败不应阻止服务启动
            pass

        # 预热 OCR 引擎（RapidOCR 三个 ONNX 模型加载，约 1~2s）。
        # 不预热时，进程内「第一个图片请求」会把这次加载算进去，2MB 截图可能冲到 6~7s；
        # 预热后首请求落回 ~4.5s、后续稳定在 2.5~3s（CPU）。启动期一次性付清加载成本，
        # OCR 不可用（缺依赖）时静默跳过，不影响纯文本 / 系统事件流程。
        try:
            from .ocr_recognizer import is_ocr_available, _load_engine

            if is_ocr_available():
                _load_engine()
        except Exception:  # noqa: BLE001 - 预加载失败不应阻止服务启动
            pass

        # 预热多平台预计算所需的 35k 用户数据 / 富画像 / 交互源，避免首个跨平台请求
        # 在主线程串行加载（约 0.5~1s）。与 compute_event_intermediates 使用相同默认
        # product_name（abc_reading）与 limit=None，命中同一模块级缓存；失败静默跳过。
        try:
            from .data_loader import DataLoader

            _dl = DataLoader(workspace_root=PROJECT_ROOT)
            _dl.load_profiles()
            try:
                _dl.load_enriched_graph_attributes()
            except Exception:  # noqa: BLE001 - 富画像缺失不影响其余功能
                pass
            try:
                _dl.load_interactions(limit_sources=None, limit_records_per_source=0)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 - 预加载失败不应阻止服务启动
            pass

        # 预热跨平台进程池的 worker：让每个 worker 进程提前加载 35k 画像/富画像/交互，
        # 把首个跨平台请求里 ~15s 的冷启动（worker 首次加载）挪到服务启动时一次性付清，
        # 之后跨平台墙钟稳定在 ~2~3s（参见 eval_hot_events._run_hot_event_multiplatform_evaluation）。
        # 进程池创建/预热失败均静默跳过，不影响其余功能与启动。
        try:
            from .eval_hot_events import _warm_platform_pool

            _warm_platform_pool(PROJECT_ROOT)
        except Exception:  # noqa: BLE001
            pass

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "workspace_root": str(PROJECT_ROOT),
            "default_events_path": str(DEFAULT_EVENTS_PATH),
            "web_runs_root": str(WEB_RUNS_ROOT),
            # 全链路默认不调用大模型：图像识别走本地 OCR，事件解析走规则，
            # 文案走占位文本。只有显式把对应开关打开才会请求外部 API。
            "llm_default_enabled": False,
            "image_recognition": "local_ocr",
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

    @app.get("/api/nodes/{user_id}")
    def get_node_raw(user_id: str) -> dict[str, Any]:
        """返回某数字人在底层画像库（数据库）中的原始记录（全部字段）。

        用于前端点击 / 悬停数字人时展示其数据库原始数据。直接返回画像文件里该
        ``user_id`` 对应的原始节点（含所有原始字段，如简介 / 兴趣 / 平台 / 关注 /
        邻居关系等），不做字段裁剪。进程内缓存一次加载，后续直接命中。
        """
        global _NODE_RAW_CACHE, _NODE_RAW_CACHE_ROOT
        if _NODE_RAW_CACHE is None or _NODE_RAW_CACHE_ROOT != PROJECT_ROOT:
            loader = DataLoader(workspace_root=PROJECT_ROOT)
            _NODE_RAW_CACHE = loader._read_json(loader._preferred_profile_path())
            _NODE_RAW_CACHE_ROOT = PROJECT_ROOT
        raw = _NODE_RAW_CACHE.get(user_id)
        if raw is None:
            raise HTTPException(status_code=404, detail=f"node not found: {user_id}")
        return raw

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

    @app.post("/api/runs/text-image", response_model=RunStatus)
    def run_text_image_event(
        event_text: str = Form(...),
        image: UploadFile | None = File(None),
        options: str = Form("{}"),
    ) -> dict[str, Any]:
        """用户输入：文字必选 + 图片可选。

        文字直接作为事件描述；若附带图片，后端先用本地 OCR 识别图片上的文字，
        再与用户输入文字融合为一段完整的事件描述，交给与纯文本相同的规则事件解析
        链路（``_event_from_text_request``）处理。下游选人 / 策略 / 输出流程完全
        一致，因此这里只是"事件解析方式"多了 OCR 融合这一步，不改变后续任何逻辑。
        """

        parsed_options = _parse_options(options)
        _validate_dispatch_options(parsed_options)

        if not event_text or not event_text.strip():
            raise HTTPException(status_code=400, detail="event_text 不能为空。")

        has_image = image is not None and getattr(image, "filename", "")
        status = _create_run(
            mode="image" if has_image else "text",
            total=1,
            dispatch_mode=parsed_options.dispatch_mode,
        )
        image_path: Path | None = None
        if has_image:
            suffix = Path(image.filename or "").suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image type: {suffix or image.filename}",
                )
            run_dir = _run_dir(status["run_id"])
            upload_dir = run_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            image_path = upload_dir / f"input{suffix}"
            with image_path.open("wb") as handle:
                shutil.copyfileobj(image.file, handle)

        _submit_job(
            status["run_id"],
            lambda: _run_text_image_job(
                run_id=status["run_id"],
                event_text=event_text,
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

    @app.post("/api/test-runs")
    def create_test_run(request: TestRunRequest) -> dict[str, Any]:
        """提交一次"总体测试"：异步执行，立即返回 queued 状态与 test_run_id。

        每个子任务独立运行并复用现有计时（stage_times / per_platform），后台线程
        串行跑完所有子任务后汇总；前端凭 test_run_id 轮询 GET /api/test-runs/{id}。
        """
        if not request.subtasks:
            raise HTTPException(status_code=400, detail="subtasks 不能为空。")
        # 同步校验每个子任务配置（非法直接 400，不进入后台）。
        for spec in request.subtasks:
            _subtask_to_run_options(spec)

        test_run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        base_dir = TEST_RUNS_ROOT / _safe_run_id(test_run_id)
        (base_dir / "output").mkdir(parents=True, exist_ok=True)
        (base_dir / "trace").mkdir(parents=True, exist_ok=True)
        now = _now()
        payload: dict[str, Any] = {
            "test_run_id": test_run_id,
            "label": (request.label or "").strip() or None,
            "state": "queued",
            "total_subtasks": len(request.subtasks),
            "completed": 0,
            "failed": 0,
            "subtasks": [],
            "summary": None,
            "output_dir": str(base_dir),
            "created_at": now,
            "updated_at": now,
            "error": None,
        }
        _write_test_run(test_run_id, payload)
        _EXECUTOR.submit(_test_run_job_wrapper, test_run_id, request)
        return dict(payload)

    @app.get("/api/test-runs")
    def list_test_runs() -> dict[str, Any]:
        """总体测试历史列表（与 /api/runs 完全隔离）。"""
        TEST_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        merged: dict[str, dict[str, Any]] = {}
        for path in TEST_RUNS_ROOT.glob("*/test_run.json"):
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = str(stored.get("test_run_id") or path.parent.name)
            merged[key] = stored
        with _TEST_RUN_LOCK:
            merged.update({key: dict(value) for key, value in _TEST_RUNS.items()})
        items = [_test_run_history_item(payload) for payload in merged.values()]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"test_runs": items[:50]}

    @app.get("/api/test-runs/{test_run_id}")
    def get_test_run(test_run_id: str) -> dict[str, Any]:
        """单条总体测试详情（含每子任务 stage_times / summary），供前端轮询。"""
        return _read_test_run(test_run_id)

    @app.delete("/api/test-runs/{test_run_id}")
    def delete_test_run(test_run_id: str) -> dict[str, Any]:
        payload = _read_test_run(test_run_id)
        if str(payload.get("state")) in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail=f"测试 {test_run_id} 仍在进行（state={payload.get('state')}），请等待结束后再删除。",
            )
        base_dir = _test_run_dir(test_run_id)
        if base_dir.exists():
            shutil.rmtree(base_dir)
        with _TEST_RUN_LOCK:
            _TEST_RUNS.pop(test_run_id, None)
        return {"deleted": True, "test_run_id": test_run_id}


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
        base_trace_dir = _run_dir(run_id) / "trace"
        # 1) 单平台 / 多人协同：trace 直接落在 trace/<event_id>。
        direct = base_trace_dir / event_id
        if direct.exists():
            return {
                "run_id": run_id,
                "event_id": event_id,
                "files": _collect_trace_files(direct),
            }
        # 2) 跨平台：trace 按平台分目录存放在 trace/<platform>/<event_id>，
        #    聚合并以「平台名/」作为文件名前缀，避免不同平台文件互相覆盖。
        files: list[dict[str, Any]] = []
        if base_trace_dir.exists():
            for platform_dir in sorted(base_trace_dir.iterdir()):
                if not platform_dir.is_dir():
                    continue
                candidate = platform_dir / event_id
                if candidate.exists() and candidate.is_dir():
                    files.extend(
                        _collect_trace_files(candidate, prefix=f"{platform_dir.name}/")
                    )
        if files:
            return {"run_id": run_id, "event_id": event_id, "files": files}
        raise HTTPException(status_code=404, detail=f"Trace not found: {event_id}")

    @app.get("/api/runs/{run_id}/files/{event_id}")
    def download_result(run_id: str, event_id: str) -> FileResponse:
        output_path = _resolve_result_output_path(run_id, event_id)
        if output_path is None:
            raise HTTPException(status_code=404, detail=f"Result not found: {event_id}")
        return FileResponse(output_path, filename=output_path.name, media_type="application/json")

    @app.get("/api/runs/{run_id}/image")
    def serve_run_image(run_id: str) -> FileResponse:
        """返回本次运行上传的识别原图（纯图 / 图文任务都会落到 uploads/input.*）。"""
        run_dir = _run_dir(run_id)
        upload_dir = run_dir / "uploads"
        if upload_dir.is_dir():
            for suffix in SUPPORTED_IMAGE_SUFFIXES:
                candidate = upload_dir / f"input{suffix}"
                if candidate.is_file():
                    return FileResponse(candidate)
        raise HTTPException(status_code=404, detail="no image for this run")

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
    """从 hot_event 集合里**不放回**地随机抽 ``count`` 条事件。

    用于总体测试：一次测试里每个子任务的事件尽量互不相同，避免同一事件的
    画像 / LLM 缓存被"预热"后拉低后续轮次耗时，从而更准确地反映配置差异。

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


def _elapsed_seconds(created_at: str | None) -> float | None:
    """从 ``created_at``（收到请求那一刻）到现在经过的秒数。

    解析失败时返回 None——耗时是展示信息，不该因为它把整个 run 判成失败。
    """

    if not created_at:
        return None
    try:
        start = datetime.fromisoformat(str(created_at))
    except ValueError:
        return None
    return round(max((datetime.now().astimezone() - start).total_seconds(), 0.0), 3)


def _run_job_wrapper(run_id: str, job: Any) -> None:
    created_at = _read_status(run_id).get("created_at")
    try:
        _update_status(run_id, state="running")
        job()
        status = _read_status(run_id)
        final_state = "failed" if status.get("completed", 0) == 0 and status.get("failed", 0) else "completed"
        _update_status(
            run_id,
            state=final_state,
            current_event_id=None,
            elapsed_seconds=_elapsed_seconds(created_at),
        )
    except Exception as exc:  # pragma: no cover - defensive boundary for background jobs
        _append_error(run_id, event_id=None, message=f"{type(exc).__name__}: {exc}")
        _update_status(
            run_id,
            state="failed",
            current_event_id=None,
            elapsed_seconds=_elapsed_seconds(created_at),
        )


def _run_image_job(*, run_id: str, image_path: Path, options: RunOptions) -> None:
    _update_status(run_id, current_event_id="image_recognition")
    reference_events = load_hot_events(DEFAULT_EVENTS_PATH)
    t0 = time.perf_counter()
    events = load_image_events(
        image=image_path,
        image_dir=None,
        reference_events=reference_events,
        workspace_root=PROJECT_ROOT,
        event_limit=1,
        event_id=options.event_id,
        use_llm=options.use_llm,
    )
    ocr_seconds = time.perf_counter() - t0
    for ev in events:
        meta = ev.get("image_recognition")
        if isinstance(meta, dict):
            meta["ocr_seconds"] = round(ocr_seconds, 3)
    _update_status(run_id, total=len(events))
    _run_events_job(run_id=run_id, events=events, options=options)


def _fuse_text_and_ocr(
    event_text: str, image_path: Path | None
) -> tuple[str, str | None]:
    """把用户输入文字与图片 OCR 文字融合成一段事件描述。

    返回 ``(融合后文本, OCR 原文或 None)``，便于上层把识别到的文字回显到结果里。
    融合策略：用户输入为主，OCR 文本作为补充内容追加在后面。图片缺失或 OCR
    不可用时，仅保留用户文字（不抛异常，由上层决定是否降级）。
    """

    text = (event_text or "").strip()
    if image_path is None:
        return text, None
    try:
        outcome = run_ocr(image_path)
    except Exception:
        return text, None
    ocr_text = "\n".join(outcome.kept_texts).strip()
    if not ocr_text:
        return text, None
    if not text:
        return ocr_text, ocr_text
    return f"{text}\n{ocr_text}", ocr_text


def _run_text_image_job(
    *,
    run_id: str,
    event_text: str,
    image_path: Path | None,
    options: RunOptions,
) -> None:
    """文字 + 可选图片：OCR 融合后走与纯文本完全相同的事件解析与下游流程。"""

    _update_status(run_id, current_event_id="image_recognition")
    t0 = time.perf_counter()
    try:
        combined_text, ocr_text = _fuse_text_and_ocr(event_text, image_path)
    except Exception as exc:  # OCR 异常不应让整次运行崩溃，降级为仅用文字
        combined_text = (event_text or "").strip()
        ocr_text = None
        if not combined_text:
            raise ValueError(f"文字输入为空且图片识别失败：{exc}") from exc
        ocr_seconds = None
    else:
        ocr_seconds = time.perf_counter() - t0
    _update_status(run_id, current_event_id=None)

    if not combined_text.strip():
        raise ValueError("文字输入与图片识别结果均为空，无法构建事件。")
    request = TextRunRequest(event_text=combined_text, options=options)
    event = _event_from_text_request(request)
    if image_path is not None:
        event["source_type"] = "image"
        event["source_image"] = str(image_path)
        event["image_recognition"] = {
            "method": "ocr_local",
            "ocr_seconds": round(ocr_seconds, 3) if ocr_seconds is not None else None,
            "text": ocr_text,
        }
    _run_events_job(run_id=run_id, events=[event], options=options)


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
                    trace_dir=effective_trace_dir,
                    water_army_enabled=options.water_army_enabled,
                    water_army_count=options.water_army_count,
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
                    trace_dir=effective_trace_dir,
                    target_user_id=options.target_user_id,
                    dispatch_platform=options.dispatch_platform
                    if options.dispatch_mode == "multi_user"
                    else None,
                    single_target_mode=options.single_target_mode,
                    target_old_post_id=options.target_old_post_id,
                    water_army_enabled=options.water_army_enabled,
                    water_army_count=options.water_army_count,
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
        # 单次事件的完整五维调度策略产物：前端「具体结果」卡片直接展开查看。
        "strategy_output": payload,
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
        summary["per_platform"] = _cross_platform_per_platform(payload)
        # 跨平台模式顶层没有单一「五维调度策略」，selected ids 由各平台结果整合。
        selected_ids = [
            str(uid)
            for pr in (payload.get("各平台结果") or [])
            for uid in (
                (pr.get("五维调度策略") or {})
                .get("目标对象", {})
                .get("选取数字人id组", []) or []
            )
        ]
        summary["selected_digital_human_ids"] = selected_ids
        return summary

    if dispatch_mode == "single_target":
        diagnostic = payload.get("定向分发诊断") or payload.get("数字人注意诊断")
        if isinstance(diagnostic, dict):
            summary["target_dispatch"] = {
                "target_user_id": diagnostic.get("target_user_id"),
                "target_user_name": diagnostic.get("target_user_name"),
                "target_platform": diagnostic.get("target_platform"),
                "target_role": diagnostic.get("target_role"),
                "target_old_post_id": diagnostic.get("target_old_post_id"),
                "warnings": diagnostic.get("warnings") or diagnostic.get("target_warning") or [],
            }
    support_block = payload.get("特殊支持节点")
    if isinstance(support_block, dict):
        summary["support_nodes"] = {
            "enabled": support_block.get("enabled"),
            "count": support_block.get("count"),
            "platforms": support_block.get("platforms"),
            "core_actor_count": support_block.get("core_actor_count"),
        }
    return summary


def _cross_platform_per_platform(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从多平台聚合结果里取出逐平台人数与选取 id。

    多平台模式把每个平台的完整五维策略放在 ``各平台结果``，结果列表只需窄化出
    平台、目标/选取人数与 id，供结果列表与前端按平台展示。
    """
    platform_results = payload.get("各平台结果") if isinstance(payload, dict) else None
    if not isinstance(platform_results, list):
        return []

    per_platform: list[dict[str, Any]] = []
    for pr in platform_results:
        if not isinstance(pr, dict):
            continue
        ids = [
            str(u)
            for u in (
                (pr.get("五维调度策略") or {})
                .get("目标对象", {})
                .get("选取数字人id组", []) or []
            )
        ]
        per_platform.append(
            {
                "platform_key": pr.get("平台"),
                "requested_count": pr.get("目标人数"),
                "selected_count": pr.get("选取人数"),
                "shortfall_count": max(
                    0, int(pr.get("目标人数", 0)) - int(pr.get("选取人数", 0))
                ),
                "selected_user_ids": ids,
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


def _collect_trace_files(trace_dir: Path, prefix: str = "") -> list[dict[str, Any]]:
    """读取某个 trace 目录下的 *.json，整理成前端 TraceSection 需要的结构。"""
    files: list[dict[str, Any]] = []
    if not trace_dir.exists():
        return files
    for path in sorted(trace_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            payload = path.read_text(encoding="utf-8")
        files.append({"name": prefix + path.name, "payload": payload})
    return files


def _aggregate_stage_timings(
    *, iterations: int, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """汇总一组运行结果的阶段耗时：各阶段平均 / 总用时 / 每次迭代总用时。

    输入 ``results`` 是各次运行（事件 / 子任务轮次）的列表，每项含
    ``ok`` / ``stage_times`` / ``elapsed_seconds``；只统计 ``ok=True`` 的项。
    输出结构可直接被前端汇总面板消费（含 ``stage_averages`` / ``stage_min_max`` /
    ``per_iteration`` 等）。复用既有计时字段，不新增任何计时测量。
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
            if isinstance(duration, (int, float)):
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


# --- 总体测试（test-run）：多子任务组合批量评测，复用现有计时与运行函数 ---


def _test_run_dir(test_run_id: str) -> Path:
    return TEST_RUNS_ROOT / _safe_run_id(test_run_id)


def _test_run_path(test_run_id: str) -> Path:
    return _test_run_dir(test_run_id) / "test_run.json"


def _write_test_run(test_run_id: str, payload: dict[str, Any]) -> None:
    snapshot = copy.deepcopy(payload)
    with _TEST_RUN_LOCK:
        _TEST_RUNS[test_run_id] = snapshot
    path = _test_run_path(test_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_test_run(test_run_id: str) -> dict[str, Any]:
    with _TEST_RUN_LOCK:
        cached = _TEST_RUNS.get(test_run_id)
        if cached is not None:
            return dict(cached)
    path = _test_run_path(test_run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Test run not found: {test_run_id}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Test run 状态文件损坏: {test_run_id}") from exc
    with _TEST_RUN_LOCK:
        _TEST_RUNS[test_run_id] = payload
    return dict(payload)


def _update_test_run(test_run_id: str, **updates: Any) -> dict[str, Any]:
    with _TEST_RUN_LOCK:
        payload = dict(_TEST_RUNS.get(test_run_id) or {})
        payload.update(updates)
        payload["updated_at"] = _now()
        snapshot = dict(payload)
        _TEST_RUNS[test_run_id] = snapshot
    path = _test_run_path(test_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def _test_run_history_item(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    return {
        "test_run_id": payload.get("test_run_id"),
        "label": payload.get("label"),
        "state": payload.get("state") or "running",
        "total_subtasks": payload.get("total_subtasks"),
        "completed": payload.get("completed") or 0,
        "failed": payload.get("failed") or 0,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "total_seconds": summary.get("total_seconds"),
        "average_seconds": summary.get("average_seconds"),
        "output_dir": payload.get("output_dir"),
        "error": payload.get("error"),
    }


def _make_event_from_image(image_path: Path) -> dict[str, Any]:
    """含图子任务：直接对随机图片跑本地 OCR，把识别文字当事件（不回匹配数据集）。"""
    outcome = run_ocr(image_path)
    ocr_text = "\n".join(getattr(outcome, "kept_texts", []) or []).strip()
    if not ocr_text:
        raise ValueError(f"OCR 未从图片识别到文字：{image_path}")
    return _normalize_hot_event(
        {
            "event_id": _safe_event_id(f"image_event_{_timestamp_id()}"),
            "domain": "general",
            "event_title": ocr_text[:48] or "图片事件",
            "event_summary": ocr_text,
            "target": "引导公众理解事件影响并进行理性讨论。",
            "is_synthetic": True,
            "opinion_variants": [ocr_text],
            "source_type": "image",
            "source_image": str(image_path),
            "image_recognition": {"method": "ocr_local", "text": ocr_text},
        }
    )


def _subtask_to_run_options(spec: "SubTaskSpec") -> "RunOptions":
    """把子任务配置映射成现有 RunOptions（严格校验，非法直接抛 400）。

    各子任务只暴露用户真正要配的字段：时间窗口、特殊支持节点（水军）、LLM 开关；
    其余运行参数（风险等级、选人上限、频率上限、可选平台等）走 RunOptions 默认值。
    """
    common = dict(
        campaign_window_hours=spec.campaign_window_hours,
        water_army_enabled=spec.water_army_enabled,
        water_army_count=spec.water_army_count,
        use_llm=spec.use_llm,
    )
    if spec.kind == "single_target":
        if not spec.target_user_id:
            raise HTTPException(status_code=400, detail="单人定向（single_target）必须指定 target_user_id。")
        return RunOptions(
            dispatch_mode="single_target",
            target_user_id=spec.target_user_id,
            single_target_mode=spec.single_target_mode,
            target_old_post_id=spec.target_old_post_id,
            platform_specs=[],  # type: ignore[list-item]
            dispatch_platform=None,
            max_selected_nodes=1,
            **common,
        )
    if spec.kind == "multi_user":
        dispatch_platform = spec.platform_specs[0].platform_key if spec.platform_specs else None
        requested = spec.platform_specs[0].requested_count if spec.platform_specs else 1
        return RunOptions(
            dispatch_mode="multi_user",
            dispatch_platform=dispatch_platform,
            platform_specs=[],  # type: ignore[list-item]
            max_selected_nodes=int(requested),
            **common,
        )
    # cross_platform
    if not spec.platform_specs:
        raise HTTPException(status_code=400, detail="跨平台（cross_platform）必须至少指定一个 platform_specs。")
    max_count = max((p.requested_count for p in spec.platform_specs), default=5)
    return RunOptions(
        dispatch_mode="cross_platform",
        platform_specs=list(spec.platform_specs),
        max_selected_nodes=max_count,
        **common,
    )


def _run_test_run_job(*, test_run_id: str, request: "TestRunRequest") -> None:
    hot_events = load_hot_events(DEFAULT_EVENTS_PATH)
    image_dir = PROJECT_ROOT / "eval" / "image"
    image_paths = sorted(image_dir.glob("*.png"))
    base_dir = _test_run_dir(test_run_id)

    subtask_results: list[dict[str, Any]] = []
    for si, spec in enumerate(request.subtasks):
        options = _subtask_to_run_options(spec)
        # 抽事件：含图则随机抽一张图 OCR 成事件；否则从库里随机抽。
        events: list[dict[str, Any]] = []
        image_failed = False
        for _ in range(int(spec.event_count)):
            if spec.with_image and image_paths:
                try:
                    events.append(_make_event_from_image(random.choice(image_paths)))
                    continue
                except Exception as exc:
                    image_failed = True
                    subtask_results.append(
                        {
                            "index": si,
                            "kind": spec.kind,
                            "ok": False,
                            "error": f"图片事件构建失败：{type(exc).__name__}: {exc}",
                            "stage_times_list": [],
                            "per_event": [],
                            "subtask_summary": None,
                        }
                    )
                    break
            if not hot_events:
                break
            events.append(copy.deepcopy(random.choice(hot_events)))

        if not events:
            if not image_failed:
                subtask_results.append(
                    {
                        "index": si,
                        "kind": spec.kind,
                        "ok": False,
                        "error": "无可用事件（库为空且无图片）",
                        "stage_times_list": [],
                        "per_event": [],
                        "subtask_summary": None,
                    }
                )
            _update_test_run(test_run_id, completed=si + 1, subtasks=list(subtask_results))
            continue

        sub_output = base_dir / "output" / f"sub_{si:02d}"
        sub_trace = base_dir / "trace" / f"sub_{si:02d}"
        sub_output.mkdir(parents=True, exist_ok=True)
        sub_trace.mkdir(parents=True, exist_ok=True)

        per_event: list[dict[str, Any]] = []
        stage_times_list: list[dict[str, Any]] = []
        # 每事件的真实墙钟耗时（event_total），与 per_event 卡片保持一致；
        # 图表与总体汇总都复用这一份，避免"阶段之和"被当成计时导致不一致。
        elapsed_list: list[float] = []
        sub_error: str | None = None
        for ei, ev in enumerate(events):
            ev_out = sub_output / f"ev_{ei:02d}"
            ev_trace = sub_trace / f"ev_{ei:02d}"
            ev_out.mkdir(parents=True, exist_ok=True)
            ev_trace.mkdir(parents=True, exist_ok=True)
            try:
                if spec.kind == "cross_platform":
                    output_path, payload, stage_times = _run_hot_event_multiplatform_evaluation(
                        workspace_root=PROJECT_ROOT,
                        output_dir=ev_out,
                        trace_dir=ev_trace,
                        hot_event=ev,
                        platform_specs=list(options.platform_specs),
                        profile_limit=options.profile_limit,
                        max_selected_nodes=options.max_selected_nodes,
                        risk_level=options.risk_level,
                        campaign_window_hours=options.campaign_window_hours,
                        max_frequency_per_day=options.max_frequency_per_day,
                        allowed_platforms=options.allowed_platforms,
                        use_llm=options.use_llm,
                        water_army_enabled=options.water_army_enabled,
                        water_army_count=options.water_army_count,
                    )
                else:
                    output_path, payload, stage_times = _run_hot_event_evaluation(
                        workspace_root=PROJECT_ROOT,
                        output_dir=ev_out,
                        trace_dir=ev_trace,
                        hot_event=ev,
                        profile_limit=options.profile_limit,
                        max_selected_nodes=options.max_selected_nodes,
                        risk_level=options.risk_level,
                        campaign_window_hours=options.campaign_window_hours,
                        max_frequency_per_day=options.max_frequency_per_day,
                        allowed_platforms=options.allowed_platforms,
                        use_llm=options.use_llm,
                        target_user_id=options.target_user_id,
                        dispatch_platform=options.dispatch_platform if options.dispatch_mode == "multi_user" else None,
                        single_target_mode=options.single_target_mode,
                        water_army_enabled=options.water_army_enabled,
                        water_army_count=options.water_army_count,
                    )
                # 统一压平成 dict[str, float]，用于聚合与前端柱状图：
                #  - 只保留「真实子阶段」（事件解析 / 特征构建 / 选人 / 内容生成 …），
                #    **剔除汇总项** event_total（墙钟合计）与 pipeline_total（子阶段之和），
                #    否则柱状图会把"合计"当成一条阶段，使各阶段之和看起来翻倍。
                #  - 跨平台：substage_totals 是各平台 per-stage 之和，除以平台数得到单平台代表值，
                #    使各阶段之和≈墙钟，不会被并行放大成 N 倍。
                # 耗时直接取管线自己测的 event_total（墙钟），是"这件事花了多久"的真相；
                # 只有这样 total_seconds / 各阶段之和才与真实耗时一致，不会被并行或汇总项放大。
                if spec.kind == "cross_platform" and isinstance(stage_times, dict):
                    substage = stage_times.get("substage_totals") or {}
                    per_platform = stage_times.get("per_platform") or []
                    n_platforms = max(len(per_platform), 1)
                    flat_stage_times = {
                        k: float(v) / n_platforms
                        for k, v in substage.items()
                        if k not in ("event_total", "pipeline_total")
                    }
                else:
                    flat_stage_times = {
                        k: float(v)
                        for k, v in (stage_times or {}).items()
                        if k not in ("event_total", "pipeline_total")
                    }
                event_total_val = (
                    float(stage_times.get("event_total", 0.0))
                    if isinstance(stage_times, dict) and stage_times.get("event_total")
                    else 0.0
                )
                elapsed_seconds = event_total_val or float(sum(flat_stage_times.values()))
                summary = _result_summary(
                    output_path=output_path,
                    payload=payload,
                    dispatch_mode=options.dispatch_mode,
                    stage_times=flat_stage_times,
                )
                stage_times_list.append(flat_stage_times)
                elapsed_list.append(elapsed_seconds)
                per_event.append(
                    {
                        "event_id": str(ev.get("event_id")),
                        "event_title": ev.get("event_title"),
                        "stage_times": flat_stage_times,
                        "elapsed_seconds": elapsed_seconds,
                        "summary": summary,
                    }
                )
            except Exception as exc:
                sub_error = f"{type(exc).__name__}: {exc}"
                stage_times_list.append({})
                elapsed_list.append(0.0)
                per_event.append(
                    {
                        "event_id": str(ev.get("event_id")),
                        "stage_times": {},
                        "summary": None,
                        "error": sub_error,
                    }
                )

        # 复用现有压测汇总：对同一子任务的各事件 stage_times 求平均。
        # 注意：elapsed_seconds 必须用真实墙钟 event_total（与 per_event 卡片一致），
        # 不能用 sum(stage_times) —— 那样会把"剔除了 event_total 的阶段明细"误当计时。
        iterations = [
            {
                "ok": bool(st),
                "iteration": i,
                "stage_times": st,
                "elapsed_seconds": float(elapsed_list[i]),
            }
            for i, st in enumerate(stage_times_list)
        ]
        subtask_summary = _aggregate_stage_timings(iterations=len(iterations), results=iterations)
        subtask_results.append(
            {
                "index": si,
                "kind": spec.kind,
                "single_target_mode": spec.single_target_mode,
                "with_image": spec.with_image,
                "target_user_id": spec.target_user_id,
                "platform_specs": [p.model_dump() for p in spec.platform_specs],
                "event_count": len(events),
                "ok": sub_error is None,
                "error": sub_error,
                "stage_times_list": stage_times_list,
                "event_elapsed": list(elapsed_list),
                "per_event": per_event,
                "subtask_summary": subtask_summary,
            }
        )
        _update_test_run(
            test_run_id,
            completed=si + 1,
            subtasks=list(subtask_results),
        )

    # 总体汇总：跨所有子任务、所有事件。
    flat: list[dict[str, Any]] = []
    for st in subtask_results:
        for stt, el in zip(st.get("stage_times_list", []), st.get("event_elapsed", [])):
            flat.append(
                {"ok": bool(stt), "iteration": len(flat), "stage_times": stt, "elapsed_seconds": float(el)}
            )
    overall_summary = _aggregate_stage_timings(iterations=len(flat), results=flat)
    succeeded = sum(1 for st in subtask_results if st.get("ok"))
    _update_test_run(
        test_run_id,
        state="completed" if succeeded > 0 else "failed",
        failed=sum(1 for st in subtask_results if not st.get("ok")),
        subtasks=list(subtask_results),
        summary=overall_summary,
        error=None if succeeded > 0 else "所有子任务都失败了",
    )


def _test_run_job_wrapper(test_run_id: str, request: "TestRunRequest") -> None:
    try:
        _update_test_run(test_run_id, state="running")
        _run_test_run_job(test_run_id=test_run_id, request=request)
    except Exception as exc:  # pragma: no cover - 后台任务边界
        _update_test_run(test_run_id, state="failed", error=f"{type(exc).__name__}: {exc}")
