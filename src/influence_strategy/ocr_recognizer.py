"""本地 OCR 图像事件识别 —— 不依赖任何大模型，只读取图片上的文字。

设计背景
--------
原先的图像识别调用视觉大模型（``generate_json_with_image``），直接把图片理解成
事件标题、摘要、领域等结构化字段。改用本地部署后，模型只允许读取**文字**，
因此链路变成：

    图片 ──► OCR 提取文字块 ──► 规则重组事件字段

OCR 只负责"看见字"，不理解语义。把碎片文字重组成「标题 / 摘要 / 关键词 / 领域」
这一步由本模块的规则完成 —— 这是纯文本启发式，没有任何模型参与。

为什么不用现有的 ``RuleBasedEventParser``
-----------------------------------------
它的 ``EVENT_TYPE_RULES`` 是为阅读/亲子场景定制的分类（亲子共读、英语启蒙、
阅读打卡……），拿去解析「红海航运」「AI 治理」这类新闻截图会全部落到
``general_influence_event``。所以这里单独维护一套面向新闻截图的领域词典。

速度
----
RapidOCR（PP-OCR ONNX）在 CPU 上处理一张 1672×941 的截图约 3~5 秒（冷启动含模型
加载时更高），瓶颈是文本行数量而非图片尺寸 —— 每行都要过一次识别网络。缩放收益
有限，但能把超大图（手机长截图）拉回可控范围。

已做的两个提速点：
  * 关闭角度分类器（``use_cls=False``）：本系统只处理横排新闻截图，方向分类是多余的
    逐行开销，关掉后单图稳定快约 0.1~0.2s，对横排文字精度无影响。
  * Web 服务启动时预加载引擎（见 ``web_app._warmup_local_models``），把模型加载的
    一次性成本从「首个图片请求」挪到「服务启动」，使首请求落回 ~4.5s、后续热路径
    稳定在 2.5~3s。
图片识别是一次性请求，Web 端走异步任务队列，这个耗时可以接受。
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

#: 送进 OCR 前把长边压到该值。原图 1672×941 压缩后识别质量无明显下降。
MAX_IMAGE_SIDE = 1280

#: 文本块数量下限：低于它说明图片里几乎没有可用文字。
MIN_BLOCKS = 3

#: 判为"长句"的字符数下限。长句承载事件陈述，短块多为版面标签或实体名。
LONG_BLOCK_MIN_CHARS = 12

#: 判为"短实体"的字符数上限（下限 2 字，单字基本是噪声）。
SHORT_BLOCK_MAX_CHARS = 10

#: 版面噪声：电视台标、栏目名、状态栏。这些出现在图上但不是事件内容。
NOISE_EXACT = frozenset(
    {
        "newslive",
        "livenews",
        "newsupdate",
        "新闻播报",
        "最新动态",
        "最新观察",
        "国际新闻",
        "国际观察",
        "新闻",
        "播报",
        "快讯",
        "直播",
        "回放",
    }
)

#: 允许保留的纯英文短语（专业缩写本身是有信息的实体）。
ASCII_ALLOWLIST = frozenset({"ai", "gpt", "cpu", "gpu", "ceo", "cfo", "5g", "6g", "vr", "ar"})

#: 时间戳 / 数字型状态栏。
_TIME_PATTERN = re.compile(r"^\d{1,2}[:：]\d{2}([:：]\d{2})?$")
_PURE_NUMBER = re.compile(r"^[\d\s.,%％+\-—/]+$")
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

#: 领域锚文本（纯语义相似度）。
#:
#: 取代原先的关键词命中计数（``DOMAIN_RULES``）。每个领域用一段自然语言描述其语义
#: 中心，解析时由 ``SemanticClassifier`` 把事件文本与各锚文本编码到同一语义空间，
#: 取余弦相似度最高的领域；编码器不可用时回落 ``general``。键与视觉识别原先产出的
#: domain 枚举保持一致，避免下游字段口径变化。
DOMAIN_ANCHORS: dict[str, str] = {
    "military": "军事、战争、护航、军舰、导弹、冲突、海域、安全形势、军演、兵力。",
    "technology": "人工智能、AI、芯片、算力、算法、大模型、技术、半导体、生成式。",
    "politics": "政府、治理、监管、政策、法规、法治、协商、磋商、规则、合规。",
    "finance": "金融、银行、利率、股市、汇率、投资、债券、资本。",
    "energy": "能源、石油、油价、天然气、电力、燃油、煤炭、新能源。",
    "international_relations": "国际、多国、跨境、谈判、关税、外交、全球、协定。",
    "climate": "气候、碳中和、碳排放、环保、减排、绿色能源。",
    "cybersecurity": "网络安全、数据安全、黑客、攻击、隐私、加密、泄露。",
    "public_policy": "公共政策、民生、医疗、教育、社保、就业、住房。",
    # 兜底领域：事件文本在语义空间里最贴近它时，回落到通用领域。
    "general": "一般性的新闻与公共话题，无法归入上述特定领域。",
}

#: 领域分类器懒加载缓存（进程内只构建一次）。
_DOMAIN_CLASSIFIER: Any = None
_DOMAIN_CLASSIFIER_READY = False


def _get_domain_classifier(workspace_root: str | Path | None = None) -> Any:
    """构建（并缓存）领域语义分类器；本地编码器不可用时返回 ``None``。"""

    global _DOMAIN_CLASSIFIER, _DOMAIN_CLASSIFIER_READY
    if _DOMAIN_CLASSIFIER_READY:
        return _DOMAIN_CLASSIFIER

    encoder = None
    try:
        from pathlib import Path as _P
        from .local_models import load_local_models

        root = workspace_root or _P(__file__).resolve().parents[2]
        models = load_local_models(root)
        encoder = models.encoder if models else None
    except Exception:  # noqa: BLE001 - 取不到编码器不应中断解析
        encoder = None

    from .local_semantic import SemanticClassifier

    _DOMAIN_CLASSIFIER = SemanticClassifier.build(encoder, DOMAIN_ANCHORS)
    _DOMAIN_CLASSIFIER_READY = True
    return _DOMAIN_CLASSIFIER


@dataclass(slots=True)
class OcrBlock:
    """一行 OCR 结果。``score`` 是该行的识别置信度（0~1）。"""

    text: str
    score: float


@dataclass(slots=True)
class OcrOutcome:
    """一次 OCR 的完整结果，含被过滤掉的噪声，便于排查。"""

    blocks: list[OcrBlock]
    kept_texts: list[str]
    dropped_texts: list[str]
    mean_score: float

    @property
    def has_text(self) -> bool:
        return len(self.blocks) >= MIN_BLOCKS


_engine: Any = None
_engine_lock = threading.Lock()


def _load_engine() -> Any:
    """懒加载 RapidOCR 单例。

    RapidOCR 初始化会加载 3 个 ONNX 模型（检测/方向/识别），约 0.9s、几十 MB 内存，
    因此进程内只建一次。Web 服务是多线程的，用锁保护。

    默认关闭角度分类器（``use_cls=False``）：本系统只处理横排新闻截图，方向分类是
    多余的逐行开销；关掉后每个文本行少跑一次分类网络，单图识别稳定快约 0.1~0.2s，
    对横排文字的识别精度无影响。
    """

    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR(use_cls=False)
            logger.info("RapidOCR 引擎已加载（use_cls=False）")
    return _engine


def is_ocr_available() -> bool:
    """OCR 依赖是否就绪（``rapidocr_onnxruntime`` + onnxruntime）。"""

    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def run_ocr(image_path: str | Path) -> OcrOutcome:
    """对图片做 OCR，返回保序的文字块。

    失败时抛 ``RuntimeError``，由调用方决定如何降级 —— 本函数不吞异常，
    避免"看起来识别成功但其实是空结果"。
    """

    path = Path(image_path)
    if not path.exists():
        raise RuntimeError(f"图片不存在: {path}")

    engine = _load_engine()
    prepared = _prepare_image(path)

    try:
        raw, _elapse = engine(prepared)
    finally:
        if prepared != str(path):
            Path(prepared).unlink(missing_ok=True)

    blocks: list[OcrBlock] = []
    for item in raw or []:
        # RapidOCR 返回 [box, text, score]
        if len(item) < 3:
            continue
        text = str(item[1] or "").strip()
        if not text:
            continue
        try:
            score = float(item[2])
        except (TypeError, ValueError):
            score = 0.0
        blocks.append(OcrBlock(text=text, score=score))

    kept = [block for block in blocks if not is_noise(block.text)]
    dropped = [block.text for block in blocks if is_noise(block.text)]
    mean_score = sum(block.score for block in kept) / len(kept) if kept else 0.0
    return OcrOutcome(
        blocks=blocks,
        kept_texts=[block.text for block in kept],
        dropped_texts=dropped,
        mean_score=mean_score,
    )


def _prepare_image(path: Path) -> str:
    """必要时缩放图片，返回可直接交给 OCR 的路径。"""

    try:
        from PIL import Image
    except Exception:
        return str(path)

    try:
        with Image.open(path) as image:
            width, height = image.size
            longest = max(width, height)
            if longest <= MAX_IMAGE_SIDE:
                return str(path)
            scale = MAX_IMAGE_SIDE / float(longest)
            resized = image.convert("RGB").resize(
                (max(int(width * scale), 1), max(int(height * scale), 1)),
                Image.LANCZOS,
            )
            tmp_path = path.with_name(f".ocr_tmp_{path.stem}.png")
            resized.save(tmp_path)
            return str(tmp_path)
    except Exception as exc:
        logger.warning("图片预处理失败，按原图识别: %s", exc)
        return str(path)


def is_noise(text: str) -> bool:
    """判断一行文字是否是版面噪声（台标、时间、纯数字、单字）。"""

    value = text.strip()
    if len(value) < 2:
        return True
    if value.lower() in NOISE_EXACT:
        return True
    if _TIME_PATTERN.match(value):
        return True
    if _PURE_NUMBER.match(value):
        return True
    if _PUNCT_ONLY.match(value):
        return True
    # 纯 ASCII 且不在白名单：多为台标缩写或装饰性英文，保留价值低于中文实体。
    if value.isascii() and len(value) <= 8 and value.lower() not in ASCII_ALLOWLIST:
        return True
    return False


def infer_domain(texts: Iterable[str], workspace_root: str | Path | None = None) -> str:
    """按语义相似度推断领域；本地编码器不可用时回落 ``general``。

    原先是关键词命中计数（``DOMAIN_RULES``），现改为 ``SemanticClassifier``：
    把每个领域用一段锚文本表示，与事件文本在语义空间里取余弦相似度最高的领域。
    """

    joined = " ".join(texts)
    clf = _get_domain_classifier(workspace_root)
    if clf is None:
        return "general"
    label, _ = clf.classify(joined, default_label="general", margin=0.03)
    return label


def build_event_fields(texts: Sequence[str]) -> dict[str, Any]:
    """把 OCR 文字块重组成事件字段（纯规则，无模型）。

    规则依据实测截图：
    * **短块**（2~10 字、无标点）几乎都是实体或栏目标签 ——「红海」「苏伊士运河」
      「护航动态」，直接当关键词；
    * **长块**（>=12 字）是事件陈述 —— 取最长的作为标题，其余拼成摘要与观点变体。
    """

    cleaned = [text.strip() for text in texts if text and not is_noise(text)]
    if not cleaned:
        raise ValueError("没有可用的 OCR 文本")

    long_blocks = [text for text in cleaned if len(text) >= LONG_BLOCK_MIN_CHARS]
    short_blocks = [
        text
        for text in cleaned
        if len(text) <= SHORT_BLOCK_MAX_CHARS and len(text) >= 2
    ]

    # 标题：最长的长句。没有长句时退化为最长的短块。
    title_pool = long_blocks or short_blocks
    title = max(title_pool, key=len) if title_pool else cleaned[0]

    # 摘要：只拼**完整句子**（不以逗号/顿号结尾）。
    # 版面里常有被框线切开的长句，前半截以"，"结尾、"后半截"另起一块，
    # 直接首尾相接会得到"…与企业提升链路承压…"这种断裂文本。
    summary_parts: list[str] = []
    for text in long_blocks:
        if text == title or not _is_complete_sentence(text):
            continue
        if summary_parts and sum(len(item) for item in summary_parts) + len(text) > 220:
            break
        summary_parts.append(text.rstrip("。."))
    summary = "。".join(summary_parts) if summary_parts else title

    # 关键词：短块（实体）+ 长块里的四字以上术语，去重保序。
    keywords: list[str] = []
    for text in short_blocks + long_blocks:
        for candidate in ([text] if text in short_blocks else _split_phrases(text)):
            if len(candidate) < 2 or candidate in keywords:
                continue
            keywords.append(candidate)
        if len(keywords) >= 8:
            break
    keywords = keywords[:8]

    # 观点变体：除标题外的长句，最多 2 条；不足时用摘要兜底。
    variants = [text for text in long_blocks if text != title][:2]
    if not variants:
        variants = [summary]

    domain = infer_domain(cleaned)

    return {
        "event_title": title,
        "event_summary": summary,
        "domain": domain,
        "keywords": keywords,
        "target": f"引导公众理解「{title}」的核心事实与影响，并进行理性讨论。",
        "opinion_variants": variants,
    }


def _is_complete_sentence(text: str) -> bool:
    """判断文本块是否构成完整句子（未以逗号、顿号、冒号等连接符结尾）。

    被版面框线切断的句子通常以"，"结尾，接到摘要里会造成句子断裂。
    """

    return not text.rstrip().endswith(("，", ",", "、", "；", ";", "：", ":"))


def _split_phrases(text: str) -> list[str]:
    """把长句按中文标点切成短语，用于补充关键词。"""

    parts = re.split(r"[，。；：、,;:!?！？\s]+", text)
    return [part for part in parts if 4 <= len(part) <= 12]


def compute_confidence(outcome: OcrOutcome, fields: dict[str, Any]) -> float:
    """置信度 = OCR 平均分行置信度 × 文本充分度折扣。

    文字太少（只有零星几个词）时即便单行置信度高，整体事件描述也必然残缺，
    因此按长句数量打折，让下游能识别出"这张图信息不足"。
    """

    base = outcome.mean_score if outcome.mean_score > 0 else 0.5
    long_count = sum(1 for text in outcome.kept_texts if len(text) >= LONG_BLOCK_MIN_CHARS)
    if long_count >= 3:
        factor = 1.0
    elif long_count == 2:
        factor = 0.92
    elif long_count == 1:
        factor = 0.82
    else:
        factor = 0.62
    confidence = base * factor
    if len(str(fields.get("event_summary") or "")) < 20:
        confidence *= 0.85
    return max(0.05, min(0.98, confidence))
