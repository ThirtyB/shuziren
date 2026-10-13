# 图像输入评测能力设计规格

日期：2026-07-06

## 背景

当前数字人项目的评测入口只接受文本事件数据，默认从 `eval/hot_event_opinion_variants.json` 读取 hot_event JSON。现有流水线已经形成稳定链路：

```text
hot_event JSON
  -> hot_event_to_pipeline_payload()
  -> event_parser
  -> feature_builder
  -> scorer
  -> selector
  -> strategy_generator
  -> structured JSON output
```

现在 `eval/image/` 下已经放入新闻播报风格的图像数据。新需求是让 eval 能正确接受图像输入，识别图像中的事件，并像文本输入一样继续执行后续策略生成。

## 目标

1. 支持单张图像和图像目录作为 eval 输入。
2. 从图像中识别事件标题、摘要、领域、关键词、传播目标和观点变体。
3. 优先使用支持视觉输入的 LLM，例如当前配置中的 `qwen3.7-plus`。
4. 视觉 LLM 失败时提供 OCR/规则兜底路径。
5. 识别出的图像事件优先匹配现有 hot_event 数据集，匹配不到时生成新的 `image_event_xxx` 事件。
6. 复用现有文本评测流水线，不把图像复杂性扩散到 scorer、selector、strategy_generator。
7. 输出可诊断 trace，便于检查图像被识别成什么事件、是否匹配到已有 hot_event、后续流水线如何处理。

## 非目标

1. 第一版不建设图片标注平台。
2. 第一版不引入图像 embedding 检索。
3. 第一版不把图片本身传入后续 scorer、selector 或 strategy_generator。
4. OCR 是兜底增强，不要求第一版必须离线可用。
5. 不改变现有文本 JSON eval 的默认运行方式。

## 推荐方案

采用“图像归一化层 + 匹配层 + trace”的方案。

新增能力只放在 eval 输入归一化层，不改主策略链路。图像先被转换成标准 hot_event dict，再交给现有 `hot_event_to_pipeline_payload()` 和后续流水线。

```text
eval/image/*.png
  -> image_event_loader
      -> vision LLM event recognition
      -> OCR/rule fallback
      -> hot_event dataset matching
      -> standard hot_event dict
  -> hot_event_to_pipeline_payload()
  -> existing strategy pipeline
```

## 模块设计

新增模块建议命名为：

```text
src/influence_strategy/image_event_loader.py
```

### ImageEventRecognizer

负责单张图片识别。

输入：

- 图片路径
- workspace root
- 可选 vision client
- 是否启用 LLM

输出：

- `event_title`
- `event_summary`
- `domain`
- `keywords`
- `target`
- `opinion_variants`
- `confidence`
- `source_image`
- `method`
- `warnings`

主路径使用视觉 LLM。当前项目配置中 `.env` 包含：

```env
MODEL_NAME=qwen3.7-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

该模型支持视觉输入，但现有 `OpenAICompatibleLLMClient.generate_json()` 只发送纯文本 messages。因此需要新增图像 JSON 调用方法，支持 OpenAI-compatible 的 image input 格式。

### HotEventMatcher

负责把图像识别结果匹配到现有 `eval/hot_event_opinion_variants.json`。

匹配材料：

- 图像识别出的标题
- 图像识别出的摘要
- 图像识别出的关键词
- 现有 hot_event 的 `event_title`
- 现有 hot_event 的 `event_summary`
- 现有 hot_event 的 `target`
- 现有 hot_event 的 `opinion_variants`

推荐阈值：

```text
match_threshold = 0.72
```

匹配成功时，复用已有 hot_event 的 `event_id`、`domain`、`target` 和 `opinion_variants`，同时保留图像识别元信息。

匹配失败时，生成 `image_event_001`、`image_event_002` 这类事件 ID，并使用图像识别出的字段。

### load_image_events()

负责把单图或目录批量转换为 hot_event dict 列表。

行为：

- `--image` 只处理单张图片。
- `--image-dir` 按文件名排序读取目录中的 `.png`、`.jpg`、`.jpeg`、`.webp`。
- `--event-limit` 在图像模式下限制处理图片数量。
- 返回值可直接传给现有评测执行函数。

## CLI 设计

保留当前文本入口：

```powershell
python run_eval.py
python run_eval.py --input eval/hot_event_opinion_variants.json
```

新增图像入口：

```powershell
python run_eval.py --image eval/image/4.png
python run_eval.py --image-dir eval/image
python run_eval.py --image-dir eval/image --event-limit 4
python run_eval.py --image eval/image/4.png --disable-llm
```

参数说明：

- `--image`：单张图片输入，适合调试。
- `--image-dir`：目录批量输入，适合 eval 数据集。
- `--input`：图像模式下作为可匹配的参考 hot_event 数据集。
- `--event-id`：图像模式下筛选匹配后的事件或生成出的 image event。
- `--event-limit`：图像模式下限制图片数量。
- `--disable-llm`：禁用策略链路 LLM，同时禁用视觉 LLM；此时只能走 OCR/规则兜底。

建议新增可选环境变量：

```env
VISION_LLM_API_KEY=
VISION_LLM_BASE_URL=
VISION_LLM_MODEL=
OCR_ENABLED=true
```

如果未配置 `VISION_*`，默认复用现有 LLM/DashScope 配置。

## 标准事件结构

图像识别后输出 hot_event-like dict：

```json
{
  "event_id": "hot_event_005",
  "domain": "technology",
  "event_title": "高端AI芯片供应与出口限制牵动产业链",
  "event_summary": "从图片识别出的事件摘要",
  "target": "说明事件对产业链、企业和公众讨论的影响",
  "is_synthetic": false,
  "source_type": "image",
  "source_image": "eval/image/4.png",
  "image_recognition": {
    "method": "vision_llm",
    "confidence": 0.91,
    "matched_event_id": "hot_event_005",
    "match_score": 0.87,
    "match_threshold": 0.72,
    "fallback_used": false,
    "warnings": []
  },
  "opinion_variants": []
}
```

## 数据流

1. `run_eval.py` 判断是否传入 `--image` 或 `--image-dir`。
2. 图像模式下读取 `--input` 指向的文本 hot_event 数据集作为匹配参考。
3. 对每张图片调用 `ImageEventRecognizer`。
4. 调用 `HotEventMatcher` 尝试匹配现有 hot_event。
5. 匹配成功时复用现有事件主体字段；匹配失败时生成新的 image event。
6. 生成的 hot_event dict 进入现有 `_run_hot_event_evaluation()`。
7. 最终输出路径和当前文本 eval 保持一致：

```text
eval/output/<event_id>_strategy_output.json
```

## 错误处理

错误按单图隔离，尽量不让整批 eval 因单张图片失败而中断。

- `vision_llm_failed`：视觉模型调用失败，尝试 OCR/规则兜底。
- `ocr_unavailable`：本地没有 OCR 能力，返回低置信度 fallback 或清晰错误。
- `low_confidence`：识别置信度或匹配分数过低，生成 `image_event_xxx`，不强行匹配。
- `invalid_json_response`：视觉 LLM 未返回合法 JSON，尝试从文本中抽取 JSON；仍失败则兜底。
- `unsupported_image_type`：跳过不支持的文件类型。
- `empty_image_dir`：目录为空时给出清晰错误。

## Trace 设计

延续现有 trace 目录：

```text
tests/pipeline_step_outputs/<event_id>/
  00_image_input.json
  00_image_recognition.json
  00_hot_event_input.json
  01_pipeline_payload.json
  02_event_parser_output.json
  03_feature_builder_output.json
  04_scorer_output.json
  05_selector_output.json
  06_strategy_generator_output.json
  07_final_output.json
```

`00_image_recognition.json` 示例：

```json
{
  "source_image": "eval/image/4.png",
  "method": "vision_llm",
  "fallback_used": false,
  "recognized_event": {
    "event_title": "高端AI芯片供应与出口限制牵动产业链",
    "domain": "technology",
    "keywords": ["AI芯片", "出口管制", "替代方案", "本土化供应"]
  },
  "matched_event": {
    "event_id": "hot_event_005",
    "match_score": 0.87,
    "threshold": 0.72
  },
  "warnings": []
}
```

## 测试计划

### 单元测试：图像识别归一化

使用 fake vision client 返回固定 JSON，断言识别结果能生成标准 hot_event dict。

### 单元测试：匹配逻辑

给定图像识别文本和现有 hot_event 列表，断言：

- 芯片供应图片能匹配到 `hot_event_005`。
- 低相似度样例生成 `image_event_001`。
- 匹配结果包含 `match_score` 和 `match_threshold`。

### CLI 测试

覆盖：

- `--image`
- `--image-dir`
- `--event-limit`
- `--disable-llm`

### 端到端轻量测试

使用 fake image recognizer 生成事件，再跑现有最小 workspace，验证最终仍输出：

```text
action_schema_v5_five_dimensions_minimal
```

## 验收标准

1. `python run_eval.py --image eval/image/4.png` 能识别芯片事件并生成策略输出。
2. `eval/image/4.png` 优先匹配到 `hot_event_005`。
3. `python run_eval.py --image-dir eval/image --event-limit 4` 能批量处理图片。
4. 文本 eval 的默认命令行为保持不变。
5. 每个图像事件的 trace 中能看到识别结果、匹配分数、兜底状态和最终 hot_event 输入。
6. 视觉模型不可用时，错误信息清晰，且不影响其他图片继续处理。
7. 新增单元测试和轻量端到端测试通过。

## Implementation Artifacts

- Design spec: `docx/image-eval-input-design.md`
- Implementation plan: `docx/image-eval-input-plan.md`
- Image loader module: `src/influence_strategy/image_event_loader.py`
- CLI entry: `run_eval.py --image eval/image/4.png`
- Batch CLI entry: `run_eval.py --image-dir eval/image --event-limit 4`
