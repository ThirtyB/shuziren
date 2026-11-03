# abc_reading 影响力事件分发策略原型

> 围绕一条清晰、可解释的流水线，将一条影响力事件转化为结构化的数字人分发策略：
> 解析事件 → 构建候选特征 → 评分 → 选人并分配角色 → 生成可执行的分发方案。

本项目当前目标是**影响力事件分发策略生成**（而非商品推荐），重点放在规则可解释、LLM 作为增强层、以及结构化输出上。

---

## 1. 项目简介

**核心目标**

1. 将自然语言事件描述解析为可计算的结构化事件对象。
2. 基于用户画像、互动图谱与邻居特征构建候选节点特征。
3. 对候选节点进行主题相关性、扩散能力与风险控制评分。
4. 选择主选数字人并分配角色（支持一人多身份的兼任分配）。
5. 输出可执行的五维调度策略（角色 / 平台 / 时间 / 目标对象 / 频率）。

**当前已具备的能力**

- 规则驱动的基础流水线，LLM 仅在关键节点做语义增强。
- 三种分发模式：单人定向、多人协同、跨平台协同。
- **多身份分配**：选人后按门槛追加兼任角色，并保证核心角色不被自指动作污染。
- 图像事件输入：可选接入视觉模型识别事件内容。
- 200 条热点事件批量评测与中间结果追踪。
- 轻量 Web 控制台（单事件 / 图片 / 批量评测）。

---

## 2. 流水线总览

```text
event_input
   -> data_loader          读取 profile / interaction / derived graph
   -> event_parser        事件解析为结构化对象（可选 LLM 增强）
   -> feature_builder     规则召回 + 候选池 LLM 语义增强
   -> scorer              综合特征分、语义分、风险惩罚打分
   -> selector            角色化选人 + 多身份追加分配
   -> strategy_generator  生成时间 / 频率 / 平台 / 协同方案
   -> eval_hot_events     组装五维调度策略并写盘
```

`run_eval.py` 支持带 LLM 的热点评测链路；`main.py` 用于单事件基线；Web 控制台运行结果写入 `outputs/web_runs/`。

---

## 3. 环境准备

推荐使用 `conda`：

```bash
conda env create -f environment.yml
conda activate abc-reading-strategy
```

也可使用项目自带的依赖声明直接安装：

```bash
pip install -e ".[semantic,ocr]"
```

主要依赖：`python>=3.11`、`pydantic`、`networkx`、`pandas`、`scikit-learn`、
`fastapi`、`uvicorn`、`rapidocr_onnxruntime`（可选 OCR）等，详见 `pyproject.toml`。

---

## 4. 快速上手

### 运行测试

```bash
conda run -n abc-reading-strategy python -m pytest
```

### 单事件基线

```bash
python main.py --event-text "围绕亲子阅读与英语启蒙做一次传播活动，提升讨论度并控制刷屏风险"
# 或读取事件文件
python main.py --event-file your_event.json
```

结果写入 `outputs/strategy/` 下的 `.json` 与 `.md`。

### 热点批量评测

```bash
# 规则模式（不调用 LLM）
python run_eval.py --disable-llm

# 启用 LLM 增强
python run_eval.py

# 只跑指定事件 / 限制条数
python run_eval.py --event-id hot_event_001
python run_eval.py --event-limit 3
```

常用参数：`--profile-limit`、`--risk-level`、`--max-selected-nodes`、
`--max-frequency-per-day`、`--campaign-window-hours`、`--allowed-platforms`、
`--output-dir`、`--trace-dir`。

### Web 控制台

后端：

```bash
conda activate abc-reading-strategy
uvicorn --app-dir src influence_strategy.web_app:app --reload --host 127.0.0.1 --port 8000
```

前端：

```bash
cd web/frontend
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。200 条批量评测默认读取 `eval/hot_event_opinion_variants_200.json`。

---

## 5. 选人策略与多身份分配

角色分为四类：**首发节点**、**扩散节点**、**互动节点**、**支持节点**。
选人时按重要性依次分配（首发 → 扩散 → 互动 → 支持），后选角色只能从前一轮
的剩余候选中取，从而保证关键角色优先获得最匹配的节点。

**准入门槛**：每个角色的门槛取"该角色已选组内按同一排序键的最低分"。
因此该角色的既有成员天然满足，不存在"自己达不到自己门槛"的自相矛盾。

**多身份追加分配**（选人之后统一追加，不改变人数与主角色）：

- 对名单中每个节点，检查其是否达到其它角色的门槛，达到则追加为兼任身份。
- 首发节点**不参与**兼任分配：其配额恒 ≥ 1 且有兜底选人，不存在缺口。
- 当某事件支持节点配额为 0（小规模事件常见）时，由综合分最低的节点兼任支持，
  保证四类角色在结果中都有代表。
- 兼任只在"门槛达标"时发生，未达标不强行凑数。

**动作叠加与自指保护**：

- 兼任者会执行其承担的全部角色动作，每日频次按所承担角色**求和**。
- 扩散 / 二次扩散环节会跳过"转发自己的帖子"这一自指动作；该动作视为未发生，
  其下游依赖不受影响，避免悬空依赖。

---

## 6. 分发模式

| 模式 | 说明 |
| --- | --- |
| 单人定向 (`single_target`) | 锁定一名指定路人，数字人围绕其进行协同传播；不参与多身份分配。 |
| 多人协同 (`multi_user`) | 按角色配额从候选池挑选多名数字人，并追加兼任身份。 |
| 跨平台协同 (`cross_platform`) | 在多个平台上分别选人编排，自动识别跨平台重复数字人。 |

---

## 7. LLM 接入

启用 LLM 增强时，在工作目录创建 `.env`，字段参考 [.env.example](./.env.example)：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-llm-endpoint.example.com/v1
LLM_MODEL=your-model-name
```

也兼容 `DEEPSEEK_*` / `OPENAI_*` 系列变量名。图像事件可配置独立的
`VISION_LLM_*` 变量（未配置时回退到通用 LLM 设置）。

**LLM 主要用在以下环节**（前三个影响"选谁"，最后一个影响"怎么写"）：

1. `event_parser`：事件结构化增强（事件类型、目标受众、关键词、叙事框架等）。
2. `feature_builder`：对缩小后的候选池做语义相关性 / 受众契合度判断。
3. `selector`：对候选清单做重排与角色推荐。
4. 文案生成：为已选数字人生成发帖内容、互动提示与协同说明。

`data_loader` 与 `scorer` 不直接调用 LLM，但 `scorer` 会消费 `feature_builder`
返回的语义分数。

---

## 8. 输出结构

`run_eval.py` 产出贴近实施方案的结构，关键字段位于五维调度策略下：

| 字段 | 含义 |
| --- | --- |
| `目标对象.选取数字人id组` | 去重后的主选数字人 id 列表（即实际人数）。 |
| `目标对象.数字人分工` | 每个数字人的编号、主角色、职责、人设、平台，以及 `兼任角色` / `兼任节点角色`。 |
| `目标对象.兼任汇总` | 兼任人数、各角色准入门槛、兼任明细，用于解释"为什么这样分配"。 |
| `时间.数字人角色` | 每个数字人在时间轴上的主角色与兼任角色。 |
| `频率.数字人频率` | 每个数字人按所承担角色求和后的每日条数。 |
| `平台` / `风险` 等 | 平台分布、风险护栏与节奏建议。 |

每次运行还会在 `tests/pipeline_step_outputs/<event_id>/` 下写入逐阶段中间产物，
是分析"为什么选中这个节点"的主要调试入口。

---

## 9. 核心模块说明

| 模块 | 作用 |
| --- | --- |
| `data_loader` | 读取 profile / interaction / derived graph 数据。 |
| `event_parser` | 将事件解析为结构化事件对象。 |
| `feature_builder` | 规则召回 + LLM 语义增强，生成节点特征。 |
| `scorer` | 综合特征、语义与风险打分。 |
| `selector` | 角色化选人并追加多身份分配。 |
| `strategy_generator` | 生成时间 / 频率 / 平台 / 协同方案。 |
| `pipeline` | 串联整条流水线。 |
| `eval_hot_events` | 热点评测、输出整理与追踪写盘。 |
| `web_app` | FastAPI 后端，提供单事件 / 图片 / 批量评测接口。 |
| `llm_client` / `prompts` | LLM 调用封装与提示词模板。 |

---

## 10. 设计原则

- 先把可解释的基线流程跑通，再让 LLM 作为增强层叠加。
- 不让 LLM 直接扫描全量数据集，而是先用规则缩小候选范围再精排。
- 不让高粉丝量单独决定入选，主题相关性必须参与门控。
- 输出必须结构化，便于接入 API、前端或自动化流程。
- 多人场景的"兼任"只追加不覆盖，人数、主角色与配额始终可控。

---

## 11. 局限与后续

**当前局限**

- 数据集与真实热点事件在画像层面并非完全贴合，部分泛画像高影响力账号可能进入候选。
- 暂未引入向量检索 / embedding 召回层，语义召回依赖 LLM 在规则候选池上的判断。
- 单人定向模式不启用多身份分配。

**后续方向**

1. 强化对"主题不相关高影响力节点"的抑制规则。
2. 引入 embedding 检索层，先做语义召回再做 LLM 精排。
3. 为不同事件类型设计专用提示词模板。
4. 增加缓存机制，降低大规模评测时的 LLM 调用成本。

---

## 12. 相关文档

- [2026-06-02_热点评测流水线改造说明.md](./2026-06-02_热点评测流水线改造说明.md)
- [改进2026-05-31.md](./改进2026-05-31.md)
- [影响力事件分发策略初步方案.md](./影响力事件分发策略初步方案.md)
- [项目流程与技术说明.md](./项目流程与技术说明.md)
