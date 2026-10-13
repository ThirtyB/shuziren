# abc_reading 影响力事件分发策略原型

这是一个基于 `abc_reading` 数据集构建的影响力事件分发策略原型项目。当前目标不是做商品推荐，而是围绕一条清晰的流水线，完成：

- 输入一条事件信息
- 解析事件目标与约束
- 从数据集中筛选合适的数字人节点
- 输出结构化分发策略

当前项目已经支持：

- 规则驱动的基础流水线
- 在关键节点接入 LLM 做语义增强
- 热点事件批量评测
- 中间结果追踪与调试


## 项目目标

本项目聚焦以下问题：

1. 将自然语言事件描述转成可计算的结构化事件对象。
2. 基于用户画像、互动图谱与邻居特征构建候选节点特征。
3. 对候选节点进行主题相关性、扩散能力和风险控制评分。
4. 选择主选数字人与备选数字人。
5. 输出可执行的结构化分发策略。


## 当前状态

当前版本已经完成的核心链路：

```text
event_input
  -> data_loader
  -> event_parser
  -> feature_builder
  -> scorer
  -> selector
  -> strategy_generator
  -> structured JSON output
```

其中 `run_eval.py` 已支持带 LLM 的热点评测链路，`main.py` 当前主要用于直接跑单事件基线流程。


## 目录结构

```text
abc_reading_workspace/
  data/
    raw/                         # 原始数据
    derived/                     # 衍生数据，如加入 neighbor 的画像文件
  eval/
    hot_event_opinion_variants.json
    output/                      # 热点评测输出 JSON
  outputs/
    strategy/                    # main.py 直接运行时的输出
    web_runs/                    # Web 控制台运行输出
  scripts/                       # 数据预处理脚本
  src/influence_strategy/
    data_loader.py
    event_parser.py
    feature_builder.py
    scorer.py
    selector.py
    strategy_generator.py
    pipeline.py
    eval_hot_events.py
    llm_client.py
    prompts.py
    web_app.py
    reporting.py
    models.py
  web/frontend/                  # React/Vite 前端
  tests/
    pipeline_step_outputs/       # 热点评测中间过程追踪输出
    test_*.py
    visualize_*.py
  main.py
  run_eval.py
  environment.yml
  pyproject.toml
  .env.example
```


## 数据集使用范围

当前项目只使用 `abc_reading` 相关文件：

- `data/raw/abc_reading_profile.graph.anon`
- `data/raw/abc_reading_interaction.graph.anon`
- `data/raw/abc_reading_product_info.json`
- `data/derived/abc_reading_profile_with_neighbors.graph.anon`
- `data/derived/abc_reading_profile_compact.graph.anon`
- `data/derived/weibo_profile_reworked.graph.anon`
- `data/derived/weibo_profile_reworked_with_neighbors.graph.anon`

### 微博重构数据集

当前运行时的 `DataLoader` 会优先读取 `data/derived/weibo_profile_reworked_with_neighbors.graph.anon`（如果存在），否则回退到 `weibo_profile_reworked.graph.anon`。
可以使用以下命令重新生成：

```powershell
C:\Users\82039\.conda\envs\learnAgent\python.exe scripts\build_weibo_reworked_dataset.py
C:\Users\82039\.conda\envs\learnAgent\python.exe scripts\build_weibo_reworked_neighbors.py
```

该脚本会保留原始数据不变，删除未重写的空画像低价值节点，并自动将一部分空画像节点重写为更适合热点事件模拟的公共议题微博用户。

生成文件包括：
- `data/derived/weibo_profile_reworked.graph.anon`
- `data/derived/weibo_profile_reworked_summary.json`
- `data/derived/weibo_profile_rewrite_manifest.jsonl`
- `data/derived/weibo_profile_reworked_with_neighbors.graph.anon`
- `data/derived/weibo_profile_reworked_graph_summary.json`
- `data/derived/weibo_graph_rewrite_manifest.jsonl`

### 200 条热点事件主题画像补强

`eval/hot_event_opinion_variants_200.json` 将原来的 10 条热点事件扩展到 200 条。为了让候选数字人覆盖这些新增主题，可以在 `weibo_profile_reworked_with_neighbors.graph.anon` 中挑选低频、描述较少但仍有连接关系的节点，将其改造成按主题簇组织的候选数字人，并同步扩展连接关系：

```powershell
C:\Users\82039\.conda\envs\learnAgent\python.exe scripts\build_hot200_clustered_digital_humans.py
```

该脚本会直接更新 `data/derived/weibo_profile_reworked_with_neighbors.graph.anon`，但会先生成 `.bak` 备份。默认参数会选取约 30 个主题簇，替换 150 个原有低价值节点，并为它们补充同簇连接、语义相近原节点连接和跨簇桥接连接；所有受影响节点的 `graph_attributes` 和 `neighbors` 会从关系图重新计算。

辅助输出包括：

- `data/derived/weibo_profile_reworked_with_neighbors.graph.anon.bak`
- `data/derived/weibo_profile_reworked_hot200_clustered_manifest.jsonl`
- `data/derived/weibo_profile_reworked_hot200_clustered_summary.json`
- `data/derived/hot200_topic_cluster_summary.json`

项目中虽然仍沿用 `abc_reading` 这个数据来源，但任务目标已经明确从“商品推荐”切换为“影响力事件分发策略生成”。


## 环境准备

当前推荐使用 `conda`：

```powershell
conda env create -f environment.yml
conda activate abc-reading-strategy
```

当前主要依赖包括：

- `python=3.11`
- `pydantic`
- `pytest`
- `rapidfuzz`
- `networkx`
- `pandas`
- `matplotlib`
- `fastapi`
- `uvicorn`
- `python-multipart`


## Web 控制台

项目提供一个轻量 Web 控制台，支持单事件 JSON/表单输入、图片输入和 200 条热点事件一键评测。Web 运行结果默认写入 `outputs/web_runs/`，不会写入 `eval/output/`。

后端启动：

```powershell
conda activate abc-reading-strategy
uvicorn --app-dir src influence_strategy.web_app:app --reload --host 127.0.0.1 --port 8000
```

前端启动：

```powershell
cd web/frontend
npm install
npm run dev
```

浏览器访问：

```text
http://127.0.0.1:5173
```

前端默认启用 LLM，页面右上角可以手动关闭。200 条批量评测默认读取 `eval/hot_event_opinion_variants_200.json`。


## LLM 配置

如果要启用 LLM，请在工作目录创建 `.env`，字段可参考 [.env.example](./.env.example)：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

兼容的环境变量包括：

- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`
- `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`


## 核心模块说明

| 模块 | 作用 | 主要输出 |
| --- | --- | --- |
| `data_loader` | 读取 profile、interaction、derived graph 数据 | `DatasetBundle` / 原始对象 |
| `event_parser` | 将事件解析成结构化事件对象 | `ParsedEvent` |
| `feature_builder` | 为节点构建规则特征与 LLM 语义特征 | `FeatureBuildResult` |
| `scorer` | 基于特征、风险和语义相关性打分 | `ScoreResult` |
| `selector` | 选择主选节点、备选节点并分配角色 | `SelectionResult` |
| `strategy_generator` | 生成时间、频率、平台、协同方案 | `StrategyResult` |
| `pipeline` | 串联整条流水线 | `StrategyResult` / `PipelineArtifacts` |
| `eval_hot_events` | 热点事件评测、输出整理、追踪写盘 | JSON 输出 |
| `llm_client` | OpenAI-compatible LLM 调用封装 | 结构化 JSON 响应 |
| `prompts` | event parser / feature / selector 的提示词模板 | prompt text |


## LLM 用在什么地方

当前 LLM 主要用于 4 个环节，其中前 3 个会直接影响“选谁”，最后 1 个影响“怎么写”。

### 1. `event_parser`

规则解析完成后，可再调用一次 LLM，对事件做结构化增强，补充：

- `event_type`
- `target_goal`
- `target_audience`
- `extracted_keywords`
- `semantic_tags`
- `narrative_frames`
- `target_roles`
- `dispatch_preferences`

### 2. `feature_builder`

先用规则方法从全量节点中做第一轮候选召回，再把缩小后的候选池交给 LLM 做语义增强。LLM 会写回：

- `event_semantic_relevance_score`
- `audience_fit_score`
- `role_fit_score`
- `narrative_fit_score`
- `risk_conflict_score`
- `novelty_score`
- `llm_feature_score`

### 3. `selector`

`scorer` 先给出可选节点集合，`selector` 再把 shortlist 交给 LLM 进行重排，输出：

- `selected_order`
- `fallback_order`
- `recommended_role`
- 简短 `reasoning`

### 4. 最终文案生成

在热点评测输出阶段，LLM 会为已选中的数字人生成：

- `发帖内容`
- `目标群体画像`
- `互动提示`
- `协同说明`

说明：

- `data_loader` 不调用 LLM。
- `scorer` 不直接调用 LLM，但会消费 `feature_builder` 返回的 LLM 语义分数。
- `strategy_generator` 也不直接调用 LLM，它只负责把选中结果翻译成结构化策略。


## 节点检索与筛选机制

当前不是“让 LLM 直接扫描全量节点”，而是：

```text
全量规则召回 -> 候选池构造 -> LLM 语义增强 -> 打分 -> 角色化选择
```

具体过程如下。

### Step 1. `data_loader` 读取全量节点

读取：

- `profile`
- `interaction`
- `enriched_profile`

### Step 2. `event_parser` 生成事件查询条件

最关键的是生成：

- `extracted_keywords`
- `semantic_tags`
- `target_roles`
- `risk_level`

### Step 3. `feature_builder` 做第一轮召回与特征构建

会综合考虑：

- `user_description`
- `user_interests`
- 邻居数量与互动图结构
- 评论 / 转发行为
- 关键词命中情况

主要得到：

- `matched_keywords`
- `keyword_hit_count`
- `topic_match_score`
- `influence_score`
- `diffusion_score`
- `activity_score`
- `stability_score`

### Step 4. 构造候选池

不是简单取前 `N`，而是按不同目标分桶构造候选：

- `topic_bucket`：更偏主题相关
- `interaction_bucket`：更偏互动承接
- `diffusion_bucket`：更偏扩散能力
- `baseline_bucket`：规则高分补充

同时会限制“高影响力但泛化”的账号不要垄断候选池。

### Step 5. `feature_builder` 用 LLM 做候选精炼

只对缩小后的候选池做语义判断，而不是对整个数据集调用 LLM。

### Step 6. `scorer` 做最终门控

这里会综合：

- 规则特征分
- LLM 语义分
- 风险惩罚

并判断：

- `eligible`
- `manual_review_required`
- `final_score`

### Step 7. `selector` 选主选与备选

优先保证角色覆盖，再决定：

- `core_publish_node`
- `interaction_response_node`
- `amplification_node`
- `support_node`

一句话概括：当前的节点检索是“规则召回 + 小候选池 LLM 精排”，而不是“LLM 直接面对全库”。


## 运行方式

### 1. 运行测试

```powershell
conda run -n abc-reading-strategy python -m pytest
```

### 2. 运行单事件基线流程

`main.py` 适合输入一条事件，直接得到结构化策略结果。

```powershell
python main.py --event-text "希望围绕亲子阅读与英语启蒙做一次传播活动，提升讨论度并控制刷屏风险"
```

也可以使用事件文件：

```powershell
python main.py --event-file your_event.json
```

说明：

- `main.py` 当前默认走规则基线流程。
- 它会输出 JSON 和 Markdown 两份文件到 `outputs/strategy/`。

### 3. 运行热点评测

规则模式：

```powershell
python run_eval.py --disable-llm
```

启用 LLM：

```powershell
python run_eval.py
```

只跑单个事件：

```powershell
python run_eval.py --event-id hot_event_001
```

限制只跑前 3 条热点：

```powershell
python run_eval.py --event-limit 3
```

### 4. 常用参数

`main.py` 与 `run_eval.py` 常见参数包括：

- `--workspace-root`
- `--profile-limit`
- `--risk-level`
- `--max-selected-nodes`
- `--max-frequency-per-day`
- `--campaign-window-hours`
- `--allowed-platforms`

`run_eval.py` 额外支持：

- `--event-id`
- `--event-limit`
- `--output-dir`
- `--trace-dir`
- `--disable-llm`


## 输出结果

### `main.py` 输出

默认写入：

```text
outputs/strategy/strategy_<event_id>.json
outputs/strategy/strategy_<event_id>.md
```

JSON 主体是完整的 `StrategyResult`，包含：

- `event`
- `product_context`
- `selection_summary`
- `summary`
- `stage_plans`
- `selected_nodes`
- `fallback_nodes`
- `strategy`

### `run_eval.py` 输出

默认写入：

```text
eval/output/<event_id>_strategy_output.json
```

当前 `run_eval.py` 输出使用的是更贴近实施方案的结构：

- `事件名称`
- `选取数字人id组`
- 每个数字人节点的动作清单

### 中间结果追踪

每次运行 `run_eval.py`，都会在：

```text
tests/pipeline_step_outputs/<event_id>/
```

下写入：

- `00_hot_event_input.json`
- `01_pipeline_payload.json`
- `02_event_parser_output.json`
- `03_feature_builder_output.json`
- `04_scorer_output.json`
- `05_selector_output.json`
- `06_strategy_generator_output.json`
- `07_final_output.json`

这是当前最重要的调试入口，建议在分析“为什么选中这个节点”时优先查看这组文件。


## 可视化与调试

项目中保留了一组轻量可视化脚本：

```powershell
python tests\visualize_feature_builder.py
python tests\visualize_scorer.py
python tests\visualize_selector.py
python tests\visualize_strategy_generator.py
```


## 当前规则与设计原则

当前版本遵循以下原则：

- 先把可解释的基线流程跑通，再让 LLM 作为增强层加入。
- 不让 LLM 直接扫描整个大数据集，而是先用规则方法缩小候选范围。
- 不让高粉丝量单独决定是否入选，主题相关性必须参与门控。
- 输出必须结构化，便于后续接入 API、前端或自动化流程。


## 当前局限

虽然当前流水线已经可运行，但仍有这些限制：

- 数据集本身来自 `abc_reading`，账号画像与国际热点事件天然并不完全匹配。
- 某些高活跃、泛画像账号仍可能在部分事件中进入候选池。
- `main.py` 当前默认未开放端到端 LLM 开关，带 LLM 的完整链路主要集中在 `run_eval.py`。
- 暂未引入向量检索或 embedding 召回层。


## 后续改进方向

建议下一步优先考虑：

1. 继续强化对“生活方式 / 泛账号 / 主题不相关高影响力节点”的抑制规则。
2. 引入 embedding 检索层，先做语义召回，再做 LLM 精排。
3. 为军事、金融、网络安全、气候、公共政策等不同事件类型设计专用提示词模板。
4. 增加缓存机制，降低大规模热点评测时的 LLM 调用成本。
5. 给 `main.py` 增加显式 `--use-llm` 开关，使单事件入口也能跑完整增强链路。


## 相关文档

如果需要查看本轮热点评测改造的更详细说明，可参考：

- [2026-06-02_热点评测流水线改造说明.md](./2026-06-02_热点评测流水线改造说明.md)
- [改进2026-05-31.md](./改进2026-05-31.md)
- [影响力事件分发策略初步方案.md](./影响力事件分发策略初步方案.md)
#   s h u z i r e n  
 