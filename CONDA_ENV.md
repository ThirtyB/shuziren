# Conda 环境说明

本项目建议使用独立 conda 环境运行，当前本地验证环境名为 `now2`。

## 推荐创建方式

如果 conda 源支持 Python 3.11，优先使用项目自带的 `environment.yml`：

```bash
cd /data/zxm/now/abc_reading_workspace
conda env create -n now2 -f environment.yml
conda activate now2
```

如果已经存在 `now2`，可以更新依赖：

```bash
conda activate now2
python -m pip install "pydantic>=2.8,<3" "pytest>=8,<9" "rapidfuzz>=3.9,<4" "networkx>=3.3,<4" "pandas>=2.2,<3" "matplotlib>=3.9,<4"
```

## 本机当前运行方式

本机 conda 默认源无法直接解析 `python=3.11` 时，可以先克隆已有环境，再安装依赖：

```bash
conda create -y -n now2 --clone now
conda activate now2
python -m pip install "pydantic>=2.8,<3" "pytest>=8,<9" "rapidfuzz>=3.9,<4" "networkx>=3.3,<4" "pandas>=2.2,<3" "matplotlib>=3.9,<4"
```

## 运行测试

```bash
cd /data/zxm/now/abc_reading_workspace
source /data/anaconda3/etc/profile.d/conda.sh
conda activate now2
python -m pytest
```

## 运行 eval 生成

默认启用 `.env` 中配置的大模型：

```bash
python run_eval.py --event-id hot_event_001
```

如需关闭大模型、只使用规则模板兜底：

```bash
python run_eval.py --event-id hot_event_001 --disable-llm
```

## 环境变量

大模型配置文件位于：

```text
src/influence_strategy/.env
```

常用字段：

```text
OPENAI_API_KEY=...
OPENAI_API_BASE=...
MODEL_NAME=...
```

`.env` 文件包含密钥或本地配置，已经在 `.gitignore` 中忽略，不要提交到 git。
