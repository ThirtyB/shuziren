import sys
from pathlib import Path

# 将项目的 src 目录加入 Python 模块搜索路径
workspace_root = Path(__file__).resolve().parent
src_path = workspace_root / "src"
sys.path.insert(0, str(src_path))

from influence_strategy.llm_client import OpenAICompatibleLLMClient


print("开始读取千问配置……")

client = OpenAICompatibleLLMClient.from_env_files(workspace_root)

if client is None:
    print("没有读取到 LLM 配置，请检查 .env 文件。")
else:
    print("LLM 配置读取成功")
    print("模型：", client.model)
    print("Base URL：", client.base_url)

    print("开始调用千问接口……")

    result = client.generate_json(
        system_prompt="你是一个事件传播策略分析助手。",
        user_prompt="""
请分析下面的事件，并严格输出JSON，不要输出额外解释。

事件：
学校准备开展亲子阅读与英语启蒙活动，希望提高家长参与度，同时控制刷屏风险。

请输出以下字段：
{
  "event_type": "",
  "target_audience": [],
  "goals": [],
  "risk_level": "",
  "keywords": []
}
""",
    )

    print("千问返回结果：")
    print(result)