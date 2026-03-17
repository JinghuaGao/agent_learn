import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import (
    fs_edit_json,
    fs_list_tree,
    fs_pwd,
    fs_read,
    pdf_search,
)


def create_model_client() -> OpenAIChatCompletionClient:
    # 支持两种模型端：
    # 1) 本地自部署 Qwen（无 key）
    # 2) NVIDIA/OpenAI-compatible（有 key）
    use_local_model = os.getenv("USE_LOCAL_MODEL", "1") == "1"

    if use_local_model:
        # 你给的地址是 chat/completions 终端。
        # OpenAI-compatible client 的 base_url 需要填到 /v1。
        local_base_url = os.getenv("LOCAL_BASE_URL", "http://121.37.90.146:55803/v1")
        local_model = os.getenv("LOCAL_MODEL", "qwen3-32b")

        return OpenAIChatCompletionClient(
            model=local_model,
            api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),
            base_url=local_base_url,
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),
            top_p=float(os.getenv("MODEL_TOP_P", "0.95")),
            max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "20480")),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": False,
                "family": "unknown",
            },
        )

    # 统一从环境变量读取密钥：避免硬编码，便于安全轮换
    api_key = (
        os.getenv("NV_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:
        raise RuntimeError("缺少 API Key。请设置 NVIDIA_API_KEY（或 NV_API_KEY）。")

    # 使用 OpenAI-compatible 接口接入 NVIDIA 端点
    # 如需切换供应商，只需改 base_url/model/KEY，不改 Agent 逻辑
    base_url = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("OPENAI_MODEL", "minimaxai/minimax-m2.5")

    # 生成参数可通过环境变量覆盖，便于实验
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.01"))
    top_p = float(os.getenv("MODEL_TOP_P", "0.1"))
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "81920"))

    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # 非 OpenAI 官方模型必须显式声明 model_info
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": False,
            "family": "unknown",
        },
    )


# Run the agent and stream the messages to the console.
async def main() -> None:
    # 1) 初始化模型客户端
    model_client = create_model_client()

    # 2) 初始化 Agent，并注册两个工具（粗读 + 精读检索）
    agent = AssistantAgent(
        name="pdf_reader_agent",
        model_client=model_client,
        tools=[
            fs_pwd,
            fs_list_tree,
            fs_read,
            fs_edit_json,
            pdf_search,
        ],
        system_message=(
            "你是科研论文助理。"
            "你拥有最小化工具集：fs_pwd / fs_list_tree / fs_read / pdf_search / fs_edit_json。"
            "默认资料目录是 /Users/jiean/agent_learn/refer_docs/。"
            ""
            "工作流程（第一性原理）："
            "1) 先确认当前路径：fs_pwd；再用 fs_list_tree 查目录结构。"
            "2) 再读文件内容：文本/JSON 用 fs_read；PDF 用 fs_read 或 pdf_search。"
            "3) 需要安全修改 JSON 时，使用 fs_edit_json(file_path, key_path, value_json)。"
            "4) 用户提问论文问题时：先用 pdf_search 找证据页，再给结论和页码。"
            "5) 证据不足时明确说明，不要编造。"
            "6) 只能通过真实工具调用，不要输出伪 <tool_call> 文本。"
        ),
        reflect_on_tool_use=True,
        model_client_stream=True,  # Enable streaming tokens from the model client.
    )

    # 3) 准备演示任务（可替换成你自己的 PDF 路径和问题）
    default_pdf = Path("/Users/jiean/agent_learn/refer_docs").resolve()
    task = (
        "大模型在望远镜探测方面能有什么应用呢？"
    )
    # 4) 运行并流式输出（这是最基础的“可观测性”入口）
    # 说明：Console 是展示层，不是必须；你后续可以替换成日志系统。
    try:
        await Console(agent.run_stream(task=task))
    finally:
        # 5) 无论成功失败都关闭连接，避免资源泄露
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
