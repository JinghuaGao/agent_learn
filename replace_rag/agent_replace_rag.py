import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import (
    build_pdf_metadata_index,
    discover_pdf_folders,
    get_pdf_main_content,
    search_pdf_pages,
    upsert_pdf_metadata,
)


def create_model_client() -> OpenAIChatCompletionClient:
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
    temperature = float(os.getenv("MODEL_TEMPERATURE", "1"))
    top_p = float(os.getenv("MODEL_TOP_P", "0.95"))
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "8192"))

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
            discover_pdf_folders,
            get_pdf_main_content,
            search_pdf_pages,
            build_pdf_metadata_index,
            upsert_pdf_metadata,
        ],
        system_message=(
            "你是科研论文助理，负责在本地 PDF 数据集中完成检索与问答。"
            "默认文档目录是 /Users/jiean/agent_learn/refer_docs/。"
            ""
            "可用工具与职责："
            "1) discover_pdf_folders(root_path, hint, max_depth, max_results)：当路径不明确或用户路径可能写错时，先发现候选 PDF 目录。"
            "2) build_pdf_metadata_index(folder_path, index_file, max_pages, force)：批量建立/更新元数据索引（首选先做这一步）。"
            "3) get_pdf_main_content(file_path, max_pages)：读取单篇 PDF 前几页，用于快速理解主题并生成摘要。"
            "4) upsert_pdf_metadata(index_file, file_path, abstract, keywords)：把你生成的摘要与关键词写回索引。"
            "5) search_pdf_pages(file_path, query, top_k, page_start, page_end)：针对具体问题逐页定位证据。"
            ""
            "工作流程（必须遵守）："
            "A. 若路径不确定，先调用 discover_pdf_folders。"
            "B. 若索引不存在或用户要求更新，先调用 build_pdf_metadata_index。"
            "C. 用户提出问题后，先判断是否需要查 PDF；需要时优先从相关论文中调用 search_pdf_pages 获取证据。"
            "D. 证据不足时明确说明不足，不要编造。"
            "E. 输出尽量简洁，并给出证据页码与文件名。"
        ),
        reflect_on_tool_use=True,
        model_client_stream=True,  # Enable streaming tokens from the model client.
    )

    # 3) 准备演示任务（可替换成你自己的 PDF 路径和问题）
    default_pdf = Path("./papers/LLM-Agent.pdf").resolve()
    task = (
        f"请调用工具读取这个 PDF 的前 3 页: {default_pdf}。"
        "然后输出 200 字以内摘要和 5 个关键词。"
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
