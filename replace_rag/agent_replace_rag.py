import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools import (
    docling_read_pdf,
    fs_edit_json,
    fs_list_tree,
    fs_pwd,
    fs_read,
    metadata_check_freshness,
    metadata_compact_previews,
    metadata_retrieve_top_docs,
    pdf_search,
    pptx_read,
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
            metadata_check_freshness,
            metadata_compact_previews,
            metadata_retrieve_top_docs,
            docling_read_pdf,
            pdf_search,
            pptx_read,
        ],
        system_message=(
            "你是科研论文助理。你可以通过参考资料目录中的文件来回答用户的问题。"
            "默认资料目录是 /Users/jiean/agent_learn/refer_docs/。"
            "默认元数据是 /Users/jiean/agent_learn/refer_docs/pdf_metadata_index.json, 包含了每个 PDF 的路径、页数、主要内容等。"
            "你拥有最小化工具集:fs_pwd / fs_list_tree / fs_read / metadata_check_freshness / metadata_compact_previews / metadata_retrieve_top_docs / docling_read_pdf / pdf_search / pptx_read / fs_edit_json。"

            ""
            "工作流程（必须遵守）："
            "1) 先确认目录与元数据可用（必要时调用 fs_pwd/fs_list_tree/fs_read）。"
            "2) 若怀疑元数据过期，先调用 metadata_check_freshness 判断是否最新。"
            "2.1) 若 metadata 过大或 preview_text 过长，先调用 metadata_compact_previews 压缩索引，再继续检索。"
            "3) 回答论文问题前，必须先调用 metadata_retrieve_top_docs(query, max_docs<=3) 做文档级筛选。"
            "4) 只在筛出的最多 3 篇文档中继续阅读：PDF 优先用 docling_read_pdf（必要时 pdf_search 做页级证据定位）；PPTX 使用 pptx_read。"
            "5) 需要安全修改 JSON 时，使用 fs_edit_json(file_path, key_path, value_json)。"
            "6) 证据不足时明确说明，不要编造。"
            "7) 只能通过真实工具调用，不要输出伪 <tool_call> 文本。"
            "8) 若已经有足够证据，请停止继续调用工具并直接输出最终答案。"
            "9) 最终答案必须按以下固定结构输出："
            "【结论】...\n"
            "【证据】按要点列出，每条包含来源文件名与页码/位置\n"
            "【不确定性】信息缺口与置信度说明\n"
            "【后续建议】如需补充检索，给出下一步。"
            "10) 若没有找到有效证据，必须输出“证据不足”，并说明已尝试的检索范围。"
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=5,  # 允许单 Agent 在一次任务中进行多轮工具调用
        model_client_stream=True,  # Enable streaming tokens from the model client.
    )

    # 3) 准备演示任务（可替换成你自己的 PDF 路径和问题）
    default_pdf = Path("/Users/jiean/agent_learn/refer_docs").resolve()
    task = (
        "“GeoChemAD 的组成特征有哪些？（区域、采样来源、目标元素、子集数量）”"
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
