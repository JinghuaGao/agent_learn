#Travel Planning
#In this example, we’ll walk through the process of creating a sophisticated travel planning system using AgentChat. Our travel planner will utilize multiple AI agents, each with a specific role, to collaboratively create a comprehensive travel itinerary.

#First, let us import the necessary modules.

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
#Defining Agents
#In the next section we will define the agents that will be used in the travel planning team.

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


model_client = create_model_client()

planner_agent = AssistantAgent(
    "planner_agent",
    model_client=model_client,
    description="A helpful assistant that can plan trips.",
    system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
)

local_agent = AssistantAgent(
    "local_agent",
    model_client=model_client,
    description="A local assistant that can suggest local activities or places to visit.",
    system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
)

language_agent = AssistantAgent(
    "language_agent",
    model_client=model_client,
    description="A helpful assistant that can provide language tips for a given destination.",
    system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
)

travel_summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=model_client,
    description="A helpful assistant that can summarize the travel plan.",
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
)
termination = TextMentionTermination("TERMINATE")
group_chat = RoundRobinGroupChat(
    [planner_agent, local_agent, language_agent, travel_summary_agent], termination_condition=termination
)


async def main() -> None:
    await Console(group_chat.run_stream(task="我想去海口玩浆板，顺便吃一些当地的美食，还要环海南自驾，帮我规划一下行程吧！"))
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())