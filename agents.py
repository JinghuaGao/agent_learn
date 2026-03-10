# multi_agent_system.py
# Manager + 4 Worker Agents 协作系统
# 任务：检查 OPENROUTER_API_KEY，如果不存在则添加

import os
import json
import subprocess
import re
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# ============ 0. 共享状态定义 ============

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]  # 对话历史
    task: str           # 当前任务描述
    worker_result: str  # Worker Agent 返回的结果
    next_step: str      # Manager 决定的下一步
    done: bool          # 任务是否完成

# ============ 1. 基础 LLM 配置 ============

def get_llm():
    """获取 NVIDIA LLM"""
    api_key = os.getenv("NV_API_KEY") or os.getenv("NVIDIA_API_KEY")
    model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    return ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.2,
        max_tokens=1024
    )

# ============ 2. 四个 Worker Agent（每个是一个 LLM + 专用提示词） ============

def create_worker_agent(name: str, system_prompt: str, tools_description: str):
    """创建 Worker Agent 的调用函数"""
    llm = get_llm()
    
    def agent_node(state: AgentState):
        # Worker 只看 Manager 分配的具体任务
        task = state.get("task", "无任务")
        
        messages = [
            SystemMessage(content=f"""你是 {name}。
{system_prompt}

可用工具：{tools_description}

重要：你只能使用上述工具完成任务，不要解释过程，直接给出工具调用或结果。
如果不需要工具，直接回答。

当前任务：{task}"""),
            HumanMessage(content=f"请执行：{task}")
        ]
        
        response = llm.invoke(messages)
        return {"worker_result": response.content}
    
    return agent_node

# Read Agent - 专门读取文件
read_agent = create_worker_agent(
    "ReadAgent",
    "你专门负责读取文件内容。使用 read 工具查看文件。",
    "read(file_path) - 读取指定文件内容"
)

# Write Agent - 专门写入文件  
write_agent = create_worker_agent(
    "WriteAgent", 
    "你专门负责写入文件。使用 write 工具创建或覆盖文件。",
    "write(file_path, content) - 写入内容到文件"
)

# Edit Agent - 专门编辑文件
edit_agent = create_worker_agent(
    "EditAgent",
    "你专门负责编辑文件。使用 edit 工具精确替换内容。",
    "edit(file_path, old_str, new_str) - 替换文件中的文本"
)

# Exec Agent - 专门执行命令
exec_agent = create_worker_agent(
    "ExecAgent",
    "你专门负责执行 shell 命令。使用 bash 工具运行命令。",
    "bash(command) - 执行 shell 命令"
)

# ============ 3. 工具执行器（真正干活的代码） ============

def execute_tool_call(tool_str: str) -> str:
    """解析并执行工具调用"""
    try:
        # 尝试解析 JSON 格式的工具调用
        if "{" in tool_str:
            start = tool_str.find("{")
            end = tool_str.rfind("}")
            data = json.loads(tool_str[start:end+1])
            
            tool_name = data.get("tool") or data.get("name")
            args = data.get("args") or data.get("arguments") or data
            
            # 执行工具
            if tool_name == "read" or tool_name == "ReadAgent":
                path = args.get("file_path") or args.get("path")
                with open(os.path.expanduser(path), 'r') as f:
                    return f.read()
                    
            elif tool_name == "write" or tool_name == "WriteAgent":
                path = args.get("file_path") or args.get("path")
                content = args.get("content")
                with open(os.path.expanduser(path), 'w') as f:
                    f.write(content)
                return f"成功写入 {path}"
                
            elif tool_name == "edit" or tool_name == "EditAgent":
                path = args.get("file_path") or args.get("path")
                old = args.get("old_str") or args.get("old")
                new = args.get("new_str") or args.get("new")
                with open(os.path.expanduser(path), 'r') as f:
                    content = f.read()
                content = content.replace(old, new, 1)
                with open(os.path.expanduser(path), 'w') as f:
                    f.write(content)
                return f"成功编辑 {path}"
                
            elif tool_name == "bash" or tool_name == "ExecAgent":
                cmd = args.get("command") or args.get("cmd")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                return result.stdout + result.stderr
                
        # 如果没有 JSON，尝试直接执行（ExecAgent 可能直接给命令）
        if "export" in tool_str or "echo" in tool_str or "grep" in tool_str:
            lines = tool_str.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    result = subprocess.run(line, shell=True, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout:
                        return result.stdout
            return "命令执行完成"
            
        return f"无法解析工具调用：{tool_str[:100]}"
        
    except Exception as e:
        return f"工具执行错误：{str(e)}"

# ============ 4. Manager Agent（协调员） ============

def manager_node(state: AgentState):
    """Manager Agent：分析任务，决定调用哪个 Worker"""
    llm = get_llm()
    
    # 构建 Manager 的上下文
    context = f"""
当前任务：{state.get("task", "无")}
上一步结果：{state.get("worker_result", "无")}
已完成步骤：{[m.content for m in state["messages"][-3:]]}

你是 Manager Agent，负责协调 4 个 Worker：
- ReadAgent：读取文件
- WriteAgent：写入文件  
- EditAgent：编辑文件
- ExecAgent：执行命令

你的职责：
1. 分析当前进度，决定下一步调用哪个 Worker
2. 为 Worker 生成具体的任务描述（包含工具调用格式）
3. 如果任务完成，直接输出最终结果

按以下格式回复：
NEXT: <Worker名称> | TASK: <具体任务描述>
或
DONE: <最终结果>
"""
    
    messages = [
        SystemMessage(content="你是 Manager Agent，负责协调多个 Worker Agent 完成任务。"),
        HumanMessage(content=context)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # 解析 Manager 的决策
    if "DONE:" in content:
        return {
            "next_step": "END",
            "messages": [AIMessage(content=content.replace("DONE:", "").strip())],
            "done": True
        }
    elif "NEXT:" in content:
        # 解析 NEXT: Worker | TASK: xxx
        match = re.search(r"NEXT:\s*(\w+).*TASK:\s*(.+)", content, re.DOTALL)
        if match:
            worker = match.group(1).strip()
            task = match.group(2).strip()
            return {
                "next_step": worker,
                "task": task,
                "messages": [AIMessage(content=f"Manager 决定：调用 {worker}，任务：{task}")]
            }
    
    # 默认让 ExecAgent 尝试
    return {
        "next_step": "ExecAgent",
        "task": content,
        "messages": [AIMessage(content=f"Manager：{content}")]
    }

# ============ 5. Worker 执行节点（包装真实工具执行） ============

def worker_wrapper(worker_name: str):
    """包装 Worker，执行工具并返回结果"""
    def node(state: AgentState):
        print(f"\n🔧 {worker_name} 执行任务: {state['task'][:80]}...")
        
        # 先让 Worker LLM 生成工具调用
        if worker_name == "ReadAgent":
            result = read_agent(state)
        elif worker_name == "WriteAgent":
            result = write_agent(state)
        elif worker_name == "EditAgent":
            result = edit_agent(state)
        elif worker_name == "ExecAgent":
            result = exec_agent(state)
        else:
            return {"worker_result": "未知 Worker"}
        
        tool_call = result.get("worker_result", "")
        print(f"   {worker_name} 生成调用: {tool_call[:100]}...")
        
        # 执行真实工具
        execution_result = execute_tool_call(tool_call)
        print(f"   执行结果: {execution_result[:100]}...")
        
        return {
            "worker_result": execution_result,
            "messages": [AIMessage(content=f"{worker_name} 结果: {execution_result}")]
        }
    return node

# ============ 6. 构建 LangGraph 拓扑图 ============

def create_workflow():
    """创建多 Agent 协作图"""
    
    # 定义节点
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Manager", manager_node)
    workflow.add_node("ReadAgent", worker_wrapper("ReadAgent"))
    workflow.add_node("WriteAgent", worker_wrapper("WriteAgent"))
    workflow.add_node("EditAgent", worker_wrapper("EditAgent"))
    workflow.add_node("ExecAgent", worker_wrapper("ExecAgent"))
    
    # 从 Manager 出发的条件边
    def route_from_manager(state: AgentState):
        next_step = state.get("next_step", "END")
        if state.get("done") or next_step == "END":
            return END
        return next_step
    
    workflow.set_entry_point("Manager")
    
    # Manager → Workers
    workflow.add_conditional_edges(
        "Manager",
        route_from_manager,
        {
            "ReadAgent": "ReadAgent",
            "WriteAgent": "WriteAgent",
            "EditAgent": "EditAgent",
            "ExecAgent": "ExecAgent",
            END: END
        }
    )
    
    # Workers → Manager（完成后回报）
    for worker in ["ReadAgent", "WriteAgent", "EditAgent", "ExecAgent"]:
        workflow.add_edge(worker, "Manager")
    
    return workflow.compile()

# ============ 7. 运行系统 ============

def main():
    print("=" * 70)
    print("🤖 Multi-Agent System: Manager + 4 Workers")
    print("任务：检查 ~/.zshrc 中的 OPENROUTER_API_KEY，不存在则添加")
    print("=" * 70)
    
    # 检查 API Key
    if not (os.getenv("NV_API_KEY") or os.getenv("NVIDIA_API_KEY")):
        print("❌ 请先设置 NV_API_KEY 或 NVIDIA_API_KEY")
        return

    print(f"🧠 使用模型: {os.getenv('NVIDIA_MODEL', 'meta/llama-3.1-70b-instruct')}")
    
    # 创建图
    app = create_workflow()
    
    # 初始状态
    initial_state = {
        "messages": [],
        "task": "检查 ~/.zshrc 中是否存在 OPENROUTER_API_KEY 环境变量设置。如果不存在，添加 export OPENROUTER_API_KEY='sk-or-2131321' 到文件末尾。如果已存在，报告当前值。",
        "worker_result": "",
        "next_step": "",
        "done": False
    }
    
    print(f"\n📋 初始任务: {initial_state['task'][:80]}...")
    print("\n🚀 启动多 Agent 协作...")
    print("-" * 70)
    
    # 运行图（会自动循环直到 END）
    final_state = None
    for event in app.stream(initial_state):
        for node_name, state in event.items():
            print(f"\n📍 节点 [{node_name}]:")
            if "messages" in state and state["messages"]:
                msg = state["messages"][-1]
                content = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"   {content[:150]}...")
        final_state = state
    
    print("\n" + "=" * 70)
    print("✅ 任务完成！")
    print("-" * 70)
    
    # 输出最终结果
    if final_state and final_state.get("messages"):
        final_msg = final_state["messages"][-1]
        print(f"📝 最终结果: {final_msg.content if hasattr(final_msg, 'content') else final_msg}")
    
    # 验证实际结果
    print("\n🔍 验证 ~/.zshrc 中的 OPENROUTER_API_KEY:")
    try:
        with open(os.path.expanduser("~/.zshrc"), 'r') as f:
            content = f.read()
            if "OPENROUTER_API_KEY" in content:
                # 提取值
                match = re.search(r'OPENROUTER_API_KEY[=:]\s*["\']?([^"\']+)["\']?', content)
                if match:
                    print(f"   ✅ 找到: OPENROUTER_API_KEY={match.group(1)}")
                else:
                    print("   ✅ 找到设置，但无法解析值")
            else:
                print("   ❌ 未找到 OPENROUTER_API_KEY")
    except Exception as e:
        print(f"   验证失败: {e}")

if __name__ == "__main__":
    main()
