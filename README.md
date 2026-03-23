# Agent Learn

从零开始学习 Agent，到构造 Agent，再到提升 Agent。这个仓库是我的实战学习基地，目标是成为 **Agent 构造和维修大师**。

## 我的目标

我会沿着 3 个阶段持续进阶：

1. **了解 Agent**
   - 理解 Agent 的核心概念：`状态`、`工具`、`规划`、`记忆`、`评估`
   - 学会单 Agent 的基本工作流
2. **构造 Agent**
   - 构建可运行的多 Agent 系统（Manager + Workers）
   - 让 Agent 可以读写文件、执行命令、完成端到端任务
3. **提升 Agent**
   - 提升稳定性、可观测性、可维护性
   - 引入测试、日志、错误恢复与性能优化

## 仓库内容

- [agents.py](agents.py)：当前的多 Agent 协作示例（Manager + 4 Workers）
- [agent_replace_rag.py](agent_replace_rag.py)：单 Agent + PDF 标题工具（AutoGen 最小 Demo）
- [papers/](papers/)：阅读与沉淀 Agent 相关论文/资料

## 当前系统简介

`agents.py` 包含：

- `Manager Agent`：负责分解任务、决策下一步
- `ReadAgent`：读取文件
- `WriteAgent`：写入文件
- `EditAgent`：编辑文件
- `ExecAgent`：执行命令

系统基于 `LangGraph` 组织流程节点，形成 Manager 与 Worker 的循环协作。

## 快速开始

### 1) 安装依赖（示例）

根据代码中的导入，至少需要：

- `langgraph`
- `langchain-core`
- `langchain-nvidia-ai-endpoints`

### 2) 配置环境变量

- `NV_API_KEY` 或 `NVIDIA_API_KEY`
- 可选：`NVIDIA_MODEL`（默认 `meta/llama-3.1-70b-instruct`）

### 3) 运行

直接执行 `agents.py`。

### 4) 单 Agent + PDF 标题工具（AutoGen 最小 Demo）

安装依赖（示例）：

- `pyautogen`
- `pypdf`

执行：

- `python agent_replace_rag.py --pdf-dir ./papers`

## 学习路线（持续更新）

- [ ] 单 Agent 最小闭环（Plan → Act → Observe）
- [ ] 工具调用协议标准化（JSON schema）
- [ ] 多 Agent 协作策略优化（任务拆解与路由）
- [ ] 引入记忆机制（短期/长期）
- [ ] 增加评估基准与回归测试
- [ ] 故障诊断与自修复机制
- [ ] 成本/时延优化

## 当前进度（2026-03-14）

- ✅ 已完成一个 `Manager + 4 Workers` 的多 Agent 基础框架（LangGraph 编排）
- ✅ 已跑通基础工具链路：读文件 / 写文件 / 编辑文件 / 执行命令
- ✅ 已开始沉淀资料：新增 Anthropic《Building Effective AI Agents》
- ✅ 明确区分：`工具增强 LLM` ≠ `完整 Agent`（关键差异是自主决策循环与状态管理）
- ✅ 梳理 AutoGen 组件关系：`autogen-agentchat`（框架核心） + `autogen-ext`（模型/MCP扩展） + `autogenstudio`（可选 GUI）
- ✅ 确认技术路线：优先从 AutoGen 最小单 Agent 闭环入手，再逐步接入高德 MCP
- ✅ 明确模型接入策略：可使用 OpenAI-compatible 方式接 NVIDIA API，不强依赖 OpenAI 官方 API
- ✅ 文献学习进度：`papers/LLM-Agent.pdf` 已学习至第 3 章
- 🔄 下一步将围绕开源项目 `AutoGen` 做对照实践，重点关注：
   - 可观测（日志、追踪、指标）
   - 可控制（权限边界、策略约束、人工干预）
   - 鲁棒性（重试、回退、异常恢复）
   - 工业落地（评估、测试、部署、成本治理）

详细任务见 [TODO.md](TODO.md)。

## 学习日记

- [2026-03-17 学习记录](LEARNING_LOG_2026-03-17.md)

## 本地智慧问答系统计划（Gradio + Milvus + 本地 LLM）

基于当前进展，下一阶段按“最小可用闭环”推进：

1. **前端交互（Gradio）**
   - 提供单轮问答输入框与答案展示区。
   - 展示检索证据片段与来源（如文件名/切片序号）。

2. **检索层（Milvus）**
   - 使用已验证可连通的 Milvus 服务（`121.37.90.146:19530`）。
   - 默认集合：`kb_chunks`，向量维度：`1024`。
   - 问题向量化后执行 top-k 检索，返回 `content` 作为证据上下文。

3. **生成层（本地 LLM）**
   - 使用 OpenAI-compatible 本地模型服务：
     - `LOCAL_BASE_URL=http://121.37.90.146:55803/v1`
     - `LOCAL_MODEL=qwen3-32b`
   - 将“用户问题 + 检索证据”拼接后生成最终回答。

4. **评估层（Ragas）**
   - 先构建小规模测试集（手工 + 半自动）。
   - 再执行自动评估，输出指标与结果表（CSV）。
   - 指标优先：`faithfulness`、`context_precision`、`context_recall`（有 LLM 评委时）。

5. **迭代目标**
   - 第一步：先跑通单轮问答闭环（可用）。
   - 第二步：补充可观测日志（检索命中、时延、失败原因）。
   - 第三步：做回归评测与参数优化（top-k、prompt、模型参数）。

## 仓库愿景

把这个仓库打造成一个可复用的 Agent 工程实践模板：

- 能快速搭建 Agent 原型
- 能稳定运行真实任务
- 能持续迭代并定位修复问题

---

如果你也在学习 Agent，欢迎一起交流和迭代。
