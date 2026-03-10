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

## 学习路线（持续更新）

- [ ] 单 Agent 最小闭环（Plan → Act → Observe）
- [ ] 工具调用协议标准化（JSON schema）
- [ ] 多 Agent 协作策略优化（任务拆解与路由）
- [ ] 引入记忆机制（短期/长期）
- [ ] 增加评估基准与回归测试
- [ ] 故障诊断与自修复机制
- [ ] 成本/时延优化

## 仓库愿景

把这个仓库打造成一个可复用的 Agent 工程实践模板：

- 能快速搭建 Agent 原型
- 能稳定运行真实任务
- 能持续迭代并定位修复问题

---

如果你也在学习 Agent，欢迎一起交流和迭代。
