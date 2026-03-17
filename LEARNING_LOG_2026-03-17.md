# 学习日记（2026-03-17）

## 今天完成了什么

- 跑通并对比了 AutoGen 两类样例：
  - `examples/travel_plan.py`（多 Agent 轮转，无外部工具）
  - `examples/stock_agent.py`（多 Agent + 外部工具）
- 明确了 `RoundRobinGroupChat` 的边界：
  - 有顺序（按列表轮转）
  - 不是严格状态机
  - 复杂流程需要额外编排器/状态机
- 回到 `replace_rag` 主线，重构为最小工具集（第一性原理）：
  - `fs_pwd`（路径确认）
  - `fs_list_tree`（目录结构）
  - `fs_read`（文本/PDF读取）
  - `pdf_search`（页级证据检索）
  - `fs_edit_json`（安全 JSON 编辑）
- 验证了一个关键事实：
  - `fs_read` 工具本身可用（直接调用可读出 PDF 内容）
  - 主要不稳定点在“模型工具调用协议漂移”（伪 `<tool_call>` 文本）

## 今天的关键认知

1. `cd` 在 Agent 中通常是“语义动作”，不是进程级目录状态。
2. 真正的工具调用以事件为准：
   - `ToolCallRequestEvent`
   - `ToolCallExecutionEvent`
3. 伪 `<tool_call>` 文本不是执行结果，只是模型输出。
4. 单 Agent 也能做复杂任务，前提是工具边界清晰、流程分层。
5. “替代 RAG”不是不要检索，而是做 Agentic Retrieval（分阶段检索 + 可观测执行）。

## 当前主要问题

- 本地模型在 function-calling 上偶发协议漂移。
- 需要继续增强可观测性（每轮 LLM 调用计数、耗时、tool 参数与结果记录）。

## 下一步计划

- 增加 `retrieve_top_documents`（文档级 top-k 召回）
- 增加 request_id/trace_id 日志
- 做单 Agent vs 多 Agent 的小规模评测（正确率/成本/时延）
