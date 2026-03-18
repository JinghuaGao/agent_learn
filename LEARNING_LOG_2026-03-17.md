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

---

## 2026-03-18（今天）补充总结

### 今天的成果

- `replace_rag` 侧完成了 metadata 体积治理的闭环：
  - 新增 `metadata_compact_previews(...)`，可批量压缩超长 `preview_text`
  - `metadata_retrieve_top_docs(...)` 在评分阶段只使用截断后的 preview，降低上下文/计算负担
  - Agent workflow 增加“metadata 过大先压缩”的步骤
- 实际验证了压缩收益：metadata 文件体积显著下降（已实测生效）。
- 清理了开发噪音文件策略：`__pycache__`、`*.pyc`、`*.bak` 已纳入忽略。
- 今天的 demo 算是成功：主流程可运行，工具链联动可用。

### 今天学到的关键点（关于 Agent“记忆”与长文档）

你的问题非常关键：500 页 PDF 明显超过单次上下文窗口，Agent 不是“全量记住”，而是“外部化记忆 + 迭代检索”。

可以把机制理解为三层：

1. **短期上下文（LLM窗口）**
   - 只保留当前轮最相关信息，不可能容纳整本书。

2. **工具态记忆（外部状态）**
   - 通过工具返回结构化信息（页码、文件路径、命中片段、摘要）。
   - “读到第几页”通常不靠模型死记，而是靠：
     - 工具参数（例如 `page_start/page_end`）
     - 工具输出中的页码
     - 外部文件（json/日志）记录进度（可恢复）。

3. **索引态记忆（metadata / 检索层）**
   - 先粗筛文档，再精读命中页，避免顺序硬读 1→500 页。
   - 必要时把“已读页范围、已提取证据、未解决问题”写入可持久化状态。

一句话：**Agent 处理超长文档的本质，是把“记忆”从上下文窗口迁移到工具与索引。**

### 我现在还不懂、后续要继续研究的点

- 如何设计稳定的“阅读状态机”：
  - 已读页集合
  - 命中证据映射（页码 → 结论）
  - 停止条件（证据充分即停）
- 如何定义工具接口，减少模型自由发挥导致的协议漂移。
- 如何做“分层摘要”防止摘要本身再次膨胀。

### TODO（下一阶段）

- [ ] 设计 `pdf_iter_read` 风格工具协议（输入：`cursor/page_window/query`，输出：`next_cursor/hits/done`）。
- [ ] 增加持久化进度文件（例如 `reading_state.json`）：记录已读页、命中页、证据片段 hash。
- [ ] 建立“先检索后阅读”标准流程：metadata top-k → `pdf_search` 命中页 → 定向 `docling_read_pdf`。
- [ ] 增加可观测性字段：`request_id`、轮次、工具耗时、token 开销。
- [ ] 做一次 300~500 页长文档压测，记录成功率/耗时/证据完整性。

### 备注

今天先收工，阶段性目标达成：**demo 成功 + 检索链路更稳 + metadata 膨胀问题得到工程化控制**。
