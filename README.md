# Agent Learn

这个仓库当前的主线不是泛泛地“学 Agent”，而是做一个真正会查资料的 Agentic RAG：

- 让大模型自己去读 PDF、PPT，后面再扩展到图片
- 让它先看“图书目录”式的 metadata，再决定读哪几份资料
- 最后把它和经典 RAG 做对比，主要靠人的主观判断看谁更强

Ragas 暂时不作为主线，先把系统跑顺、把差异看清楚。

## 当前目标

1. **经典 RAG 基线**
   - 先有一个稳定的“召回 → 拼上下文 → 直接回答”版本
2. **Agentic RAG**
   - 让 Agent 自己决定先看 metadata、再看正文、再补证据
   - 支持 PDF / PPT，后续再接图片
3. **元数据维护**
   - 维护一个可用的文档目录，避免每次都把整个资料库读一遍
4. **主观对比**
   - 比较回答是否更准、更完整、更像“会查资料的人”
   - 重点看证据质量、推理过程、稳定性和可控性

## 代码地图

- [replace_rag/agentic_rag.py](replace_rag/agentic_rag.py)：Agentic RAG 主入口
- [replace_rag/tools.py](replace_rag/tools.py)：文件读取、目录检索、metadata 维护等工具
- [replace_rag/pdf_metadata_index.json](replace_rag/pdf_metadata_index.json)：当前的文档目录
- [evaluation/simple_milvus_rag_demo.py](evaluation/simple_milvus_rag_demo.py)：经典 RAG 对照组
- [agents.py](agents.py)：另一个多 Agent 实验示例
- [papers/](papers/)：学习资料

## 现在这套系统怎么分工

### 1) 经典 RAG

`evaluation/simple_milvus_rag_demo.py` 做的是基线版本：

- 问题先转向量
- 去 Milvus 召回相关 chunk
- rerank 后拼上下文
- 直接交给 LLM 回答

它的价值不是“最强”，而是当对照组。

### 2) Agentic RAG

`replace_rag/agentic_rag.py` 做的是 Agentic 版本：

- 先检查 metadata 是否可用、是否过期
- 再从文档目录里筛候选文档
- 再读取 PDF / PPT 的具体内容
- 证据够了就停，不够就继续找

它更像一个会查资料的人，而不是只会“向量检索 + 拼上下文”的机器。

### 3) 元数据目录

`replace_rag/pdf_metadata_index.json` 相当于图书馆目录，但现在还是“半成品目录”：

- 很多条目的 `abstract` / `keywords` 还是空的
- `status` 仍然偏原始导入态
- `updated_at` 也未必反映最新状态

所以现在的重点不是继续堆检索技巧，而是先把目录维护好。

## 目前已知问题

- Agent 现在能读资料，但**还没有真正完整的 metadata 自动更新/重建流程**
- `fs_edit_json` 只能改单个 JSON 字段，不适合做全量目录重建
- 现在的 metadata 更像“原始索引”，还不是“清洗过的知识目录”
- 评测上先靠人工主观判断，不急着回到 Ragas

## 下一步优先级

1. **把 Agentic RAG 主链路跑稳**
   - metadata 召回 → 正文阅读 → 证据整合 → 最终回答
2. **补一个真正的 metadata 重建脚本**
   - 让目录能从 PDF / PPT / 图片源文件重新生成
3. **扩展到图片**
   - OCR / 图像理解 / 多模态读取
4. **做主观对比模板**
   - 统一问题集，比较经典 RAG 和 Agentic RAG 的差异

## 学习路线

- [ ] 跑通经典 RAG 与 Agentic RAG 的同题对比
- [ ] 建立清晰的 metadata 重建流程
- [ ] 支持更多文档类型：PDF / PPT / 图片
- [ ] 做人工主观评测表
- [ ] 再决定是否重新引入 Ragas 做量化评测

## 运行入口

### Agentic RAG

```bash
python replace_rag/agentic_rag.py
```

### 经典 RAG 基线

```bash
python evaluation/simple_milvus_rag_demo.py
```

## 仓库愿景

把这个仓库打造成一个可以复用、可以对比、可以持续迭代的 Agentic RAG 实战模板。

---

如果你也在做 Agentic RAG，欢迎一起把它打磨成一个真正可用的系统。
