from __future__ import annotations

"""
最小 RAG 示例（本地 LLM + Milvus）

你可以把它和“纯 LLM 调用”对比着看：

纯 LLM：
    question -> LLM -> answer

加上 RAG：
    question -> embedding -> Milvus 检索 -> retrieved_contexts -> LLM -> answer

这个脚本故意写得很直白，目标是帮助理解流程，不追求工程完整性。

默认配置：
- 本地 LLM 服务: http://121.37.90.146:55803/v1
- 本地模型: qwen3-32b
- Milvus 服务: 121.37.90.146:19530
- Milvus 集合: kb_chunks
"""

import os
import re
from typing import Any, List, Tuple

from openai import OpenAI
from pymilvus import Collection, DataType, connections


# -----------------------------
# 1) OpenAI-compatible LLM 配置
# -----------------------------
# 这些就是你熟悉的“标准请求元素”：
# - base_url
# - model
# - api_key（本地服务没有鉴权时可用 EMPTY）
LLM_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://121.37.90.146:55803/v1")
LLM_MODEL = os.getenv("LOCAL_MODEL", "qwen3-32b")
LLM_API_KEY = os.getenv("LOCAL_API_KEY", "EMPTY")


# -----------------------------
# 1.1) Embedding 模型配置
# -----------------------------
# 这一步是 RAG 和“纯 LLM 问答”相比新增的关键能力：
# 需要把 query 先编码成向量，才能去 Milvus 做相似度检索。
EMBEDDING_BASE_URL = os.getenv("LOCAL_EMBEDDING_BASE_URL", "http://121.37.90.146:55006/v1")
EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "bge-m3")
EMBEDDING_API_KEY = os.getenv("LOCAL_EMBEDDING_API_KEY", "EMPTY")


# -----------------------------
# 2) Milvus 配置
# -----------------------------
# RAG 不是“只多一个端口”，而是多了一层“检索”。
# 这层检索的核心配置包括：
# - Milvus host/port
# - 集合名
# - 向量字段
# - 文本字段
MILVUS_HOST = os.getenv("MILVUS_HOST", "121.37.90.146")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "kb_chunks")
TOP_K = int(os.getenv("MILVUS_TOP_K", "8"))
FINAL_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))


def _pick_vector_and_text_field(collection: Any) -> Tuple[str, int, str]:
    """从集合 schema 里找出向量字段、维度和文本字段。"""
    vector_field = None
    vector_dim = None
    text_field = None

    for field in collection.schema.fields:
        dtype = getattr(field, "dtype", None)
        field_name = field.name

        if int(dtype) == int(DataType.FLOAT_VECTOR):
            vector_field = field_name
            vector_dim = int(getattr(field, "params", {}).get("dim", 0) or 0)

        if field_name in {"content", "chunk_text", "text", "page_content", "document"}:
            text_field = field_name

    if not vector_field or not vector_dim:
        raise RuntimeError("未找到向量字段")
    if not text_field:
        raise RuntimeError("未找到文本字段")

    return vector_field, vector_dim, text_field


def get_query_embedding(text: str, dim: int) -> List[float]:
    """
    调用本地 embedding API，把 query 编码成向量。

    说明：
    - 你当前使用的是本地 OpenAI-compatible embedding 服务。
    - 这里要尽量保证 embedding 模型与入库时使用的一致，否则召回质量会下降。
    """
    client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vec = response.data[0].embedding
    if len(vec) != dim:
        raise RuntimeError(
            f"embedding 维度不匹配: 模型返回 {len(vec)} 维，但 Milvus 集合需要 {dim} 维"
        )
    return list(vec)


def _tokenize(text: str) -> List[str]:
    """一个非常简单的分词器，用于教学版 rerank。"""
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_\-]+", text.lower())


def simple_rerank(question: str, contexts: List[str], top_k: int = FINAL_TOP_K) -> List[Tuple[float, str]]:
    """
    一个“教学用”的简单 rerank。

    原理：
    - 先由 Milvus 做向量召回（粗召回）
    - 再用一个更细粒度的规则，对召回结果重新排序（精排）

    当前这个最小版本没有调用专门的 cross-encoder，
    而是使用“query-token 与 chunk-token 的重合度”来模拟 rerank 过程。

    生产上常见替代方案：
    - bge-reranker
    - jina-reranker
    - cross-encoder/ms-marco-*
    """
    q_tokens = set(_tokenize(question))
    scored: List[Tuple[float, str]] = []

    for ctx in contexts:
        c_tokens = set(_tokenize(ctx))
        overlap = len(q_tokens & c_tokens)
        coverage = overlap / max(1, len(q_tokens))

        # 一个非常简单的教学分数：
        # - query 命中的 token 数越多越好
        # - 覆盖率越高越好
        score = overlap + coverage
        scored.append((score, ctx))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(1, top_k)]


def retrieve_contexts(question: str, top_k: int = TOP_K) -> List[str]:
    """
    用 Milvus 检索和问题最相关的文本片段。

    这一步就是 RAG 和“纯 LLM 调用”最大的区别：
    在把问题发给模型之前，先去知识库找证据。
    """
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    vector_field, vector_dim, text_field = _pick_vector_and_text_field(collection)

    # 真实 RAG 的第一步：先把 query 编码成向量，再去向量库检索。
    query_vector = get_query_embedding(question, dim=vector_dim)

    metric_type = "COSINE"
    if collection.indexes:
        metric_type = collection.indexes[0].params.get("metric_type", metric_type)

    search_result = collection.search(
        data=[query_vector],
        anns_field=vector_field,
        param={"metric_type": metric_type, "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=[text_field],
    )

    hits = search_result[0] if search_result else []
    contexts: List[str] = []
    for hit in hits:
        entity = getattr(hit, "entity", None)
        if entity is None:
            continue
        text = entity.get(text_field)
        if isinstance(text, str) and text.strip():
            contexts.append(text.strip())

    return contexts


def build_rag_prompt(question: str, contexts: List[str]) -> str:
    """把问题和检索证据拼接成给模型的最终 prompt。"""
    evidence_block = "\n\n".join(
        [f"[证据 {i}]\n{ctx}" for i, ctx in enumerate(contexts, start=1)]
    )

    return f"""
你是一个问答助手。请严格依据给定证据回答问题。
如果证据不足，请直接说“证据不足”，不要编造。

用户问题：
{question}

检索证据：
{evidence_block}

请按下面格式回答：
【结论】...
【证据】...
【不确定性】...
""".strip()


def call_llm(prompt: str) -> str:
    """调用本地 OpenAI-compatible LLM。"""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个严谨的 RAG 问答助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def build_plain_prompt(question: str) -> str:
    """构造“纯 LLM 直答”版本的 prompt。"""
    return f"""
你是一个问答助手，请直接回答用户问题。
如果你不确定，可以说明不确定性，但不要编造引用来源。

用户问题：
{question}

请按下面格式回答：
【结论】...
【依据】...
【不确定性】...
""".strip()


def ask_plain_llm(question: str) -> None:
    """
    纯 LLM 问答流程：
    1. 输入问题
    2. 不做检索
    3. 直接调用 LLM 回答
    """
    print("\n=== A) 纯 LLM：用户问题 ===")
    print(question)

    prompt = build_plain_prompt(question)

    print("\n=== B) 纯 LLM：发给模型的 prompt（截断预览） ===")
    preview_prompt = prompt[:500] + ("..." if len(prompt) > 500 else "")
    print(preview_prompt)

    answer = call_llm(prompt)

    print("\n=== C) 纯 LLM：最终回答 ===")
    print(answer)


def ask_with_rag(question: str) -> None:
    """
    完整最小 RAG 流程：
    1. 输入问题
    2. 去 Milvus 检索相关证据
    3. 把证据和问题一起发给 LLM
    4. 打印最终回答
    """
    print("\n=== 1) 用户问题 ===")
    print(question)

    raw_contexts = retrieve_contexts(question)

    print("\n=== 2) Milvus 初次召回结果（recall） ===")
    if not raw_contexts:
        print("未检索到上下文")
        return

    for i, ctx in enumerate(raw_contexts, start=1):
        preview = ctx.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(f"[{i}] {preview}")

    reranked = simple_rerank(question, raw_contexts, top_k=FINAL_TOP_K)
    contexts = [ctx for _, ctx in reranked]

    print("\n=== 3) 简单 rerank 后保留的证据 ===")
    for i, (score, ctx) in enumerate(reranked, start=1):
        preview = ctx.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(f"[{i}] score={score:.3f} | {preview}")

    prompt = build_rag_prompt(question, contexts)

    print("\n=== 4) 发给 LLM 的 prompt（截断预览） ===")
    preview_prompt = prompt[:800] + ("..." if len(prompt) > 800 else "")
    print(preview_prompt)

    answer = call_llm(prompt)

    print("\n=== 5) 最终回答 ===")
    print(answer)


if __name__ == "__main__":
    # 你可以把这里替换成自己的问题
    demo_question = "RAG 系统里，Milvus 的作用是什么？"
    ask_plain_llm(demo_question)
    print("\n" + "#" * 72)
    print("# 下面是同一个问题的 RAG 回答")
    print("#" * 72)
    ask_with_rag(demo_question)
