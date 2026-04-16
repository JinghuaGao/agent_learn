from __future__ import annotations

"""
最小 RAG 示例（本地 LLM + Milvus）

你可以把它和“纯 LLM 调用”对比着看：

纯 LLM：
    question -> LLM -> answer

加上 RAG：
    question -> embedding -> Milvus 检索 -> retrieved_contexts -> LLM -> answer

再进一步细化成今天这版的完整链路：

    question -> embedding -> Milvus 召回(top-k) -> reranker 精排 -> 最终证据 -> LLM -> answer

这个脚本故意写得很直白，目标是帮助理解流程，不追求工程完整性。
重点不是“写得最优雅”，而是把每一步为什么存在、输入输出是什么讲清楚。

默认配置：
- 本地 LLM 服务: http://121.37.90.146:1433/v1
- 本地模型: Qwen3.5-35B-A3B
- Milvus 服务: 121.37.90.146:19530
- Milvus 集合: kb_chunks
"""

import os
import re
import json
from urllib import error as urllib_error
from urllib import request as urllib_request
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
LLM_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://121.37.90.146:1433/v1")
LLM_MODEL = os.getenv("LOCAL_MODEL", "Qwen3.5-35B-A3B")
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
# 1.2) Reranker 配置
# -----------------------------
# reranker 和 embedding 不是一回事：
# - embedding 用于召回（把问题/文档编码成向量，再做相似度搜索）
# - reranker 用于精排（直接判断 query 和候选 chunk 谁更相关）
RERANK_BASE_URL = os.getenv("LOCAL_RERANK_BASE_URL", "http://host.docker.internal:55005")
RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL_NAME", "bge-reranker-v2-m3")


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
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ja_dw_kb_chunks")
TOP_K = int(os.getenv("MILVUS_TOP_K", "8"))
FINAL_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))


def _pick_vector_and_text_field(collection: Any) -> Tuple[str, int, str]:
    """从集合 schema 里找出向量字段、维度和文本字段。

    这里先“读 schema”，是因为 Milvus 一个 collection 里可能有很多字段，
    例如：主键、时间戳、原文、向量、标签等。
    RAG 检索时必须先知道：
    - 哪个字段存的是 embedding 向量
    - 这个向量是多少维
    - 哪个字段存的是可读文本
    """
    vector_field = None
    vector_dim = None
    text_field = None

    for field in collection.schema.fields:
        # collection.schema.fields 就是这个集合的字段定义列表。
        # 这里逐个字段检查，找出：
        # 1) FLOAT_VECTOR 类型的向量字段
        # 2) 可能存 chunk 原文的文本字段
        dtype = getattr(field, "dtype", None)
        field_name = field.name

        if int(dtype) == int(DataType.FLOAT_VECTOR):
            # FLOAT_VECTOR 表示这是一个向量字段。
            # params 里通常会带 dim，也就是向量维度。
            vector_field = field_name
            vector_dim = int(getattr(field, "params", {}).get("dim", 0) or 0)

        if field_name in {"content", "chunk_text", "text", "page_content", "document"}:
            # 这些名字只是常见的文本字段命名习惯，
            # 实际项目里你可以按自己的 schema 改。
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


def rerank_contexts(question: str, contexts: List[str], top_k: int = FINAL_TOP_K) -> List[Tuple[float, str]]:
    """调用本地 reranker 服务，对 Milvus 召回结果做精排。

    这个版本优先走真正的 reranker 服务；如果本地服务不可用，
    会回退到教学版的 simple_rerank，保证 demo 还能跑通。

    可以把它理解成：
    - 输入：query + 候选 chunks
    - 处理：逐个判断“这个 chunk 是否真的在回答 query”
    - 输出：按相关性重新排序后的 chunk 列表

    这里采用的是 Cohere / OpenRouter 风格的请求体：
    - model: reranker 模型名
    - query: 用户问题
    - documents: 候选 chunk 列表
    - top_n: 只返回前 top_n 条
    """
    if not contexts:
        return []

    payload = {
        "model": RERANK_MODEL,
        "query": question,
        "documents": contexts,
        "top_n": max(1, top_k),
    }

    url = f"{RERANK_BASE_URL.rstrip('/')}/rerank"
    req = urllib_request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=60) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw_text)

        results = data.get("results", []) if isinstance(data, dict) else []
        reranked: List[Tuple[float, str]] = []

        for item in results:
            if not isinstance(item, dict):
                continue

            idx = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(contexts):
                continue

            reranked.append((float(score), contexts[idx]))

        if reranked:
            return reranked[: max(1, top_k)]

        print("[warn] reranker 返回结果为空，回退到 simple_rerank")
        return simple_rerank(question, contexts, top_k=top_k)

    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[warn] reranker 调用失败，回退到 simple_rerank: {exc}")
        return simple_rerank(question, contexts, top_k=top_k)


def retrieve_contexts(question: str, top_k: int = TOP_K) -> List[str]:
    """
    用 Milvus 检索和问题最相关的文本片段。

    这一步就是 RAG 和“纯 LLM 调用”最大的区别：
    在把问题发给模型之前，先去知识库找证据。

    这个阶段做的是“召回”，不是最终排序：
    - 用 embedding 把 query 变成向量
    - 用 Milvus 在向量空间里先找一批候选 chunk
    - 再把候选交给 reranker 精排
    """
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    # 连接上 Milvus 之后，先打开目标 collection。
    # Collection(...) 不是“搜索结果”，而是“表 / 集合对象”，
    # 后面可以从它的 schema、index、search 方法里取信息。
    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    # 先从 schema 里自动识别：
    # - 向量字段叫什么
    # - 向量多少维
    # - 文本字段叫什么
    # 这样脚本对不同知识库结构会更通用。
    vector_field, vector_dim, text_field = _pick_vector_and_text_field(collection)

    # 真实 RAG 的第一步：先把 query 编码成向量，再去向量库检索。
    query_vector = get_query_embedding(question, dim=vector_dim)

    metric_type = "COSINE"
    if collection.indexes:
        # 如果这个 collection 已经建过 index，就从 index 配置里读 metric_type。
        # 这样可以尽量跟入库时的索引配置保持一致。
        metric_type = collection.indexes[0].params.get("metric_type", metric_type)

    # 这里的 collection.search(...) 就是 Milvus 的核心向量检索调用。
    # 含义可以拆成几部分：
    # - data=[query_vector]：把“查询向量”传进去
    # - anns_field=vector_field：告诉 Milvus，用哪个向量字段去比
    # - param={...}：检索参数，包含相似度度量方式和搜索参数
    # - limit=top_k：只返回最相似的前 top_k 条
    # - output_fields=[text_field]：把命中的记录里对应的文本字段也带回来
    #
    # 也就是说：Milvus 返回的不是“纯向量”，而是“匹配到的记录”，
    # 你后面可以直接拿其中的文本字段拼进 prompt。
    search_result = collection.search(
        data=[query_vector],
        anns_field=vector_field,
        param={"metric_type": metric_type, "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=[text_field],
    )

    # search_result[0] 是这一条 query 的搜索结果列表。
    # 里面每个 hit 代表一条最相近的记录。
    hits = search_result[0] if search_result else []
    contexts: List[str] = []
    for hit in hits:
        # 每个 hit 里会带 entity，也就是被命中的那条记录本身。
        entity = getattr(hit, "entity", None)
        if entity is None:
            continue
        text = entity.get(text_field)
        if isinstance(text, str) and text.strip():
            contexts.append(text.strip())

    return contexts


def inspect_milvus_collection(
    collection_name: str = MILVUS_COLLECTION,
    host: str = MILVUS_HOST,
    port: str = MILVUS_PORT,
) -> dict:
    """不走大模型，直接查看 Milvus 集合的基本信息。

    这个小函数适合做“Milvus 信息测试 / 自检”，主要看三类东西：
    - 集合是否能连上
    - schema 里有哪些字段、字段类型是什么
    - 目前一共有多少条记录（通常也可以理解为多少条向量记录）

    返回一个字典，方便你后面打印或保存结果。
    """
    connections.connect(alias="default", host=host, port=port)
    collection = Collection(collection_name)

    # 这里不需要调用 LLM，也不需要 embedding。
    # 只是读取 Milvus 元数据，看看这个集合长什么样。
    info: dict = {
        "collection_name": collection_name,
        "host": host,
        "port": port,
        "num_entities": collection.num_entities,
        "schema": [],
        "indexes": [],
    }

    for field in collection.schema.fields:
        field_info = {
            "name": field.name,
            "dtype": str(getattr(field, "dtype", "unknown")),
            "is_primary": bool(getattr(field, "is_primary", False)),
            "auto_id": bool(getattr(field, "auto_id", False)),
            "params": dict(getattr(field, "params", {}) or {}),
        }
        info["schema"].append(field_info)

    for idx in collection.indexes:
        info["indexes"].append(
            {
                "field_name": getattr(idx, "field_name", None),
                "index_type": getattr(idx, "index_type", None),
                "params": dict(getattr(idx, "params", {}) or {}),
            }
        )

    vector_fields = [
        f
        for f in collection.schema.fields
        if int(getattr(f, "dtype", -1)) == int(DataType.FLOAT_VECTOR)
    ]
    info["vector_field_count"] = len(vector_fields)
    info["vector_fields"] = [f.name for f in vector_fields]

    print("\n=== Milvus 集合信息 ===")
    print(f"collection: {info['collection_name']}")
    print(f"address: {info['host']}:{info['port']}")
    print(f"num_entities: {info['num_entities']}")
    print(f"vector_field_count: {info['vector_field_count']}")
    print(f"vector_fields: {info['vector_fields']}")

    print("\n--- schema ---")
    for field in info["schema"]:
        print(
            f"- {field['name']}: dtype={field['dtype']}, "
            f"primary={field['is_primary']}, auto_id={field['auto_id']}, params={field['params']}"
        )

    if info["indexes"]:
        print("\n--- indexes ---")
        for idx in info["indexes"]:
            print(
                f"- field={idx['field_name']}, type={idx['index_type']}, params={idx['params']}"
            )
    else:
        print("\n--- indexes ---")
        print("(no index found)")

    return info


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
    3. 交给 reranker 重新排序
    4. 把证据和问题一起发给 LLM
    5. 打印最终回答
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

    reranked = rerank_contexts(question, raw_contexts, top_k=FINAL_TOP_K)
    contexts = [ctx for _, ctx in reranked]

    print("\n=== 3) rerank 后保留的证据 ===")
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

    only_inspect=0
    if only_inspect == True: #仅仅查看milvus 服务的属性
        inspect_milvus_collection()

    else: #直接测试rag
        # 你可以把这里替换成自己的问题
        demo_question = "司法记录器是什么？它的功能和应用场景有哪些？"
        ask_plain_llm(demo_question)
        print("\n" + "#" * 72)
        print("# 下面是同一个问题的 RAG 回答")
        print("#" * 72)
        ask_with_rag(demo_question)