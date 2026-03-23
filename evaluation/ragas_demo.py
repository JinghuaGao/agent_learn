from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics.collections import context_precision, context_recall, faithfulness
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall


# 手工样本（保留用于对比）
DEMO_ROWS: Dict[str, List[Any]] = {
    "user_input": [
        "RAG 系统里，向量检索器的主要作用是什么？",
        "为什么在 Agent-RAG 场景要保留工具调用日志？",
    ],
    "response": [
        "向量检索器把问题和文档都映射到向量空间，按相似度召回相关片段，给大模型提供证据上下文。",
        "因为日志能记录每一步工具调用和结果，便于复盘错误、定位幻觉，并做稳定性优化。",
    ],
    "reference": [
        "向量检索器通过向量相似度从知识库中检索与问题最相关的文本片段，作为生成答案的证据上下文。",
        "在 Agent-RAG 中保留工具调用日志有助于追踪决策路径、复盘失败案例、识别伪调用，并支持评估与调优。",
    ],
    "retrieved_contexts": [
        [
            "向量检索的核心是 embedding 相似度匹配。",
            "检索阶段负责提供上下文证据，生成阶段负责组织答案。",
        ],
        [
            "工具调用日志可用于观测 request_id、调用参数、返回结果与耗时。",
            "没有日志时很难定位工具调用幻觉与协议漂移问题。",
        ],
    ],
    # Non-LLM 指标（如 NonLLMContextPrecisionWithReference）需要该列
    # 这里先用简化版 gold context（每条一个参考证据片段）
    "reference_contexts": [
        [
            "向量检索器通过向量相似度检索相关文本片段，作为生成答案的证据上下文。",
        ],
        [
            "保留工具调用日志有助于追踪决策路径、复盘失败案例并识别伪调用。",
        ],
    ],
}


def _get_api_key() -> str:
    api_key = (
        os.getenv("NV_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "未找到 API Key，请设置 NV_API_KEY / NVIDIA_API_KEY / OPENAI_API_KEY 其一。"
        )
    return api_key


def _get_api_key_optional() -> str | None:
    return (
        os.getenv("NV_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def build_demo_dataset_manual() -> Dataset:
    """
    手工构造评估集（你之前的版本，保留用于对比）。
    ragas 0.4.x 默认列名：
    - user_input / response / reference / retrieved_contexts
    """
    return Dataset.from_dict(DEMO_ROWS)


def _simple_hash_embedding(text: str, dim: int = 1024) -> List[float]:
    """
    纯 Python 的演示用向量化函数（非生产）。
    用于在没有接入真实 embedding 模型时完成 Milvus 检索流程演示。
    """
    vec = [0.0] * dim
    for token in text.lower().split():
        idx = hash(token) % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _pick_text_field(collection: Any) -> str:
    candidates = ["chunk_text", "text", "content", "page_content", "document"]
    names = {f.name for f in collection.schema.fields}
    for c in candidates:
        if c in names:
            return c
    # 兜底：第一个字符串字段
    for f in collection.schema.fields:
        if str(getattr(f, "dtype", "")).lower().endswith("varchar"):
            return f.name
    raise RuntimeError("未在集合中找到可用文本字段（如 chunk_text/text/content）。")


def _pick_vector_field(collection: Any) -> str:
    for f in collection.schema.fields:
        if "FLOAT_VECTOR" in str(getattr(f, "dtype", "")):
            return f.name
    raise RuntimeError("未在集合中找到 FLOAT_VECTOR 向量字段。")


def build_demo_dataset_from_milvus(
    host: str = "121.37.90.146",
    port: str = "19530",
    collection_name: str = "kb_chunks",
    top_k: int = 2,
    vector_dim: int = 1024,
) -> Dataset:
    """
    从 Milvus 检索真实 contexts 来构造评估集。

    说明：
    - 为了“可直接跑通”，这里默认用 `_simple_hash_embedding` 生成查询向量。
    - 生产建议替换为与你入库时一致的 embedding 模型，否则召回质量会受限。
    """
    try:
        from pymilvus import Collection, connections, utility
    except Exception as exc:
        raise RuntimeError("请先安装 pymilvus：pip install pymilvus") from exc

    connections.connect(alias="default", host=host, port=port)
    if not utility.has_collection(collection_name):
        raise RuntimeError(f"Milvus 中不存在集合: {collection_name}")

    coll = Collection(collection_name)
    coll.load()

    text_field = _pick_text_field(coll)
    vector_field = _pick_vector_field(coll)

    rows = {
        "user_input": list(DEMO_ROWS["user_input"]),
        "response": list(DEMO_ROWS["response"]),
        "reference": list(DEMO_ROWS["reference"]),
        "retrieved_contexts": [],
        "reference_contexts": list(DEMO_ROWS["reference_contexts"]),
    }

    for question in rows["user_input"]:
        qvec = _simple_hash_embedding(question, dim=vector_dim)
        try:
            search_result = coll.search(
                data=[qvec],
                anns_field=vector_field,
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=[text_field],
            )
            hits = search_result[0] if search_result else []
            contexts = []
            for h in hits:
                txt = h.entity.get(text_field) if h.entity else None
                if isinstance(txt, str) and txt.strip():
                    contexts.append(txt.strip())

            if not contexts:
                contexts = ["(Milvus 未检索到有效文本，回退为空上下文)"]
        except Exception as exc:
            contexts = [f"(Milvus 检索失败: {exc})"]

        rows["retrieved_contexts"].append(contexts)

    return Dataset.from_dict(rows)


def build_demo_dataset() -> Dataset:
    """
    保持原函数名，默认仍返回手工数据，避免你看不明白。
    如需切换 Milvus，可在 main() 里设置 DATASET_SOURCE=milvus。
    """
    return build_demo_dataset_manual()


def main() -> None:
    api_key = _get_api_key_optional()
    evaluator_llm = None
    use_local_model = os.getenv("USE_LOCAL_MODEL", "0") == "1"

    dataset_source = os.getenv("DATASET_SOURCE", "manual").strip().lower()
    if dataset_source == "milvus":
        dataset = build_demo_dataset_from_milvus(
            host=os.getenv("MILVUS_HOST", "121.37.90.146"),
            port=os.getenv("MILVUS_PORT", "19530"),
            collection_name=os.getenv("MILVUS_COLLECTION", "kb_chunks"),
            top_k=int(os.getenv("MILVUS_TOP_K", "2")),
            vector_dim=int(os.getenv("MILVUS_VECTOR_DIM", "1024")),
        )
        print("[dataset] 使用 Milvus 检索上下文构建评估集")
    else:
        dataset = build_demo_dataset_manual()
        print("[dataset] 使用手工上下文构建评估集")

    if use_local_model:
        local_base_url = os.getenv("LOCAL_BASE_URL", "http://121.37.90.146:55803/v1")
        local_model = os.getenv("LOCAL_MODEL", "qwen3-32b")
        local_api_key = os.getenv("LOCAL_API_KEY", "EMPTY")

        evaluator_llm = ChatOpenAI(
            model=local_model,
            api_key=local_api_key,
            base_url=local_base_url,
            temperature=0,
        )

        print("[eval] 使用本地 OpenAI-compatible 模型进行 LLM 指标评估")
        print(f"[eval] base_url={local_base_url}, model={local_model}")
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, context_recall],
            llm=evaluator_llm,
            show_progress=True,
        )
    elif api_key:
        base_url = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
        model = os.getenv("RAGAS_EVAL_MODEL", os.getenv("OPENAI_MODEL", "minimaxai/minimax-m2.5"))

        evaluator_llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )

        print("[eval] 检测到 API Key，使用远端 LLM 指标: faithfulness/context_precision/context_recall")
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, context_recall],
            llm=evaluator_llm,
            show_progress=True,
        )
    else:
        print("[eval] 未检测到 API Key，自动切换到 Non-LLM 指标")
        print("[eval] 当前指标: non_llm_context_precision_with_reference / non_llm_context_recall")
        result = evaluate(
            dataset=dataset,
            metrics=[
                NonLLMContextPrecisionWithReference(),
                NonLLMContextRecall(),
            ],
            show_progress=True,
        )

    df = result.to_pandas()
    print("\n=== RAGAS DEMO RESULT ===")
    print(df)

    out_file = Path(__file__).with_name("ragas_demo_result.csv")
    df.to_csv(out_file, index=False)
    print(f"\n结果已保存: {out_file}")


if __name__ == "__main__":
    main()
