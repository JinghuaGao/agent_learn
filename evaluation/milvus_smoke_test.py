from __future__ import annotations

import argparse
import random
import socket
import sys
from typing import Any


def _print_header(title: str) -> None:
    print("\n" + "=" * 18 + f" {title} " + "=" * 18)


def _pick_vector_field(collection: Any) -> tuple[str, int]:
    try:
        from pymilvus import DataType  # type: ignore
        float_vector_code = int(DataType.FLOAT_VECTOR)
    except Exception:
        float_vector_code = 101

    for f in collection.schema.fields:
        dtype_name = str(getattr(f, "dtype", ""))
        dtype_val = getattr(f, "dtype", None)
        try:
            dtype_code = int(dtype_val)
        except Exception:
            dtype_code = None

        if "FLOAT_VECTOR" in dtype_name or dtype_code == float_vector_code:
            dim = int(getattr(f, "params", {}).get("dim", 0) or 0)
            return f.name, dim
    raise RuntimeError("集合中未找到 FLOAT_VECTOR 字段")


def _pick_text_fields(collection: Any) -> list[str]:
    candidates = ["chunk_text", "text", "content", "page_content", "document"]
    names = [f.name for f in collection.schema.fields]

    picked = [c for c in candidates if c in names]
    if picked:
        return picked

    # 兜底：挑所有字符串字段
    out: list[str] = []
    for f in collection.schema.fields:
        dtype_name = str(getattr(f, "dtype", "")).upper()
        if "VARCHAR" in dtype_name or "STRING" in dtype_name:
            out.append(f.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus 连通性与集合内容冒烟测试")
    parser.add_argument("--host", default="121.37.90.146", help="Milvus host")
    parser.add_argument("--port", default="19530", help="Milvus port")
    parser.add_argument("--collection", default="kb_chunks", help="要检查的集合名")
    parser.add_argument("--top-k", type=int, default=3, help="随机向量检索返回条数")
    args = parser.parse_args()

    # 1) TCP 连通性
    _print_header("TCP 连通性")
    try:
        with socket.create_connection((args.host, int(args.port)), timeout=5):
            print(f"✅ TCP 可达: {args.host}:{args.port}")
    except Exception as exc:
        print(f"❌ TCP 不可达: {args.host}:{args.port} -> {exc}")
        return 2

    # 2) SDK 层检查
    try:
        from pymilvus import Collection, connections, utility
    except Exception as exc:
        print(f"❌ 未安装/无法导入 pymilvus: {exc}")
        return 3

    _print_header("Milvus 连接")
    try:
        connections.connect(alias="default", host=args.host, port=args.port)
        print("✅ SDK 连接成功")
    except Exception as exc:
        print(f"❌ SDK 连接失败: {exc}")
        return 4

    # 3) 服务与集合概览
    _print_header("服务与集合概览")
    try:
        version = utility.get_server_version()
        print(f"server_version: {version}")
    except Exception as exc:
        print(f"server_version: (读取失败) {exc}")

    try:
        collections = utility.list_collections()
        print(f"collections({len(collections)}): {collections}")
    except Exception as exc:
        print(f"❌ 列出集合失败: {exc}")
        return 5

    if not utility.has_collection(args.collection):
        print(f"❌ 集合不存在: {args.collection}")
        return 6

    # 4) 目标集合详细信息
    _print_header(f"集合详情: {args.collection}")
    try:
        coll = Collection(args.collection)
        print(f"name: {coll.name}")
        print(f"description: {getattr(coll.schema, 'description', '')}")

        print("fields:")
        for f in coll.schema.fields:
            print(f"- {f.name}: dtype={f.dtype}, params={getattr(f, 'params', {})}, primary={getattr(f, 'is_primary', False)}")

        try:
            print(f"num_entities: {coll.num_entities}")
        except Exception as exc:
            print(f"num_entities: (读取失败) {exc}")

        try:
            print(f"indexes: {[idx.params for idx in coll.indexes]}")
        except Exception as exc:
            print(f"indexes: (读取失败) {exc}")

    except Exception as exc:
        print(f"❌ 读取集合详情失败: {exc}")
        return 7

    # 5) 随机向量检索，验证“能查到内容”
    _print_header("随机向量检索冒烟")
    try:
        vec_field, dim = _pick_vector_field(coll)
        if dim <= 0:
            raise RuntimeError(f"向量字段 {vec_field} 的 dim 非法: {dim}")
        text_fields = _pick_text_fields(coll)

        coll.load()
        probe = [random.uniform(-0.1, 0.1) for _ in range(dim)]
        metric_type = "IP"
        try:
            if coll.indexes:
                metric_type = coll.indexes[0].params.get("metric_type", metric_type)
        except Exception:
            pass

        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}

        hits = coll.search(
            data=[probe],
            anns_field=vec_field,
            param=search_params,
            limit=max(1, args.top_k),
            output_fields=text_fields,
        )

        first = hits[0] if hits else []
        print(f"✅ 检索成功: hits={len(first)}, vec_field={vec_field}, dim={dim}, metric={metric_type}")

        for i, h in enumerate(first, start=1):
            print(f"\n[{i}] id={h.id}, distance={h.distance}")
            if not text_fields:
                continue
            entity = getattr(h, "entity", None)
            if entity is None:
                continue
            for tf in text_fields:
                val = entity.get(tf)
                if isinstance(val, str) and val.strip():
                    preview = val.strip().replace("\n", " ")
                    if len(preview) > 180:
                        preview = preview[:180] + "..."
                    print(f"  {tf}: {preview}")
    except Exception as exc:
        print(f"❌ 检索测试失败: {exc}")
        return 8

    _print_header("结论")
    print("✅ Milvus 服务可连通，集合可读，检索链路可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
