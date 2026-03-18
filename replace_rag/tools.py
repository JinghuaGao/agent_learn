import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore


DEFAULT_REFER_DOCS = "/Users/jiean/agent_learn/refer_docs"
DEFAULT_METADATA_INDEX = f"{DEFAULT_REFER_DOCS}/pdf_metadata_index.json"


def _safe_resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


async def fs_pwd(path: str = ".") -> str:
    """
    Tool API: fs_pwd

    功能:
    - 返回当前/指定路径的绝对路径（类似 pwd）。

    参数:
    - path (str, optional): 目标路径。默认值 "."。

    返回:
    - str:
        - 成功: "pwd: <绝对路径>"
        - 失败: "路径不存在: <绝对路径>"

    示例:
    - 输入: fs_pwd(".")
    - 输出: "pwd: /Users/jiean/agent_learn/replace_rag"
    """
    p = _safe_resolve(path)
    if not p.exists():
        return f"路径不存在: {p}"
    return f"pwd: {p}"


async def fs_list_tree(root_path: str = ".", max_depth: int = 3, max_results: int = 300) -> str:
    """
    Tool API: fs_list_tree

    功能:
    - 递归查看目录结构（受限深度、受限结果数）。

    参数:
    - root_path (str, optional): 根目录路径。默认 "."。
    - max_depth (int, optional): 最大递归深度。默认 3。
    - max_results (int, optional): 最大输出行数。默认 300。

    返回:
    - str:
        - 成功: 树形结构文本（目录/文件逐行展示）
        - 失败: "目录不存在: <绝对路径>"

    示例:
    - 输入: fs_list_tree("/Users/jiean/agent_learn/refer_docs", max_depth=1)
    - 输出: "refer_docs/\n  pdf_metadata_index.json\n  ..."

    备注:
    - 若超出 max_results，会在尾部追加 "...(truncated)..."。
    """
    root = _safe_resolve(root_path)
    if not root.exists() or not root.is_dir():
        return f"目录不存在: {root}"

    root_depth = len(root.parts)
    items: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        depth = len(d.parts) - root_depth
        if depth > max_depth:
            continue

        indent = "  " * depth
        items.append(f"{indent}{d.name}/")
        for name in sorted(filenames):
            items.append(f"{indent}  {name}")

        if len(items) >= max_results:
            break

    if len(items) > max_results:
        items = items[:max_results] + ["...(truncated)..."]
    return "\n".join(items)


async def fs_read(file_path: str, max_chars: int = 20000, page_start: int = 1, page_end: int = 3) -> str:
    """
    Tool API: fs_read

    功能:
    - 读取文本文件或 PDF 文件内容。

    参数:
    - file_path (str): 文件路径。
    - max_chars (int, optional): 返回内容最大字符数。默认 20000。
    - page_start (int, optional): PDF 起始页（1-based，含）。默认 1。
    - page_end (int, optional): PDF 结束页（1-based，含）。默认 3。

    返回:
    - str:
        - 文本文件: 原文（可能截断）
        - PDF: "file=..., pages=..." + 各页文本
        - 失败: "文件不存在..." / "PDF 读取失败..." / "文本读取失败..."

    示例:
    - 输入: fs_read("/path/a.txt", max_chars=1000)
    - 输出: "<文件前 1000 字符>..."
    - 输入: fs_read("/path/a.pdf", page_start=1, page_end=2)
    - 输出: "file=a.pdf, pages=1-2/10\n[Page 1]..."
    """
    p = _safe_resolve(file_path)
    if not p.exists() or not p.is_file():
        return f"文件不存在: {p}"

    if p.suffix.lower() == ".pdf":
        if PdfReader is None:
            return "依赖缺失: pypdf 未安装，无法读取 PDF。请先安装 pypdf。"
        try:
            reader = PdfReader(str(p))
            total = len(reader.pages)
            if total <= 0:
                return f"PDF 无页面内容: {p.name}"
            start = max(1, page_start)
            end = min(max(start, page_end), total)
            parts: List[str] = [f"file={p.name}, pages={start}-{end}/{total}"]
            for i in range(start - 1, end):
                text = (reader.pages[i].extract_text() or "").strip()
                parts.append(f"\n[Page {i + 1}]\n{text or '(empty)'}")
            out = "\n".join(parts)
            if len(out) > max_chars:
                out = out[:max_chars] + "\n...(truncated)..."
            return out
        except Exception as exc:
            return f"PDF 读取失败: {p.name} -> {exc}"

    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "\n...(truncated)..."
        return txt
    except Exception as exc:
        return f"文本读取失败: {p.name} -> {exc}"


async def fs_edit_json(
    file_path: str,
    key_path: str,
    value_json: str,
    create_missing: bool = True,
) -> str:
    """
    Tool API: fs_edit_json

    功能:
    - 按点分路径更新 JSON 字段值，支持按需创建中间对象。

    参数:
    - file_path (str): JSON 文件路径。
    - key_path (str): 点分路径，如 "a.b.c"。
    - value_json (str): 新值（必须是合法 JSON 字符串）。
    - create_missing (bool, optional): 是否创建缺失的中间对象。默认 True。

    返回:
    - str:
        - 成功: "JSON 更新成功: ... | path=... | old=... | new=..."
        - 失败: 错误信息（路径不可达、解析失败、文件类型不符等）

    示例:
    - 输入: fs_edit_json("a.json", "config.top_k", "3")
    - 输出: "JSON 更新成功: a.json | path=config.top_k | old=1 | new=3"

    备注:
    - 当前仅支持对象路径，不支持数组索引路径。
    """
    p = _safe_resolve(file_path)
    if not p.exists() or p.suffix.lower() != ".json":
        return f"无效 JSON 文件: {p}"

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"JSON 解析失败: {p.name} -> {exc}"

    try:
        value = json.loads(value_json)
    except Exception as exc:
        return f"value_json 不是合法 JSON: {exc}"

    keys = [k for k in key_path.split(".") if k]
    if not keys:
        return "key_path 不能为空"

    cur = data
    for k in keys[:-1]:
        if isinstance(cur, dict):
            if k not in cur:
                if not create_missing:
                    return f"路径不存在: {key_path}"
                cur[k] = {}
            if not isinstance(cur[k], dict):
                return f"中间路径不是对象: {k}"
            cur = cur[k]
        else:
            return f"路径不可达: {k}"

    last = keys[-1]
    if not isinstance(cur, dict):
        return f"目标父节点不是对象: {'.'.join(keys[:-1])}"

    old = cur.get(last, None)
    cur[last] = value

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

    return f"JSON 更新成功: {p.name} | path={key_path} | old={old} | new={value}"


def _tokenize_query(text: str) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]{1,}|[a-zA-Z0-9_\-]{2,}", text.lower())
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
        "what", "when", "where", "which", "how", "why", "is", "to", "of", "in", "on",
        "一个", "一些", "这个", "那个", "什么", "如何", "以及", "或者", "可以", "我们", "你们",
    }
    return [t for t in tokens if t not in stop and len(t) > 1]


def _score_page(text: str, query_tokens: List[str]) -> float:
    if not text:
        return 0.0
    text_low = text.lower()
    token_hits = 0
    raw_score = 0.0
    for token in query_tokens:
        c = text_low.count(token)
        if c > 0:
            token_hits += 1
            raw_score += min(c, 5)
    if token_hits == 0:
        return 0.0
    coverage = token_hits / max(1, len(query_tokens))
    return raw_score * (1.0 + coverage)


def _load_metadata_items(metadata_path: str) -> List[Dict[str, Any]]:
    p = _safe_resolve(metadata_path)
    if not p.exists() or p.suffix.lower() != ".json":
        raise ValueError(f"无效 metadata JSON 路径: {p}")

    payload = json.loads(p.read_text(encoding="utf-8"))
    raw_items = payload.get("items", {})
    items: List[Dict[str, Any]] = []

    if isinstance(raw_items, dict):
        for file_path, meta in raw_items.items():
            if isinstance(meta, dict):
                item = dict(meta)
                item.setdefault("file_path", file_path)
                item.setdefault("file_name", Path(file_path).name)
                items.append(item)
    elif isinstance(raw_items, list):
        for meta in raw_items:
            if isinstance(meta, dict):
                item = dict(meta)
                fp = item.get("file_path") or item.get("path")
                if isinstance(fp, str):
                    item.setdefault("file_name", Path(fp).name)
                items.append(item)
    else:
        raise ValueError("metadata.items 既不是对象也不是数组")

    return items


def _snippet(text: str, limit: int = 240) -> str:
    txt = re.sub(r"\s+", " ", (text or "")).strip()
    if len(txt) <= limit:
        return txt
    return txt[:limit] + "...(truncated)..."


def _file_sig(path: Path) -> str:
    """生成文件签名：mtime秒:文件大小。"""
    st = path.stat()
    return f"{int(st.st_mtime)}:{st.st_size}"


async def metadata_retrieve_top_docs(
    query: str,
    metadata_path: str = DEFAULT_METADATA_INDEX,
    max_docs: int = 3,
) -> str:
    """
    Tool API: metadata_retrieve_top_docs

    功能:
    - 基于 metadata（标题/摘要/预览/关键词）做文档级粗筛，返回候选文档。

    参数:
    - query (str): 检索查询。
    - metadata_path (str, optional): metadata JSON 路径。
    - max_docs (int, optional): 返回候选数量；内部会限制到 1~3。默认 3。

    返回:
    - str:
        - 成功: 候选列表（score/file_name/file_path/abstract/preview/status）
        - 未命中: "未在 metadata 中匹配到文档..."
        - 失败: "metadata 读取失败..."

    示例:
    - 输入: metadata_retrieve_top_docs("JWST faint galaxy", max_docs=3)
    - 输出: "[1] score=...\nfile_name: ...\nfile_path: ..."

    备注:
    - 该工具是“第一阶段召回”，建议后续结合 docling_read_pdf/pdf_search 精读取证。
    """
    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return f"查询词过短或无有效 token: {query}"

    top_n = max(1, min(int(max_docs), 3))

    try:
        items = _load_metadata_items(metadata_path)
    except Exception as exc:
        return f"metadata 读取失败: {exc}"

    if not items:
        return f"metadata 为空: {metadata_path}"

    scored: List[tuple] = []
    for item in items:
        file_path = str(item.get("file_path", "")).strip()
        file_name = str(item.get("file_name", "")).strip()
        abstract = str(item.get("abstract", "")).strip()
        preview = str(item.get("preview_text", "")).strip()
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        corpus = "\n".join(
            [
                file_name,
                file_path,
                abstract,
                " ".join(str(k) for k in keywords),
                preview,
            ]
        )
        score = _score_page(corpus, query_tokens)

        if score <= 0:
            continue

        scored.append(
            (
                score,
                {
                    "file_name": file_name or Path(file_path).name,
                    "file_path": file_path,
                    "abstract": abstract,
                    "preview_text": preview,
                    "status": item.get("status", ""),
                },
            )
        )

    if not scored:
        return (
            f"未在 metadata 中匹配到文档: query={query}\n"
            f"metadata={_safe_resolve(metadata_path)}\n"
            "建议：放宽关键词、改用英文术语，或先用 fs_list_tree 查看文件池。"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = scored[:top_n]

    lines = [
        f"查询: {query}",
        f"metadata: {_safe_resolve(metadata_path)}",
        f"候选文档数: {len(hits)} (max_docs={top_n})",
        "---",
    ]
    for i, (score, doc) in enumerate(hits, start=1):
        lines.append(f"[{i}] score={score:.2f}")
        lines.append(f"file_name: {doc['file_name']}")
        lines.append(f"file_path: {doc['file_path']}")
        if doc.get("status"):
            lines.append(f"status: {doc['status']}")
        if doc.get("abstract"):
            lines.append(f"abstract: {_snippet(doc['abstract'])}")
        if doc.get("preview_text"):
            lines.append(f"preview: {_snippet(doc['preview_text'])}")
        lines.append("")
    return "\n".join(lines).strip()


async def metadata_check_freshness(
    refer_docs_dir: str = DEFAULT_REFER_DOCS,
    metadata_path: str = DEFAULT_METADATA_INDEX,
    recursive: bool = True,
    include_subdirs: bool = True,
) -> str:
    """
    Tool API: metadata_check_freshness

    功能:
    - 检查 metadata 索引与当前 PDF 文件池是否一致，判断“元数据是否最新”。

    参数:
    - refer_docs_dir (str, optional): PDF 根目录。默认 /Users/jiean/agent_learn/refer_docs。
    - metadata_path (str, optional): 元数据 JSON 路径。默认 pdf_metadata_index.json。
    - recursive (bool, optional): 是否递归扫描子目录。默认 True。
    - include_subdirs (bool, optional): False 时仅统计根目录下 PDF。默认 True。

    返回:
    - str: 结构化诊断报告，包含:
      - up_to_date: true/false
      - disk_pdf_count / metadata_pdf_count
      - missing_in_metadata（磁盘有、索引无）
      - missing_on_disk（索引有、磁盘无）
      - changed_sig（文件内容/时间变化）
      - metadata_updated_at
      - 建议动作

    示例:
    - 输入: metadata_check_freshness()
    - 输出: "up_to_date: false\nmissing_in_metadata: 30\n..."

    备注:
    - 常用于下载新论文后的“索引健康检查”。
    - 当前只做检查，不自动重建 metadata。
    """
    root = _safe_resolve(refer_docs_dir)
    meta_p = _safe_resolve(metadata_path)

    if not root.exists() or not root.is_dir():
        return f"无效 refer_docs 目录: {root}"
    if not meta_p.exists() or meta_p.suffix.lower() != ".json":
        return f"无效 metadata JSON 路径: {meta_p}"

    try:
        payload = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"metadata 解析失败: {meta_p.name} -> {exc}"

    raw_items = payload.get("items", {})
    if not isinstance(raw_items, dict):
        return "metadata.items 格式异常：预期为对象(dict)。"

    # 1) 扫描磁盘 PDF
    disk_map: Dict[str, str] = {}
    if recursive and include_subdirs:
        iterator = root.rglob("*.pdf")
    else:
        iterator = root.glob("*.pdf")

    for p in iterator:
        if p.is_file():
            rp = str(p.resolve())
            try:
                disk_map[rp] = _file_sig(p)
            except Exception:
                # 罕见 I/O 异常时跳过该文件
                continue

    # 2) 收集 metadata PDF 信息
    meta_map: Dict[str, str] = {}
    for k, v in raw_items.items():
        if not isinstance(v, dict):
            continue
        fp = str(v.get("file_path") or k)
        if not fp.lower().endswith(".pdf"):
            continue
        sig = str(v.get("file_sig", ""))
        meta_map[fp] = sig

    disk_set = set(disk_map.keys())
    meta_set = set(meta_map.keys())

    missing_in_metadata = sorted(disk_set - meta_set)
    missing_on_disk = sorted(meta_set - disk_set)

    common = disk_set & meta_set
    changed_sig = []
    for fp in sorted(common):
        ms = meta_map.get(fp, "")
        ds = disk_map.get(fp, "")
        if ms and ds and ms != ds:
            changed_sig.append(fp)

    up_to_date = len(missing_in_metadata) == 0 and len(missing_on_disk) == 0 and len(changed_sig) == 0

    # 3) metadata 时间戳健康检查（仅提醒）
    metadata_updated_at = str(payload.get("updated_at", ""))
    timestamp_note = ""
    if metadata_updated_at:
        try:
            dt = datetime.fromisoformat(metadata_updated_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_days = (now - dt).total_seconds() / 86400.0
            timestamp_note = f"metadata_age_days≈{age_days:.2f}"
        except Exception:
            timestamp_note = "metadata_age_days=unknown"
    else:
        timestamp_note = "metadata_age_days=unknown"

    def _take(xs: List[str], n: int = 8) -> List[str]:
        return xs[:n]

    lines = [
        f"refer_docs_dir: {root}",
        f"metadata_path: {meta_p}",
        f"metadata_updated_at: {metadata_updated_at or '(missing)'}",
        timestamp_note,
        f"disk_pdf_count: {len(disk_map)}",
        f"metadata_pdf_count: {len(meta_map)}",
        f"missing_in_metadata: {len(missing_in_metadata)}",
        f"missing_on_disk: {len(missing_on_disk)}",
        f"changed_sig: {len(changed_sig)}",
        f"up_to_date: {'true' if up_to_date else 'false'}",
        "---",
    ]

    if missing_in_metadata:
        lines.append("missing_in_metadata_samples:")
        lines.extend([f"- {p}" for p in _take(missing_in_metadata)])
    if missing_on_disk:
        lines.append("missing_on_disk_samples:")
        lines.extend([f"- {p}" for p in _take(missing_on_disk)])
    if changed_sig:
        lines.append("changed_sig_samples:")
        lines.extend([f"- {p}" for p in _take(changed_sig)])

    if up_to_date:
        lines.append("建议: metadata 已与当前 PDF 文件池一致，可直接检索。")
    else:
        lines.append("建议: 先重建/更新 metadata（尤其处理 missing_in_metadata 与 changed_sig）。")

    return "\n".join(lines).strip()


async def docling_read_pdf(
    file_path: str,
    max_chars: int = 20000,
    page_start: int = 1,
    page_end: int = 3,
) -> str:
    """
    Tool API: docling_read_pdf

    功能:
    - 优先用 docling 进行结构化 PDF 解析；失败时自动回退到 pypdf 分页提取。

    参数:
    - file_path (str): PDF 文件路径。
    - max_chars (int, optional): 最大返回字符数。默认 20000。
    - page_start (int, optional): fallback 模式起始页（1-based，含）。默认 1。
    - page_end (int, optional): fallback 模式结束页（1-based，含）。默认 3。

    返回:
    - str:
        - docling 成功: 包含 "method=docling" 与 markdown 内容
        - fallback 成功: 包含 "method=pypdf_fallback" 与页级文本
        - 失败: 错误信息

    示例:
    - 输入: docling_read_pdf("paper.pdf", max_chars=5000)
    - 输出: "file=paper.pdf\nmethod=docling\n---\n# ..."

    备注:
    - fallback 输出会附带 docling_error，便于排查环境依赖问题。
    """
    p = _safe_resolve(file_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        return f"无效 PDF 路径: {p}"

    docling_error = None
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        converter = DocumentConverter()
        result = converter.convert(str(p))
        markdown = (result.document.export_to_markdown() or "").strip()

        if not markdown:
            raise ValueError("docling 返回空内容")

        out = (
            f"file={p.name}\n"
            f"method=docling\n"
            "note=docling 当前返回整文 markdown，页码切片将退化为全文截断\n"
            "---\n"
            f"{markdown}"
        )
        if len(out) > max_chars:
            out = out[:max_chars] + "\n...(truncated)..."
        return out
    except Exception as exc:
        docling_error = str(exc)

    # fallback: pypdf page slicing
    if PdfReader is None:
        return (
            "docling 不可用且 pypdf 未安装，无法读取 PDF。"
            f" docling_error={docling_error}"
        )
    try:
        reader = PdfReader(str(p))
        total = len(reader.pages)
        if total <= 0:
            return f"PDF 无页面内容: {p.name}"

        start = max(1, page_start)
        end = min(max(start, page_end), total)
        parts: List[str] = [
            f"file={p.name}",
            "method=pypdf_fallback",
            f"docling_error={docling_error}",
            f"pages={start}-{end}/{total}",
            "---",
        ]
        for i in range(start - 1, end):
            text = (reader.pages[i].extract_text() or "").strip()
            parts.append(f"\n[Page {i + 1}]\n{text or '(empty)'}")

        out = "\n".join(parts)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n...(truncated)..."
        return out
    except Exception as exc:
        return f"PDF 读取失败: {p.name} -> {exc}"


async def pdf_search(file_path: str, query: str, top_k: int = 3) -> str:
    """
    Tool API: pdf_search

    功能:
    - 在单个 PDF 内做页级关键词匹配并排序，返回 top-k 证据页。

    参数:
    - file_path (str): PDF 路径。
    - query (str): 查询字符串。
    - top_k (int, optional): 返回命中页数量。默认 3。

    返回:
    - str:
        - 成功: 命中页列表（page、score、snippet）
        - 未命中: "未匹配到相关页面: ..."
        - 失败: "无效 PDF 路径..." 或 "PDF 打开失败..."

    示例:
    - 输入: pdf_search("paper.pdf", "diffusion model", top_k=2)
    - 输出: "[1] page=3, score=...\n..."

    备注:
    - 该工具适合做“证据定位”，常用于最终回答的页码依据。
    """
    p = _safe_resolve(file_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        return f"无效 PDF 路径: {p}"
    if PdfReader is None:
        return "依赖缺失: pypdf 未安装，无法执行 pdf_search。请先安装 pypdf。"

    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return f"查询词过短或无有效 token: {query}"

    try:
        reader = PdfReader(str(p))
    except Exception as exc:
        return f"PDF 打开失败: {p.name} -> {exc}"

    total_pages = len(reader.pages)
    if total_pages == 0:
        return f"PDF 无页面内容: {p.name}"

    candidates = []
    for page_num in range(1, total_pages + 1):
        text = (reader.pages[page_num - 1].extract_text() or "").strip()
        score = _score_page(text, query_tokens)
        if score <= 0:
            continue
        snippet_len = 900
        snippet = text[:snippet_len] + ("...(truncated)..." if len(text) > snippet_len else "")
        candidates.append((score, page_num, snippet))

    candidates.sort(key=lambda x: x[0], reverse=True)
    hits = candidates[: max(1, top_k)]

    if not hits:
        return f"未匹配到相关页面: {p.name}"

    lines = [
        f"文件: {p.name}",
        f"查询: {query}",
        f"总页数: {total_pages}",
        f"命中数: {len(hits)} (top_k={top_k})",
        "---",
    ]
    for i, (score, page_num, snippet) in enumerate(hits, start=1):
        lines.append(f"[{i}] page={page_num}, score={score:.2f}")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines).strip()
