import json
import os
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader


def _safe_resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


async def fs_pwd(path: str = ".") -> str:
    """返回当前/指定路径的绝对路径（pwd）。"""
    p = _safe_resolve(path)
    if not p.exists():
        return f"路径不存在: {p}"
    return f"pwd: {p}"


async def fs_list_tree(root_path: str = ".", max_depth: int = 3, max_results: int = 300) -> str:
    """查看目录结构（受限深度，等价 ls -R 的安全子集）。"""
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
    """读取文件内容：文本直接读；PDF 按页读取。"""
    p = _safe_resolve(file_path)
    if not p.exists() or not p.is_file():
        return f"文件不存在: {p}"

    if p.suffix.lower() == ".pdf":
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
    """安全编辑 JSON：按 key_path(点分路径)写入 value_json。"""
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


async def pdf_search(file_path: str, query: str, top_k: int = 3) -> str:
    """在单个 PDF 中按关键词打分，返回 top-k 页证据。"""
    p = _safe_resolve(file_path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        return f"无效 PDF 路径: {p}"

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
