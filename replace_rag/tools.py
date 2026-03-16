import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


# ============================================================
# PDF Tools（可被 AutoGen Agent 直接注册调用）
# ============================================================


async def discover_pdf_folders(
    root_path: str = ".",
    hint: str = "",
    max_depth: int = 4,
    max_results: int = 20,
) -> str:
    # Tool-0：文件发现工具
    # 作用：当用户给的目录名不准确时，在 root_path 下扫描并返回“可能的 PDF 目录”。
    # 输出包含目录路径与 PDF 数量，便于 Agent 继续调用 build/search 工具。
    root = Path(root_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return f"根目录不存在: {root}"

    hint_low = (hint or "").strip().lower()
    root_depth = len(root.parts)

    hits: List[tuple[int, str, int]] = []
    for dirpath, _, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth
        if depth > max_depth:
            continue

        pdf_count = sum(1 for n in filenames if n.lower().endswith(".pdf"))
        if pdf_count <= 0:
            continue

        path_text = str(current).lower()
        score = pdf_count
        if hint_low:
            if hint_low in path_text:
                score += 100
            else:
                # 没命中 hint 的目录也保留，但分数较低
                score -= 1

        hits.append((score, str(current), pdf_count))

    if not hits:
        return f"未发现包含 PDF 的目录: root={root}, max_depth={max_depth}"

    hits.sort(key=lambda x: x[0], reverse=True)
    top = hits[: max(1, max_results)]

    lines = [
        f"root: {root}",
        f"hint: {hint or '(none)'}",
        f"found: {len(hits)}, return: {len(top)}",
        "---",
    ]
    for i, (score, path, cnt) in enumerate(top, start=1):
        lines.append(f"[{i}] score={score}, pdf_count={cnt}, path={path}")

    return "\n".join(lines)


def _now_iso() -> str:
    """统一时间格式，便于日志与索引追踪。"""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _read_index(index_path: Path) -> Dict[str, Any]:
    """读取索引文件；不存在则返回默认结构。"""
    if not index_path.exists():
        return {"version": 1, "updated_at": _now_iso(), "items": {}}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if "items" not in data or not isinstance(data["items"], dict):
            data["items"] = {}
        return data
    except Exception:
        # 索引损坏时兜底，避免整个流程崩溃
        return {"version": 1, "updated_at": _now_iso(), "items": {}}


def _write_index_atomic(index_path: Path, data: Dict[str, Any]) -> None:
    """原子写入，降低中途崩溃导致索引损坏的风险。"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(index_path)


async def get_pdf_main_content(file_path: str, max_pages: int = 3) -> str:
    # Tool-1：粗读工具
    # 作用：提取 PDF 前 N 页主要内容，用于快速判断文档主题是否相关。
    # 特性：文本抽取失败时可尝试 OCR 回退（依赖可选）。
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or path.suffix.lower() != ".pdf":
        return f"无效 PDF 路径: {path}"

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return f"PDF 打开失败: {path.name} -> {exc}"

    # 页数不足 3 时，只读取实际页数（避免越界）
    total_pages = len(reader.pages)
    pages_to_read = min(max_pages, total_pages)
    if pages_to_read <= 0:
        return f"PDF 无页面内容: {path.name}"

    text_blocks: List[str] = []
    empty_pages: List[int] = []
    for i in range(pages_to_read):
        page_text = (reader.pages[i].extract_text() or "").strip()
        if page_text:
            text_blocks.append(f"[Page {i + 1}]\n{page_text}")
        else:
            empty_pages.append(i + 1)

    # method 用于让上层 Agent 感知数据来源质量：text / ocr / none
    method = "text"
    notes: List[str] = []

    # 当无法提取文本时，尝试 OCR（依赖可选，不强制）
    if len(text_blocks) == 0:
        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(str(path), first_page=1, last_page=pages_to_read)
            ocr_blocks: List[str] = []
            for idx, img in enumerate(images, start=1):
                ocr_text = (pytesseract.image_to_string(img) or "").strip()
                if ocr_text:
                    ocr_blocks.append(f"[Page {idx} | OCR]\n{ocr_text}")

            if ocr_blocks:
                text_blocks = ocr_blocks
                method = "ocr"
                notes.append("文本抽取失败，已使用 OCR 回退")
            else:
                method = "none"
                notes.append("文本抽取失败，OCR 也未获得内容")
        except Exception as exc:
            method = "none"
            notes.append("文本抽取失败，且 OCR 不可用。可安装: pdf2image + pytesseract（并配置 tesseract）")
            notes.append(f"OCR 错误: {exc}")

    if empty_pages and method != "ocr":
        notes.append(f"空白/不可提取页面: {empty_pages}")

    # 将多页文本拼接为单段上下文，供模型做摘要
    content = "\n\n".join(text_blocks).strip()
    if not content:
        return (
            f"文件: {path.name}\n"
            f"页数: {total_pages}, 扫描页: 1-{pages_to_read}\n"
            f"提取方式: {method}\n"
            f"说明: {'; '.join(notes) if notes else '无'}\n"
            "内容: (空)"
        )

    # 控制工具返回长度，避免一次塞太多上下文（上下文预算保护）
    max_chars = 12000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...(truncated)..."
        notes.append(f"内容已截断到 {max_chars} 字符")

    return (
        f"文件: {path.name}\n"
        f"页数: {total_pages}, 扫描页: 1-{pages_to_read}\n"
        f"提取方式: {method}\n"
        f"说明: {'; '.join(notes) if notes else '无'}\n"
        "---\n"
        f"{content}"
    )


async def build_pdf_metadata_index(
    folder_path: str,
    index_file: str = "./pdf_metadata_index.json",
    max_pages: int = 3,
    force: bool = False,
) -> str:
    # Tool-3：批量建立/更新“PDF -> 元数据”映射
    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return f"目录不存在: {root}"

    index_path = Path(index_file).expanduser().resolve()
    index = _read_index(index_path)
    items: Dict[str, Any] = index.get("items", {})

    pdf_files = sorted(root.glob("*.pdf"))
    if not pdf_files:
        return f"目录中未找到 PDF: {root}"

    created = 0
    updated = 0
    skipped = 0

    for pdf in pdf_files:
        stat = pdf.stat()
        key = str(pdf.resolve())
        current_sig = f"{int(stat.st_mtime)}:{stat.st_size}"

        old = items.get(key)
        if (not force) and old and old.get("file_sig") == current_sig:
            skipped += 1
            continue

        preview = await get_pdf_main_content(str(pdf), max_pages=max_pages)
        entry = {
            "file_name": pdf.name,
            "file_path": key,
            "file_sig": current_sig,
            "updated_at": _now_iso(),
            "abstract": old.get("abstract", "") if old else "",
            "keywords": old.get("keywords", []) if old else [],
            "status": "ready" if (old and old.get("abstract")) else "pending_summary",
            "preview_text": preview,
        }
        items[key] = entry
        if old:
            updated += 1
        else:
            created += 1

    index["version"] = 1
    index["updated_at"] = _now_iso()
    index["items"] = items
    _write_index_atomic(index_path, index)

    return (
        f"索引文件: {index_path}\n"
        f"PDF 总数: {len(pdf_files)}\n"
        f"新建: {created}, 更新: {updated}, 跳过: {skipped}\n"
        "说明: status=pending_summary 的条目需要 Agent 后续补写 abstract。"
    )


async def upsert_pdf_metadata(
    index_file: str,
    file_path: str,
    abstract: str,
    keywords: List[str] | None = None,
) -> str:
    # Tool-4：由 Agent 把生成好的摘要/关键词写回持久化索引
    index_path = Path(index_file).expanduser().resolve()
    index = _read_index(index_path)
    items: Dict[str, Any] = index.get("items", {})

    key = str(Path(file_path).expanduser().resolve())
    old = items.get(key, {})

    old.update(
        {
            "file_name": Path(key).name,
            "file_path": key,
            "abstract": (abstract or "").strip(),
            "keywords": keywords or [],
            "status": "ready" if abstract.strip() else "pending_summary",
            "updated_at": _now_iso(),
        }
    )
    items[key] = old

    index["version"] = 1
    index["updated_at"] = _now_iso()
    index["items"] = items
    _write_index_atomic(index_path, index)

    return f"已写入元数据: {Path(key).name} -> {index_path}"


def _tokenize_query(text: str) -> List[str]:
    # 查询分词（中英混合的简易实现）
    tokens = re.findall(r"[\u4e00-\u9fff]{1,}|[a-zA-Z0-9_\-]{2,}", text.lower())
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
        "what", "when", "where", "which", "how", "why", "is", "to", "of", "in", "on",
        "一个", "一些", "这个", "那个", "什么", "如何", "以及", "或者", "可以", "我们", "你们",
    }
    return [t for t in tokens if t not in stop and len(t) > 1]


def _score_page(text: str, query_tokens: List[str]) -> float:
    # 页面相关性打分：关键词命中次数 + 覆盖率
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


async def search_pdf_pages(
    file_path: str,
    query: str,
    top_k: int = 5,
    page_start: int = 1,
    page_end: int = 0,
) -> str:
    # Tool-2：精读检索工具
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or path.suffix.lower() != ".pdf":
        return f"无效 PDF 路径: {path}"

    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return f"查询词过短或无有效 token: {query}"

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return f"PDF 打开失败: {path.name} -> {exc}"

    total_pages = len(reader.pages)
    if total_pages == 0:
        return f"PDF 无页面内容: {path.name}"

    start = max(1, page_start)
    end = total_pages if page_end <= 0 else min(page_end, total_pages)
    if start > end:
        return f"页码范围无效: page_start={page_start}, page_end={page_end}, total={total_pages}"

    candidates = []
    empty_pages: List[int] = []
    for page_num in range(start, end + 1):
        text = (reader.pages[page_num - 1].extract_text() or "").strip()
        if not text:
            empty_pages.append(page_num)
            continue
        score = _score_page(text, query_tokens)
        if score <= 0:
            continue
        snippet_len = 900
        snippet = text[:snippet_len] + ("...(truncated)..." if len(text) > snippet_len else "")
        candidates.append((score, page_num, snippet))

    candidates.sort(key=lambda x: x[0], reverse=True)
    hits = candidates[: max(1, top_k)]

    if not hits:
        notes = [f"未匹配到相关页面: {path.name}"]
        if empty_pages:
            notes.append(f"空白/不可提取页面: {empty_pages[:20]}")
            notes.append("如果该 PDF 是扫描件，可考虑启用 OCR 版逐页检索。")
        return "\n".join(notes)

    lines = [
        f"文件: {path.name}",
        f"查询: {query}",
        f"页范围: {start}-{end} / 总页数: {total_pages}",
        f"命中数: {len(hits)} (top_k={top_k})",
    ]
    if empty_pages:
        lines.append(f"空白/不可提取页面(部分): {empty_pages[:20]}")
    lines.append("---")
    for i, (score, page_num, snippet) in enumerate(hits, start=1):
        lines.append(f"[{i}] page={page_num}, score={score:.2f}")
        lines.append(snippet)
        lines.append("")

    return "\n".join(lines).strip()
