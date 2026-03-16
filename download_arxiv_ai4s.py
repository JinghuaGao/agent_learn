#!/usr/bin/env python3
"""
批量下载 arXiv 上与 AI for Science 相关的论文 PDF，并生成元数据清单。

示例：
python download_arxiv_ai4s.py --max-results 30 --out-dir ./refer_docs/ai4science
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def build_query() -> str:
    # 关键词可按你的研究方向继续扩展
    terms = [
        'all:"ai for science"',
        'all:"machine learning for science"',
        'all:"scientific discovery" AND all:"machine learning"',
        'all:"foundation model" AND all:"science"',
    ]
    return " OR ".join(f"({t})" for t in terms)


def fetch_feed(search_query: str, start: int, max_results: int) -> str:
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_entries(feed_xml: str) -> List[Dict]:
    root = ET.fromstring(feed_xml)
    entries = []
    for e in root.findall("atom:entry", NS):
        entry_id = (e.findtext("atom:id", default="", namespaces=NS) or "").strip()
        title = (e.findtext("atom:title", default="", namespaces=NS) or "").strip()
        summary = (e.findtext("atom:summary", default="", namespaces=NS) or "").strip()
        published = (e.findtext("atom:published", default="", namespaces=NS) or "").strip()
        updated = (e.findtext("atom:updated", default="", namespaces=NS) or "").strip()

        authors = []
        for a in e.findall("atom:author", NS):
            name = (a.findtext("atom:name", default="", namespaces=NS) or "").strip()
            if name:
                authors.append(name)

        # 从 id 提取 arXiv id，例如 http://arxiv.org/abs/2501.01234v1
        m = re.search(r"/abs/([^/]+)$", entry_id)
        arxiv_id = m.group(1) if m else entry_id.rsplit("/", 1)[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        categories = [c.attrib.get("term", "") for c in e.findall("atom:category", NS)]

        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published,
                "updated": updated,
                "categories": categories,
                "entry_id": entry_id,
                "pdf_url": pdf_url,
            }
        )
    return entries


def safe_name(text: str, max_len: int = 120) -> str:
    t = re.sub(r"[\\/:*?\"<>|\n\r\t]+", " ", text).strip()
    t = re.sub(r"\s+", " ", t)
    return (t[:max_len]).strip() or "untitled"


def download_pdf(url: str, out_path: Path, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量下载 arXiv AI for Science 论文")
    parser.add_argument("--max-results", type=int, default=30, help="下载论文数量上限")
    parser.add_argument("--batch-size", type=int, default=100, help="每次 API 拉取条数")
    parser.add_argument("--out-dir", type=str, default="./refer_docs/ai4science", help="PDF 输出目录")
    parser.add_argument("--sleep", type=float, default=1.5, help="每次下载之间的间隔秒数")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    query = build_query()
    print(f"[INFO] Query: {query}")

    all_entries: List[Dict] = []
    seen = set()
    start = 0

    while len(all_entries) < args.max_results:
        need = min(args.batch_size, args.max_results - len(all_entries))
        xml = fetch_feed(query, start=start, max_results=need)
        entries = parse_entries(xml)
        if not entries:
            break

        newly = 0
        for e in entries:
            if e["arxiv_id"] in seen:
                continue
            seen.add(e["arxiv_id"])
            all_entries.append(e)
            newly += 1
            if len(all_entries) >= args.max_results:
                break

        print(f"[INFO] fetched start={start}, got={len(entries)}, kept={newly}, total={len(all_entries)}")
        start += len(entries)
        if len(entries) < need:
            break

    if not all_entries:
        print("[WARN] 没有检索到论文。")
        return

    manifest = []
    ok, failed = 0, 0

    for i, e in enumerate(all_entries, start=1):
        title_part = safe_name(e["title"], max_len=90)
        file_name = f"{e['arxiv_id'].replace('/', '_')} - {title_part}.pdf"
        out_path = out_dir / file_name

        row = {**e, "file_name": file_name, "file_path": str(out_path)}
        if out_path.exists() and out_path.stat().st_size > 0:
            row["download_status"] = "exists"
            manifest.append(row)
            ok += 1
            continue

        try:
            print(f"[DL] ({i}/{len(all_entries)}) {e['arxiv_id']} -> {file_name}")
            download_pdf(e["pdf_url"], out_path)
            row["download_status"] = "ok"
            ok += 1
        except Exception as ex:
            row["download_status"] = f"failed: {ex}"
            failed += 1
        manifest.append(row)
        time.sleep(max(args.sleep, 0.0))

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== DONE ===")
    print(f"输出目录: {out_dir}")
    print(f"论文记录: {len(manifest)}")
    print(f"下载成功/已存在: {ok}")
    print(f"下载失败: {failed}")
    print(f"清单文件: {manifest_path}")


if __name__ == "__main__":
    main()
