from __future__ import annotations
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from bs4 import BeautifulSoup

@dataclass
class DocChunk:
    id: str
    text: str
    source: str
    title: str | None = None
    url: str | None = None

def _read_text_file(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf_file(p: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(p)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def _strip_html(s: str) -> str:
    try:
        return BeautifulSoup(s, "html.parser").get_text(" ")
    except Exception:
        return re.sub(r"<[^>]+>", " ", s)

def load_docs(src: str | Path) -> List[Tuple[str, str]]:
    src = str(src)
    paths: List[str] = []
    if os.path.isdir(src):
        for ext in ("*.md", "*.txt", "*.html", "*.htm", "*.pdf"):
            paths.extend(glob.glob(os.path.join(src, "**", ext), recursive=True))
    else:
        paths = glob.glob(src)
    out: List[Tuple[str, str]] = []
    for p in sorted(paths):
        low = p.lower()
        try:
            if low.endswith((".md", ".txt")):
                txt = _read_text_file(p)
            elif low.endswith((".html", ".htm")):
                txt = _strip_html(_read_text_file(p))
            elif low.endswith(".pdf"):
                txt = _read_pdf_file(p)
            else:
                continue
        except Exception:
            txt = ""
        if txt and txt.strip():
            out.append((p, txt))
    return out

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text)

def split_into_chunks(text: str, chunk_tokens: int = 300, overlap_tokens: int = 40) -> List[str]:
    toks = _tokenize(text)
    chunks: List[str] = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + chunk_tokens)
        chunk = " ".join(toks[i:j])
        chunks.append(chunk)
        if j >= len(toks):
            break
        i = j - overlap_tokens
        if i <= 0:
            i = j
    cleaned = [re.sub(r"\s+", " ", c).strip() for c in chunks if c and c.strip()]
    return cleaned

def build_chunks(src: str | Path, chunk_tokens: int, overlap_tokens: int, base_url: str | None = None) -> List[DocChunk]:
    items: List[DocChunk] = []
    for path, text in load_docs(src):
        title = os.path.basename(path)
        url = None
        if base_url:
            try:
                rel = os.path.relpath(path, start=str(src))
                url = base_url.rstrip("/") + "/" + rel.replace(os.sep, "/")
            except Exception:
                url = None
        parts = split_into_chunks(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        for n, chunk in enumerate(parts):
            items.append(DocChunk(id=f"{path}#chunk-{n}", text=chunk, source=path, title=title, url=url))
    return items
