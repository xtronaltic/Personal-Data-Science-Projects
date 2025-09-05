import os
from pathlib import Path

from rag.ingest import build_chunks
from rag.embed import HashEmbedder
from rag.index import VectorIndex, save_metadata
from rag.retriever import RagStore


def test_rag_build_and_search(tmp_path: Path):
    # Prepare tiny corpus
    corp = tmp_path / "docs"
    corp.mkdir(parents=True, exist_ok=True)
    (corp / "a.txt").write_text("AlignTune is a modern LLM fine-tuning project.", encoding="utf-8")
    (corp / "b.txt").write_text("We demonstrate SFT, DPO, and SimPO with reports.", encoding="utf-8")

    chunks = build_chunks(str(corp), chunk_tokens=32, overlap_tokens=4)
    texts = [c.text for c in chunks]
    meta = [{"id": c.id, "text": c.text, "source": c.source, "title": c.title, "url": c.url} for c in chunks]

    emb = HashEmbedder(dim=64)
    vecs = emb.encode(texts)
    idx = VectorIndex(dim=vecs.shape[1], use_faiss=False)
    idx.build(vecs)

    root = tmp_path / "rag" / "indices"
    col = root / "testcol"
    col.mkdir(parents=True, exist_ok=True)
    idx.save(col)
    save_metadata(col / "meta.jsonl", meta)
    (col / "manifest.json").write_text("{}", encoding="utf-8")

    store = RagStore(root=str(root), collection="testcol", embed_backend="hash")
    res = store.search("What is AlignTune?", k=2)
    assert len(res) >= 1
    assert any("AlignTune" in r.text for r in res)

