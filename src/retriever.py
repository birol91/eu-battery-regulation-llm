"""Dense retriever over the prebuilt FAISS index."""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

MODEL_ID = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_INDEX_DIR = Path.home() / ".cache" / "eu-battery-llm" / "rag_index"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_metadata(path: Path) -> dict[int, dict]:
    by_idx: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_idx[int(row["faiss_idx"])] = row
    return by_idx


class Retriever:
    def __init__(self, index_dir: Path | None = None, use_reranker: bool = True) -> None:
        index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        index_path = index_dir / "faiss.index"
        metadata_path = index_dir / "chunks_metadata.jsonl"

        self.device = pick_device()
        print(f"[retriever] device={self.device}")

        self.index = faiss.read_index(str(index_path))
        print(f"[retriever] index loaded: ntotal={self.index.ntotal} dim={self.index.d}")

        self.metadata_by_idx = _load_metadata(metadata_path)
        print(f"[retriever] metadata loaded: rows={len(self.metadata_by_idx)}")

        if self.index.ntotal != len(self.metadata_by_idx):
            raise RuntimeError(
                f"index/metadata size mismatch: index.ntotal={self.index.ntotal} "
                f"metadata_rows={len(self.metadata_by_idx)}"
            )

        print(f"[retriever] loading model {MODEL_ID} ...")
        self.model = SentenceTransformer(MODEL_ID, device=self.device)

        self.use_reranker = use_reranker
        self.reranker: CrossEncoder | None = None
        if self.use_reranker:
            print(f"[retriever] loading reranker {RERANKER_MODEL_ID} ...")
            self.reranker = CrossEncoder(RERANKER_MODEL_ID, device=self.device)

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [BGE_QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return vec

    def retrieve(self, query: str, top_k: int = 5, similarity_floor: float = 0.45) -> list[dict]:
        q = self._embed_query(query)
        fetch_k = min(top_k * 4, 20) if self.use_reranker else top_k
        scores, indices = self.index.search(q, fetch_k)
        scores, indices = scores[0], indices[0]

        results: list[dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or float(score) < similarity_floor:
                continue
            meta = self.metadata_by_idx[int(idx)]
            results.append({
                "chunk_id": meta["chunk_id"],
                "source": meta["source"],
                "heading": meta["heading"],
                "text": meta["text"],
                "similarity_score": float(score),
                "is_primary": meta.get("is_primary", True),
            })

        if self.use_reranker and self.reranker and results:
            pairs = [[query, r["text"]] for r in results]
            rerank_scores = self.reranker.predict(pairs)
            for r, s in zip(results, rerank_scores):
                r["reranker_score"] = float(s)
            results.sort(key=lambda r: r["reranker_score"], reverse=True)
            results = results[:top_k]

        return results
