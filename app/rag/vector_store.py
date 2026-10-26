"""FAISS cosine-similarity store with JSON metadata persistence."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.ingestion import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    score: float
    source_url: str | None = None
    retrieved_at: str | None = None


class FAISSVectorStore:
    """Store normalized vectors in FAISS IndexFlatIP (cosine similarity)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index_path = directory / "index.faiss"
        self.metadata_path = directory / "metadata.json"
        self._index: object | None = None
        self._records: list[DocumentChunk] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None:
        """Add a batch of chunks and matching normalized vectors."""

        if not chunks:
            return
        faiss = self._faiss()
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError("Embedding matrix must have one vector per chunk")
        faiss.normalize_L2(matrix)
        if self._index is None:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
        elif self._index.d != matrix.shape[1]:  # type: ignore[attr-defined]
            raise ValueError("Embedding dimension does not match the existing FAISS index")
        self._index.add(matrix)  # type: ignore[attr-defined]
        self._records.extend(chunks)

    def search(self, vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """Return nearest chunks sorted by descending cosine score."""

        if self._index is None or not self._records:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1).copy()
        self._faiss().normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, len(self._records)))  # type: ignore[attr-defined]
        return [
            SearchHit(
                text=self._records[int(index)].text,
                source=self._records[int(index)].source,
                score=float(score),
                source_url=self._records[int(index)].source_url,
                retrieved_at=self._records[int(index)].retrieved_at,
            )
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    def save(self) -> None:
        """Persist vectors and source text locally; no audio or user data is stored."""

        if self._index is None:
            return
        faiss = self._faiss()
        self.directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(
                [
                    {
                        "text": row.text,
                        "source": row.source,
                        "source_url": row.source_url,
                        "retrieved_at": row.retrieved_at,
                    }
                    for row in self._records
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """Load persisted files if both are present and valid."""

        if not self.index_path.is_file() or not self.metadata_path.is_file():
            return False
        try:
            faiss = self._faiss()
            index = faiss.read_index(str(self.index_path))
            rows = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if index.ntotal != len(rows):
                raise ValueError("FAISS index and metadata have different record counts")
            self._index = index
            self._records = [
                DocumentChunk(
                    text=row["text"],
                    source=row["source"],
                    source_url=row.get("source_url"),
                    retrieved_at=row.get("retrieved_at"),
                )
                for row in rows
            ]
            return True
        except Exception:
            logger.exception("Stored FAISS index could not be loaded; rebuilding it")
            self._index = None
            self._records = []
            return False

    @staticmethod
    def _faiss() -> object:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS is unavailable; install project dependencies with `pip install -e .`") from exc
        return faiss
