"""Lazy RAG index initialization and score-thresholded retrieval."""

import logging
import math
import os
import re
import shutil
import threading
import unicodedata
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from typing import Protocol

import numpy as np

from app.models.schemas import RetrievedDocument
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.ingestion import DocumentChunk, load_markdown_documents
from app.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)
_STOP_WORDS = frozenset(
    "a ai alors au aucun aux avec avoir bon car ce ceci cela ces cet cette ceux chez comme comment dans de des "
    "dit du elle en encore est et etre eux fait faites faire ici il ils je l la le les leur lui ma mais me meme mes "
    "moi mon ne nos notre nous on ou par parce pas pour qu que quelle quelles quel quels qui sa sans se ses son "
    "sont sous sur ta te tes toi ton tu un une vos votre vous y euh hein genre voila merci bonjour ok".split()
)


class RAGFailure(RuntimeError):
    """Raised when embedding, index loading or vector search cannot run."""


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class VectorStore(Protocol):
    def __len__(self) -> int: ...
    def add(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, top_k: int): ...
    def load(self) -> bool: ...
    def save(self) -> None: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]: ...


class RAGRetriever:
    """Select dense FAISS, BM25, hybrid RRF, and an optional candidate reranker."""

    def __init__(
        self,
        knowledge_base: Path,
        vector_store_dir: Path,
        *,
        model_name: str,
        dense_min_free_disk_mb: int = 1536,
        top_k: int = 4,
        min_score: float = 0.30,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        force_lexical: bool = False,
        strategy: str = "auto",
        reranker_model: str = "",
        reranker: Reranker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.min_score = min_score
        self.dense_min_free_disk_mb = max(0, dense_min_free_disk_mb)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        selected_strategy = "bm25" if force_lexical else strategy.strip().lower()
        if selected_strategy not in {"auto", "bm25", "dense", "hybrid"}:
            raise ValueError("RAG strategy must be auto, bm25, dense or hybrid")
        self.strategy = selected_strategy
        self.force_lexical = selected_strategy == "bm25"
        self.reranker_model = reranker_model.strip()
        self.reranker = reranker
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder(model_name)
        self.vector_store = vector_store if vector_store is not None else FAISSVectorStore(vector_store_dir)
        self._lock = threading.Lock()
        self._initialized = False
        self._check_disk_space = embedder is None
        self._dense_failure: str | None = None
        self._lexical_chunks: list[DocumentChunk] | None = None
        self.mode = "faiss"

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """Load or build the local index, then return qualifying search hits."""

        query = query.strip()
        if not query:
            return []
        if self.strategy == "bm25":
            return self._maybe_rerank(query, self._retrieve_lexical(query))
        if self.strategy == "hybrid":
            lexical = self._retrieve_lexical(query)
            if self._dense_failure is not None:
                return self._lexical_with_mode(query, "hybrid_bm25_fallback", lexical)
            try:
                dense = self._retrieve_dense(query)
            except Exception as exc:
                self._remember_dense_failure(exc)
                return self._lexical_with_mode(query, "hybrid_bm25_fallback", lexical)
            fused = self._reciprocal_rank_fusion(lexical, dense)
            self.mode = "hybrid_rrf"
            return self._maybe_rerank(query, fused)
        if self._dense_failure is not None:
            return self._lexical_with_mode(query, "bm25_fallback")
        try:
            documents = self._retrieve_dense(query)
        except Exception as exc:
            self._remember_dense_failure(exc)
            return self._lexical_with_mode(query, "bm25_fallback")
        self.mode = "faiss"
        return self._maybe_rerank(query, documents)

    def _lexical_with_mode(
        self,
        query: str,
        mode: str,
        documents: list[RetrievedDocument] | None = None,
    ) -> list[RetrievedDocument]:
        results = documents if documents is not None else self._retrieve_lexical(query)
        self.mode = mode
        return self._maybe_rerank(query, results)

    def _retrieve_dense(self, query: str) -> list[RetrievedDocument]:
        self._check_embedding_disk_space()
        self._ensure_index()
        if len(self.vector_store) == 0:
            return []
        query_vector = self.embedder.encode([query])
        hits = self.vector_store.search(query_vector[0], self.top_k)
        return [
            RetrievedDocument(
                text=hit.text,
                source=hit.source,
                score=hit.score,
                rank=rank,
                source_url=getattr(hit, "source_url", None),
                retrieved_at=getattr(hit, "retrieved_at", None),
            )
            for rank, hit in enumerate(hits, start=1)
            if hit.score >= self.min_score
        ]

    def _remember_dense_failure(self, error: Exception) -> None:
        self._dense_failure = type(error).__name__
        logger.warning(
            "Dense RAG unavailable (%s); switching to local BM25 retrieval",
            self._dense_failure,
            exc_info=True,
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        lexical: list[RetrievedDocument], dense: list[RetrievedDocument], *, rank_constant: int = 60
    ) -> list[RetrievedDocument]:
        """Fuse two ranked lists without pretending their scores share a scale."""

        fused: dict[tuple[str, str], tuple[RetrievedDocument, float]] = {}
        for results in (lexical, dense):
            for rank, document in enumerate(results, start=1):
                key = (document.source, document.text)
                contribution = 1.0 / (rank_constant + rank)
                previous = fused.get(key)
                fused[key] = (document, contribution + (previous[1] if previous else 0.0))
        ordered = sorted(fused.values(), key=lambda item: item[1], reverse=True)
        return [
            document.model_copy(update={"score": score, "rank": rank})
            for rank, (document, score) in enumerate(ordered, 1)
        ]

    def _maybe_rerank(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        if not documents or (self.reranker is None and not self.reranker_model):
            return documents
        try:
            if self.reranker is None:
                from app.rag.reranker import CrossEncoderReranker

                self.reranker = CrossEncoderReranker(self.reranker_model)
            reranked = self.reranker.rerank(query, documents)
            self.mode = f"{self.mode}_reranked"
            return reranked[: self.top_k]
        except Exception as exc:
            logger.warning("Optional RAG reranker unavailable (%s); keeping original ranking", type(exc).__name__)
            self.mode = f"{self.mode}_rerank_unavailable"
            return documents

    def build_index(self, force: bool = False) -> int:
        """Build and persist the FAISS index; useful for the provided CLI script."""

        with self._lock:
            if self._initialized and not force:
                return len(self.vector_store)
            if not force and self.vector_store.load() and len(self.vector_store):
                self._initialized = True
                return len(self.vector_store)
            chunks = load_markdown_documents(
                self.knowledge_base,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            if not chunks:
                logger.warning("Knowledge base contains no Markdown documents: %s", self.knowledge_base)
                self._initialized = True
                return 0
            vectors = self.embedder.encode([chunk.text for chunk in chunks])
            self.vector_store.add(chunks, vectors)
            self.vector_store.save()
            self._initialized = True
            logger.info("RAG index built with %d chunks", len(chunks))
            return len(chunks)

    def _ensure_index(self) -> None:
        if self._initialized:
            return
        self.build_index()

    def _check_embedding_disk_space(self) -> None:
        """Avoid starting a large model download when its cache drive is nearly full."""

        if not self._check_disk_space or self.dense_min_free_disk_mb == 0:
            return
        cache_path = self._huggingface_cache_path()
        probe = cache_path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        free_mb = shutil.disk_usage(probe).free // (1024 * 1024)
        if free_mb < self.dense_min_free_disk_mb:
            raise RuntimeError(
                f"Only {free_mb} MB free on the Hugging Face cache drive; "
                f"dense retrieval requires at least {self.dense_min_free_disk_mb} MB"
            )

    @staticmethod
    def _huggingface_cache_path() -> Path:
        configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if configured:
            return Path(configured).expanduser()
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home).expanduser() / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    def _retrieve_lexical(self, query: str) -> list[RetrievedDocument]:
        """Search Markdown locally with BM25, without model downloads or FAISS."""

        try:
            if self._lexical_chunks is None:
                self._lexical_chunks = load_markdown_documents(
                    self.knowledge_base,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )
            chunks = self._lexical_chunks
            self.mode = "bm25"
            if not chunks:
                return []
            vocabulary = set().union(*(_terms(chunk.text) for chunk in chunks))
            query_terms = _match_query_terms(_terms(query), vocabulary)
            if not query_terms:
                return []
            tokenized = [_terms(chunk.text) for chunk in chunks]
            document_frequency: Counter[str] = Counter()
            for tokens in tokenized:
                document_frequency.update(set(tokens))
            average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
            scored: list[tuple[float, DocumentChunk]] = []
            for chunk, tokens in zip(chunks, tokenized, strict=True):
                frequencies = Counter(tokens)
                score = 0.0
                for term in query_terms:
                    frequency = frequencies[term]
                    if not frequency:
                        continue
                    df = document_frequency[term]
                    inverse_frequency = math.log1p((len(chunks) - df + 0.5) / (df + 0.5))
                    length_adjustment = 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1))
                    score += inverse_frequency * frequency * 2.5 / (frequency + length_adjustment)
                if score > 0:
                    scored.append((score, chunk))
            scored.sort(key=lambda item: item[0], reverse=True)
            results: list[RetrievedDocument] = []
            for rank, (raw_score, chunk) in enumerate(scored[: self.top_k], start=1):
                normalized_score = raw_score / (raw_score + 0.5)
                if normalized_score < self.min_score:
                    continue
                results.append(
                    RetrievedDocument(
                        text=chunk.text,
                        source=chunk.source,
                        score=normalized_score,
                        rank=rank,
                        source_url=chunk.source_url,
                        retrieved_at=chunk.retrieved_at,
                    )
                )
            logger.info("RAG retrieval mode=bm25 results=%d", len(results))
            return results
        except Exception as exc:
            logger.exception("Local BM25 fallback failed")
            raise RAGFailure("RAG fallback failed; check that the knowledge base contains readable Markdown") from exc


def _terms(text: str) -> list[str]:
    """Normalize accents and tokenize French/English text for lexical retrieval."""

    normalized = unicodedata.normalize("NFKD", text.casefold())
    unaccented = "".join(character for character in normalized if not unicodedata.combining(character))
    return [term for term in re.findall(r"[a-z0-9]{2,}", unaccented) if term not in _STOP_WORDS]


def _match_query_terms(query_terms: list[str], vocabulary: set[str]) -> list[str]:
    """Map obvious ASR spelling variants to nearby document terms."""

    matched: list[str] = []
    for term in query_terms:
        if term in vocabulary:
            matched.append(term)
            continue
        if len(term) < 5:
            continue
        candidates = [word for word in vocabulary if abs(len(word) - len(term)) <= 2]
        close = get_close_matches(term, candidates, n=1, cutoff=0.72)
        if close:
            matched.append(close[0])
    return list(dict.fromkeys(matched))
