"""Sentence Transformers adapter with a lazy model load."""

import threading

import numpy as np


class SentenceTransformerEmbedder:
    """Create multilingual dense embeddings using a configurable local model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: object | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("sentence-transformers is unavailable; install project dependencies") from exc
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text into normalized float32 vectors."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._get_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
