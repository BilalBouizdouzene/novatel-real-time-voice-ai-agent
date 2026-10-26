"""Optional cross-encoder reranking for a small RAG candidate set."""

from app.models.schemas import RetrievedDocument


class CrossEncoderReranker:
    """Lazily load Sentence Transformers' CrossEncoder only when configured."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("A cross-encoder model name is required")
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to use the configured reranker") from exc
            self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, document.text) for document in documents])
        ranked = sorted(zip(documents, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
        return [
            document.model_copy(update={"score": float(score), "rank": rank})
            for rank, (document, score) in enumerate(ranked, start=1)
        ]
