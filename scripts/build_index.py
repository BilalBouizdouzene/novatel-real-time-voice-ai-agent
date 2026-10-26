"""Build and persist the local Sentence Transformers + FAISS knowledge index."""

from app.config import get_settings
from app.rag.retriever import RAGRetriever


def main() -> None:
    settings = get_settings()
    retriever = RAGRetriever(
        knowledge_base=settings.knowledge_base_dir,
        vector_store_dir=settings.vector_store_dir,
        model_name=settings.embedding_model,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    count = retriever.build_index(force=True)
    print(f"Index FAISS construit : {count} chunks. Dossier : {settings.vector_store_dir}")


if __name__ == "__main__":
    main()
