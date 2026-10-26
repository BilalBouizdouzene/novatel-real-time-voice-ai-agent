from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.pipeline.intents import NovaTelIntent, classify_intent, retrieval_hint
from app.rag.ingestion import DocumentChunk, chunk_text, load_markdown_documents
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import SearchHit


class KeywordEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [float("facture" in lower or "paiement" in lower), float("résiliation" in lower or "résilier" in lower)]
            )
        return np.asarray(vectors, dtype=np.float32)


class MemoryStore:
    def __init__(self):
        self.records: list[DocumentChunk] = []
        self.vectors = np.empty((0, 2), dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def load(self):
        return bool(self.records)

    def add(self, chunks, vectors):
        self.records.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors])

    def save(self):
        return None

    def search(self, vector, top_k):
        scores = self.vectors @ vector
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchHit(self.records[i].text, self.records[i].source, float(scores[i])) for i in order]


def test_chunking_keeps_source_and_respects_size() -> None:
    chunks = chunk_text("A" * 80, "faq.md", chunk_size=30, overlap=5)
    assert len(chunks) >= 3
    assert all(chunk.source == "faq.md" for chunk in chunks)
    assert all(len(chunk.text) <= 30 for chunk in chunks)


def test_retrieval_returns_relevant_chunk_and_filters_unknown(tmp_path: Path) -> None:
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "billing.md").write_text("Consultez votre facture dans la rubrique facturation.\n", encoding="utf-8")
    (docs / "cancel.md").write_text("Pour résilier, ouvrez la rubrique abonnement.\n", encoding="utf-8")
    store = MemoryStore()
    retriever = RAGRetriever(
        docs,
        tmp_path / "index",
        model_name="unused-in-test",
        top_k=2,
        min_score=0.7,
        embedder=KeywordEmbedder(),
        vector_store=store,
    )
    billing = retriever.retrieve("Où trouver ma facture ?")
    assert billing and billing[0].source == "billing.md"
    assert billing[0].rank == 1
    assert retriever.retrieve("Quelle est la météo ?") == []


def test_empty_knowledge_base_returns_no_documents(tmp_path: Path) -> None:
    retriever = RAGRetriever(
        tmp_path / "missing",
        tmp_path / "index",
        model_name="unused-in-test",
        embedder=KeywordEmbedder(),
        vector_store=MemoryStore(),
    )
    assert retriever.retrieve("question") == []


def test_retriever_falls_back_to_bm25_when_dense_model_fails(tmp_path: Path) -> None:
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "billing.md").write_text(
        "Pour obtenir un remboursement, contactez le service facturation avec votre numéro de commande.",
        encoding="utf-8",
    )

    class BrokenEmbedder:
        def encode(self, _texts):
            raise OSError("model download failed")

    retriever = RAGRetriever(
        docs,
        tmp_path / "index",
        model_name="unused-in-test",
        embedder=BrokenEmbedder(),
        vector_store=MemoryStore(),
    )
    results = retriever.retrieve("Comment obtenir un remboursement ?")

    assert retriever.mode == "bm25_fallback"
    assert results and results[0].source == "billing.md"
    assert retriever.retrieve("Quelle est la météo ?") == []
    assert retriever.retrieve("J'en...") == []


def test_bm25_matches_minor_asr_spelling_errors(tmp_path: Path) -> None:
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "subscriptions.md").write_text(
        "Pour résilier votre abonnement, ouvrez la rubrique Abonnement puis Résilier.",
        encoding="utf-8",
    )

    class BrokenEmbedder:
        def encode(self, _texts):
            raise OSError("dense model unavailable")

    retriever = RAGRetriever(
        docs,
        tmp_path / "index",
        model_name="unused-in-test",
        embedder=BrokenEmbedder(),
        vector_store=MemoryStore(),
    )
    results = retriever.retrieve("Comment résidiez mon abonement ?")

    assert results
    assert results[0].source == "subscriptions.md"


def test_noisy_subscription_question_still_retrieves_cancellation_guidance() -> None:
    knowledge_base = Path(__file__).resolve().parents[1] / "knowledge_base" / "novatel"

    class BrokenEmbedder:
        def encode(self, _texts):
            raise OSError("dense retrieval disabled in test")

    query = "Comment je paris de diriez mon abonnement ?"
    intent = classify_intent(query)
    retriever = RAGRetriever(
        knowledge_base,
        knowledge_base.parent / "test-index-not-written",
        model_name="unused-in-test",
        embedder=BrokenEmbedder(),
        vector_store=MemoryStore(),
    )

    assert intent == NovaTelIntent.PLAN_INFORMATION
    results = retriever.retrieve(f"{query}\n{retrieval_hint(intent, query)}")

    assert results
    assert results[0].source == "cancellation.md"


def test_low_disk_space_uses_bm25_without_loading_embedding_model(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "billing.md").write_text("La facture se trouve dans votre espace client.", encoding="utf-8")
    monkeypatch.setattr(
        "app.rag.retriever.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=100 * 1024 * 1024),
    )
    retriever = RAGRetriever(
        docs,
        tmp_path / "index",
        model_name="model-is-not-loaded-in-this-test",
        dense_min_free_disk_mb=500,
        vector_store=MemoryStore(),
    )

    results = retriever.retrieve("Où trouver ma facture ?")

    assert retriever.mode == "bm25_fallback"
    assert results and results[0].source == "billing.md"


def test_runtime_corpus_is_novatel_only_and_searchable() -> None:
    knowledge_base = Path(__file__).resolve().parents[1] / "knowledge_base" / "novatel"
    chunks = load_markdown_documents(knowledge_base)
    sources = {chunk.source for chunk in chunks}
    assert {
        "faq.md",
        "mobile_plans.md",
        "internet_plans.md",
        "billing.md",
        "payments.md",
        "cancellation.md",
        "refunds.md",
        "address_change.md",
        "network_support.md",
        "roaming.md",
        "customer_service.md",
        "escalation_policy.md",
    } <= sources
    assert all("public_guidance/" not in source for source in sources)

    class BrokenEmbedder:
        def encode(self, _texts):
            raise OSError("dense retrieval disabled in test")

    retriever = RAGRetriever(
        knowledge_base,
        knowledge_base.parent / "test-index-not-written",
        model_name="unused-in-test",
        embedder=BrokenEmbedder(),
        vector_store=MemoryStore(),
    )
    results = retriever.retrieve("Comment résilier mon forfait NovaTel ?")
    assert results
    assert results[0].source == "cancellation.md"


def test_novatel_support_corpus_covers_all_major_topics() -> None:
    knowledge_base = Path(__file__).resolve().parents[1] / "knowledge_base" / "novatel"

    class BrokenEmbedder:
        def encode(self, _texts):
            raise OSError("dense retrieval disabled in test")

    retriever = RAGRetriever(
        knowledge_base,
        knowledge_base.parent.parent / "test-index-not-written",
        model_name="unused-in-test",
        embedder=BrokenEmbedder(),
        vector_store=MemoryStore(),
        top_k=8,
    )
    cases = [
        ("Pourquoi ma facture est plus élevée ce mois-ci ?", "billing.md"),
        ("Mon paiement par carte est refusé, que faire ?", "payments.md"),
        ("Comment modifier mon adresse de contact ?", "address_change.md"),
        ("Comment changer mon offre mobile ?", "mobile_plans.md"),
        ("Je veux résilier mon forfait", "cancellation.md"),
        ("Je n'ai plus internet depuis ce matin", "network_support.md"),
        ("Comment obtenir un remboursement ?", "refunds.md"),
        ("Quel forfait mobile choisir selon l'enveloppe de données ?", "mobile_plans.md"),
        ("Pouvez-vous garantir une remise de 70 % ?", "mobile_plans.md"),
        ("Quelle enveloppe data est incluse en itinérance ?", "roaming.md"),
        ("Je veux parler à un conseiller", "escalation_policy.md"),
        ("Où voir le statut de ma demande ?", "escalation_policy.md"),
    ]

    for query, expected_source in cases:
        results = retriever.retrieve(query)
        assert any(result.source == expected_source for result in results), (query, results)
