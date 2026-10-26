import json
import wave
from pathlib import Path

import numpy as np
import pytest

from app.models.schemas import RetrievedDocument
from app.monitoring.evaluation import evaluate_text_cases
from app.rag.ingestion import DocumentChunk
from app.rag.retriever import RAGRetriever
from app.speech.audio_utils import float32_to_pcm16_bytes, pcm16_bytes_to_float32
from app.speech.evaluation import add_white_noise_snr
from app.tools.demo_ticket import DemoTicketService
from scripts.load_test_websocket import percentile, read_audio


class KeywordEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float("facture" in text.lower()), float("paiement" in text.lower())] for text in texts],
            dtype=np.float32,
        )


class MemoryStore:
    def __init__(self) -> None:
        self.records: list[DocumentChunk] = []
        self.vectors = np.empty((0, 2), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def load(self) -> bool:
        return bool(self.records)

    def add(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None:
        self.records.extend(chunks)
        self.vectors = np.vstack((self.vectors, vectors))

    def save(self) -> None:
        return None

    def search(self, vector: np.ndarray, top_k: int):
        scores = self.vectors @ vector
        order = np.argsort(scores)[::-1][:top_k]
        return [
            type(
                "Hit",
                (),
                {"text": self.records[index].text, "source": self.records[index].source, "score": float(scores[index])},
            )()
            for index in order
        ]


def test_demo_ticket_sqlite_is_idempotent_and_audited(tmp_path: Path) -> None:
    database = tmp_path / "tickets.sqlite3"
    service = DemoTicketService(database)
    request_id = "stable-confirmation-id"
    service.record_event("session-a", request_id, "billing_question", "confirmation_requested")

    first = service.create("billing_question", idempotency_key=request_id, session_id="session-a")
    second = service.create("billing_question", idempotency_key=request_id, session_id="session-a")

    assert first == second
    assert len(service.list_tickets()) == 1
    assert [event["event"] for event in service.list_audit_events()] == ["confirmation_requested", "ticket_created"]
    reopened = DemoTicketService(database)
    assert reopened.list_tickets()[0]["ticket_id"] == first.ticket_id


def test_ticket_idempotency_key_cannot_change_category() -> None:
    service = DemoTicketService()
    service.create("billing_question", idempotency_key="request-1")
    with pytest.raises(ValueError, match="different ticket category"):
        service.create("network_issue", idempotency_key="request-1")


def test_hybrid_retrieval_fuses_dense_and_bm25_ranks(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "billing.md").write_text("Facture dans la rubrique facturation.", encoding="utf-8")
    (kb / "payments.md").write_text("Paiement par carte et prélèvement.", encoding="utf-8")
    retriever = RAGRetriever(
        kb,
        tmp_path / "index",
        model_name="unused-test-model",
        strategy="hybrid",
        top_k=2,
        min_score=0.0,
        embedder=KeywordEmbedder(),
        vector_store=MemoryStore(),
    )

    results = retriever.retrieve("facture paiement")

    assert {document.source for document in results} == {"billing.md", "payments.md"}
    assert retriever.mode == "hybrid_rrf"
    assert all(0 < document.score < 0.04 for document in results)


@pytest.mark.parametrize(
    ("strategy", "expected_mode"),
    [("dense", "bm25_fallback"), ("hybrid", "hybrid_bm25_fallback")],
)
def test_requested_dense_strategies_report_bm25_fallback(tmp_path: Path, strategy: str, expected_mode: str) -> None:
    kb = tmp_path / strategy
    kb.mkdir()
    (kb / "cancellation.md").write_text("Pour résilier un abonnement, ouvrez votre espace client.", encoding="utf-8")

    class BrokenEmbedder:
        def encode(self, _texts: list[str]) -> np.ndarray:
            raise OSError("intentionally unavailable in this test")

    retriever = RAGRetriever(
        kb,
        tmp_path / f"{strategy}-index",
        model_name="unused",
        strategy=strategy,
        min_score=0.0,
        embedder=BrokenEmbedder(),
        vector_store=MemoryStore(),
    )

    results = retriever.retrieve("Comment résilier mon abonnement ?")

    assert results[0].source == "cancellation.md"
    assert retriever.mode == expected_mode


def test_configured_reranker_can_change_order_without_model_download(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "billing.md").write_text("Facture facturation de votre compte.", encoding="utf-8")
    (kb / "payments.md").write_text("Facture liée à un paiement par carte.", encoding="utf-8")

    baseline = RAGRetriever(
        kb,
        tmp_path / "baseline-index",
        model_name="unused",
        strategy="bm25",
        min_score=0,
    ).retrieve("facture")

    class ReverseReranker:
        def rerank(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
            return [
                document.model_copy(update={"score": float(rank), "rank": rank})
                for rank, document in enumerate(reversed(documents), start=1)
            ]

    retriever = RAGRetriever(
        kb,
        tmp_path / "index",
        model_name="unused",
        strategy="bm25",
        min_score=0,
        reranker=ReverseReranker(),
    )

    reranked = retriever.retrieve("facture")
    assert len(baseline) == 2
    assert reranked[0].source == baseline[-1].source
    assert retriever.mode == "bm25_reranked"


def test_noise_augmentation_is_deterministic_and_keeps_input_unchanged() -> None:
    original = float32_to_pcm16_bytes(np.full(16_000, 0.1, dtype=np.float32))

    first = add_white_noise_snr(original, 20, seed=42)
    second = add_white_noise_snr(original, 20, seed=42)

    assert first == second
    assert first != original
    assert len(first) == len(original)
    assert len(pcm16_bytes_to_float32(original)) == 16_000


def test_noise_augmentation_rejects_empty_and_silent_audio() -> None:
    with pytest.raises(ValueError, match="empty audio"):
        add_white_noise_snr(b"", 10)
    with pytest.raises(ValueError, match="silent audio"):
        add_white_noise_snr(bytes(320), 10)


def test_robustness_benchmark_cases_are_valid_jsonl() -> None:
    path = Path(__file__).resolve().parents[1] / "evals" / "robustness_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cases) >= 12
    assert any("injection" in case["id"] for case in cases)
    assert any(case["expected_intent"] == "out_of_scope" for case in cases)
    assert all(case.get("query") and case.get("expected_intent") for case in cases)


def test_robustness_corpus_runs_without_model_downloads() -> None:
    root = Path(__file__).resolve().parents[1]
    cases_path = root / "evals" / "robustness_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = RAGRetriever(
        root / "knowledge_base" / "novatel",
        Path("unused-index"),
        model_name="not-loaded-in-bm25-mode",
        strategy="bm25",
    )

    report = evaluate_text_cases(cases, retriever)

    assert report["intent_accuracy"] == 1.0
    assert report["retrieval_hit_at_k"] == 1.0
    assert report["abstention_accuracy"] == 1.0


def test_load_test_percentile_and_wav_format_validation(tmp_path: Path) -> None:
    assert percentile([100, 1, 50], 0.5) == 50
    assert percentile([], 0.95) is None
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(float32_to_pcm16_bytes(np.ones(1024, dtype=np.float32) * 0.1))
    audio, rate = read_audio(audio_path)
    assert rate == 16_000
    assert len(audio) == 2048

    invalid_path = tmp_path / "wrong-rate.wav"
    with wave.open(str(invalid_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)
        wav_file.writeframes(bytes(256))
    with pytest.raises(ValueError, match="16 kHz"):
        read_audio(invalid_path)
