"""Small deterministic intent/retrieval benchmark for the local demo corpus."""

import time
from collections.abc import Iterable
from typing import Protocol

from app.pipeline.intents import NovaTelIntent, classify_intent, retrieval_hint
from app.pipeline.quick_replies import quick_reply


class Retriever(Protocol):
    mode: str

    def retrieve(self, query: str): ...


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def evaluate_text_cases(cases: Iterable[dict[str, object]], retriever: Retriever) -> dict[str, object]:
    """Measure intent routing, source hit-rate, abstention, and local turn latency."""

    results: list[dict[str, object]] = []
    turn_latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        query = str(case["query"])
        actual_intent = "social" if quick_reply(query) is not None else classify_intent(query).value
        expected_intent = str(case["expected_intent"])
        retrieval_expected = bool(case.get("retrieval_expected", False))
        expected_sources = {str(source) for source in case.get("expected_sources", [])}  # type: ignore[union-attr]
        documents = []
        retrieval_mode = "not_searched"
        if retrieval_expected and actual_intent not in {NovaTelIntent.OUT_OF_SCOPE.value, "social"}:
            hint = retrieval_hint(NovaTelIntent(actual_intent), query)
            documents = retriever.retrieve(f"{query}\n{hint}" if hint else query)
            retrieval_mode = getattr(retriever, "mode", "unknown")
        source_ranks = [doc.rank for doc in documents if doc.source in expected_sources]
        rank = min(source_ranks) if source_ranks else None
        latency_ms = (time.perf_counter() - started) * 1000
        turn_latencies.append(latency_ms)
        results.append(
            {
                "id": str(case.get("id", "unnamed")),
                "intent_expected": expected_intent,
                "intent_actual": actual_intent,
                "intent_correct": actual_intent == expected_intent,
                "retrieval_expected": retrieval_expected,
                "retrieved_sources": [doc.source for doc in documents],
                "expected_source_rank": rank,
                "retrieval_hit": rank is not None if retrieval_expected else not documents,
                "latency_ms": round(latency_ms, 2),
                "retrieval_mode": retrieval_mode,
            }
        )

    intent_correct = [bool(result["intent_correct"]) for result in results]
    retrieval_cases = [result for result in results if result["retrieval_expected"]]
    abstention_cases = [result for result in results if not result["retrieval_expected"]]
    hit_ranks = [int(result["expected_source_rank"]) for result in retrieval_cases if result["expected_source_rank"]]
    return {
        "case_count": len(results),
        "intent_accuracy": sum(intent_correct) / len(intent_correct) if intent_correct else None,
        "retrieval_hit_at_k": sum(bool(row["retrieval_hit"]) for row in retrieval_cases) / len(retrieval_cases)
        if retrieval_cases
        else None,
        "mean_reciprocal_rank": sum(1 / rank for rank in hit_ranks) / len(retrieval_cases)
        if retrieval_cases
        else None,
        "abstention_accuracy": sum(bool(row["retrieval_hit"]) for row in abstention_cases) / len(abstention_cases)
        if abstention_cases
        else None,
        "turn_latency_ms": {
            "p50": _percentile(turn_latencies, 0.50),
            "p95": _percentile(turn_latencies, 0.95),
        },
        "latency_scope": "Intent routing plus local retrieval only; excludes microphone, ASR, LLM, TTS and network.",
        "cases": results,
    }
