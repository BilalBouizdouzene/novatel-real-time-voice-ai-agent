"""Monotonic-clock latency tracking for one spoken turn."""

import logging
import time

from app.models.schemas import LatencyMetrics

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Collect measured stage durations without synthesizing placeholder data."""

    def __init__(self, started_at: float | None = None) -> None:
        self.started_at = started_at if started_at is not None else time.perf_counter()
        self.metrics = LatencyMetrics()

    @staticmethod
    def elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000

    def log(self) -> None:
        """Log actual measurements; optional streaming metrics stay nullable."""

        m = self.metrics
        logger.info(
            "LATENCY | VAD %.1f ms | ASR %.1f ms | RAG %.1f ms | LLM %.1f ms | TTS %.1f ms | TOTAL %.1f ms",
            m.vad_ms,
            m.asr_ms,
            m.retrieval_ms,
            m.llm_ms,
            m.tts_ms,
            m.total_ms,
        )
