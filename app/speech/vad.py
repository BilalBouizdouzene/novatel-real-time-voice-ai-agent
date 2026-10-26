"""Stateful Silero VAD wrapper that yields complete PCM speech segments."""

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.speech.audio_utils import audio_duration_seconds, pcm16_bytes_to_float32

logger = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512  # Silero's supported 16 kHz window (32 ms).


@dataclass
class SpeechSegment:
    audio: bytes
    started_at: float
    vad_ms: float
    duration: float


@dataclass
class VADUpdate:
    events: list[str] = field(default_factory=list)
    segments: list[SpeechSegment] = field(default_factory=list)


class StreamingSileroVAD:
    """Run Silero on fixed windows and apply configurable speech/silence rules.

    ``probability_fn`` exists for deterministic unit tests. Production code uses
    ``from_silero`` so a missing model package is reported instead of silently
    substituting an energy threshold.
    """

    def __init__(
        self,
        probability_fn: Callable[[np.ndarray], float],
        *,
        threshold: float = 0.45,
        min_speech_ms: int = 200,
        min_silence_ms: int = 650,
        max_speech_ms: int = 20_000,
        pre_roll_ms: int = 200,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")
        self._probability_fn = probability_fn
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_speech_ms = max_speech_ms
        self._pre_roll_windows = max(1, round(pre_roll_ms / 32))
        self._pending = bytearray()
        self._pre_roll: deque[tuple[bytes, float]] = deque(maxlen=self._pre_roll_windows)
        self._speaking = False
        self._segment = bytearray()
        self._speech_started_at = 0.0
        self._speech_samples = 0
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._vad_processing_ms = 0.0

    @classmethod
    def from_silero(cls, **options: int | float) -> "StreamingSileroVAD":
        """Load the real Silero model; never replace it with a fake VAD."""

        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is unavailable; install project dependencies with `pip install -e .`"
            ) from exc

        model = load_silero_vad(onnx=False)
        if hasattr(model, "eval"):
            model.eval()

        def infer(window: np.ndarray) -> float:
            tensor = torch.from_numpy(window.astype(np.float32, copy=False))
            with torch.inference_mode():
                probability = model(tensor, SAMPLE_RATE)
            return float(probability.item())

        return cls(infer, **options)

    def feed(self, pcm16: bytes) -> VADUpdate:
        """Accept an arbitrary-sized PCM16 chunk and return detected events."""

        if len(pcm16) % 2:
            raise ValueError("PCM16 chunks must contain an even number of bytes")
        self._pending.extend(pcm16)
        update = VADUpdate()
        window_bytes = WINDOW_SAMPLES * 2
        while len(self._pending) >= window_bytes:
            raw_window = bytes(self._pending[:window_bytes])
            del self._pending[:window_bytes]
            window = pcm16_bytes_to_float32(raw_window)
            inference_started = time.perf_counter()
            probability = self._probability_fn(window)
            infer_ms = (time.perf_counter() - inference_started) * 1000
            is_speech = probability >= self.threshold
            frame_ms = WINDOW_SAMPLES / SAMPLE_RATE * 1000

            if not self._speaking:
                self._pre_roll.append((raw_window, infer_ms))
                if is_speech:
                    self._speaking = True
                    self._speech_started_at = time.perf_counter()
                    self._segment = bytearray(b"".join(frame for frame, _ in self._pre_roll))
                    self._speech_samples = len(self._segment) // 2
                    self._voiced_ms = frame_ms
                    self._silence_ms = 0.0
                    self._vad_processing_ms = sum(latency for _, latency in self._pre_roll)
                    update.events.append("speech_started")
                    logger.info("VAD | speech_started")
                continue

            self._vad_processing_ms += infer_ms
            self._segment.extend(raw_window)
            self._speech_samples += WINDOW_SAMPLES
            if is_speech:
                self._voiced_ms += frame_ms
                self._silence_ms = 0.0
            else:
                self._silence_ms += frame_ms

            segment_ms = self._speech_samples / SAMPLE_RATE * 1000
            if self._silence_ms >= self.min_silence_ms or segment_ms >= self.max_speech_ms:
                voiced_ms = self._voiced_ms
                segment = self._finish_segment()
                update.events.append("speech_ended")
                if segment is not None:
                    update.segments.append(segment)
                    logger.info("VAD | duration=%.2fs", segment.duration)
                else:
                    update.events.append("speech_discarded")
                    logger.info(
                        "VAD | speech_discarded voiced_ms=%.0f minimum_ms=%d segment_ms=%.0f",
                        voiced_ms,
                        self.min_speech_ms,
                        segment_ms,
                    )
        return update

    def reset(self) -> None:
        """Discard all buffered samples and conversation-local VAD state."""

        self._pending.clear()
        self._pre_roll.clear()
        self._speaking = False
        self._segment.clear()
        self._speech_samples = 0
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._vad_processing_ms = 0.0

    def _finish_segment(self) -> SpeechSegment | None:
        audio = bytes(self._segment)
        duration = audio_duration_seconds(len(audio) // 2, SAMPLE_RATE)
        segment: SpeechSegment | None = None
        if self._voiced_ms >= self.min_speech_ms:
            segment = SpeechSegment(
                audio=audio,
                started_at=self._speech_started_at,
                vad_ms=self._vad_processing_ms,
                duration=duration,
            )
        self._speaking = False
        self._segment.clear()
        self._pre_roll.clear()
        self._speech_samples = 0
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._vad_processing_ms = 0.0
        return segment
