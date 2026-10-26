import numpy as np

from app.speech.audio_utils import float32_to_pcm16_bytes
from app.speech.vad import StreamingSileroVAD


def _frame(value: float = 0.0) -> bytes:
    return float32_to_pcm16_bytes(np.full(512, value, dtype=np.float32))


def test_silence_does_not_create_speech_segment() -> None:
    vad = StreamingSileroVAD(lambda _: 0.01, min_speech_ms=64, min_silence_ms=64)
    result = vad.feed(_frame() * 20)
    assert result.events == []
    assert result.segments == []


def test_speech_and_silence_produce_one_complete_segment() -> None:
    probabilities = iter([0.9] * 12 + [0.05] * 20)
    vad = StreamingSileroVAD(lambda _: next(probabilities), threshold=0.5, min_speech_ms=100, min_silence_ms=128)
    result = vad.feed(_frame(0.1) * 32)
    assert result.events == ["speech_started", "speech_ended"]
    assert len(result.segments) == 1
    assert result.segments[0].duration > 0.3
    assert result.segments[0].audio


def test_empty_audio_is_safe() -> None:
    vad = StreamingSileroVAD(lambda _: 0.0)
    assert vad.feed(b"").events == []


def test_too_short_speech_is_discarded() -> None:
    probabilities = iter([0.9, 0.05, 0.05])
    vad = StreamingSileroVAD(lambda _: next(probabilities), min_speech_ms=200, min_silence_ms=64)
    result = vad.feed(_frame(0.1) * 3)
    assert result.events == ["speech_started", "speech_ended", "speech_discarded"]
    assert result.segments == []
