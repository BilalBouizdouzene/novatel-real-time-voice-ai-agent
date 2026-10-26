import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app.speech.asr import ASRFailure, ASRService
from app.speech.audio_utils import float32_to_pcm16_bytes


class FakeModel:
    def transcribe(self, audio, **kwargs):
        assert isinstance(audio, np.ndarray)
        assert kwargs["vad_filter"] is False
        assert kwargs["beam_size"] == 1
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["initial_prompt"] is None
        return [SimpleNamespace(text=" Bonjour le monde. ")], SimpleNamespace(language="fr", language_probability=0.98)


def test_transcribe_returns_text_language_and_measured_duration() -> None:
    service = ASRService(model_factory=lambda *args, **kwargs: FakeModel())
    assert not service.is_model_loaded
    audio = float32_to_pcm16_bytes(np.zeros(16_000, dtype=np.float32))
    result = asyncio.run(service.transcribe(audio))
    assert result.text == "Bonjour le monde."
    assert result.language == "fr"
    assert result.language_probability == 0.98
    assert result.duration == 1.0
    assert result.processing_time >= 0
    assert service.is_model_loaded


def test_model_can_be_loaded_before_first_voice_turn() -> None:
    service = ASRService(model_factory=lambda *args, **kwargs: FakeModel())

    service.load_model()

    assert service.is_model_loaded


def test_empty_audio_is_rejected() -> None:
    service = ASRService(model_factory=lambda *args, **kwargs: FakeModel())
    with pytest.raises(ASRFailure, match="empty audio"):
        asyncio.run(service.transcribe(b""))


def test_model_failure_is_reported_as_asr_failure() -> None:
    class BrokenModel:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("model unavailable")

    service = ASRService(model_factory=lambda *args, **kwargs: BrokenModel())
    audio = float32_to_pcm16_bytes(np.zeros(256, dtype=np.float32))
    with pytest.raises(ASRFailure, match="Speech recognition failed"):
        asyncio.run(service.transcribe(audio))
