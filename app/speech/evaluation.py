"""Reference-based WER/CER helpers, kept separate from production inference."""

import re

import numpy as np

from app.speech.audio_utils import float32_to_pcm16_bytes, pcm16_bytes_to_float32


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER from a reference transcript and a hypothesis."""

    reference_words = re.findall(r"\w+", reference.casefold(), flags=re.UNICODE)
    hypothesis_words = re.findall(r"\w+", hypothesis.casefold(), flags=re.UNICODE)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute CER after case-folding and removing whitespace."""

    reference_chars = list("".join(reference.casefold().split()))
    hypothesis_chars = list("".join(hypothesis.casefold().split()))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return _edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def add_white_noise_snr(pcm16_audio: bytes, snr_db: float, *, seed: int = 0) -> bytes:
    """Return an in-memory noisy PCM16 copy at the requested signal/noise ratio."""

    samples = pcm16_bytes_to_float32(pcm16_audio)
    if samples.size == 0:
        raise ValueError("Cannot augment empty audio")
    signal_rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if signal_rms <= 1e-8:
        raise ValueError("Cannot add SNR-controlled noise to silent audio")
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = np.random.default_rng(seed).normal(0.0, noise_rms, samples.size).astype(np.float32)
    return float32_to_pcm16_bytes(np.clip(samples + noise, -1.0, 1.0))
