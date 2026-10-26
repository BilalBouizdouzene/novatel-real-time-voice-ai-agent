"""Audio conversion helpers used at the WebSocket boundary and by models."""

import wave
from io import BytesIO

import numpy as np


def pcm16_bytes_to_float32(audio: bytes) -> np.ndarray:
    """Convert little-endian mono PCM16 bytes to normalized float32 samples."""

    if len(audio) % 2:
        raise ValueError("PCM16 audio must contain an even number of bytes")
    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Clip normalized float samples and encode little-endian PCM16."""

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio using polyphase filtering, with a NumPy fallback."""

    audio = np.asarray(samples, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate or audio.size == 0:
        return audio.copy()
    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(source_rate, target_rate)
        return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)
    except ImportError:
        output_length = round(audio.size * target_rate / source_rate)
        old_positions = np.arange(audio.size, dtype=np.float64) / source_rate
        new_positions = np.arange(output_length, dtype=np.float64) / target_rate
        return np.interp(new_positions, old_positions, audio).astype(np.float32)


def normalize_audio(samples: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Normalize a non-silent signal to a requested peak amplitude."""

    audio = np.asarray(samples, dtype=np.float32)
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum <= 1e-8:
        return audio.copy()
    return (audio * (peak / maximum)).astype(np.float32)


def calculate_rms(samples: np.ndarray) -> float:
    """Return the root-mean-square amplitude of a signal."""

    audio = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def audio_duration_seconds(sample_count: int, sample_rate: int) -> float:
    """Convert sample count to duration, guarding against invalid rates."""

    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    return sample_count / sample_rate


def wav_bytes_to_pcm16(audio: bytes, target_rate: int = 24_000) -> tuple[bytes, int]:
    """Convert a PCM WAV into mono PCM16 for low-overhead WebSocket streaming."""

    try:
        with wave.open(BytesIO(audio), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("Compressed TTS WAV is not supported for PCM streaming")
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            raw = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError("TTS backend returned an invalid WAV file") from exc
    if channels < 1 or sample_rate < 1:
        raise ValueError("TTS WAV must contain audio channels at a valid sample rate")

    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        integers = (
            octets[:, 0].astype(np.int32) | (octets[:, 1].astype(np.int32) << 8) | (octets[:, 2].astype(np.int32) << 16)
        )
        integers = (integers ^ 0x800000) - 0x800000
        samples = integers.astype(np.float32) / 8_388_608.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    else:
        raise ValueError(f"Unsupported TTS WAV sample width: {sample_width} bytes")

    if samples.size % channels:
        raise ValueError("TTS WAV has an incomplete audio frame")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != target_rate:
        samples = resample_audio(samples, sample_rate, target_rate)
    return float32_to_pcm16_bytes(samples), target_rate
