"""Small helpers for real-time mono PCM16 delivery."""

from collections.abc import Iterator


def iter_pcm_frames(pcm: bytes, *, sample_rate: int, frame_ms: int = 40) -> Iterator[bytes]:
    """Yield PCM16 mono frames without dropping a short final frame."""

    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError("sample_rate and frame_ms must be positive")
    frame_bytes = max(2, int(sample_rate * frame_ms / 1000) * 2)
    frame_bytes -= frame_bytes % 2
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        if frame:
            yield frame
