"""Download and load the configured faster-whisper model before a voice session."""

import logging
import time

from app.config import get_settings
from app.speech.asr import ASRService


def main() -> None:
    """Warm the same Whisper configuration used by the FastAPI application."""

    settings = get_settings()
    service = ASRService(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        cpu_threads=settings.whisper_cpu_threads,
        beam_size=settings.whisper_beam_size,
        initial_prompt=settings.whisper_initial_prompt,
    )
    print(
        f"Preparing Whisper model '{settings.whisper_model}' on {service.device} "
        f"({service.compute_type}). The first run may download model files.",
        flush=True,
    )
    started_at = time.perf_counter()
    try:
        service.load_model()
    except Exception as exc:
        logging.exception("Whisper model preparation failed")
        raise SystemExit(
            f"Whisper could not be prepared ({type(exc).__name__}: {exc}). "
            "Check internet access, free disk space, and the configured model name."
        ) from exc
    print(f"Whisper model ready in {time.perf_counter() - started_at:.1f}s. You can start the server.", flush=True)


if __name__ == "__main__":
    main()
