"""Lazy faster-whisper transcription service with automatic CPU/GPU settings."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable

from app.models.schemas import TranscriptionResult
from app.speech.audio_utils import audio_duration_seconds, pcm16_bytes_to_float32

logger = logging.getLogger(__name__)


class ASRFailure(RuntimeError):
    """Raised when the configured speech recognition engine cannot transcribe."""


class ASRService:
    """Transcribe PCM16 audio with faster-whisper, loading its model on demand."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "fr",
        cpu_threads: int = 1,
        beam_size: int = 1,
        initial_prompt: str = "",
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device, self.compute_type = self._resolve_device(device, compute_type)
        self.language = language or None
        self.cpu_threads = max(1, cpu_threads)
        self.beam_size = max(1, beam_size)
        self.initial_prompt = initial_prompt.strip() or None
        self._model_factory = model_factory
        self._model: object | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def is_model_loaded(self) -> bool:
        """Whether this process has finished loading the Whisper model."""

        return self._model is not None

    def load_model(self) -> None:
        """Download if needed and load the configured model into memory."""

        self._get_model()

    @staticmethod
    def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
        selected_device = device
        if selected_device == "auto":
            try:
                import ctranslate2

                selected_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except (ImportError, RuntimeError):
                selected_device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if selected_device == "cuda" else "int8"
        return selected_device, compute_type

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                factory = self._model_factory
                if factory is None:
                    try:
                        from faster_whisper import WhisperModel
                    except ImportError as exc:
                        raise ASRFailure("faster-whisper is unavailable; install project dependencies") from exc
                    factory = WhisperModel
                logger.info(
                    "Loading Whisper model=%s device=%s compute_type=%s",
                    self.model_name,
                    self.device,
                    self.compute_type,
                )
                load_started = time.perf_counter()
                self._model = factory(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=1,
                )
                logger.info(
                    "Whisper model ready model=%s load_seconds=%.1f",
                    self.model_name,
                    time.perf_counter() - load_started,
                )
        return self._model

    async def transcribe(self, pcm16_audio: bytes, sample_rate: int = 16_000) -> TranscriptionResult:
        """Run inference off the event loop and return transcript plus timings."""

        return await asyncio.to_thread(self._transcribe_sync, pcm16_audio, sample_rate)

    def _transcribe_sync(self, pcm16_audio: bytes, sample_rate: int) -> TranscriptionResult:
        if not pcm16_audio:
            raise ASRFailure("Cannot transcribe empty audio")
        if sample_rate != 16_000:
            raise ASRFailure("faster-whisper input must be resampled to 16 kHz")
        samples = pcm16_bytes_to_float32(pcm16_audio)
        started = time.perf_counter()
        try:
            # Keep concurrent sessions from exhausting CPU memory with parallel decodes.
            with self._inference_lock:
                model = self._get_model()
                logger.info(
                    "ASR | transcribing audio_seconds=%.2f beam_size=%d",
                    audio_duration_seconds(samples.size, sample_rate),
                    self.beam_size,
                )
                segments, info = model.transcribe(  # type: ignore[attr-defined]
                    samples,
                    language=self.language,
                    vad_filter=False,
                    beam_size=self.beam_size,
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
        except ASRFailure:
            raise
        except Exception as exc:
            logger.exception("ASR transcription failed")
            raise ASRFailure("Speech recognition failed") from exc
        processing_time = time.perf_counter() - started
        logger.info(
            "ASR completed model=%s audio_seconds=%.2f processing_seconds=%.2f chars=%d",
            self.model_name,
            audio_duration_seconds(samples.size, sample_rate),
            processing_time,
            len(text),
        )
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=audio_duration_seconds(samples.size, sample_rate),
            processing_time=processing_time,
        )
