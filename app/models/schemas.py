"""Typed payloads exchanged by the API, pipeline and frontend."""

from enum import StrEnum

from pydantic import BaseModel, Field


class AgentState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_ASR = "processing_asr"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    SPEAKING = "speaking"


class RetrievedDocument(BaseModel):
    text: str
    source: str
    score: float
    rank: int
    source_url: str | None = None
    retrieved_at: str | None = None


class TranscriptionResult(BaseModel):
    text: str
    language: str | None = None
    language_probability: float | None = None
    duration: float
    processing_time: float


class AudioResult(BaseModel):
    audio: bytes
    sample_rate: int
    duration: float
    processing_time: float
    content_type: str = "audio/wav"
    backend: str


class LatencyMetrics(BaseModel):
    vad_ms: float = 0.0
    asr_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    time_to_first_audio_ms: float | None = None


class ClientMessage(BaseModel):
    type: str = Field(description="start, stop, clear or ping")
