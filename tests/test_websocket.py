import io
import time
import wave

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.schemas import AudioResult, RetrievedDocument, TranscriptionResult
from app.speech.vad import SpeechSegment, VADUpdate


class OneTurnVAD:
    def feed(self, _audio):
        return VADUpdate(
            events=["speech_started", "speech_ended"],
            segments=[SpeechSegment(b"pcm", time.perf_counter(), 0.5, 0.6)],
        )

    def reset(self):
        return None


class FakeASR:
    async def transcribe(self, _audio, sample_rate=16_000):
        return TranscriptionResult(text="Où trouver ma facture ?", language="fr", duration=0.6, processing_time=0.01)


class FakeRetriever:
    def retrieve(self, _query):
        return [
            RetrievedDocument(
                text="Les factures se trouvent dans Facturation.",
                source="faq.md",
                score=0.8,
                rank=1,
            )
        ]


class FakeLLM:
    async def stream_chat(self, *_args):
        yield "Consultez la rubrique Facturation."


class FakeTTS:
    async def synthesize(self, _text):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22_050)
            wav_file.writeframes(b"\x00\x00" * 2205)
        return AudioResult(
            audio=buffer.getvalue(),
            sample_rate=22_050,
            duration=0.3,
            processing_time=0.02,
            backend="test",
        )


def test_websocket_transports_events_and_binary_audio() -> None:
    app = create_app(Settings())
    app.state.vad_factory = OneTurnVAD
    app.state.asr = FakeASR()
    app.state.retriever = FakeRetriever()
    app.state.llm = FakeLLM()
    app.state.tts = FakeTTS()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/voice") as websocket:
            assert websocket.receive_json()["type"] == "session"
            websocket.send_json({"type": "start"})
            assert websocket.receive_json()["type"] == "state"
            websocket.send_bytes(b"\x00\x00")

            event_types = []
            audio_frames = []
            audio_metadata = None
            for _ in range(30):
                message = websocket.receive()
                if message.get("bytes") is not None:
                    audio_frames.append(message["bytes"])
                    continue
                event = message.get("text")
                if event:
                    import json

                    parsed = json.loads(event)
                    event_types.append(parsed["type"])
                    if parsed["type"] == "tts_audio_start":
                        audio_metadata = parsed
                    if parsed["type"] == "metrics":
                        break

            assert len(b"".join(audio_frames)) == 4_800  # 0.1 seconds of 24 kHz mono PCM16.
            assert audio_metadata["content_type"] == "audio/pcm;format=s16le;channels=1"
            assert audio_metadata["sample_rate"] == 24_000
            assert "speech_started" in event_types
            assert "transcription" in event_types
            assert "retrieval" in event_types
            assert "llm_response" in event_types
            assert "metrics" in event_types
            websocket.send_json({"type": "stop"})
