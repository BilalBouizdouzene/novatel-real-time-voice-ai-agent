import asyncio
import io
import json
import time
import wave
from pathlib import Path

import httpx
import numpy as np

from app.models.schemas import AudioResult, TranscriptionResult
from app.pipeline.state import ConversationSession
from app.pipeline.voice_agent import VoiceAgentPipeline
from app.speech.audio_utils import wav_bytes_to_pcm16
from app.speech.vad import SpeechSegment
from app.tools.support_handoff import CRMWebhookClient, SupportCaseService


def test_support_case_is_idempotent_claimable_and_persists_consented_messages(tmp_path: Path) -> None:
    service = SupportCaseService(tmp_path / "support.sqlite3")
    case = service.create_case(
        session_id="session-a",
        category="billing_question",
        summary="Customer asks about a duplicate payment.",
        idempotency_key="confirmed-request-1",
    )
    assert (
        service.create_case(
            session_id="session-a",
            category="billing_question",
            summary="different retry text is ignored",
            idempotency_key="confirmed-request-1",
        )
        == case
    )
    assert service.claim(case.case_id, "agent-a").status == "in_progress"
    assert service.claim(case.case_id, "agent-b") is None
    message = service.record_message(case.case_id, "advisor", "I can help with that payment.")
    assert service.list_messages(case.case_id) == [message]
    assert service.resolve(case.case_id, "agent-b") is None
    assert service.resolve(case.case_id, "agent-a").status == "resolved"

    reopened = SupportCaseService(tmp_path / "support.sqlite3")
    assert reopened.get_case(case.case_id).status == "resolved"
    assert [event["event"] for event in reopened.list_audit_events(case.case_id)] == [
        "case_created",
        "case_claimed",
        "message_advisor",
        "case_resolved",
    ]


def test_disconnect_requeues_unfinished_agent_cases(tmp_path: Path) -> None:
    service = SupportCaseService(tmp_path / "requeue.sqlite3")
    case = service.create_case(
        session_id="session-a",
        category="advisor_request",
        summary="Please call me back.",
        idempotency_key="requeue-1",
    )
    service.claim(case.case_id, "agent-a")

    released = service.release_agent_cases("agent-a")

    assert len(released) == 1
    assert released[0].status == "waiting"
    assert released[0].assigned_to is None
    assert service.list_audit_events(case.case_id)[-1]["event"] == "case_requeued"


def test_crm_webhook_sends_minimal_case_with_bearer_and_idempotency() -> None:
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"ticket_id": "CRM-501"})

    client = CRMWebhookClient(
        "https://crm.example.test/hooks/support",
        "secret-token",
        transport=httpx.MockTransport(handler),
    )
    service = SupportCaseService(":memory:")
    case = service.create_case(
        session_id="session-xyz",
        category="advisor_request",
        summary="Please connect me to an advisor.",
        idempotency_key="confirmed-2",
    )

    reference = asyncio.run(client.deliver(case))

    assert reference == "CRM-501"
    assert requests[0].headers["Authorization"] == "Bearer secret-token"
    assert requests[0].headers["Idempotency-Key"] == case.case_id
    payload = json.loads(requests[0].content)
    assert payload["event"] == "support_case.created"
    assert payload["case"]["summary"] == case.summary
    assert "conversation_history" not in payload["case"]


def test_wav_to_pcm16_normalizes_mono_rate_and_sample_width() -> None:
    samples = (np.sin(np.linspace(0, 2 * np.pi, 1600, endpoint=False)) * 20_000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(samples.tobytes())

    pcm, sample_rate = wav_bytes_to_pcm16(buffer.getvalue(), target_rate=24_000)

    assert sample_rate == 24_000
    assert len(pcm) // 2 == 2400
    assert np.max(np.abs(np.frombuffer(pcm, dtype="<i2"))) > 0


def test_human_handoff_requires_confirmation_then_routes_followup_to_agent_queue(tmp_path: Path) -> None:
    class SequenceASR:
        def __init__(self) -> None:
            self.items = iter(
                [
                    "Je veux parler à un conseiller.",
                    "Oui.",
                    "J’ai aussi une question sur ma facture.",
                ]
            )

        async def transcribe(self, _audio, sample_rate=16_000):
            return TranscriptionResult(text=next(self.items), language="fr", duration=1, processing_time=0.01)

    class NeverRetriever:
        def retrieve(self, _query):
            raise AssertionError("Human handoff does not need RAG")

    class NeverLLM:
        async def stream_chat(self, *_args):
            raise AssertionError("Human handoff must not call the LLM")
            yield "unreachable"

    class FakeTTS:
        async def synthesize(self, text):
            return AudioResult(audio=b"wav", sample_rate=24_000, duration=0.2, processing_time=0.01, backend="test")

    async def run() -> None:
        events = []
        audio = []

        async def send_event(event):
            events.append(event)

        async def send_audio(data):
            audio.append(data)

        service = SupportCaseService(tmp_path / "pipeline-support.sqlite3")
        session = ConversationSession()
        pipeline = VoiceAgentPipeline(
            session,
            object(),
            SequenceASR(),
            NeverRetriever(),
            NeverLLM(),
            FakeTTS(),
            send_event,
            send_audio,
            support_case_service=service,
        )

        for _ in range(3):
            await asyncio.wait_for(
                pipeline._process_segment(SpeechSegment(b"pcm", time.perf_counter(), 1, 1)), timeout=3
            )

        case = service.list_open_cases()[0]
        assert case.status == "waiting"
        assert case.crm_status == "not_configured"
        assert case.category == "advisor_request"
        assert len(service.list_messages(case.case_id)) == 2
        assert [event["type"] for event in events].count("handoff_prompt") == 1
        assert any(event["type"] == "support_case" and event["case_id"] == case.case_id for event in events)
        assert any(event["type"] == "handoff_message_sent" for event in events)
        assert session.active_handoff_case_id == case.case_id
        assert len(audio) == 3  # Prompt, confirmation, and the spoken acknowledgment of the queued message.

    asyncio.run(run())


def test_yes_after_rag_abstention_opens_consented_handoff_case(tmp_path: Path) -> None:
    class SequenceASR:
        def __init__(self) -> None:
            self.items = iter(["Quelle est la politique de roaming au Japon ?", "Oui, transmettez ma demande."])

        async def transcribe(self, _audio, sample_rate=16_000):
            return TranscriptionResult(text=next(self.items), language="fr", duration=1, processing_time=0.01)

    class EmptyRetriever:
        def retrieve(self, _query):
            return []

    class NeverLLM:
        async def stream_chat(self, *_args):
            raise AssertionError("No evidence should cause abstention, not LLM generation")
            yield "unreachable"

    class FakeTTS:
        async def synthesize(self, text):
            return AudioResult(audio=b"wav", sample_rate=24_000, duration=0.2, processing_time=0.01, backend="test")

    async def run() -> None:
        events = []

        async def send_event(event):
            events.append(event)

        async def send_audio(_data):
            return None

        service = SupportCaseService(tmp_path / "knowledge-gap.sqlite3")
        pipeline = VoiceAgentPipeline(
            ConversationSession(),
            object(),
            SequenceASR(),
            EmptyRetriever(),
            NeverLLM(),
            FakeTTS(),
            send_event,
            send_audio,
            support_case_service=service,
        )
        segment = SpeechSegment(b"pcm", time.perf_counter(), 1, 1)

        await pipeline._process_segment(segment)
        assert service.list_open_cases() == []
        assert "Le souhaitez-vous" in next(event["text"] for event in events if event["type"] == "llm_response")
        await pipeline._process_segment(segment)

        case = service.list_open_cases()[0]
        assert case.category == "knowledge_gap"
        assert case.summary == "Quelle est la politique de roaming au Japon ?"
        assert len(service.list_messages(case.case_id)) == 1

    asyncio.run(run())


def test_declining_handoff_does_not_create_or_persist_a_case(tmp_path: Path) -> None:
    class SequenceASR:
        def __init__(self) -> None:
            self.items = iter(["Je veux parler à un conseiller.", "Non merci, ne transmettez rien."])

        async def transcribe(self, _audio, sample_rate=16_000):
            return TranscriptionResult(text=next(self.items), language="fr", duration=1, processing_time=0.01)

    class NeverRetriever:
        def retrieve(self, _query):
            raise AssertionError("Explicit advisor requests should not run retrieval")

    class NeverLLM:
        async def stream_chat(self, *_args):
            raise AssertionError("A declined handoff should not call the LLM")
            yield "unreachable"

    class FakeTTS:
        async def synthesize(self, text):
            return AudioResult(audio=b"wav", sample_rate=24_000, duration=0.2, processing_time=0.01, backend="test")

    async def run() -> None:
        events = []

        async def send_event(event):
            events.append(event)

        async def send_audio(_data):
            return None

        service = SupportCaseService(tmp_path / "declined-handoff.sqlite3")
        pipeline = VoiceAgentPipeline(
            ConversationSession(),
            object(),
            SequenceASR(),
            NeverRetriever(),
            NeverLLM(),
            FakeTTS(),
            send_event,
            send_audio,
            support_case_service=service,
        )
        segment = SpeechSegment(b"pcm", time.perf_counter(), 1, 1)

        await pipeline._process_segment(segment)
        await pipeline._process_segment(segment)

        assert service.list_open_cases() == []
        assert not any(event["type"] == "support_case" for event in events)
        answers = [event["text"] for event in events if event["type"] == "llm_response"]
        answer = answers[-1]
        assert "aucun dossier n’a été créé" in answer

    asyncio.run(run())
