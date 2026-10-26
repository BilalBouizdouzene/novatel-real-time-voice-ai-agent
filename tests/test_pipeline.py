import asyncio
import time

from app.models.schemas import AgentState, AudioResult, RetrievedDocument, TranscriptionResult
from app.pipeline.state import ConversationSession
from app.pipeline.voice_agent import VoiceAgentPipeline
from app.speech.vad import SpeechSegment, VADUpdate


class FakeVAD:
    def reset(self):
        return None


class FakeASR:
    async def transcribe(self, _audio, sample_rate=16_000):
        return TranscriptionResult(
            text="Comment consulter ma facture ?", language="fr", duration=1, processing_time=0.01
        )


class FakeRetriever:
    def retrieve(self, _query):
        return [RetrievedDocument(text="Les factures sont dans Facturation.", source="faq.md", score=0.9, rank=1)]


class FakeLLM:
    async def stream_chat(self, _query, _documents, _history):
        yield "Ouvrez la rubrique "
        yield "Facturation de votre espace client."


class FakeTTS:
    async def synthesize(self, _text):
        return AudioResult(audio=b"wav", sample_rate=22_050, duration=0.4, processing_time=0.02, backend="test")


def test_complete_mocked_turn_runs_through_rag_llm_and_tts() -> None:
    async def run():
        events = []
        audio = []

        async def send_event(event):
            events.append(event)

        async def send_audio(data):
            audio.append(data)

        session = ConversationSession()
        pipeline = VoiceAgentPipeline(
            session,
            FakeVAD(),
            FakeASR(),
            FakeRetriever(),
            FakeLLM(),
            FakeTTS(),
            send_event,
            send_audio,
        )
        await pipeline._process_segment(SpeechSegment(b"pcm", time.perf_counter(), 1.0, 1.0))
        assert audio == [b"wav"]
        assert [message["role"] for message in session.history] == ["user", "assistant"]
        assert session.state == AgentState.LISTENING
        event_types = [event["type"] for event in events]
        assert "transcription" in event_types
        assert "retrieval" in event_types
        assert "llm_token" in event_types
        assert "metrics" in event_types

    asyncio.run(run())


def test_vad_during_asr_does_not_cancel_transcription_and_segments_are_queued() -> None:
    class UpdatingVAD:
        def __init__(self):
            self.updates = iter(
                [
                    VADUpdate(segments=[SpeechSegment(b"first", time.perf_counter(), 1.0, 0.8)]),
                    VADUpdate(
                        events=["speech_started", "speech_ended"],
                        segments=[SpeechSegment(b"second", time.perf_counter(), 1.0, 0.8)],
                    ),
                ]
            )

        def feed(self, _audio):
            return next(self.updates)

        def reset(self):
            return None

    class SlowASR:
        def __init__(self):
            self.first_started = asyncio.Event()
            self.calls = 0

        async def transcribe(self, audio, sample_rate=16_000):
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
            await asyncio.sleep(0.05)
            text = "Bonjour" if audio == b"first" else "Merci"
            return TranscriptionResult(text=text, language="fr", duration=0.8, processing_time=0.05)

    async def run():
        events = []
        audio = []

        async def send_event(event):
            events.append(event)

        async def send_audio(data):
            audio.append(data)

        asr = SlowASR()
        pipeline = VoiceAgentPipeline(
            ConversationSession(), UpdatingVAD(), asr, FakeRetriever(), FakeLLM(), FakeTTS(), send_event, send_audio
        )
        await pipeline.ingest_audio(b"chunk one")
        await asyncio.wait_for(asr.first_started.wait(), timeout=1)
        await pipeline.ingest_audio(b"chunk two")
        assert pipeline.session.state == AgentState.PROCESSING_ASR
        await asyncio.wait_for(pipeline._processing_task, timeout=2)

        assert asr.calls == 2
        assert len([event for event in events if event["type"] == "transcription"]) == 2
        assert not any(event["type"] == "assistant_interrupted" for event in events)
        assert len(audio) == 2
        await pipeline.close()

    asyncio.run(run())


def test_barge_in_cancels_assistant_work_when_new_speech_starts() -> None:
    class SpeakingVAD:
        def feed(self, _audio):
            return VADUpdate(events=["speech_started"])

        def reset(self):
            return None

    async def run():
        events = []
        cancelled = asyncio.Event()

        async def send_event(event):
            events.append(event)

        async def send_audio(_data):
            return None

        async def response_work():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        pipeline = VoiceAgentPipeline(
            ConversationSession(),
            SpeakingVAD(),
            FakeASR(),
            FakeRetriever(),
            FakeLLM(),
            FakeTTS(),
            send_event,
            send_audio,
        )
        pipeline.session.state = AgentState.SPEAKING
        pipeline._processing_task = asyncio.create_task(response_work())
        await asyncio.sleep(0)

        await pipeline.ingest_audio(b"new speech frame")

        assert cancelled.is_set()
        assert pipeline.session.state == AgentState.LISTENING
        assert any(event["type"] == "assistant_interrupted" for event in events)
        assert any(event["type"] == "speech_started" for event in events)

    asyncio.run(run())


def test_no_relevant_document_skips_llm_but_returns_honest_answer() -> None:
    class EmptyRetriever:
        def retrieve(self, _query):
            return []

    class FailingLLM:
        async def stream_chat(self, *_args):
            raise AssertionError("LLM must not run without relevant evidence")
            yield "unreachable"

    async def run():
        events = []

        async def send_event(event):
            events.append(event)

        async def send_audio(_data):
            return None

        pipeline = VoiceAgentPipeline(
            ConversationSession(),
            FakeVAD(),
            FakeASR(),
            EmptyRetriever(),
            FailingLLM(),
            FakeTTS(),
            send_event,
            send_audio,
        )
        await pipeline._process_segment(SpeechSegment(b"pcm", time.perf_counter(), 1.0, 1.0))
        answer = next(event["text"] for event in events if event["type"] == "llm_response")
        assert "pas trouvé" in answer

    asyncio.run(run())


def test_thanks_gets_a_quick_reply_without_rag_or_llm() -> None:
    class ThanksASR:
        async def transcribe(self, _audio, sample_rate=16_000):
            return TranscriptionResult(text="Merci.", language="fr", duration=0.5, processing_time=0.01)

    class NeverRetriever:
        def retrieve(self, _query):
            raise AssertionError("RAG should not run for a simple thank-you")

    class NeverLLM:
        async def stream_chat(self, *_args):
            raise AssertionError("LLM should not run for a simple thank-you")
            yield "unreachable"

    async def run():
        events = []
        audio = []

        async def send_event(event):
            events.append(event)

        async def send_audio(data):
            audio.append(data)

        pipeline = VoiceAgentPipeline(
            ConversationSession(),
            FakeVAD(),
            ThanksASR(),
            NeverRetriever(),
            NeverLLM(),
            FakeTTS(),
            send_event,
            send_audio,
        )
        await pipeline._process_segment(SpeechSegment(b"pcm", time.perf_counter(), 0.5, 0.5))

        response = next(event for event in events if event["type"] == "llm_response")
        retrieval = next(event for event in events if event["type"] == "retrieval")
        assert "Avec plaisir" in response["text"]
        assert retrieval["mode"] == "conversation"
        assert audio == [b"wav"]

    asyncio.run(run())


def test_follow_up_reuses_previous_topic_and_grounded_documents() -> None:
    follow_up_text = "Pour vivre mon donnis plus de détails ?"

    class SequenceASR:
        def __init__(self):
            self.texts = iter(["Comment je peux résilir mon album ?", follow_up_text])

        async def transcribe(self, _audio, sample_rate=16_000):
            return TranscriptionResult(text=next(self.texts), language="fr", duration=1, processing_time=0.01)

    class ContextRetriever:
        def __init__(self):
            self.queries = []

        def retrieve(self, query):
            self.queries.append(query)
            if len(self.queries) == 1:
                return [
                    RetrievedDocument(
                        text="Pour résilier, ouvrez Abonnement puis confirmez la demande.",
                        source="subscriptions.md",
                        score=0.9,
                        rank=1,
                    )
                ]
            return []

    class RecordingLLM:
        def __init__(self):
            self.calls = []

        async def stream_chat(self, query, documents, history):
            self.calls.append((query, documents, history))
            yield "Je peux préciser les étapes de résiliation."

    async def run():
        events = []

        async def send_event(event):
            events.append(event)

        async def send_audio(_data):
            return None

        retriever = ContextRetriever()
        llm = RecordingLLM()
        pipeline = VoiceAgentPipeline(
            ConversationSession(),
            FakeVAD(),
            SequenceASR(),
            retriever,
            llm,
            FakeTTS(),
            send_event,
            send_audio,
        )
        segment = SpeechSegment(b"pcm", time.perf_counter(), 1.0, 1.0)

        await pipeline._process_segment(segment)
        await pipeline._process_segment(segment)

        assert len(retriever.queries) == 2
        assert "Comment je peux résilir mon album ?" in retriever.queries[1]
        assert follow_up_text in retriever.queries[1]
        assert len(llm.calls) == 2
        assert llm.calls[1][0] == follow_up_text
        assert llm.calls[1][1][0].source == "subscriptions.md"
        assert any(
            message["role"] == "user" and message["content"] == "Comment je peux résilir mon album ?"
            for message in llm.calls[1][2]
        )
        latest_retrieval = [event for event in events if event["type"] == "retrieval"][-1]
        assert latest_retrieval["intent"] == "cancel_subscription"
        assert latest_retrieval["query"] == follow_up_text
        assert latest_retrieval["documents"][0]["content"] == latest_retrieval["documents"][0]["text"]
        assert latest_retrieval["documents"][0]["source"] == "subscriptions.md"
        assert latest_retrieval["mode"] == "conversation_context"

    asyncio.run(run())


def test_out_of_scope_question_never_calls_retriever_or_llm() -> None:
    class CanadaQuestionASR:
        async def transcribe(self, _audio, sample_rate=16_000):
            return TranscriptionResult(
                text="Quelle est la capitale du Canada ?", language="fr", duration=1, processing_time=0.01
            )

    class NeverRetriever:
        def retrieve(self, _query):
            raise AssertionError("out-of-scope questions must not search NovaTel documents")

    class NeverLLM:
        async def stream_chat(self, *_args):
            raise AssertionError("out-of-scope questions must not call the general LLM")
            yield "unreachable"

    async def run():
        events = []

        async def send_event(event):
            events.append(event)

        async def send_audio(_data):
            return None

        pipeline = VoiceAgentPipeline(
            ConversationSession(),
            FakeVAD(),
            CanadaQuestionASR(),
            NeverRetriever(),
            NeverLLM(),
            FakeTTS(),
            send_event,
            send_audio,
        )
        await pipeline._process_segment(SpeechSegment(b"pcm", time.perf_counter(), 1.0, 1.0))

        retrieval = next(event for event in events if event["type"] == "retrieval")
        answer = next(event for event in events if event["type"] == "llm_response")
        assert retrieval["intent"] == "out_of_scope"
        assert retrieval["mode"] == "out_of_scope"
        assert not retrieval["documents"]
        assert "NovaTel" in answer["text"]

    asyncio.run(run())
