"""Orchestrate VAD, ASR, RAG, LLM and TTS for a single live connection."""

import asyncio
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol
from uuid import uuid4

from app.llm.client import LLMFailure
from app.models.schemas import AgentState, RetrievedDocument, TranscriptionResult
from app.monitoring.latency import LatencyTracker
from app.monitoring.tracing import trace_span
from app.pipeline.follow_ups import contextual_retrieval_query, is_contextual_follow_up
from app.pipeline.intents import NovaTelIntent, classify_intent, retrieval_hint
from app.pipeline.quick_replies import quick_reply
from app.pipeline.state import ConversationSession
from app.rag.retriever import RAGFailure
from app.speech.asr import ASRFailure
from app.speech.tts import TTSFailure
from app.speech.vad import SpeechSegment, StreamingSileroVAD
from app.tools.demo_ticket import (
    DemoTicketService,
    confirmation_decision,
    requested_ticket_category,
    ticket_payload,
)
from app.tools.support_handoff import (
    CRMClient,
    CRMDeliveryFailure,
    CRMWebhookClient,
    HandoffHub,
    SupportCaseService,
    serialize_case,
)

logger = logging.getLogger(__name__)
EventSender = Callable[[dict[str, object]], Awaitable[None]]
AudioSender = Callable[[bytes], Awaitable[None]]


def _take_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Return text ending at sentence punctuation; retain any unfinished tail."""

    boundaries = [match.end() for match in re.finditer(r"(?<=[.!?])(?:[\"'»”)]*)\s+", buffer)]
    if not boundaries:
        return [], buffer
    sentences: list[str] = []
    start = 0
    for boundary in boundaries:
        sentence = buffer[start:boundary].strip()
        if sentence:
            sentences.append(sentence)
        start = boundary
    return sentences, buffer[start:]


class ASR(Protocol):
    async def transcribe(self, pcm16_audio: bytes, sample_rate: int = 16_000): ...


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedDocument]: ...


class LLM(Protocol):
    def stream_chat(
        self, user_query: str, retrieved_context: list[RetrievedDocument], conversation_history: list[dict[str, str]]
    ) -> AsyncIterator[str]: ...


class TTS(Protocol):
    async def synthesize(self, text: str): ...


class VoiceAgentPipeline:
    """One pipeline instance per WebSocket; cancels response work on barge-in."""

    def __init__(
        self,
        session: ConversationSession,
        vad: StreamingSileroVAD,
        asr: ASR,
        retriever: Retriever,
        llm: LLM,
        tts: TTS,
        send_event: EventSender,
        send_audio: AudioSender,
        *,
        max_history_messages: int = 8,
        demo_ticket_service: DemoTicketService | None = None,
        support_case_service: SupportCaseService | None = None,
        crm_client: CRMClient | None = None,
        handoff_hub: HandoffHub | None = None,
        handoff_enabled: bool = True,
        output_sample_rate: int = 24_000,
    ) -> None:
        self.session = session
        self.vad = vad
        self.asr = asr
        self.retriever = retriever
        self.llm = llm
        self.tts = tts
        self.send_event = send_event
        self.send_audio = send_audio
        self.max_history_messages = max_history_messages
        self.demo_tickets = demo_ticket_service or DemoTicketService()
        self.support_cases = support_case_service
        self.crm_client = crm_client or CRMWebhookClient()
        self.handoff_hub = handoff_hub or HandoffHub()
        self.handoff_enabled = handoff_enabled
        self.output_sample_rate = output_sample_rate
        self._processing_task: asyncio.Task[None] | None = None
        self._pending_segments: deque[SpeechSegment] = deque()
        self._closed = False

    async def ingest_audio(self, pcm16_audio: bytes) -> None:
        """Consume a microphone chunk, emit VAD events and schedule completed turns."""

        if self._closed:
            return
        update = await asyncio.to_thread(self.vad.feed, pcm16_audio)
        for event in update.events:
            if event == "speech_started":
                # Only interrupt work while the assistant is speaking. A VAD
                # false positive during ASR/RAG/LLM must not cancel the user turn.
                if (
                    self.session.state == AgentState.SPEAKING
                    and self._processing_task
                    and not self._processing_task.done()
                ):
                    await self._cancel_processing_task()
                    await self._emit("assistant_interrupted")
                    self.session.state = AgentState.LISTENING
                elif self.session.state in {AgentState.IDLE, AgentState.LISTENING}:
                    self.session.state = AgentState.LISTENING
                await self._emit("speech_started", state=self.session.state.value)
            elif event == "speech_ended":
                await self._emit("speech_ended")
            elif event == "speech_discarded":
                busy = self.session.state not in {AgentState.IDLE, AgentState.LISTENING}
                if not busy:
                    self.session.state = AgentState.LISTENING
                await self._emit("speech_discarded", busy=busy, reason="below_minimum_duration")
        for segment in update.segments:
            self._pending_segments.append(segment)
            if self._processing_task is None or self._processing_task.done():
                self._processing_task = asyncio.create_task(self._process_pending_segments())

    async def _process_pending_segments(self) -> None:
        """Process complete user turns in order without replacing active work."""

        while self._pending_segments and not self._closed:
            await self._process_segment(self._pending_segments.popleft())

    async def clear_conversation(self) -> None:
        """Cancel current work and forget messages for this client only."""

        await self._cancel_processing_task()
        self._pending_segments.clear()
        self.vad.reset()
        self.session.clear()
        await self._emit("conversation_cleared", state=self.session.state.value)

    async def close(self) -> None:
        """Release per-connection work when the WebSocket closes."""

        self._closed = True
        await self._cancel_processing_task()
        self._pending_segments.clear()

    async def _cancel_processing_task(self) -> None:
        task = self._processing_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _process_segment(self, segment: SpeechSegment) -> None:
        tracker = LatencyTracker(started_at=segment.started_at)
        tracker.metrics.vad_ms = segment.vad_ms
        audio_streamed = False
        try:
            with trace_span(
                "voice.vad.segment",
                {
                    "vad.segment.duration_seconds": segment.duration,
                    "vad.processing_ms": segment.vad_ms,
                },
            ) as vad_span:
                if vad_span is not None:
                    vad_span.set_attribute("vad.segment.accepted", True)
            await self._set_state(AgentState.PROCESSING_ASR)
            asr_started = time.perf_counter()
            try:
                with trace_span(
                    "voice.asr",
                    {"session.id": self.session.session_id, "asr.model": getattr(self.asr, "model_name", None)},
                ):
                    transcription = await self._transcribe_with_progress(segment.audio)
            finally:
                tracker.metrics.asr_ms = LatencyTracker.elapsed_ms(asr_started)
            text = transcription.text.strip()
            await self._emit(
                "transcription",
                text=text,
                language=transcription.language,
                language_probability=transcription.language_probability,
            )
            if not text:
                await self._set_state(AgentState.LISTENING)
                return
            language = (transcription.language or "fr").lower()

            if self.support_cases and self.session.active_handoff_case_id:
                active_case = await asyncio.to_thread(self.support_cases.get_case, self.session.active_handoff_case_id)
                if active_case and active_case.status in {"waiting", "in_progress"}:
                    message = await asyncio.to_thread(
                        self.support_cases.record_message, active_case.case_id, "customer", text
                    )
                    if active_case.crm_status == "delivered":
                        try:
                            await self.crm_client.deliver_message(active_case, message)
                        except CRMDeliveryFailure:
                            active_case = await asyncio.to_thread(
                                self.support_cases.set_crm_delivery, active_case.case_id, "failed"
                            )
                    await self.handoff_hub.broadcast_to_agents(
                        {
                            "type": "case_message",
                            "case_id": active_case.case_id,
                            "message": {
                                "message_id": message.message_id,
                                "case_id": message.case_id,
                                "sender": message.sender,
                                "text": message.text,
                                "created_at": message.created_at,
                            },
                        }
                    )
                    await self.handoff_hub.broadcast_to_agents(
                        {"type": "case_updated", "case": serialize_case(active_case)}
                    )
                    await self._emit("handoff_message_sent", case_id=active_case.case_id)
                    ack = (
                        "Votre message a été ajouté à votre demande. Un conseiller vous répondra ici dès qu’il la "
                        "prendra en charge."
                        if language.startswith("fr")
                        else "Your message was added to the case. An advisor will reply here when they pick it up."
                    )
                    await self._speak_handoff_ack(ack, tracker)
                    tracker.metrics.total_ms = LatencyTracker.elapsed_ms(tracker.started_at)
                    tracker.log()
                    await self._emit("metrics", latencies=tracker.metrics.model_dump())
                    await self._set_state(AgentState.LISTENING)
                    return
                self.session.active_handoff_case_id = None
                if active_case:
                    await self._emit("handoff_status", case_id=active_case.case_id, status=active_case.status)

            support_turn = (
                await self._handle_support_request(text, language) if self.support_cases is not None else None
            )
            demo_turn = await self._handle_demo_ticket(text, language) if self.support_cases is None else None
            action_turn = support_turn or demo_turn
            answer = action_turn[0] if action_turn else quick_reply(text, self.session.history)
            if answer is not None:
                mode = action_turn[1] if action_turn else "conversation"
                await self._emit(
                    "retrieval",
                    query=text,
                    intent="support_handoff"
                    if support_turn
                    else "demo_tool"
                    if demo_turn
                    else NovaTelIntent.SOCIAL.value,
                    documents=[],
                    mode=mode,
                )
                if action_turn and action_turn[2]:
                    event_type = (
                        "support_case"
                        if support_turn and mode == "handoff_created"
                        else "handoff_prompt"
                        if support_turn and mode == "handoff_confirmation"
                        else "handoff_unavailable"
                        if support_turn
                        else "demo_tool"
                    )
                    await self._emit(event_type, **action_turn[2])
                await self._emit("llm_response", text=answer, grounded=False)
            else:
                intent = classify_intent(text)
                is_follow_up = is_contextual_follow_up(text)
                if (
                    intent == NovaTelIntent.OUT_OF_SCOPE
                    and is_follow_up
                    and self.session.last_intent
                    and self.session.last_intent not in {NovaTelIntent.SOCIAL.value, NovaTelIntent.OUT_OF_SCOPE.value}
                ):
                    intent = NovaTelIntent(self.session.last_intent)

                if intent == NovaTelIntent.OUT_OF_SCOPE:
                    self.session.last_rag_query = None
                    self.session.last_retrieved_documents.clear()
                    self.session.last_intent = None
                    answer = (
                        "I’m Nova, NovaTel’s voice assistant. I can help with NovaTel plans, bills, payments, "
                        "network issues, refunds, or account requests. Would you like help with one of those, "
                        "or to contact an advisor?"
                        if language.startswith("en")
                        else "Je suis Nova, l’assistante vocale NovaTel. Je peux vous aider avec les forfaits, "
                        "les factures, les paiements, le réseau, les remboursements ou le suivi d’une demande. "
                        "Souhaitez-vous parler à un conseiller ?"
                    )
                    await self._emit("retrieval", query=text, intent=intent.value, documents=[], mode="out_of_scope")
                    await self._emit("llm_response", text=answer, grounded=False)
                else:
                    retrieval_query = contextual_retrieval_query(text, self.session.last_rag_query)
                    hint = retrieval_hint(intent, text)
                    if hint:
                        retrieval_query = f"{retrieval_query}\n{hint}"
                    await self._set_state(AgentState.RETRIEVING)
                    retrieval_started = time.perf_counter()
                    try:
                        with trace_span(
                            "voice.retrieval",
                            {"session.id": self.session.session_id, "rag.intent": intent.value},
                        ) as retrieval_span:
                            documents = await asyncio.to_thread(self.retriever.retrieve, retrieval_query)
                            if retrieval_span is not None:
                                retrieval_span.set_attribute("rag.document_count", len(documents))
                    finally:
                        tracker.metrics.retrieval_ms = LatencyTracker.elapsed_ms(retrieval_started)

                    retrieval_mode = getattr(self.retriever, "mode", "unknown")
                    if not documents and is_follow_up and self.session.last_retrieved_documents:
                        documents = list(self.session.last_retrieved_documents)
                        retrieval_mode = "conversation_context"
                    self.session.last_rag_query = (self.session.last_rag_query or text) if is_follow_up else text
                    self.session.last_intent = intent.value
                    self.session.last_retrieved_documents = list(documents)
                    await self._emit(
                        "retrieval",
                        query=text,
                        intent=intent.value,
                        documents=[{**doc.model_dump(), "content": doc.text} for doc in documents],
                        mode=retrieval_mode,
                    )

                    if documents:
                        answer, audio_streamed = await self._generate_answer(text, documents, tracker)
                    else:
                        if self.support_cases and self.handoff_enabled:
                            self.session.pending_ticket_category = "knowledge_gap"
                            self.session.pending_ticket_request_id = uuid4().hex
                            self.session.pending_ticket_summary = text
                            answer = (
                                "I couldn’t find this information in NovaTel’s knowledge base, "
                                "and I don’t want to guess. "
                                "I can send your question to an advisor. Would you like me to do that?"
                                if language.startswith("en")
                                else "Je n’ai pas trouvé cette information dans la base de connaissances NovaTel "
                                "et je préfère "
                                "ne pas deviner. Je peux transmettre votre question à un conseiller. "
                                "Le souhaitez-vous ?"
                            )
                        else:
                            answer = (
                                "I couldn’t find this information in NovaTel’s knowledge base, and advisor handoff is "
                                "not enabled on this installation. I don’t want to guess."
                                if language.startswith("en")
                                else "Je n’ai pas trouvé cette information dans la base NovaTel et le transfert "
                                "conseiller "
                                "n’est pas activé sur cette installation. Je préfère ne pas deviner."
                            )
                        await self._emit("llm_response", text=answer, grounded=False)

            self.session.history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": answer}])
            if self.max_history_messages:
                history_limit = self.max_history_messages - (self.max_history_messages % 2)
                self.session.history = self.session.history[-history_limit:] if history_limit else []
            if not audio_streamed:
                await self._set_state(AgentState.SPEAKING)
                tts_started = time.perf_counter()
                try:
                    with trace_span("voice.tts", {"session.id": self.session.session_id}):
                        tts_result = await self.tts.synthesize(answer)
                finally:
                    tracker.metrics.tts_ms = LatencyTracker.elapsed_ms(tts_started)
                tracker.metrics.time_to_first_audio_ms = LatencyTracker.elapsed_ms(tracker.started_at)
                await self._emit(
                    "tts_audio_start",
                    content_type="audio/pcm;format=s16le;channels=1",
                    sample_rate=self.output_sample_rate,
                    duration=tts_result.duration,
                    backend=tts_result.backend,
                    chunked=False,
                )
                await self.send_audio(tts_result.audio)
            await self._emit("tts_audio_end")
            tracker.metrics.total_ms = LatencyTracker.elapsed_ms(tracker.started_at)
            tracker.log()
            await self._emit("metrics", latencies=tracker.metrics.model_dump())
            await self._set_state(AgentState.LISTENING)
        except asyncio.CancelledError:
            raise
        except (ASRFailure, TTSFailure) as exc:
            logger.warning("Voice turn failed: %s", type(exc).__name__)
            await self._emit_error(exc)
            tracker.metrics.total_ms = LatencyTracker.elapsed_ms(tracker.started_at)
            await self._emit("metrics", latencies=tracker.metrics.model_dump())
            await self._set_state(AgentState.LISTENING)
        except Exception as exc:
            logger.exception("Unhandled voice pipeline failure")
            await self._emit_error(exc)
            tracker.metrics.total_ms = LatencyTracker.elapsed_ms(tracker.started_at)
            await self._emit("metrics", latencies=tracker.metrics.model_dump())
            await self._set_state(AgentState.LISTENING)

    async def _transcribe_with_progress(self, audio: bytes) -> TranscriptionResult:
        """Keep the UI informed while Whisper loads its weights or decodes speech."""

        task = asyncio.create_task(self.asr.transcribe(audio))
        started_at = time.perf_counter()
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=5.0)
                if task.done():
                    break
                model_loaded = bool(getattr(self.asr, "is_model_loaded", True))
                await self._emit(
                    "asr_progress",
                    stage="transcribing" if model_loaded else "loading_model",
                    elapsed_seconds=round(time.perf_counter() - started_at),
                )
            return await task
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _handle_demo_ticket(self, text: str, language: str) -> tuple[str, str, dict[str, object] | None] | None:
        """Prepare or confirm one explicitly requested, idempotent demo ticket."""

        pending_category = self.session.pending_ticket_category
        if pending_category:
            decision = confirmation_decision(text)
            if decision is None:
                prompt = (
                    "Pour continuer, dites oui pour confirmer le ticket de démonstration, ou non pour annuler."
                    if language.startswith("fr")
                    else "Say yes to confirm the demo ticket, or no to cancel it."
                )
                if self.session.pending_ticket_request_id:
                    await asyncio.to_thread(
                        self.demo_tickets.record_event,
                        self.session.session_id,
                        self.session.pending_ticket_request_id,
                        pending_category,
                        "confirmation_reprompted",
                    )
                return prompt, "tool_confirmation", {"action": "create_demo_ticket", "status": "awaiting_confirmation"}
            request_id = self.session.pending_ticket_request_id
            if not decision:
                if request_id:
                    await asyncio.to_thread(
                        self.demo_tickets.record_event,
                        self.session.session_id,
                        request_id,
                        pending_category,
                        "declined",
                    )
                self.session.pending_ticket_category = None
                self.session.pending_ticket_request_id = None
                response = (
                    "D’accord, je n’ai créé aucun ticket."
                    if language.startswith("fr")
                    else "Understood. I did not create a ticket."
                )
                return response, "demo_tool", {"action": "create_demo_ticket", "status": "cancelled"}
            request_id = request_id or uuid4().hex
            ticket = await asyncio.to_thread(
                self.demo_tickets.create,
                pending_category,
                idempotency_key=request_id,
                session_id=self.session.session_id,
            )
            self.session.pending_ticket_category = None
            self.session.pending_ticket_request_id = None
            payload = ticket_payload(ticket)
            self.session.demo_tickets.append(payload)
            response = (
                f"Le ticket de démonstration {ticket.ticket_id} est créé dans cette session locale. "
                "Aucun CRM ni compte client réel n’a été contacté."
                if language.startswith("fr")
                else f"Demo ticket {ticket.ticket_id} was created in this local session. "
                "No real customer account or CRM was contacted."
            )
            return response, "demo_tool", {"action": "create_demo_ticket", **payload}

        intent = classify_intent(text)
        category = requested_ticket_category(text, intent)
        if category is None:
            return None
        self.session.pending_ticket_category = category
        request_id = uuid4().hex
        self.session.pending_ticket_request_id = request_id
        await asyncio.to_thread(
            self.demo_tickets.record_event,
            self.session.session_id,
            request_id,
            category,
            "confirmation_requested",
        )
        prompt = (
            f"Je peux préparer un ticket de démonstration pour « {category.replace('_', ' ')} ». "
            "Voulez-vous confirmer son ouverture ? Aucun compte réel ne sera modifié."
            if language.startswith("fr")
            else f"I can prepare a demo ticket for {category.replace('_', ' ')}. "
            "Do you confirm? No real account will be changed."
        )
        return prompt, "tool_confirmation", {"action": "create_demo_ticket", "status": "awaiting_confirmation"}

    async def _handle_support_request(
        self, text: str, language: str
    ) -> tuple[str, str, dict[str, object] | None] | None:
        """Ask consent, create a durable case and optionally deliver it to a CRM."""

        assert self.support_cases is not None
        pending_category = self.session.pending_ticket_category
        if pending_category:
            decision = confirmation_decision(text)
            if decision is None:
                if language.startswith("fr"):
                    prompt = (
                        "Pour transmettre votre demande et les prochains messages à un conseiller, dites oui pour "
                        "confirmer ou non pour annuler."
                    )
                else:
                    prompt = (
                        "Your request and following messages can be shared with an advisor. "
                        "Say yes to confirm or no to cancel."
                    )
                return prompt, "handoff_confirmation", {"status": "awaiting_confirmation"}
            request_id = self.session.pending_ticket_request_id or uuid4().hex
            summary = self.session.pending_ticket_summary or text
            self.session.pending_ticket_category = None
            self.session.pending_ticket_request_id = None
            self.session.pending_ticket_summary = None
            if not decision:
                return (
                    "D’accord, je n’ai rien transmis et aucun dossier n’a été créé."
                    if language.startswith("fr")
                    else "Understood. I did not share your request or create a case.",
                    "handoff_cancelled",
                    {"status": "cancelled"},
                )

            case = await asyncio.to_thread(
                self.support_cases.create_case,
                session_id=self.session.session_id,
                tenant_id=self.session.tenant_id,
                category=pending_category,
                summary=summary,
                idempotency_key=request_id,
            )
            await asyncio.to_thread(self.support_cases.record_message, case.case_id, "customer", summary)
            if self.crm_client.configured:
                try:
                    reference = await self.crm_client.deliver(case)
                    case = await asyncio.to_thread(
                        self.support_cases.set_crm_delivery, case.case_id, "delivered", reference
                    )
                except CRMDeliveryFailure:
                    case = await asyncio.to_thread(self.support_cases.set_crm_delivery, case.case_id, "failed")
            self.session.active_handoff_case_id = case.case_id
            await self.handoff_hub.broadcast_to_agents(
                {
                    "type": "queue_updated",
                    "cases": [
                        serialize_case(item) for item in await asyncio.to_thread(self.support_cases.list_open_cases)
                    ],
                }
            )
            if case.crm_status == "delivered":
                reference = case.crm_reference or case.case_id
                if language.startswith("fr"):
                    response = (
                        f"Votre demande a été transmise au service client sous la référence {reference}. "
                        "Vous pouvez continuer à parler ici si un conseiller rejoint cette session."
                    )
                else:
                    response = (
                        f"Your request was sent to customer support under reference {reference}. "
                        "You can keep speaking here if an advisor joins this session."
                    )
            elif case.crm_status == "failed":
                response = (
                    f"Votre demande est conservée dans la file locale sous la référence {case.case_id}, "
                    "mais le CRM n’a pas confirmé sa réception. Un conseiller doit consulter l’inbox de démonstration."
                    if language.startswith("fr")
                    else f"Your request is in the local queue as {case.case_id}, but the CRM did not confirm delivery. "
                    "An advisor must check the demo inbox."
                )
            else:
                response = (
                    f"Votre demande est enregistrée dans la file locale sous la référence {case.case_id}. "
                    "Le CRM externe n’est pas configuré ; un conseiller doit consulter l’inbox de démonstration."
                    if language.startswith("fr")
                    else f"Your request is queued locally as {case.case_id}. The external CRM is not configured; "
                    "an advisor must check the demo inbox."
                )
            return response, "handoff_created", serialize_case(case)

        intent = classify_intent(text)
        category = requested_ticket_category(text, intent)
        if category is None and intent != NovaTelIntent.HUMAN_AGENT:
            return None
        if not self.handoff_enabled:
            response = (
                "Le transfert vers un conseiller n’est pas activé sur cette installation."
                if language.startswith("fr")
                else "Advisor handoff is not enabled on this installation."
            )
            return response, "handoff_unavailable", {"status": "disabled"}
        category = category or "advisor_request"
        self.session.pending_ticket_category = category
        self.session.pending_ticket_request_id = uuid4().hex
        self.session.pending_ticket_summary = text
        if language.startswith("fr"):
            prompt = (
                "Je peux transmettre votre dernière demande à un conseiller. Si vous confirmez, votre demande et les "
                "messages échangés avec le conseiller seront conservés dans le dossier, puis transmis au CRM s’il "
                "est configuré. Confirmez-vous ?"
            )
        else:
            prompt = (
                "I can hand your request to an advisor. If you confirm, your request and messages exchanged with "
                "the advisor will be stored in the case and sent to the CRM if configured. Do you agree?"
            )
        return prompt, "handoff_confirmation", {"status": "awaiting_confirmation", "category": category}

    async def _speak_handoff_ack(self, text: str, tracker: LatencyTracker) -> None:
        """Acknowledge a queued customer message while a human advisor responds."""

        await self._emit("llm_response", text=text, grounded=False)
        await self._set_state(AgentState.SPEAKING)
        started = time.perf_counter()
        try:
            result = await self.tts.synthesize(text)
        except TTSFailure as exc:
            tracker.metrics.tts_ms = LatencyTracker.elapsed_ms(started)
            await self._emit_error(exc)
            return
        tracker.metrics.tts_ms = LatencyTracker.elapsed_ms(started)
        tracker.metrics.time_to_first_audio_ms = LatencyTracker.elapsed_ms(tracker.started_at)
        await self._emit(
            "tts_audio_start",
            content_type="audio/pcm;format=s16le;channels=1",
            sample_rate=self.output_sample_rate,
            duration=result.duration,
            backend=result.backend,
            chunked=False,
        )
        await self.send_audio(result.audio)
        await self._emit("tts_audio_end")

    async def _generate_answer(
        self,
        text: str,
        documents: list[RetrievedDocument],
        tracker: LatencyTracker,
    ) -> tuple[str, bool]:
        await self._set_state(AgentState.GENERATING)
        llm_started = time.perf_counter()
        llm_processing_seconds = 0.0
        pieces: list[str] = []
        pending_speech = ""
        audio_started = False
        stream = self.llm.stream_chat(
            text,
            documents,
            self.session.bounded_history(self.max_history_messages),
        )
        try:
            with trace_span("voice.llm", {"session.id": self.session.session_id}):
                while True:
                    token_started = time.perf_counter()
                    try:
                        token = await stream.__anext__()
                    except StopAsyncIteration:
                        break
                    llm_processing_seconds += time.perf_counter() - token_started
                    if tracker.metrics.time_to_first_token_ms is None:
                        tracker.metrics.time_to_first_token_ms = LatencyTracker.elapsed_ms(llm_started)
                    pieces.append(token)
                    await self._emit("llm_token", text=token)
                    pending_speech += token
                    completed, pending_speech = _take_complete_sentences(pending_speech)
                    for sentence in completed:
                        await self._synthesize_and_send(sentence, tracker, audio_started=audio_started)
                        audio_started = True
                if pending_speech.strip():
                    await self._synthesize_and_send(pending_speech, tracker, audio_started=audio_started)
                    audio_started = True
        finally:
            close_stream = getattr(stream, "aclose", None)
            if close_stream is not None:
                try:
                    await close_stream()
                except Exception:
                    logger.debug("Could not close LLM stream cleanly", exc_info=True)
            tracker.metrics.llm_ms = llm_processing_seconds * 1000
        answer = "".join(pieces).strip()
        if not answer:
            raise RuntimeError("LLM completed without returning response text")
        await self._emit("llm_response", text=answer, grounded=True)
        return answer, audio_started

    async def _synthesize_and_send(self, text: str, tracker: LatencyTracker, *, audio_started: bool) -> None:
        """Synthesize and deliver one complete sentence before the LLM finishes its turn."""

        chunk = text.strip()
        if not chunk:
            return
        started = time.perf_counter()
        with trace_span("voice.tts.chunk", {"session.id": self.session.session_id}):
            result = await self.tts.synthesize(chunk)
        tracker.metrics.tts_ms += LatencyTracker.elapsed_ms(started)
        if not audio_started:
            await self._set_state(AgentState.SPEAKING)
            tracker.metrics.time_to_first_audio_ms = LatencyTracker.elapsed_ms(tracker.started_at)
            await self._emit(
                "tts_audio_start",
                content_type="audio/pcm;format=s16le;channels=1",
                sample_rate=self.output_sample_rate,
                duration=result.duration,
                backend=result.backend,
                chunked=True,
            )
        await self.send_audio(result.audio)

    async def _set_state(self, state: AgentState) -> None:
        self.session.state = state
        await self._emit("state", state=state.value)

    async def _emit(self, event_type: str, **payload: object) -> None:
        if self._closed:
            return
        await self.send_event({"type": event_type, "session_id": self.session.session_id, **payload})

    async def _emit_error(self, error: Exception) -> None:
        messages = {
            ASRFailure: "La transcription a échoué. Vérifiez le modèle Whisper et réessayez.",
            LLMFailure: "Le fournisseur LLM est inaccessible. Vérifiez son URL et que le modèle est démarré.",
            RAGFailure: str(error),
            TTSFailure: str(error),
        }
        message = next(
            (description for failure_type, description in messages.items() if isinstance(error, failure_type)),
            "Une erreur est survenue pendant le traitement de la demande.",
        )
        await self._emit(
            "error",
            code=type(error).__name__,
            message=message,
        )
