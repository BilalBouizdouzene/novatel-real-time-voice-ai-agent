"""WebSocket protocol: JSON control/events and streamed binary PCM responses."""

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.models.schemas import AgentState
from app.pipeline.state import ConversationSession
from app.pipeline.voice_agent import VoiceAgentPipeline
from app.speech.audio_utils import wav_bytes_to_pcm16
from app.speech.streaming import iter_pcm_frames

logger = logging.getLogger(__name__)


def register_websocket_route(app: FastAPI) -> None:
    """Attach the per-connection voice endpoint to an application instance."""

    @app.websocket("/ws/voice")
    async def voice_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()

        async def send_event(event: dict[str, object]) -> None:
            async with send_lock:
                await websocket.send_json(event)

        async def send_pcm(audio: bytes) -> None:
            for frame in iter_pcm_frames(
                audio,
                sample_rate=settings.tts_stream_sample_rate,
                frame_ms=settings.tts_audio_frame_ms,
            ):
                async with send_lock:
                    await websocket.send_bytes(frame)

        async def send_audio(audio: bytes) -> None:
            pcm, _sample_rate = await asyncio.to_thread(wav_bytes_to_pcm16, audio, 24_000)
            await send_pcm(pcm)

        settings = app.state.settings
        session = ConversationSession(tenant_id=websocket.query_params.get("tenant_id", "default")[:80] or "default")
        pipeline: VoiceAgentPipeline | None = None
        try:
            try:
                vad_factory = app.state.vad_factory
                vad = await asyncio.to_thread(vad_factory)
                pipeline = VoiceAgentPipeline(
                    session=session,
                    vad=vad,
                    asr=app.state.asr,
                    retriever=app.state.retriever,
                    llm=app.state.llm,
                    tts=app.state.tts,
                    send_event=send_event,
                    send_audio=send_audio,
                    max_history_messages=settings.conversation_history_messages,
                    demo_ticket_service=app.state.demo_ticket_service,
                    support_case_service=app.state.support_case_service,
                    crm_client=app.state.crm_client,
                    handoff_hub=app.state.handoff_hub,
                    handoff_enabled=bool(settings.agent_dashboard_token),
                    output_sample_rate=settings.tts_stream_sample_rate,
                )
            except Exception as exc:
                logger.exception("Could not initialize speech pipeline")
                await send_event(
                    {
                        "type": "error",
                        "code": "pipeline_unavailable",
                        "message": str(exc),
                        "session_id": session.session_id,
                    }
                )
                await websocket.close(code=1011)
                return

            await app.state.handoff_hub.register_customer(session.session_id, send_event, send_pcm)
            await send_event({"type": "session", "session_id": session.session_id, "state": session.state.value})
            active = False
            while True:
                incoming = await websocket.receive()
                if incoming.get("type") == "websocket.disconnect":
                    break
                binary = incoming.get("bytes")
                if binary is not None:
                    if not active:
                        await send_event(
                            {"type": "error", "code": "not_started", "message": "Envoyez start avant les chunks audio."}
                        )
                        continue
                    if len(binary) > settings.max_audio_chunk_bytes:
                        await send_event(
                            {"type": "error", "code": "chunk_too_large", "message": "Chunk audio trop volumineux."}
                        )
                        continue
                    try:
                        await pipeline.ingest_audio(binary)
                    except ValueError as exc:
                        await send_event({"type": "error", "code": "invalid_audio", "message": str(exc)})
                    continue

                raw_text = incoming.get("text")
                if raw_text is None:
                    continue
                try:
                    message = json.loads(raw_text)
                except json.JSONDecodeError:
                    await send_event(
                        {"type": "error", "code": "invalid_json", "message": "Message de contrôle JSON invalide."}
                    )
                    continue
                if not isinstance(message, dict):
                    await send_event(
                        {"type": "error", "code": "invalid_message", "message": "Un objet JSON est attendu."}
                    )
                    continue
                command = message.get("type")
                if command == "start":
                    active = True
                    session.state = AgentState.LISTENING
                    await send_event({"type": "state", "state": session.state.value})
                elif command == "stop":
                    active = False
                    await pipeline.close()
                    session.state = AgentState.IDLE
                    await send_event({"type": "state", "state": session.state.value})
                    break
                elif command == "clear":
                    await pipeline.clear_conversation()
                elif command == "ping":
                    await send_event({"type": "pong"})
                else:
                    await send_event({"type": "error", "code": "unknown_command", "message": "Commande inconnue."})
        except WebSocketDisconnect as exc:
            logger.info(
                "WebSocket disconnected session=%s close_code=%s",
                session.session_id,
                exc.code,
            )
        except Exception:
            logger.exception("WebSocket handler failed session=%s", session.session_id)
            try:
                await send_event(
                    {
                        "type": "error",
                        "code": "websocket_error",
                        "message": "La connexion temps réel a été interrompue.",
                    }
                )
            except Exception:
                logger.debug("Could not send a WebSocket error event", exc_info=True)
        finally:
            await app.state.handoff_hub.unregister_customer(session.session_id)
            if pipeline is not None:
                await pipeline.close()
