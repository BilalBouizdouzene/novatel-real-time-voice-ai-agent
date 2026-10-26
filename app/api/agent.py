"""Authenticated agent inbox and realtime, consented customer handoff."""

import asyncio
import logging
import re
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.integrations.twilio_voice import TwilioVoiceFailure
from app.speech.audio_utils import wav_bytes_to_pcm16
from app.speech.streaming import iter_pcm_frames
from app.tools.support_handoff import (
    CRMDeliveryFailure,
    SupportCase,
    SupportCaseService,
    SupportMessage,
    serialize_case,
)

logger = logging.getLogger(__name__)


def register_agent_routes(app: FastAPI) -> None:
    """Attach the real-time, token-protected operator inbox."""

    @app.get("/agent", include_in_schema=False)
    async def agent_page():
        return FileResponse(app.state.settings.frontend_dir / "agent.html")

    @app.websocket("/ws/agent")
    async def agent_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        settings = app.state.settings
        if not settings.agent_dashboard_token:
            await websocket.send_json(
                {"type": "error", "message": "Configurez AGENT_DASHBOARD_TOKEN dans .env pour activer l’inbox."}
            )
            await websocket.close(code=1008)
            return
        try:
            auth = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        except (TimeoutError, WebSocketDisconnect, ValueError):
            await websocket.close(code=1008)
            return
        hub = app.state.handoff_hub
        if (
            not isinstance(auth, dict)
            or auth.get("type") != "auth"
            or not hub.authenticate(str(auth.get("token", "")), settings.agent_dashboard_token)
        ):
            await websocket.send_json({"type": "error", "message": "Jeton agent invalide."})
            await websocket.close(code=1008)
            return

        display_name = re.sub(r"[^\w .'-]", "", str(auth.get("name", "Advisor")))[:40].strip() or "Advisor"
        agent_id = f"agent-{uuid4().hex}"
        await hub.register_agent(websocket)
        service = app.state.support_case_service
        try:
            await hub.send_agent(websocket, {"type": "agent_ready", "agent_id": agent_id, "name": display_name})
            await _send_queue(websocket, app)
            while True:
                command = await websocket.receive_json()
                if not isinstance(command, dict):
                    continue
                case_id = str(command.get("case_id", ""))[:40]
                if command.get("type") == "refresh":
                    await _send_queue(websocket, app)
                elif command.get("type") == "claim":
                    case = await asyncio.to_thread(service.claim, case_id, agent_id)
                    if case is None:
                        await hub.send_agent(
                            websocket, {"type": "action_error", "message": "Case already assigned or closed."}
                        )
                        continue
                    await hub.send_customer_event(
                        case.session_id,
                        {
                            "type": "handoff_status",
                            "case_id": case.case_id,
                            "status": "in_progress",
                            "agent": display_name,
                        },
                    )
                    await hub.broadcast_to_agents({"type": "case_updated", "case": serialize_case(case)})
                    await _send_case_messages(websocket, service, case.case_id)
                    await hub.send_agent(websocket, {"type": "action_ok", "message": f"You claimed {case.case_id}."})
                elif command.get("type") == "reply":
                    case = await asyncio.to_thread(service.get_case, case_id)
                    message_text = str(command.get("text", "")).strip()[:4000]
                    if not message_text or case is None or case.status != "in_progress" or case.assigned_to != agent_id:
                        await hub.send_agent(
                            websocket, {"type": "action_error", "message": "Claim this open case before replying."}
                        )
                        continue
                    message = await asyncio.to_thread(service.record_message, case_id, "advisor", message_text)
                    case = await _deliver_crm_message(app, case, message)
                    await hub.send_customer_event(
                        case.session_id,
                        {"type": "advisor_message", "case_id": case_id, "text": message.text, "agent": display_name},
                    )
                    await _speak_to_customer(app, case.session_id, message.text)
                    await hub.broadcast_to_agents(
                        {"type": "case_message", "case_id": case_id, "message": _message_json(message)}
                    )
                    await hub.broadcast_to_agents({"type": "case_updated", "case": serialize_case(case)})
                elif command.get("type") == "call_customer":
                    case = await asyncio.to_thread(service.get_case, case_id)
                    to_number = str(command.get("to_number", "")).strip()
                    if case is None or case.status != "in_progress" or case.assigned_to != agent_id:
                        await hub.send_agent(
                            websocket,
                            {"type": "action_error", "message": "Claim this case before starting a call."},
                        )
                        continue
                    if not re.fullmatch(r"\+[1-9]\d{7,14}", to_number):
                        await hub.send_agent(
                            websocket,
                            {
                                "type": "action_error",
                                "message": "Use a valid phone number in E.164 format, for example +33753414072.",
                            },
                        )
                        continue
                    twilio_client = app.state.twilio_client
                    if not twilio_client.configured:
                        await hub.send_agent(
                            websocket,
                            {"type": "action_error", "message": "Twilio phone calling is not fully configured."},
                        )
                        continue
                    try:
                        result = await twilio_client.create_call(to_number=to_number, case_id=case_id)
                    except TwilioVoiceFailure as exc:
                        await hub.send_agent(websocket, {"type": "action_error", "message": str(exc)})
                        continue
                    await hub.send_agent(
                        websocket,
                        {
                            "type": "call_started",
                            "case_id": case_id,
                            "call_sid": result["call_sid"],
                            "status": result["status"],
                            "message": "Twilio call queued.",
                        },
                    )
                elif command.get("type") == "resolve":
                    case = await asyncio.to_thread(service.resolve, case_id, agent_id)
                    if case is None:
                        await hub.send_agent(
                            websocket,
                            {"type": "action_error", "message": "Only the assigned advisor can close this case."},
                        )
                        continue
                    await hub.send_customer_event(
                        case.session_id,
                        {"type": "handoff_status", "case_id": case_id, "status": "resolved"},
                    )
                    await hub.broadcast_to_agents({"type": "case_updated", "case": serialize_case(case)})
                    await hub.send_agent(websocket, {"type": "action_ok", "message": f"Resolved {case.case_id}."})
                elif command.get("type") == "retry_crm":
                    case = await asyncio.to_thread(service.get_case, case_id)
                    if (
                        case is None
                        or case.status != "in_progress"
                        or case.assigned_to != agent_id
                        or case.crm_status != "failed"
                    ):
                        await hub.send_agent(
                            websocket,
                            {"type": "action_error", "message": "Claim a case with a failed CRM delivery first."},
                        )
                        continue
                    if not app.state.crm_client.configured:
                        await hub.send_agent(
                            websocket,
                            {"type": "action_error", "message": "CRM integration is not configured."},
                        )
                        continue
                    try:
                        reference = await app.state.crm_client.deliver(case)
                    except CRMDeliveryFailure:
                        updated = await asyncio.to_thread(service.set_crm_delivery, case_id, "failed")
                        await hub.broadcast_to_agents({"type": "case_updated", "case": serialize_case(updated)})
                        await hub.send_agent(
                            websocket,
                            {"type": "action_error", "message": "CRM retry failed; the local case is still safe."},
                        )
                        continue
                    updated = await asyncio.to_thread(service.set_crm_delivery, case_id, "delivered", reference)
                    messages = await asyncio.to_thread(service.list_messages, case_id)
                    try:
                        # The case-created event already contains the initial summary.
                        for message in messages[1:]:
                            await app.state.crm_client.deliver_message(updated, message)
                    except CRMDeliveryFailure:
                        updated = await asyncio.to_thread(service.set_crm_delivery, case_id, "failed", reference)
                        await hub.broadcast_to_agents({"type": "case_updated", "case": serialize_case(updated)})
                        await hub.send_agent(
                            websocket,
                            {
                                "type": "action_error",
                                "message": "CRM accepted the case, but a message retry failed; local history is safe.",
                            },
                        )
                        continue
                    await hub.broadcast_to_agents({"type": "case_updated", "case": serialize_case(updated)})
                    await hub.send_agent(
                        websocket,
                        {"type": "action_ok", "message": f"CRM accepted {case_id}."},
                    )
                else:
                    await hub.send_agent(websocket, {"type": "action_error", "message": "Unknown agent command."})
        except WebSocketDisconnect:
            logger.info("Agent inbox disconnected agent=%s", agent_id)
        except Exception:
            logger.exception("Agent inbox failed agent=%s", agent_id)
            try:
                await hub.send_agent(websocket, {"type": "error", "message": "Agent connection failed."})
            except Exception:
                pass
        finally:
            await hub.unregister_agent(websocket)
            released = await asyncio.to_thread(service.release_agent_cases, agent_id)
            if released:
                cases = await asyncio.to_thread(service.list_open_cases)
                await hub.broadcast_to_agents(
                    {"type": "queue_updated", "cases": [serialize_case(case) for case in cases]}
                )


async def _send_queue(websocket: WebSocket, app: FastAPI) -> None:
    service = app.state.support_case_service
    cases = await asyncio.to_thread(service.list_open_cases)
    messages = await asyncio.gather(*(asyncio.to_thread(service.list_messages, case.case_id) for case in cases))
    await app.state.handoff_hub.send_agent(
        websocket,
        {
            "type": "queue",
            "cases": [
                {**serialize_case(case), "messages": [_message_json(message) for message in case_messages]}
                for case, case_messages in zip(cases, messages, strict=True)
            ],
        },
    )


async def _send_case_messages(websocket: WebSocket, service: SupportCaseService, case_id: str) -> None:
    messages = await asyncio.to_thread(service.list_messages, case_id)
    await websocket.app.state.handoff_hub.send_agent(
        websocket,
        {"type": "case_history", "case_id": case_id, "messages": [_message_json(message) for message in messages]},
    )


async def _speak_to_customer(app: FastAPI, session_id: str, text: str) -> None:
    try:
        result = await app.state.tts.synthesize(text)
        pcm, sample_rate = await asyncio.to_thread(
            wav_bytes_to_pcm16, result.audio, app.state.settings.tts_stream_sample_rate
        )
        await app.state.handoff_hub.send_customer_event(
            session_id,
            {
                "type": "tts_audio_start",
                "content_type": "audio/pcm;format=s16le;channels=1",
                "sample_rate": sample_rate,
                "backend": result.backend,
                "chunked": False,
            },
        )
        for frame in iter_pcm_frames(pcm, sample_rate=sample_rate, frame_ms=app.state.settings.tts_audio_frame_ms):
            await app.state.handoff_hub.send_customer_audio(session_id, frame)
        await app.state.handoff_hub.send_customer_event(session_id, {"type": "tts_audio_end"})
    except Exception as exc:
        logger.info("Advisor TTS unavailable; text remains available error_type=%s", type(exc).__name__)
        await app.state.handoff_hub.send_customer_event(
            session_id,
            {
                "type": "error",
                "code": "TTSFailure",
                "message": "Le message du conseiller reste visible, mais la synthèse vocale est indisponible.",
            },
        )


async def _deliver_crm_message(app: FastAPI, case: SupportCase, message: SupportMessage) -> SupportCase:
    if not app.state.crm_client.configured or case.crm_status != "delivered":
        return case
    try:
        await app.state.crm_client.deliver_message(case, message)
    except CRMDeliveryFailure:
        logger.info("CRM message event was not delivered; local case conversation remains available")
        return await asyncio.to_thread(app.state.support_case_service.set_crm_delivery, case.case_id, "failed")
    return case


def _message_json(message: object) -> dict[str, str]:
    return {
        "message_id": message.message_id,
        "case_id": message.case_id,
        "sender": message.sender,
        "text": message.text,
        "created_at": message.created_at,
    }
