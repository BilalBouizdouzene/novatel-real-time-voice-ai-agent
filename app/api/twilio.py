"""Optional Twilio Trial routes for a real phone handoff demonstration."""

import hmac
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.integrations.twilio_voice import TwilioVoiceClient, TwilioVoiceFailure

logger = logging.getLogger(__name__)


class CallRequest(BaseModel):
    to_number: str
    case_id: str = ""


def register_twilio_routes(app, client: TwilioVoiceClient) -> None:
    router = APIRouter()

    def require_admin(request: Request) -> None:
        expected = request.app.state.settings.agent_dashboard_token
        provided = request.headers.get("X-Agent-Token", "")
        if not expected or not hmac.compare_digest(provided.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Advisor token required")

    @router.post("/api/twilio/call")
    async def create_call(payload: CallRequest, request: Request):
        require_admin(request)
        try:
            result = await client.create_call(to_number=payload.to_number, case_id=payload.case_id)
        except TwilioVoiceFailure as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.info("Twilio demo call queued case_id=%s", payload.case_id[:80])
        return result

    @router.post("/twilio/voice", response_class=Response)
    async def voice_webhook(request: Request):
        if not client.configured:
            return Response(
                "<Response><Say>Phone demo is not configured.</Say></Response>", media_type="application/xml"
            )
        case_id = request.query_params.get("case_id", "")
        return Response(client.twiml_for_handoff(case_id=case_id), media_type="application/xml")

    @router.post("/twilio/status")
    async def status_webhook(request: Request):
        # Avoid requiring python-multipart just for this optional callback.
        body = (await request.body()).decode("utf-8", errors="ignore")
        status = "unknown"
        for item in body.split("&"):
            if item.startswith("CallStatus="):
                status = item.split("=", 1)[1][:40]
                break
        logger.info("Twilio call status received status=%s", status)
        return {"ok": True}

    app.include_router(router)
