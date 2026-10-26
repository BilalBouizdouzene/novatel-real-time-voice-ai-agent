import io
import json
import wave
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.main import create_app
from app.models.schemas import AudioResult
from app.speech.vad import VADUpdate
from app.tools.support_handoff import CRMWebhookClient


class VADUpdateOnly:
    def feed(self, _audio):
        return VADUpdate()

    def reset(self):
        return None


class AdvisorTTS:
    async def synthesize(self, text):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(b"\x00\x00" * 2400)
        return AudioResult(
            audio=buffer.getvalue(),
            sample_rate=24_000,
            duration=0.1,
            processing_time=0.01,
            backend="test",
        )


def test_agent_auth_claims_case_and_retries_idempotent_crm_delivery(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def crm_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ticket_id": "CRM-777"})

    settings = Settings(
        _env_file=None,
        frontend_dir=PROJECT_ROOT / "frontend",
        support_cases_db_path=tmp_path / "agent-cases.sqlite3",
        agent_dashboard_token="agent-test-token-that-is-long-enough",
        crm_webhook_url="https://crm.example.test/support",
        crm_webhook_bearer_token="crm-test-secret-that-is-long-enough",
    )
    app = create_app(settings)
    app.state.crm_client = CRMWebhookClient(
        settings.crm_webhook_url,
        settings.crm_webhook_bearer_token,
        transport=httpx.MockTransport(crm_handler),
    )
    app.state.vad_factory = VADUpdateOnly
    app.state.tts = AdvisorTTS()
    service = app.state.support_case_service

    with TestClient(app) as client:
        with client.websocket_connect("/ws/voice") as customer:
            session = customer.receive_json()
            customer.send_json({"type": "start"})
            assert customer.receive_json()["type"] == "state"
            case = service.create_case(
                session_id=session["session_id"],
                category="advisor_request",
                summary="Please connect me to an advisor.",
                idempotency_key="request-test",
            )
            service.set_crm_delivery(case.case_id, "failed")
            service.record_message(case.case_id, "customer", case.summary)
            service.record_message(case.case_id, "customer", "The issue is still happening.")

            with client.websocket_connect("/ws/agent") as websocket:
                websocket.send_json({"type": "auth", "token": "wrong", "name": "Agent"})
                assert websocket.receive_json()["type"] == "error"

            with client.websocket_connect("/ws/agent") as websocket:
                websocket.send_json(
                    {"type": "auth", "token": settings.agent_dashboard_token, "name": "Amina"}
                )
                assert websocket.receive_json()["type"] == "agent_ready"
                queue = websocket.receive_json()
                assert queue["type"] == "queue"
                assert queue["cases"][0]["case_id"] == case.case_id

                websocket.send_json({"type": "claim", "case_id": case.case_id})
                claim_events = [websocket.receive_json() for _ in range(3)]
                assert {event["type"] for event in claim_events} == {"case_updated", "case_history", "action_ok"}
                assert customer.receive_json()["type"] == "handoff_status"

                websocket.send_json({"type": "reply", "case_id": case.case_id, "text": "I will help you."})
                customer_events = []
                pcm_frames = []
                while True:
                    incoming = customer.receive()
                    if incoming.get("bytes") is not None:
                        pcm_frames.append(incoming["bytes"])
                    elif incoming.get("text"):
                        event = json.loads(incoming["text"])
                        customer_events.append(event)
                        if event["type"] == "tts_audio_end":
                            break
                assert [event["type"] for event in customer_events] == [
                    "advisor_message",
                    "tts_audio_start",
                    "tts_audio_end",
                ]
                assert len(b"".join(pcm_frames)) == 4_800
                reply_events = [websocket.receive_json() for _ in range(2)]
                assert {event["type"] for event in reply_events} == {"case_message", "case_updated"}

                websocket.send_json({"type": "retry_crm", "case_id": case.case_id})
                retry_events = [websocket.receive_json() for _ in range(2)]
                assert {event["type"] for event in retry_events} == {"case_updated", "action_ok"}

                websocket.send_json({"type": "resolve", "case_id": case.case_id})
                resolve_events = [websocket.receive_json() for _ in range(2)]
                assert {event["type"] for event in resolve_events} == {"case_updated", "action_ok"}
                assert customer.receive_json()["status"] == "resolved"

    assert service.get_case(case.case_id).crm_status == "delivered"
    assert service.get_case(case.case_id).crm_reference == "CRM-777"
    assert len(requests) == 3
    assert requests[0].headers["Authorization"] == "Bearer crm-test-secret-that-is-long-enough"
    assert requests[0].headers["Idempotency-Key"] == case.case_id
    assert json.loads(requests[0].content)["event"] == "support_case.created"
    assert [json.loads(request.content)["event"] for request in requests[1:]] == [
        "support_case.message",
        "support_case.message",
    ]
    assert [json.loads(request.content)["message"]["text"] for request in requests[1:]] == [
        "The issue is still happening.",
        "I will help you.",
    ]
