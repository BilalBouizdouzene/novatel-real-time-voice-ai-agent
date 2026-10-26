"""Minimal Twilio Voice REST adapter; no Twilio SDK is required."""

from __future__ import annotations

import logging
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class TwilioVoiceFailure(RuntimeError):
    """Raised when Twilio cannot create the demonstration call."""


class TwilioVoiceClient:
    """Create an outbound trial call and point it at our TwiML webhook."""

    def __init__(
        self,
        *,
        account_sid: str,
        auth_token: str,
        api_key_sid: str = "",
        api_key_secret: str = "",
        from_number: str,
        advisor_number: str,
        webhook_base_url: str,
        enabled: bool = False,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.account_sid = account_sid.strip()
        self.auth_token = auth_token.strip()
        self.api_key_sid = api_key_sid.strip()
        self.api_key_secret = api_key_secret.strip()
        self.from_number = from_number.strip()
        self.advisor_number = advisor_number.strip()
        self.webhook_base_url = webhook_base_url.rstrip("/")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    @property
    def configured(self) -> bool:
        has_auth = bool(self.auth_token)
        has_api_key = bool(self.api_key_sid and self.api_key_secret)
        return self.enabled and all(
            (
                self.account_sid,
                self.from_number,
                self.advisor_number,
                self.webhook_base_url.startswith("https://"),
            )
        ) and (has_auth or has_api_key)

    async def create_call(self, *, to_number: str, case_id: str = "") -> dict[str, str]:
        if not self.configured:
            raise TwilioVoiceFailure("Twilio is not fully configured")
        if not to_number.strip().startswith("+"):
            raise TwilioVoiceFailure("Twilio numbers must use E.164 format, for example +33123456789")
        encoded_case = quote(case_id[:80], safe="")
        voice_url = f"{self.webhook_base_url}/twilio/voice"
        if encoded_case:
            voice_url += f"?case_id={encoded_case}"
        endpoint = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Calls.json"
        payload = {
            "To": to_number.strip(),
            "From": self.from_number,
            "Url": voice_url,
            "StatusCallback": f"{self.webhook_base_url}/twilio/status",
            "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
        }
        auth = (
            (self.api_key_sid, self.api_key_secret)
            if self.api_key_sid and self.api_key_secret
            else (self.account_sid, self.auth_token)
        )
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=False, transport=self.transport
            ) as client:
                response = await client.post(endpoint, data=payload, auth=auth)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            # Twilio returns a useful JSON error code/message. Keep the
            # credentials private while surfacing enough detail to fix setup.
            status_code = exc.response.status_code
            twilio_code = ""
            twilio_message = ""
            try:
                error_body = exc.response.json()
                twilio_code = str(error_body.get("code", ""))
                twilio_message = str(error_body.get("message", ""))
            except ValueError:
                twilio_message = exc.response.text[:240].strip()
            request_id = exc.response.headers.get("Twilio-Request-Id", "")
            logger.warning(
                "Twilio rejected call status=%s code=%s message=%s request_id=%s",
                status_code,
                twilio_code or "unknown",
                twilio_message or "unknown",
                request_id or "unknown",
            )
            detail = f"HTTP {status_code}"
            if twilio_code:
                detail += f", code {twilio_code}"
            if twilio_message:
                detail += f": {twilio_message}"
            raise TwilioVoiceFailure(f"Twilio a refusé l'appel ({detail})") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise TwilioVoiceFailure(f"Twilio call creation failed ({type(exc).__name__})") from exc
        sid = str(body.get("sid", ""))
        if not sid:
            raise TwilioVoiceFailure("Twilio did not return a call SID")
        return {"call_sid": sid, "status": str(body.get("status", "queued"))}

    def twiml_for_handoff(self, *, case_id: str = "") -> str:
        """Return safe TwiML for the trial handoff demonstration."""

        safe_case = case_id.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        message = "Vous êtes mis en relation avec le support NovaTel."
        if safe_case:
            message += f" Référence {safe_case}."
        advisor = self.advisor_number.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Response>"
            f"<Say language=\"fr-FR\">{message}</Say>"
            f"<Dial>{advisor}</Dial>"
            "</Response>"
        )
