"""Server-side HubSpot ticket synchronization for consented support handoffs."""

import logging
from typing import Any

import httpx

from app.tools.support_handoff import CRMDeliveryFailure, SupportCase, SupportMessage

logger = logging.getLogger(__name__)


class HubSpotCRMClient:
    """Create HubSpot tickets and attach consented conversation turns as notes.

    Ticket subjects and note bodies include stable NovaTel IDs. Before creating
    an object, the client searches for that exact marker so retrying a delivery
    does not normally create duplicate tickets or timeline notes.
    """

    API_ROOT = "https://api.hubapi.com"
    NOTE_TO_TICKET_ASSOCIATION_TYPE = 228

    def __init__(
        self,
        private_app_token: str,
        *,
        timeout_seconds: float = 8.0,
        ticket_pipeline: str = "0",
        ticket_stage: str = "1",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.private_app_token = private_app_token.strip()
        self.timeout_seconds = timeout_seconds
        self.ticket_pipeline = ticket_pipeline.strip() or "0"
        self.ticket_stage = ticket_stage.strip() or "1"
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.private_app_token)

    async def deliver(self, case: SupportCase) -> str:
        """Find or create the HubSpot ticket for a consented local case."""

        if not self.configured:
            raise CRMDeliveryFailure("HubSpot private app token is not configured")

        subject = f"NovaTel voice case {case.case_id}"
        existing = await self._find_object("tickets", "subject", subject)
        if existing:
            return existing

        content = (
            f"NovaTel voice support case: {case.case_id}\n"
            f"Category: {case.category}\n"
            f"Created: {case.created_at}\n\n"
            f"Customer summary:\n{case.summary}"
        )
        body = await self._request(
            "POST",
            "/crm/v3/objects/tickets",
            json={
                "properties": {
                    "subject": subject,
                    "content": content,
                    "hs_pipeline": self.ticket_pipeline,
                    "hs_pipeline_stage": self.ticket_stage,
                }
            },
        )
        ticket_id = body.get("id")
        if not ticket_id:
            raise CRMDeliveryFailure("HubSpot ticket response did not include an ID")
        reference = str(ticket_id)
        logger.info("CRM ticket delivered provider=hubspot case_id=%s reference=%s", case.case_id, reference)
        return reference

    async def deliver_message(self, case: SupportCase, message: SupportMessage) -> None:
        """Append an idempotent, consented turn to the HubSpot ticket timeline."""

        if not self.configured:
            raise CRMDeliveryFailure("HubSpot private app token is not configured")
        if not case.crm_reference:
            raise CRMDeliveryFailure("HubSpot ticket reference is missing for this support case")

        author = "Customer" if message.sender == "customer" else "Advisor"
        note_body = f"NovaTel voice case {case.case_id} · message {message.message_id}\n{author}: {message.text}"
        existing = await self._find_object(
            "notes",
            "hs_note_body",
            message.message_id,
            operator="CONTAINS_TOKEN",
        )
        if existing:
            return

        body = await self._request(
            "POST",
            "/crm/v3/objects/notes",
            json={
                "properties": {
                    "hs_timestamp": message.created_at,
                    "hs_note_body": note_body,
                },
                "associations": [
                    {
                        "to": {"id": case.crm_reference},
                        "types": [
                            {
                                "associationCategory": "HUBSPOT_DEFINED",
                                "associationTypeId": self.NOTE_TO_TICKET_ASSOCIATION_TYPE,
                            }
                        ],
                    }
                ],
            },
        )
        if not body.get("id"):
            raise CRMDeliveryFailure("HubSpot note response did not include an ID")
        logger.info("CRM message delivered provider=hubspot case_id=%s", case.case_id)

    async def _find_object(
        self,
        object_type: str,
        property_name: str,
        value: str,
        *,
        operator: str = "EQ",
    ) -> str | None:
        body = await self._request(
            "POST",
            f"/crm/v3/objects/{object_type}/search",
            json={
                "filterGroups": [{"filters": [{"propertyName": property_name, "operator": operator, "value": value}]}],
                "properties": [property_name],
                "limit": 1,
            },
        )
        results = body.get("results", [])
        if not isinstance(results, list) or not results:
            return None
        object_id = results[0].get("id")
        return str(object_id) if object_id else None

    async def _request(self, method: str, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.private_app_token}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = await client.request(
                    method,
                    f"{self.API_ROOT}{path}",
                    headers=headers,
                    json=json,
                )
            response.raise_for_status()
            body = response.json() if response.content else {}
            if not isinstance(body, dict):
                raise ValueError("HubSpot returned a non-object JSON response")
            return body
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "HubSpot CRM request failed method=%s path=%s error_type=%s",
                method,
                path,
                type(exc).__name__,
            )
            raise CRMDeliveryFailure(f"HubSpot request failed ({type(exc).__name__})") from exc
