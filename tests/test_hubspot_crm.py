import asyncio
import json

import httpx
import pytest

from app.tools.hubspot_crm import HubSpotCRMClient
from app.tools.support_handoff import CRMDeliveryFailure, SupportCase, SupportMessage


def _case(*, reference: str | None = None) -> SupportCase:
    return SupportCase(
        case_id="case-abc123",
        session_id="session-1",
        category="billing_question",
        summary="The customer asks about a duplicate invoice.",
        status="waiting",
        created_at="2026-10-12T10:00:00+00:00",
        crm_status="delivered" if reference else "not_configured",
        crm_reference=reference,
    )


def test_hubspot_creates_ticket_and_linked_message_note() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/tickets/search") or request.url.path.endswith("/notes/search"):
            return httpx.Response(200, json={"results": []})
        if request.url.path.endswith("/objects/tickets"):
            return httpx.Response(201, json={"id": "ticket-901"})
        if request.url.path.endswith("/objects/notes"):
            return httpx.Response(201, json={"id": "note-44"})
        return httpx.Response(404)

    client = HubSpotCRMClient(
        "private-app-secret",
        transport=httpx.MockTransport(handler),
    )
    case = _case()
    message = SupportMessage(
        message_id="message-xyz",
        case_id=case.case_id,
        sender="advisor",
        text="I will check the invoice with you.",
        created_at="2026-10-12T10:01:00+00:00",
    )

    async def run() -> str:
        ticket_id = await client.deliver(case)
        await client.deliver_message(_case(reference=ticket_id), message)
        return ticket_id

    assert asyncio.run(run()) == "ticket-901"
    assert [request.method for request in requests] == ["POST", "POST", "POST", "POST"]
    assert all(request.headers["Authorization"] == "Bearer private-app-secret" for request in requests)

    ticket_payload = json.loads(requests[1].content)
    assert ticket_payload["properties"]["subject"] == "NovaTel voice case case-abc123"
    assert ticket_payload["properties"]["hs_pipeline"] == "0"
    assert ticket_payload["properties"]["hs_pipeline_stage"] == "1"
    assert "duplicate invoice" in ticket_payload["properties"]["content"]

    note_payload = json.loads(requests[3].content)
    assert note_payload["properties"]["hs_timestamp"] == message.created_at
    assert "I will check the invoice with you." in note_payload["properties"]["hs_note_body"]
    assert note_payload["associations"][0]["to"]["id"] == "ticket-901"
    assert note_payload["associations"][0]["types"][0]["associationTypeId"] == 228
    note_search = json.loads(requests[2].content)
    assert note_search["filterGroups"][0]["filters"][0]["operator"] == "CONTAINS_TOKEN"
    assert note_search["filterGroups"][0]["filters"][0]["value"] == message.message_id


def test_hubspot_search_avoids_duplicate_ticket_and_note_on_retry() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/tickets/search"):
            return httpx.Response(200, json={"results": [{"id": "existing-ticket"}]})
        if request.url.path.endswith("/notes/search"):
            return httpx.Response(200, json={"results": [{"id": "existing-note"}]})
        return httpx.Response(500)

    client = HubSpotCRMClient("private-app-secret", transport=httpx.MockTransport(handler))
    case = _case(reference="existing-ticket")
    message = SupportMessage("message-xyz", case.case_id, "customer", "Still need help.", case.created_at)

    async def run() -> str:
        reference = await client.deliver(case)
        await client.deliver_message(case, message)
        return reference

    assert asyncio.run(run()) == "existing-ticket"
    assert len(requests) == 2
    assert all(request.url.path.endswith("/search") for request in requests)


def test_hubspot_http_errors_are_safely_wrapped() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "private token rejected"})

    client = HubSpotCRMClient("private-app-secret", transport=httpx.MockTransport(handler))
    with pytest.raises(CRMDeliveryFailure, match="HubSpot request failed"):
        asyncio.run(client.deliver(_case()))


def test_hubspot_message_requires_a_persisted_external_ticket_id() -> None:
    client = HubSpotCRMClient("private-app-secret")
    message = SupportMessage("message-1", "case-abc123", "customer", "Hello", _case().created_at)
    with pytest.raises(CRMDeliveryFailure, match="reference is missing"):
        asyncio.run(client.deliver_message(_case(), message))
