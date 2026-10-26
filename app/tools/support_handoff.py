"""Consent-gated support cases, a local live-agent queue, and CRM webhook delivery.

The local SQLite queue remains available when no CRM is configured. A configured
HTTPS webhook can forward the minimum case payload to an existing helpdesk or
contact-center integration without coupling the agent to one CRM vendor.
"""

import asyncio
import hmac
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupportCase:
    """A durable support request, created only after explicit user consent."""

    case_id: str
    session_id: str
    category: str
    summary: str
    status: str
    created_at: str
    assigned_to: str | None = None
    crm_status: str = "not_configured"
    crm_reference: str | None = None
    tenant_id: str = "default"


@dataclass(frozen=True)
class SupportMessage:
    """A consented transcript message in a human handoff conversation."""

    message_id: str
    case_id: str
    sender: str
    text: str
    created_at: str


class CRMDeliveryFailure(RuntimeError):
    """Raised when a configured CRM webhook cannot accept a support case."""


class CRMClient(Protocol):
    """Provider-neutral operations used by the voice pipeline and advisor inbox."""

    @property
    def configured(self) -> bool: ...

    async def deliver(self, case: SupportCase) -> str | None: ...

    async def deliver_message(self, case: SupportCase, message: SupportMessage) -> None: ...


class SupportCaseService:
    """Store consented handoff requests and agent actions in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            if self.database_path != ":memory:":
                Path(self.database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.database_path, timeout=10, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS support_cases (
                    case_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    assigned_to TEXT,
                    crm_status TEXT NOT NULL DEFAULT 'not_configured',
                    crm_reference TEXT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    idempotency_key TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS support_case_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES support_cases(case_id),
                    event TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS support_case_messages (
                    message_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES support_cases(case_id),
                    sender TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(support_cases)")}
            if "tenant_id" not in columns:
                self._connection.execute(
                    "ALTER TABLE support_cases ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
                self._connection.commit()
        return self._connection

    @staticmethod
    def _from_row(row: sqlite3.Row) -> SupportCase:
        return SupportCase(**{key: row[key] for key in SupportCase.__dataclass_fields__})

    def create_case(
        self,
        *,
        session_id: str,
        tenant_id: str = "default",
        category: str,
        summary: str,
        idempotency_key: str,
    ) -> SupportCase:
        """Create one queued case; repeated confirmation processing is idempotent."""

        safe_summary = " ".join(summary.split())[:1000]
        if not safe_summary:
            safe_summary = "Customer requested help from a NovaTel advisor."
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock:
            db = self._db()
            row = db.execute("SELECT * FROM support_cases WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if row is not None:
                if row["category"] != category or row["session_id"] != session_id:
                    raise ValueError("An idempotency key cannot be reused for another support case")
                return self._from_row(row)
            case = SupportCase(
                case_id=f"NT-{uuid4().hex[:10].upper()}",
                session_id=session_id,
                category=category,
                summary=safe_summary,
                status="waiting",
                created_at=now,
                tenant_id=tenant_id.strip()[:80] or "default",
            )
            db.execute(
                "INSERT INTO support_cases "
                "(case_id,session_id,category,summary,status,created_at,tenant_id,idempotency_key) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    case.case_id,
                    case.session_id,
                    case.category,
                    case.summary,
                    case.status,
                    case.created_at,
                    case.tenant_id,
                    idempotency_key,
                ),
            )
            self._audit(db, case.case_id, "case_created", "customer")
            db.commit()
            return case

    def get_case(self, case_id: str) -> SupportCase | None:
        with self._lock:
            row = self._db().execute("SELECT * FROM support_cases WHERE case_id = ?", (case_id,)).fetchone()
            return self._from_row(row) if row is not None else None

    def list_open_cases(self) -> list[SupportCase]:
        with self._lock:
            rows = (
                self._db()
                .execute("SELECT * FROM support_cases WHERE status IN ('waiting','in_progress') ORDER BY created_at")
                .fetchall()
            )
            return [self._from_row(row) for row in rows]

    def claim(self, case_id: str, agent: str) -> SupportCase | None:
        """Atomically assign a waiting case, rejecting competing agent claims."""

        with self._lock:
            db = self._db()
            db.execute(
                "UPDATE support_cases SET status='in_progress', assigned_to=? WHERE case_id=? AND status='waiting'",
                (agent, case_id),
            )
            row = db.execute("SELECT * FROM support_cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None or row["status"] == "resolved":
                db.rollback()
                return None
            if row["assigned_to"] != agent:
                db.rollback()
                return None
            self._audit(db, case_id, "case_claimed", agent)
            db.commit()
            return self._from_row(row)

    def resolve(self, case_id: str, agent: str) -> SupportCase | None:
        with self._lock:
            db = self._db()
            db.execute(
                "UPDATE support_cases SET status='resolved' WHERE case_id=? AND assigned_to=? AND status='in_progress'",
                (case_id, agent),
            )
            row = db.execute("SELECT * FROM support_cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None or row["status"] != "resolved" or row["assigned_to"] != agent:
                db.rollback()
                return None
            self._audit(db, case_id, "case_resolved", agent)
            db.commit()
            return self._from_row(row)

    def release_agent_cases(self, agent: str) -> list[SupportCase]:
        """Return an advisor's unfinished cases to the shared queue on disconnect."""

        with self._lock:
            db = self._db()
            rows = db.execute(
                "SELECT case_id FROM support_cases WHERE assigned_to=? AND status='in_progress'", (agent,)
            ).fetchall()
            case_ids = [row["case_id"] for row in rows]
            if not case_ids:
                return []
            db.execute(
                "UPDATE support_cases SET status='waiting', assigned_to=NULL "
                "WHERE assigned_to=? AND status='in_progress'",
                (agent,),
            )
            for case_id in case_ids:
                self._audit(db, case_id, "case_requeued", "system")
            db.commit()
            return [
                self._from_row(db.execute("SELECT * FROM support_cases WHERE case_id=?", (case_id,)).fetchone())
                for case_id in case_ids
            ]

    def set_crm_delivery(self, case_id: str, status: str, reference: str | None = None) -> SupportCase:
        if status not in {"delivered", "failed", "not_configured"}:
            raise ValueError("Unsupported CRM delivery status")
        with self._lock:
            db = self._db()
            db.execute(
                "UPDATE support_cases SET crm_status=?, crm_reference=? WHERE case_id=?",
                (status, reference, case_id),
            )
            self._audit(db, case_id, f"crm_{status}", "system")
            db.commit()
            row = db.execute("SELECT * FROM support_cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            return self._from_row(row)

    def record_message(self, case_id: str, sender: str, text: str) -> SupportMessage:
        """Persist a bounded chat message after the customer has consented to handoff."""

        if sender not in {"customer", "advisor"}:
            raise ValueError("Support message sender must be customer or advisor")
        safe_text = " ".join(text.split())[:4000]
        if not safe_text:
            raise ValueError("Support message cannot be empty")
        message = SupportMessage(
            message_id=uuid4().hex,
            case_id=case_id,
            sender=sender,
            text=safe_text,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        with self._lock:
            db = self._db()
            if db.execute("SELECT 1 FROM support_cases WHERE case_id=?", (case_id,)).fetchone() is None:
                raise KeyError(case_id)
            db.execute(
                "INSERT INTO support_case_messages(message_id,case_id,sender,text,created_at) VALUES (?,?,?,?,?)",
                (message.message_id, message.case_id, message.sender, message.text, message.created_at),
            )
            self._audit(db, case_id, f"message_{sender}", sender)
            db.commit()
        return message

    def list_messages(self, case_id: str) -> list[SupportMessage]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT * FROM support_case_messages WHERE case_id=? ORDER BY rowid", (case_id,)
                )
                .fetchall()
            )
            return [SupportMessage(**dict(row)) for row in rows]

    def list_audit_events(self, case_id: str) -> list[dict[str, str | int]]:
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT audit_id,case_id,event,actor,created_at FROM support_case_audit "
                    "WHERE case_id=? ORDER BY audit_id",
                    (case_id,),
                )
                .fetchall()
            )
            return [dict(row) for row in rows]

    @staticmethod
    def _audit(db: sqlite3.Connection, case_id: str, event: str, actor: str) -> None:
        db.execute(
            "INSERT INTO support_case_audit(case_id,event,actor,created_at) VALUES (?,?,?,?)",
            (case_id, event, actor, datetime.now(UTC).isoformat(timespec="seconds")),
        )


class CRMWebhookClient:
    """Forward a minimum, consented case payload to a configured CRM adapter."""

    def __init__(
        self,
        url: str = "",
        bearer_token: str = "",
        timeout_seconds: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url.strip()
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.url)

    async def deliver(self, case: SupportCase) -> str | None:
        """POST one idempotent case to a webhook; never follow redirects."""

        if not self.configured:
            return None
        if not self.bearer_token:
            raise CRMDeliveryFailure("CRM webhook is missing its bearer credential")
        headers = {"Idempotency-Key": case.case_id}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        payload: dict[str, Any] = {
            "event": "support_case.created",
            "case": {
                "id": case.case_id,
                "tenant_id": case.tenant_id,
                "category": case.category,
                "summary": case.summary,
                "created_at": case.created_at,
                "source": "novatel_voice_agent",
            },
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=False, transport=self.transport
            ) as client:
                response = await client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("CRM webhook delivery failed case_id=%s error_type=%s", case.case_id, type(exc).__name__)
            raise CRMDeliveryFailure(type(exc).__name__) from exc
        reference: str | None = None
        try:
            body = response.json()
            if isinstance(body, dict):
                candidate = body.get("ticket_id") or body.get("case_id") or body.get("id")
                reference = str(candidate)[:120] if candidate is not None else None
        except (ValueError, TypeError):
            pass
        return reference

    async def deliver_message(self, case: SupportCase, message: SupportMessage) -> None:
        """Notify the CRM integration of a new consented customer/advisor message."""

        if not self.configured:
            return
        if not self.bearer_token:
            raise CRMDeliveryFailure("CRM webhook is missing its bearer credential")
        headers = {"Idempotency-Key": message.message_id}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        payload = {
            "event": "support_case.message",
            "case": {"id": case.case_id, "crm_reference": case.crm_reference},
            "message": {
                "id": message.message_id,
                "sender": message.sender,
                "text": message.text,
                "created_at": message.created_at,
            },
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=False, transport=self.transport
            ) as client:
                response = await client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "CRM case message delivery failed case_id=%s error_type=%s", case.case_id, type(exc).__name__
            )
            raise CRMDeliveryFailure(type(exc).__name__) from exc


class HandoffHub:
    """In-process realtime delivery for one FastAPI worker."""

    def __init__(self) -> None:
        self._customers: dict[str, tuple[Any, Any]] = {}
        self._agents: dict[WebSocket, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def register_customer(self, session_id: str, send_event: Any, send_audio: Any) -> None:
        async with self._lock:
            self._customers[session_id] = (send_event, send_audio)

    async def unregister_customer(self, session_id: str) -> None:
        async with self._lock:
            self._customers.pop(session_id, None)

    async def register_agent(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._agents[websocket] = asyncio.Lock()

    async def unregister_agent(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._agents.pop(websocket, None)

    async def send_agent(self, websocket: WebSocket, event: dict[str, object]) -> None:
        async with self._lock:
            lock = self._agents.get(websocket)
        if lock is None:
            return
        async with lock:
            await websocket.send_json(event)

    async def send_customer_event(self, session_id: str, event: dict[str, object]) -> bool:
        async with self._lock:
            connection = self._customers.get(session_id)
        if connection is None:
            return False
        try:
            await connection[0](event)
            return True
        except Exception:
            logger.info("Customer handoff connection is no longer available")
            return False

    async def send_customer_audio(self, session_id: str, audio: bytes) -> bool:
        async with self._lock:
            connection = self._customers.get(session_id)
        if connection is None:
            return False
        try:
            await connection[1](audio)
            return True
        except Exception:
            logger.info("Could not deliver advisor audio to customer")
            return False

    async def broadcast_to_agents(self, event: dict[str, object]) -> None:
        async with self._lock:
            agents = list(self._agents.items())

        async def deliver(websocket: WebSocket, lock: asyncio.Lock) -> None:
            try:
                async with lock:
                    await websocket.send_json(event)
            except Exception:
                await self.unregister_agent(websocket)

        await asyncio.gather(*(deliver(websocket, lock) for websocket, lock in agents))

    @staticmethod
    def authenticate(provided: str, expected: str) -> bool:
        return bool(expected) and hmac.compare_digest(provided.encode(), expected.encode())


def serialize_case(case: SupportCase) -> dict[str, str | None]:
    """Return a JSON-safe support case for the UI and websocket protocol."""

    return asdict(case)
