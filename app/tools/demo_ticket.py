"""SQLite-backed demo tickets with explicit consent and idempotent creation.

This service is deliberately local: it never calls a CRM, billing system, or
customer account. Only the synthetic ticket and audit events are persisted.
"""

import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.pipeline.intents import NovaTelIntent, normalize_intent_text

_TICKET_REQUEST = re.compile(
    r"\b(?:ouvrir|creer|faire|preparer|signaler|declarer)\b.{0,50}\b(?:ticket|reclamation|incident|panne|demande)\b"
    r"|\b(?:ticket|reclamation)\b",
)
_CONFIRM = {
    "oui",
    "oui je confirme",
    "je confirme",
    "confirme",
    "confirmer",
    "d accord",
    "ok",
    "okay",
    "yes",
    "confirm",
}
_DECLINE = {"non", "non merci", "je refuse", "annule", "annuler", "stop", "no", "no thanks", "cancel"}


@dataclass(frozen=True)
class DemoTicket:
    ticket_id: str
    category: str
    status: str
    created_at: str


class DemoTicketService:
    """Persist synthetic tickets in SQLite; idempotency prevents duplicate actions."""

    def __init__(self, database_path: str | Path = ":memory:") -> None:
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
                CREATE TABLE IF NOT EXISTS demo_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS demo_ticket_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    ticket_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(idempotency_key, event)
                );
                """
            )
        return self._connection

    def record_event(self, session_id: str, idempotency_key: str, category: str, event: str) -> None:
        """Add an audit event at most once for one logical ticket request."""

        with self._lock:
            db = self._db()
            db.execute(
                "INSERT OR IGNORE INTO demo_ticket_audit "
                "(event, idempotency_key, session_id, category, created_at) VALUES (?, ?, ?, ?, ?)",
                (event, idempotency_key, session_id, category, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            db.commit()

    def create(
        self,
        category: str,
        *,
        idempotency_key: str | None = None,
        session_id: str = "local-demo",
    ) -> DemoTicket:
        """Create or return the ticket associated with an explicit confirmed request."""

        key = idempotency_key or uuid4().hex
        with self._lock:
            db = self._db()
            existing = db.execute("SELECT * FROM demo_tickets WHERE idempotency_key = ?", (key,)).fetchone()
            if existing is not None:
                if existing["category"] != category:
                    raise ValueError("An idempotency key cannot be reused for a different ticket category")
                return DemoTicket(
                    existing["ticket_id"], existing["category"], existing["status"], existing["created_at"]
                )

            ticket = DemoTicket(
                ticket_id=f"DEMO-{uuid4().hex[:8].upper()}",
                category=category,
                status="created_in_demo_session",
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            db.execute(
                "INSERT INTO demo_tickets "
                "(ticket_id, category, status, created_at, session_id, idempotency_key) VALUES (?, ?, ?, ?, ?, ?)",
                (ticket.ticket_id, ticket.category, ticket.status, ticket.created_at, session_id, key),
            )
            db.execute(
                "INSERT OR IGNORE INTO demo_ticket_audit "
                "(event, idempotency_key, session_id, category, ticket_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "ticket_created",
                    key,
                    session_id,
                    category,
                    ticket.ticket_id,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            db.commit()
            return ticket

    def list_tickets(self) -> list[dict[str, str]]:
        """Return synthetic records for local inspection and tests."""

        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT ticket_id, category, status, created_at, session_id FROM demo_tickets ORDER BY created_at"
                )
                .fetchall()
            )
            return [dict(row) for row in rows]

    def list_audit_events(self) -> list[dict[str, str | int | None]]:
        """Return ticket workflow audit events without conversation content."""

        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT audit_id, event, idempotency_key, session_id, category, ticket_id, created_at "
                    "FROM demo_ticket_audit ORDER BY audit_id"
                )
                .fetchall()
            )
            return [dict(row) for row in rows]


def requested_ticket_category(text: str, intent: NovaTelIntent) -> str | None:
    """Return a category only when the user explicitly asks to open/report a case."""

    if not _TICKET_REQUEST.search(normalize_intent_text(text)):
        return None
    if intent == NovaTelIntent.OUT_OF_SCOPE:
        return "service_request"
    return intent.value


def confirmation_decision(text: str) -> bool | None:
    """Interpret clear natural yes/no confirmations without guessing on mixed answers."""

    normalized = normalize_intent_text(text)
    if normalized in _CONFIRM:
        return True
    if normalized in _DECLINE:
        return False
    negatives = {"mais", "non", "pas", "jamais", "attends", "attendez", "annule"}
    tokens = normalized.split()
    if tokens and tokens[0] in {"oui", "yes", "ok", "okay"}:
        return None if negatives.intersection(tokens[1:]) else True
    if normalized.startswith(("d accord ", "je confirme ", "je suis d accord ", "j accepte ", "je veux bien ")):
        return True
    if normalized.startswith(("non ", "no ", "je refuse ", "je ne veux pas ", "je ne suis pas d accord ")):
        return False
    return None


def ticket_payload(ticket: DemoTicket) -> dict[str, str]:
    """Serialize only the synthetic ticket fields sent to the client."""

    return asdict(ticket)
