"""State and bounded conversational history for one WebSocket connection."""

from dataclasses import dataclass, field
from uuid import uuid4

from app.models.schemas import AgentState, RetrievedDocument


@dataclass
class ConversationSession:
    """All mutable conversational data is scoped to one client connection."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = "default"
    state: AgentState = AgentState.IDLE
    history: list[dict[str, str]] = field(default_factory=list)
    last_rag_query: str | None = None
    last_retrieved_documents: list[RetrievedDocument] = field(default_factory=list)
    last_intent: str | None = None
    pending_ticket_category: str | None = None
    pending_ticket_request_id: str | None = None
    pending_ticket_summary: str | None = None
    active_handoff_case_id: str | None = None
    demo_tickets: list[dict[str, str]] = field(default_factory=list)

    def bounded_history(self, max_messages: int) -> list[dict[str, str]]:
        """Return the last complete messages supplied to the model."""

        even_limit = max_messages - (max_messages % 2)
        return self.history[-even_limit:] if even_limit else []

    def clear(self) -> None:
        """Forget this session's messages and reset its state."""

        self.history.clear()
        self.last_rag_query = None
        self.last_retrieved_documents.clear()
        self.last_intent = None
        self.pending_ticket_category = None
        self.pending_ticket_request_id = None
        self.pending_ticket_summary = None
        self.demo_tickets.clear()
        self.state = AgentState.IDLE
