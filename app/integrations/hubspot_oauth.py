"""Encrypted HubSpot public-app OAuth storage and CRM adapter.

HubSpot private-app tokens are intentionally kept as the simple single-portal
option. This module is only used when ``CRM_PROVIDER=hubspot_oauth`` and stores
one encrypted refresh token per tenant in a local SQLite database.
"""

from __future__ import annotations

import hmac
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from app.tools.hubspot_crm import HubSpotCRMClient
from app.tools.support_handoff import CRMDeliveryFailure, SupportCase, SupportMessage


class HubSpotOAuthStore:
    """Persist short-lived OAuth credentials without storing them in plaintext."""

    AUTH_URL = "https://app.hubspot.com/oauth/authorize"
    TOKEN_URL = "https://api.hubapi.com/oauth/v3/token"
    ACCOUNT_URL = "https://api.hubapi.com/account-info/v3/details"

    def __init__(self, database_path: str | Path, *, encryption_key: str = "", timeout_seconds: float = 8.0) -> None:
        self.database_path = str(database_path)
        self.timeout_seconds = timeout_seconds
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._fernet = None
        if encryption_key.strip():
            try:
                from cryptography.fernet import Fernet
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Install the integrations extra to enable HubSpot OAuth: pip install -e .[integrations]"
                ) from exc
            try:
                self._fernet = Fernet(encryption_key.strip().encode())
            except Exception as exc:
                raise ValueError("HUBSPOT_OAUTH_ENCRYPTION_KEY is not a valid Fernet key") from exc

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    @staticmethod
    def authenticate_token(provided: str, expected: str) -> bool:
        return bool(expected) and hmac.compare_digest(provided.encode(), expected.encode())

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            if self.database_path != ":memory:":
                Path(self.database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.database_path, timeout=10, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS oauth_states (
                    state TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hubspot_connections (
                    tenant_id TEXT PRIMARY KEY,
                    portal_id TEXT NOT NULL,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    scopes TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
        return self._connection

    def create_authorization_url(
        self, *, tenant_id: str, client_id: str, redirect_uri: str, scopes: str, ttl_seconds: int = 600
    ) -> str:
        if not self.enabled:
            raise RuntimeError("HubSpot OAuth is disabled; configure HUBSPOT_OAUTH_ENCRYPTION_KEY")
        state = secrets.token_urlsafe(32)
        with self._lock:
            self._db().execute(
                "INSERT INTO oauth_states(state,tenant_id,expires_at) VALUES (?,?,?)",
                (state, tenant_id, time.time() + ttl_seconds),
            )
            self._db().commit()
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scopes,
            "state": state,
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    def consume_state(self, state: str) -> str:
        with self._lock:
            row = self._db().execute("SELECT tenant_id,expires_at FROM oauth_states WHERE state=?", (state,)).fetchone()
            self._db().execute("DELETE FROM oauth_states WHERE state=?", (state,))
            self._db().commit()
        if row is None or float(row["expires_at"]) < time.time():
            raise ValueError("OAuth state is invalid or expired")
        return str(row["tenant_id"])

    def _encrypt(self, value: str) -> str:
        if self._fernet is None:
            raise RuntimeError("HubSpot OAuth encryption is disabled")
        return self._fernet.encrypt(value.encode()).decode()

    def _decrypt(self, value: str) -> str:
        if self._fernet is None:
            raise RuntimeError("HubSpot OAuth encryption is disabled")
        return self._fernet.decrypt(value.encode()).decode()

    async def exchange_code_for_tenant(
        self,
        *,
        tenant_id: str,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: str,
    ) -> dict[str, str]:
        if not self.enabled:
            raise RuntimeError("HubSpot OAuth is disabled")
        payload = {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "code": code,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=False) as client:
            response = await client.post(self.TOKEN_URL, data=payload)
            response.raise_for_status()
            token_body = response.json()
            access_token = str(token_body.get("access_token", ""))
            refresh_token = str(token_body.get("refresh_token", ""))
            if not access_token or not refresh_token:
                raise ValueError("HubSpot OAuth response did not contain both tokens")
            account = await client.get(self.ACCOUNT_URL, headers={"Authorization": f"Bearer {access_token}"})
            account.raise_for_status()
            account_body = account.json()
        portal_id = str(account_body.get("portalId") or account_body.get("portal_id") or "unknown")
        self.save_tokens(
            tenant_id=tenant_id,
            portal_id=portal_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(token_body.get("expires_in", 1800)),
            scopes=str(token_body.get("scope", scopes)),
        )
        return {"tenant_id": tenant_id, "portal_id": portal_id, "scopes": str(token_body.get("scope", scopes))}

    def save_tokens(
        self,
        *,
        tenant_id: str,
        portal_id: str,
        access_token: str,
        refresh_token: str,
        expires_in: int,
        scopes: str,
    ) -> None:
        now = time.time()
        with self._lock:
            self._db().execute(
                """INSERT INTO hubspot_connections
                   (tenant_id,portal_id,access_token,refresh_token,expires_at,scopes,updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(tenant_id) DO UPDATE SET portal_id=excluded.portal_id,
                   access_token=excluded.access_token,refresh_token=excluded.refresh_token,
                   expires_at=excluded.expires_at,scopes=excluded.scopes,updated_at=excluded.updated_at""",
                (
                    tenant_id,
                    portal_id,
                    self._encrypt(access_token),
                    self._encrypt(refresh_token),
                    now + max(60, expires_in - 60),
                    scopes,
                    now,
                ),
            )
            self._db().commit()

    async def access_token(self, *, tenant_id: str, client_id: str, client_secret: str, scopes: str) -> str:
        with self._lock:
            row = self._db().execute("SELECT * FROM hubspot_connections WHERE tenant_id=?", (tenant_id,)).fetchone()
        if row is None:
            raise CRMDeliveryFailure(f"No HubSpot OAuth connection exists for tenant {tenant_id}")
        if float(row["expires_at"]) > time.time():
            return self._decrypt(row["access_token"])
        refresh_token = self._decrypt(row["refresh_token"])
        payload = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=False) as client:
                response = await client.post(self.TOKEN_URL, data=payload)
                response.raise_for_status()
            body = response.json()
            access_token = str(body.get("access_token", ""))
            if not access_token:
                raise ValueError("HubSpot refresh did not return an access token")
            self.save_tokens(
                tenant_id=tenant_id,
                portal_id=str(row["portal_id"]),
                access_token=access_token,
                refresh_token=str(body.get("refresh_token", refresh_token)),
                expires_in=int(body.get("expires_in", 1800)),
                scopes=str(body.get("scope", scopes)),
            )
            return access_token
        except (httpx.HTTPError, ValueError) as exc:
            raise CRMDeliveryFailure(f"HubSpot OAuth refresh failed ({type(exc).__name__})") from exc

    def connections(self) -> list[dict[str, str]]:
        with self._lock:
            rows = self._db().execute(
                "SELECT tenant_id,portal_id,scopes,updated_at FROM hubspot_connections ORDER BY tenant_id"
            ).fetchall()
        return [
            {
                "tenant_id": str(row["tenant_id"]),
                "portal_id": str(row["portal_id"]),
                "scopes": str(row["scopes"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]


class OAuthHubSpotCRMClient:
    """CRM adapter that selects a HubSpot public-app token per tenant."""

    def __init__(
        self,
        store: HubSpotOAuthStore,
        *,
        client_id: str,
        client_secret: str,
        scopes: str,
        timeout_seconds: float = 8.0,
        ticket_pipeline: str = "0",
        ticket_stage: str = "1",
    ) -> None:
        self.store = store
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes
        self.timeout_seconds = timeout_seconds
        self.ticket_pipeline = ticket_pipeline
        self.ticket_stage = ticket_stage

    @property
    def configured(self) -> bool:
        return self.store.enabled and bool(self.client_id and self.client_secret)

    async def _client(self, tenant_id: str) -> HubSpotCRMClient:
        token = await self.store.access_token(
            tenant_id=tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=self.scopes,
        )
        return HubSpotCRMClient(
            token,
            timeout_seconds=self.timeout_seconds,
            ticket_pipeline=self.ticket_pipeline,
            ticket_stage=self.ticket_stage,
        )

    async def deliver(self, case: SupportCase) -> str | None:
        return await (await self._client(case.tenant_id)).deliver(case)

    async def deliver_message(self, case: SupportCase, message: SupportMessage) -> None:
        await (await self._client(case.tenant_id)).deliver_message(case, message)
