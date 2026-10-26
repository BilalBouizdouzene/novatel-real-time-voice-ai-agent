"""HubSpot public-app OAuth endpoints for multi-portal installations."""

import logging
import re
from html import escape

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.integrations.hubspot_oauth import HubSpotOAuthStore

logger = logging.getLogger(__name__)


def register_hubspot_oauth_routes(app, store: HubSpotOAuthStore) -> None:
    """Register OAuth routes without exposing access or refresh tokens."""

    router = APIRouter()

    @router.get("/auth/hubspot/start")
    async def start(request: Request, tenant_id: str = "default"):
        settings = request.app.state.settings
        if not store.enabled or not settings.hubspot_oauth_client_id or not settings.hubspot_oauth_redirect_uri:
            raise HTTPException(status_code=503, detail="HubSpot OAuth is not configured")
        safe_tenant = re.sub(r"[^A-Za-z0-9_.:-]", "", tenant_id)[:80] or "default"
        url = store.create_authorization_url(
            tenant_id=safe_tenant,
            client_id=settings.hubspot_oauth_client_id,
            redirect_uri=settings.hubspot_oauth_redirect_uri,
            scopes=settings.hubspot_oauth_scopes,
        )
        return RedirectResponse(url=url, status_code=307)

    @router.get("/auth/hubspot/callback", response_class=HTMLResponse)
    async def callback(request: Request, code: str = "", state: str = "", error: str = ""):
        if error:
            return HTMLResponse(
                f"<h1>HubSpot authorization declined</h1><p>{escape(error[:200])}</p>", status_code=400
            )
        if not code or not state:
            return HTMLResponse("<h1>Invalid HubSpot OAuth callback</h1>", status_code=400)
        settings = request.app.state.settings
        try:
            tenant_id = store.consume_state(state)
            connection = await store.exchange_code_for_tenant(
                tenant_id=tenant_id,
                code=code,
                client_id=settings.hubspot_oauth_client_id,
                client_secret=settings.hubspot_oauth_client_secret,
                redirect_uri=settings.hubspot_oauth_redirect_uri,
                scopes=settings.hubspot_oauth_scopes,
            )
        except Exception:
            logger.warning("HubSpot OAuth callback failed", exc_info=True)
            return HTMLResponse(
                "<h1>HubSpot connection failed</h1><p>Check the server logs and redirect URI configuration.</p>",
                status_code=400,
            )
        return HTMLResponse(
            "<h1>HubSpot connected</h1>"
            f"<p>Tenant <strong>{connection['tenant_id']}</strong> is connected to portal "
            f"<strong>{connection['portal_id']}</strong>. You can close this tab.</p>"
        )

    def _authorized(request: Request) -> None:
        expected = request.app.state.settings.agent_dashboard_token
        provided = request.headers.get("X-Agent-Token", "")
        if not expected or not HubSpotOAuthStore.authenticate_token(provided, expected):
            raise HTTPException(status_code=401, detail="Advisor token required")

    @router.get("/api/hubspot/connections")
    async def connections(request: Request):
        _authorized(request)
        return {"connections": store.connections()}

    app.include_router(router)
