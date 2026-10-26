"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.agent import register_agent_routes
from app.api.hubspot_oauth import register_hubspot_oauth_routes
from app.api.routes import router
from app.api.twilio import register_twilio_routes
from app.api.webrtc import register_webrtc_route
from app.api.websocket import register_websocket_route
from app.config import Settings, get_settings
from app.integrations.hubspot_oauth import HubSpotOAuthStore, OAuthHubSpotCRMClient
from app.integrations.twilio_voice import TwilioVoiceClient
from app.llm.client import OpenAICompatibleLLMClient
from app.logging_config import configure_logging
from app.monitoring.tracing import configure_tracing
from app.rag.retriever import RAGRetriever
from app.speech.asr import ASRService
from app.speech.tts import create_tts_service
from app.speech.vad import StreamingSileroVAD
from app.tools.demo_ticket import DemoTicketService
from app.tools.hubspot_crm import HubSpotCRMClient
from app.tools.support_handoff import CRMWebhookClient, HandoffHub, SupportCaseService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an app with model loading deferred until a voice session needs it."""

    selected = settings or get_settings()
    configure_logging(selected.log_level)
    configure_tracing(
        service_name=selected.app_name,
        enabled=selected.otel_enabled,
        otlp_traces_endpoint=selected.otel_exporter_otlp_traces_endpoint,
    )
    application = FastAPI(title=selected.app_name, version="0.1.0")
    application.state.settings = selected
    application.state.asr = ASRService(
        model_name=selected.whisper_model,
        device=selected.whisper_device,
        compute_type=selected.whisper_compute_type,
        language=selected.whisper_language,
        cpu_threads=selected.whisper_cpu_threads,
        beam_size=selected.whisper_beam_size,
        initial_prompt=selected.whisper_initial_prompt,
    )
    application.state.retriever = RAGRetriever(
        knowledge_base=selected.knowledge_base_dir,
        vector_store_dir=selected.vector_store_dir,
        model_name=selected.embedding_model,
        dense_min_free_disk_mb=selected.rag_dense_min_free_disk_mb,
        top_k=selected.rag_top_k,
        min_score=selected.rag_min_score,
        chunk_size=selected.rag_chunk_size,
        chunk_overlap=selected.rag_chunk_overlap,
        strategy=selected.rag_strategy,
        reranker_model=selected.rag_reranker_model,
    )
    application.state.demo_ticket_service = DemoTicketService(selected.demo_tickets_db_path)
    application.state.support_case_service = SupportCaseService(selected.support_cases_db_path)
    application.state.hubspot_oauth_store = HubSpotOAuthStore(
        selected.hubspot_oauth_db_path,
        encryption_key=selected.hubspot_oauth_encryption_key,
        timeout_seconds=selected.crm_webhook_timeout_seconds,
    )
    if selected.resolved_crm_provider == "hubspot":
        application.state.crm_client = HubSpotCRMClient(
            selected.hubspot_private_app_token,
            timeout_seconds=selected.crm_webhook_timeout_seconds,
            ticket_pipeline=selected.hubspot_ticket_pipeline,
            ticket_stage=selected.hubspot_ticket_stage,
        )
    elif selected.resolved_crm_provider == "hubspot_oauth":
        application.state.crm_client = OAuthHubSpotCRMClient(
            application.state.hubspot_oauth_store,
            client_id=selected.hubspot_oauth_client_id,
            client_secret=selected.hubspot_oauth_client_secret,
            scopes=selected.hubspot_oauth_scopes,
            timeout_seconds=selected.crm_webhook_timeout_seconds,
            ticket_pipeline=selected.hubspot_ticket_pipeline,
            ticket_stage=selected.hubspot_ticket_stage,
        )
    elif selected.resolved_crm_provider == "webhook":
        application.state.crm_client = CRMWebhookClient(
            url=selected.crm_webhook_url,
            bearer_token=selected.crm_webhook_bearer_token,
            timeout_seconds=selected.crm_webhook_timeout_seconds,
        )
    else:
        application.state.crm_client = CRMWebhookClient()
    application.state.handoff_hub = HandoffHub()
    application.state.llm = OpenAICompatibleLLMClient(
        base_url=selected.llm_base_url,
        model=selected.llm_model,
        api_key=selected.llm_api_key,
        timeout_seconds=selected.llm_timeout_seconds,
    )
    application.state.tts = create_tts_service(selected)
    application.state.vad_factory = lambda: StreamingSileroVAD.from_silero(
        threshold=selected.vad_threshold,
        min_speech_ms=selected.vad_min_speech_ms,
        min_silence_ms=selected.vad_min_silence_ms,
        max_speech_ms=selected.vad_max_speech_ms,
        pre_roll_ms=selected.vad_pre_roll_ms,
    )
    application.include_router(router)
    register_websocket_route(application)
    register_webrtc_route(application)
    register_agent_routes(application)
    register_hubspot_oauth_routes(application, application.state.hubspot_oauth_store)
    application.state.twilio_client = TwilioVoiceClient(
        account_sid=selected.twilio_account_sid,
        auth_token=selected.twilio_auth_token,
        api_key_sid=selected.twilio_api_key_sid,
        api_key_secret=selected.twilio_api_key_secret,
        from_number=selected.twilio_from_number,
        advisor_number=selected.twilio_advisor_number,
        webhook_base_url=selected.twilio_webhook_base_url,
        enabled=selected.twilio_enabled,
    )
    register_twilio_routes(application, application.state.twilio_client)
    application.mount("/static", StaticFiles(directory=selected.frontend_dir), name="static")
    return application


app = create_app()
