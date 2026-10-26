"""Centralized, environment-driven application settings."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings; secrets are read only from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NovaTel Real-Time Voice AI Agent"
    environment: str = "development"
    log_level: str = "INFO"
    otel_enabled: bool = False
    otel_exporter_otlp_traces_endpoint: str = ""

    host: str = "0.0.0.0"
    port: int = 8000
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    knowledge_base_dir: Path = PROJECT_ROOT / "knowledge_base" / "novatel"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "novatel_faiss"

    max_audio_chunk_bytes: int = Field(default=32_000, ge=2)

    vad_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    vad_min_speech_ms: int = Field(default=200, ge=0)
    vad_min_silence_ms: int = Field(default=650, ge=100)
    vad_max_speech_ms: int = Field(default=20_000, ge=1000)
    vad_pre_roll_ms: int = Field(default=200, ge=0)

    whisper_model: str = "tiny"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_language: str = "fr"
    whisper_cpu_threads: int = Field(default=1, ge=1, le=32)
    whisper_beam_size: int = Field(default=1, ge=1, le=5)
    whisper_initial_prompt: str = (
        "Service client NovaTel en français. Mots importants : abonnement, forfait, résilier, résiliation, facture, "
        "prélèvement, paiement, remboursement, réseau, itinérance. Exemple : comment résilier mon abonnement."
    )

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_dense_min_free_disk_mb: int = Field(default=1536, ge=0)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_min_score: float = Field(default=0.30, ge=-1.0, le=1.0)
    rag_chunk_size: int = Field(default=1000, ge=200)
    rag_chunk_overlap: int = Field(default=150, ge=0)
    rag_strategy: str = "auto"
    rag_reranker_model: str = ""
    demo_tickets_db_path: Path = PROJECT_ROOT / "data" / "novatel_demo_tickets.sqlite3"
    support_cases_db_path: Path = PROJECT_ROOT / "data" / "support_cases.sqlite3"
    crm_provider: str = "auto"
    crm_webhook_url: str = ""
    crm_webhook_bearer_token: str = ""
    crm_webhook_timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    hubspot_private_app_token: str = ""
    hubspot_ticket_pipeline: str = "0"
    hubspot_ticket_stage: str = "1"
    hubspot_oauth_client_id: str = ""
    hubspot_oauth_client_secret: str = ""
    hubspot_oauth_redirect_uri: str = ""
    hubspot_oauth_scopes: str = (
        "crm.objects.tickets.read crm.objects.tickets.write "
        "crm.objects.notes.read crm.objects.notes.write"
    )
    hubspot_oauth_db_path: Path = PROJECT_ROOT / "data" / "hubspot_oauth.sqlite3"
    hubspot_oauth_encryption_key: str = ""
    agent_dashboard_token: str = ""

    voice_transport: str = "auto"  # auto, websocket, webrtc
    webrtc_enabled: bool = True
    tts_stream_sample_rate: int = Field(default=24_000, ge=8_000, le=48_000)
    tts_audio_frame_ms: int = Field(default=40, ge=10, le=200)

    twilio_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_api_key_sid: str = ""
    twilio_api_key_secret: str = ""
    twilio_from_number: str = ""
    twilio_advisor_number: str = ""
    twilio_webhook_base_url: str = ""

    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = ""
    llm_model: str = "gemini-3.1-flash-lite"
    llm_timeout_seconds: float = 90.0
    conversation_history_messages: int = Field(default=8, ge=0, le=30)

    tts_backend: str = "auto"  # auto, piper, espeak, windows, disabled
    piper_executable: str = "piper"
    piper_voice_model: str = ""
    piper_voice_config: str = ""
    espeak_executable: str = "espeak-ng"
    windows_powershell_executable: str = "powershell.exe"
    tts_language: str = "fr"
    tts_voice: str = ""

    @model_validator(mode="after")
    def validate_pipeline_limits(self) -> "Settings":
        """Reject incompatible chunking and speech-duration settings early."""

        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.vad_min_speech_ms >= self.vad_max_speech_ms:
            raise ValueError("VAD_MIN_SPEECH_MS must be smaller than VAD_MAX_SPEECH_MS")
        if self.rag_strategy.lower() not in {"auto", "bm25", "dense", "hybrid"}:
            raise ValueError("RAG_STRATEGY must be auto, bm25, dense or hybrid")
        provider = self.crm_provider.strip().lower()
        if provider not in {"auto", "none", "webhook", "hubspot", "hubspot_oauth"}:
            raise ValueError("CRM_PROVIDER must be auto, none, webhook, hubspot or hubspot_oauth")
        if provider == "auto" and self.crm_webhook_url and self.hubspot_private_app_token:
            raise ValueError("Configure only one CRM provider when CRM_PROVIDER=auto")
        if provider == "hubspot" and not self.hubspot_private_app_token.strip():
            raise ValueError("HUBSPOT_PRIVATE_APP_TOKEN is required when CRM_PROVIDER=hubspot")
        if provider == "hubspot_oauth":
            required = {
                "HUBSPOT_OAUTH_CLIENT_ID": self.hubspot_oauth_client_id,
                "HUBSPOT_OAUTH_CLIENT_SECRET": self.hubspot_oauth_client_secret,
                "HUBSPOT_OAUTH_REDIRECT_URI": self.hubspot_oauth_redirect_uri,
                "HUBSPOT_OAUTH_ENCRYPTION_KEY": self.hubspot_oauth_encryption_key,
            }
            missing = [name for name, value in required.items() if not value.strip()]
            if missing:
                raise ValueError(f"Missing HubSpot OAuth settings: {', '.join(missing)}")
        if self.voice_transport.lower() not in {"auto", "websocket", "webrtc"}:
            raise ValueError("VOICE_TRANSPORT must be auto, websocket or webrtc")
        if self.twilio_enabled:
            required_twilio = {
                "TWILIO_ACCOUNT_SID": self.twilio_account_sid,
                "TWILIO_FROM_NUMBER": self.twilio_from_number,
                "TWILIO_ADVISOR_NUMBER": self.twilio_advisor_number,
                "TWILIO_WEBHOOK_BASE_URL": self.twilio_webhook_base_url,
            }
            missing = [name for name, value in required_twilio.items() if not value.strip()]
            if missing:
                raise ValueError(f"Missing Twilio settings: {', '.join(missing)}")
            has_auth_token = bool(self.twilio_auth_token.strip())
            has_api_key = bool(self.twilio_api_key_sid.strip() and self.twilio_api_key_secret.strip())
            if not has_auth_token and not has_api_key:
                raise ValueError(
                    "Configure TWILIO_AUTH_TOKEN or both TWILIO_API_KEY_SID and TWILIO_API_KEY_SECRET"
                )
            if bool(self.twilio_api_key_sid.strip()) != bool(self.twilio_api_key_secret.strip()):
                raise ValueError("TWILIO_API_KEY_SID and TWILIO_API_KEY_SECRET must be configured together")
            if not self.twilio_webhook_base_url.strip().lower().startswith("https://"):
                raise ValueError("TWILIO_WEBHOOK_BASE_URL must use HTTPS")
        if provider == "webhook" and not self.crm_webhook_url.strip():
            raise ValueError("CRM_WEBHOOK_URL is required when CRM_PROVIDER=webhook")
        if self.agent_dashboard_token and len(self.agent_dashboard_token) < 24:
            raise ValueError("AGENT_DASHBOARD_TOKEN must contain at least 24 characters")
        if self.crm_webhook_url and provider in {"auto", "webhook"}:
            parsed = urlparse(self.crm_webhook_url)
            local_hosts = {"localhost", "127.0.0.1", "::1"}
            if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in local_hosts):
                raise ValueError("CRM_WEBHOOK_URL must use HTTPS (HTTP is allowed only for localhost)")
            if not parsed.hostname:
                raise ValueError("CRM_WEBHOOK_URL must be an absolute URL")
            if parsed.username or parsed.password:
                raise ValueError("CRM_WEBHOOK_URL must not contain embedded credentials")
            if parsed.query or parsed.fragment:
                raise ValueError("CRM_WEBHOOK_URL must not contain a query string or fragment")
            if len(self.crm_webhook_bearer_token) < 24:
                raise ValueError("CRM_WEBHOOK_BEARER_TOKEN is required when CRM_WEBHOOK_URL is configured")
        return self

    @property
    def resolved_crm_provider(self) -> str:
        """Resolve backward-compatible auto selection for CRM configuration."""

        provider = self.crm_provider.strip().lower()
        if provider != "auto":
            return provider
        if self.hubspot_private_app_token.strip():
            return "hubspot"
        if self.crm_webhook_url.strip():
            return "webhook"
        return "none"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
