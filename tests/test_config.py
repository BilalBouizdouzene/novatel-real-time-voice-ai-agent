import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings


def test_default_application_isolated_to_novatel_corpus_and_index() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "NovaTel Real-Time Voice AI Agent"
    assert settings.knowledge_base_dir == PROJECT_ROOT / "knowledge_base" / "novatel"
    assert settings.vector_store_dir == PROJECT_ROOT / "data" / "novatel_faiss"
    assert settings.knowledge_base_dir.is_dir()


def test_vad_defaults_accept_short_greetings_without_long_end_of_turn_delay() -> None:
    settings = Settings(_env_file=None)

    assert settings.vad_threshold == 0.45
    assert settings.vad_min_speech_ms == 200
    assert settings.vad_min_silence_ms == 650


def test_external_crm_webhook_requires_https_and_bearer_secret() -> None:
    with pytest.raises(ValidationError, match="CRM_WEBHOOK_BEARER_TOKEN is required"):
        Settings(_env_file=None, crm_webhook_url="https://crm.example.test/hook")

    with pytest.raises(ValidationError, match="must use HTTPS"):
        Settings(
            _env_file=None,
            crm_webhook_url="http://crm.example.test/hook",
            crm_webhook_bearer_token="crm-secret-long-enough-for-test",
        )

    settings = Settings(
        _env_file=None,
        crm_webhook_url="http://localhost:9000/hook",
        crm_webhook_bearer_token="crm-secret-long-enough-for-test",
    )
    assert settings.crm_webhook_url.startswith("http://localhost")

    with pytest.raises(ValidationError, match="query string or fragment"):
        Settings(
            _env_file=None,
            crm_webhook_url="https://crm.example.test/hook?token=secret",
            crm_webhook_bearer_token="crm-secret-long-enough-for-test",
        )


def test_hubspot_crm_provider_requires_private_app_token_and_auto_selects_provider() -> None:
    with pytest.raises(ValidationError, match="HUBSPOT_PRIVATE_APP_TOKEN is required"):
        Settings(_env_file=None, crm_provider="hubspot")

    settings = Settings(
        _env_file=None,
        crm_provider="hubspot",
        hubspot_private_app_token="private-app-token",
    )
    assert settings.resolved_crm_provider == "hubspot"
    assert Settings(_env_file=None).resolved_crm_provider == "none"
