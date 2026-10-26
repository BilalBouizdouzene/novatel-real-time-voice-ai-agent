from app.config import Settings
from app.integrations.twilio_voice import TwilioVoiceClient
from app.speech.streaming import iter_pcm_frames


def test_pcm_framing_preserves_audio_and_uses_configured_duration() -> None:
    audio = bytes(range(256)) * 30
    frames = list(iter_pcm_frames(audio, sample_rate=24_000, frame_ms=40))

    assert b"".join(frames) == audio
    assert len(frames[0]) == 1_920  # 40 ms of mono PCM16 at 24 kHz.


def test_twilio_client_is_disabled_by_default_and_escapes_case_in_twiml() -> None:
    client = TwilioVoiceClient(
        account_sid="AC123",
        auth_token="secret",
        from_number="+10000000000",
        advisor_number="+33123456789",
        webhook_base_url="https://demo.example",
    )

    assert client.configured is False
    assert "&lt;unsafe&gt;" in client.twiml_for_handoff(case_id="<unsafe>")


def test_twilio_client_accepts_api_key_credentials() -> None:
    client = TwilioVoiceClient(
        account_sid="AC123",
        auth_token="",
        api_key_sid="SK123",
        api_key_secret="secret",
        from_number="+10000000000",
        advisor_number="+33123456789",
        webhook_base_url="https://demo.example",
        enabled=True,
    )

    assert client.configured is True


def test_transport_settings_have_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.voice_transport == "auto"
    assert settings.tts_stream_sample_rate == 24_000
    assert settings.tts_audio_frame_ms == 40
