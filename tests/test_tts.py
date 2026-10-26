from types import SimpleNamespace

from app.speech.tts import WindowsSystemSpeechTTSService, create_tts_service


def test_auto_selects_windows_speech_when_other_engines_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.speech.tts.shutil.which",
        lambda executable: "powershell.exe" if executable == "powershell.exe" else None,
    )
    settings = SimpleNamespace(
        tts_backend="auto",
        piper_executable="piper",
        piper_voice_model="",
        piper_voice_config="",
        espeak_executable="espeak-ng",
        windows_powershell_executable="powershell.exe",
        tts_language="fr",
        tts_voice="",
    )

    service = create_tts_service(settings)

    assert isinstance(service, WindowsSystemSpeechTTSService)
