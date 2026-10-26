"""Offline speech synthesis adapters for Piper and eSpeak NG."""

import asyncio
import base64
import logging
import shutil
import tempfile
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path

from app.models.schemas import AudioResult

logger = logging.getLogger(__name__)


class TTSFailure(RuntimeError):
    """Raised when the configured speech engine cannot synthesize audio."""


class TTSService(ABC):
    """Shared asynchronous TTS service contract."""

    @abstractmethod
    async def synthesize(self, text: str) -> AudioResult:
        """Synthesize speech and return WAV bytes with measured metadata."""


class PiperTTSService(TTSService):
    """Run the Piper CLI with an explicitly configured local voice model."""

    def __init__(self, executable: str, model_path: str, config_path: str = "") -> None:
        self.executable = executable
        self.model_path = Path(model_path)
        self.config_path = config_path

    async def synthesize(self, text: str) -> AudioResult:
        if not text.strip():
            raise TTSFailure("Cannot synthesize empty text")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="voice-agent-tts-") as temp_dir:
            output_path = Path(temp_dir) / "response.wav"
            command = [self.executable, "--model", str(self.model_path), "--output_file", str(output_path)]
            if self.config_path:
                command.extend(["--config", self.config_path])
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await process.communicate(text.encode("utf-8"))
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                raise
            if process.returncode != 0 or not output_path.is_file():
                details = stderr.decode("utf-8", errors="replace")[-400:]
                raise TTSFailure(f"Piper synthesis failed: {details}")
            return _audio_result(output_path, time.perf_counter() - started, "piper")


class EspeakTTSService(TTSService):
    """Offline fallback using eSpeak NG; voice quality is intentionally simple."""

    def __init__(self, executable: str, voice: str = "fr") -> None:
        self.executable = executable
        self.voice = voice

    async def synthesize(self, text: str) -> AudioResult:
        if not text.strip():
            raise TTSFailure("Cannot synthesize empty text")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="voice-agent-tts-") as temp_dir:
            output_path = Path(temp_dir) / "response.wav"
            process = await asyncio.create_subprocess_exec(
                self.executable,
                "-v",
                self.voice,
                "-w",
                str(output_path),
                "--stdin",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await process.communicate(text.encode("utf-8"))
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                raise
            if process.returncode != 0 or not output_path.is_file():
                details = stderr.decode("utf-8", errors="replace")[-400:]
                raise TTSFailure(f"eSpeak synthesis failed: {details}")
            return _audio_result(output_path, time.perf_counter() - started, "espeak-ng")


class WindowsSystemSpeechTTSService(TTSService):
    """Synthesize a real WAV with a French voice installed in Windows SAPI."""

    def __init__(self, executable: str, voice_name: str = "") -> None:
        self.executable = executable
        self.voice_name = voice_name

    async def synthesize(self, text: str) -> AudioResult:
        if not text.strip():
            raise TTSFailure("Cannot synthesize empty text")
        started = time.perf_counter()
        text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        voice_b64 = base64.b64encode(self.voice_name.encode("utf-8")).decode("ascii")
        with tempfile.TemporaryDirectory(prefix="voice-agent-tts-") as temp_dir:
            output_path = Path(temp_dir) / "response.wav"
            output_b64 = base64.b64encode(str(output_path).encode("utf-8")).decode("ascii")
            script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$text = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{text_b64}'))
$voiceName = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{voice_b64}'))
$outputPath = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{output_b64}'))
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{
    $voices = @($synth.GetInstalledVoices() | Where-Object {{ $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'fr*' }})
    if ($voices.Count -eq 0) {{ throw 'No French Windows speech voice is installed.' }}
    $selected = $null
    if ($voiceName) {{
        $selected = $voices | Where-Object {{ $_.VoiceInfo.Name -eq $voiceName }} | Select-Object -First 1
    }}
    if (-not $selected) {{ $selected = $voices[0] }}
    $synth.SelectVoice($selected.VoiceInfo.Name)
    $synth.SetOutputToWaveFile($outputPath)
    $synth.Speak($text)
}} finally {{
    $synth.Dispose()
}}
"""
            process = await asyncio.create_subprocess_exec(
                self.executable,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                raise
            except TimeoutError as exc:
                process.kill()
                await process.wait()
                raise TTSFailure("Windows System.Speech timed out after 45 seconds") from exc
            if process.returncode != 0 or not output_path.is_file():
                details = stderr.decode("utf-8", errors="replace")[-400:]
                raise TTSFailure(f"Windows System.Speech failed: {details}")
            return _audio_result(output_path, time.perf_counter() - started, "windows-system-speech")


class UnavailableTTSService(TTSService):
    """Explicit failure path when no real synthesis engine is installed."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def synthesize(self, text: str) -> AudioResult:
        raise TTSFailure(self.reason)


def create_tts_service(settings: object) -> TTSService:
    """Select an installed engine without claiming unavailable engines work."""

    backend = settings.tts_backend.lower()  # type: ignore[attr-defined]
    piper_path = shutil.which(settings.piper_executable)  # type: ignore[attr-defined]
    model_path = settings.piper_voice_model  # type: ignore[attr-defined]
    espeak_path = shutil.which(settings.espeak_executable)  # type: ignore[attr-defined]
    powershell_path = shutil.which(settings.windows_powershell_executable)  # type: ignore[attr-defined]

    if backend == "windows":
        if powershell_path:
            logger.info("TTS backend selected: Windows System.Speech")
            return WindowsSystemSpeechTTSService(powershell_path, settings.tts_voice)  # type: ignore[attr-defined]
        return UnavailableTTSService("Windows System.Speech requires Windows PowerShell")

    if backend in {"auto", "piper"} and piper_path and model_path and Path(model_path).is_file():
        logger.info("TTS backend selected: Piper")
        return PiperTTSService(piper_path, model_path, settings.piper_voice_config)  # type: ignore[attr-defined]
    if backend == "piper":
        reason = "Piper was requested but its executable or configured voice model is missing"
        return UnavailableTTSService(reason)
    if backend in {"auto", "espeak"} and espeak_path:
        logger.info("TTS backend selected: eSpeak NG (offline fallback)")
        voice = settings.tts_voice or settings.tts_language  # type: ignore[attr-defined]
        return EspeakTTSService(espeak_path, voice)
    if backend == "auto" and powershell_path:
        logger.info("TTS backend selected: Windows System.Speech (French system voice)")
        return WindowsSystemSpeechTTSService(powershell_path, settings.tts_voice)  # type: ignore[attr-defined]
    if backend == "disabled":
        return UnavailableTTSService("TTS is disabled by configuration")
    if backend == "espeak":
        return UnavailableTTSService("eSpeak NG was requested but its executable is not installed")
    return UnavailableTTSService("No TTS engine found; install Piper with a voice model or install eSpeak NG")


def _audio_result(path: Path, processing_time: float, backend: str) -> AudioResult:
    audio = path.read_bytes()
    if not audio:
        raise TTSFailure(f"{backend} returned an empty audio file")
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            duration = wav_file.getnframes() / sample_rate
    except (wave.Error, ZeroDivisionError) as exc:
        raise TTSFailure(f"{backend} did not produce a valid WAV file") from exc
    return AudioResult(
        audio=audio,
        sample_rate=sample_rate,
        duration=duration,
        processing_time=processing_time,
        backend=backend,
    )
