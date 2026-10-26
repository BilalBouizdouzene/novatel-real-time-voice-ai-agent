"""Small HTTP endpoints for health, runtime settings and the static client."""

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Report process health without loading large speech or embedding models."""

    settings = request.app.state.settings
    asr = request.app.state.asr
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "speech_models": "loaded_on_demand",
        "whisper_model": settings.whisper_model,
        "whisper_model_loaded": bool(getattr(asr, "is_model_loaded", False)),
    }


@router.get("/api/config")
async def public_config(request: Request) -> dict[str, object]:
    """Return only client-facing settings; never expose API credentials."""

    settings = request.app.state.settings
    webrtc_available = bool(settings.webrtc_enabled)
    try:
        import aiortc  # noqa: F401
    except ImportError:
        webrtc_available = False
    return {
        "sample_rate": 16_000,
        "websocket_path": "/ws/voice",
        "webrtc_path": "/ws/webrtc",
        "voice_transport": settings.voice_transport,
        "webrtc_available": webrtc_available,
        "tts_sample_rate": settings.tts_stream_sample_rate,
        "tts_frame_ms": settings.tts_audio_frame_ms,
    }


@router.get("/", include_in_schema=False)
async def index(request: Request) -> FileResponse:
    """Serve the dependency-free browser client."""

    return FileResponse(request.app.state.settings.frontend_dir / "index.html")
