"""Optional WebRTC transport for bidirectional browser audio.

The existing WebSocket transport remains the deterministic fallback. This
module is deliberately lazy-imported so a normal local install does not need
native aiortc wheels just to serve the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
from fractions import Fraction

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.models.schemas import AgentState
from app.pipeline.state import ConversationSession
from app.pipeline.voice_agent import VoiceAgentPipeline
from app.speech.audio_utils import float32_to_pcm16_bytes, pcm16_bytes_to_float32, resample_audio, wav_bytes_to_pcm16

logger = logging.getLogger(__name__)


def register_webrtc_route(app: FastAPI) -> None:
    """Attach a non-trickle ICE WebRTC signaling endpoint when dependencies exist."""

    @app.websocket("/ws/webrtc")
    async def webrtc_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
            from av import AudioFrame, AudioResampler
        except ImportError:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "webrtc_unavailable",
                    "message": "WebRTC optionnel indisponible. Installez : pip install -e .[realtime]",
                }
            )
            await websocket.close(code=1013)
            return

        settings = app.state.settings
        session = ConversationSession(tenant_id=websocket.query_params.get("tenant_id", "default")[:80] or "default")
        state = {"active": False}
        send_lock = asyncio.Lock()
        pc = RTCPeerConnection()
        pipeline: VoiceAgentPipeline | None = None
        inbound_tasks: list[asyncio.Task[None]] = []

        class PCMOutputTrack(MediaStreamTrack):
            """A paced mono PCM16 track; silence keeps the peer connection alive."""

            kind = "audio"

            def __init__(self) -> None:
                super().__init__()
                self.rate = 48_000
                self.samples = 960  # 20 ms at the browser/WebRTC clock rate.
                self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=120)
                self._buffer = bytearray()
                self._pts = 0

            async def push(self, pcm: bytes) -> None:
                for offset in range(0, len(pcm), self.samples * 2):
                    frame = pcm[offset : offset + self.samples * 2]
                    if not frame:
                        continue
                    try:
                        self._queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        logger.warning("WebRTC TTS queue full; dropping oldest audio frame")
                        try:
                            self._queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self._queue.put_nowait(frame)

            async def recv(self):
                while len(self._buffer) < self.samples * 2:
                    try:
                        self._buffer.extend(await asyncio.wait_for(self._queue.get(), timeout=0.02))
                    except TimeoutError:
                        break
                data = bytes(self._buffer[: self.samples * 2])
                del self._buffer[: self.samples * 2]
                data += b"\x00" * (self.samples * 2 - len(data))
                frame = AudioFrame(format="s16", layout="mono", samples=self.samples)
                frame.planes[0].update(data)
                frame.sample_rate = self.rate
                frame.pts = self._pts
                frame.time_base = Fraction(1, self.rate)
                self._pts += self.samples
                await asyncio.sleep(self.samples / self.rate)
                return frame

        output_track = PCMOutputTrack()
        pc.addTrack(output_track)

        async def send_event(event: dict[str, object]) -> None:
            async with send_lock:
                await websocket.send_json(event)

        async def send_audio(audio: bytes) -> None:
            pcm, _ = await asyncio.to_thread(wav_bytes_to_pcm16, audio, output_track.rate)
            await output_track.push(pcm)

        async def send_handoff_audio(audio: bytes) -> None:
            # The advisor inbox currently sends the configured WebSocket PCM
            # rate; adapt it to WebRTC's 48 kHz clock before queueing it.
            pcm = await asyncio.to_thread(
                lambda: float32_to_pcm16_bytes(
                    resample_audio(pcm16_bytes_to_float32(audio), settings.tts_stream_sample_rate, output_track.rate)
                )
            )
            await output_track.push(pcm)

        async def consume_audio(track) -> None:
            resampler = AudioResampler(format="s16", layout="mono", rate=16_000)
            while True:
                frame = await track.recv()
                converted = resampler.resample(frame)
                if not isinstance(converted, list):
                    converted = [converted]
                if state["active"] and pipeline is not None:
                    for item in converted:
                        await pipeline.ingest_audio(bytes(item.planes[0]))

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                inbound_tasks.append(asyncio.create_task(consume_audio(track)))

        try:
            offer = await websocket.receive_json()
            if not isinstance(offer, dict) or offer.get("type") != "offer":
                await send_event({"type": "error", "code": "invalid_offer", "message": "WebRTC offer attendu."})
                return
            vad = await asyncio.to_thread(app.state.vad_factory)
            pipeline = VoiceAgentPipeline(
                session=session,
                vad=vad,
                asr=app.state.asr,
                retriever=app.state.retriever,
                llm=app.state.llm,
                tts=app.state.tts,
                send_event=send_event,
                send_audio=send_audio,
                max_history_messages=settings.conversation_history_messages,
                demo_ticket_service=app.state.demo_ticket_service,
                support_case_service=app.state.support_case_service,
                crm_client=app.state.crm_client,
                handoff_hub=app.state.handoff_hub,
                handoff_enabled=bool(settings.agent_dashboard_token),
                output_sample_rate=output_track.rate,
            )
            await app.state.handoff_hub.register_customer(session.session_id, send_event, send_handoff_audio)
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=str(offer.get("sdp", "")), type=str(offer.get("sdp_type", "offer")))
            )
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            for _ in range(100):
                if pc.iceGatheringState == "complete":
                    break
                await asyncio.sleep(0.01)
            await send_event(
                {
                    "type": "answer",
                    "sdp": pc.localDescription.sdp,
                    "sdp_type": pc.localDescription.type,
                }
            )
            await send_event({"type": "session", "session_id": session.session_id, "state": session.state.value})
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw_text = message.get("text")
                if raw_text is None:
                    continue
                command = json.loads(raw_text)
                if not isinstance(command, dict):
                    continue
                kind = command.get("type")
                if kind == "start":
                    state["active"] = True
                    session.state = AgentState.LISTENING
                    await send_event({"type": "state", "state": session.state.value})
                elif kind == "stop":
                    state["active"] = False
                    if pipeline is not None:
                        await pipeline.close()
                    await send_event({"type": "state", "state": AgentState.IDLE.value})
                    break
                elif kind == "clear" and pipeline is not None:
                    await pipeline.clear_conversation()
                elif kind == "ping":
                    await send_event({"type": "pong"})
                else:
                    await send_event({"type": "error", "code": "unknown_command", "message": "Commande inconnue."})
        except WebSocketDisconnect:
            logger.info("WebRTC disconnected session=%s", session.session_id)
        except Exception:
            logger.exception("WebRTC handler failed session=%s", session.session_id)
            try:
                await send_event(
                    {"type": "error", "code": "webrtc_error", "message": "La session WebRTC a été interrompue."}
                )
            except Exception:
                logger.debug("Could not report WebRTC error", exc_info=True)
        finally:
            for task in inbound_tasks:
                task.cancel()
            await asyncio.gather(*inbound_tasks, return_exceptions=True)
            if pipeline is not None:
                await pipeline.close()
            await app.state.handoff_hub.unregister_customer(session.session_id)
            await pc.close()
