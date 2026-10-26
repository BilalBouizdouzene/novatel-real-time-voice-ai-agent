"""Measure live WebSocket/session capacity against an already running backend."""

import argparse
import asyncio
import json
import statistics
import time
import wave
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile for small benchmark samples."""

    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def read_audio(path: Path) -> tuple[bytes, int]:
    """Validate the same mono PCM16/16 kHz format sent by the browser."""

    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != 16_000:
            raise ValueError("Load-test WAV must be mono, PCM16, 16 kHz")
        data = wav_file.readframes(wav_file.getnframes())
    if not data:
        raise ValueError("Load-test WAV contains no samples")
    return data, 16_000


async def _one_session(
    url: str,
    audio: bytes | None,
    silence_duration: float,
    end_silence_ms: int,
    timeout_seconds: float,
) -> dict[str, object]:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install the development dependency: pip install -e '.[dev]'") from exc

    started = time.perf_counter()
    queue: asyncio.Queue[dict[str, object] | Exception] = asyncio.Queue()
    session_ready = asyncio.Event()
    metrics: list[dict[str, object]] = []
    errors: list[str] = []

    async with websockets.connect(url, max_size=2**22, open_timeout=timeout_seconds) as socket:
        connected = time.perf_counter()

        async def consume() -> None:
            try:
                async for raw in socket:
                    if not isinstance(raw, str):
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "session":
                        session_ready.set()
                    if event.get("type") == "metrics":
                        metrics.append(event.get("latencies", {}))
                        await queue.put(event)
                    elif event.get("type") == "error":
                        errors.append(str(event.get("code", "unknown_error")))
                        await queue.put(event)
            except Exception as exc:
                await queue.put(exc)

        reader = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(session_ready.wait(), timeout_seconds)
            session_ready_at = time.perf_counter()
            await socket.send(json.dumps({"type": "start"}))
            if audio is None:
                frame = bytes(1024 * 2)
                for _ in range(max(1, round(silence_duration / 0.064))):
                    await socket.send(frame)
                    await asyncio.sleep(0.064)
            else:
                chunk_bytes = 1024 * 2  # 64 ms at mono PCM16/16 kHz.
                for offset in range(0, len(audio), chunk_bytes):
                    await socket.send(audio[offset : offset + chunk_bytes])
                    await asyncio.sleep(0.064)
                silence_frame = bytes(chunk_bytes)
                for _ in range(max(1, round(end_silence_ms / 64))):
                    await socket.send(silence_frame)
                    await asyncio.sleep(0.064)

                while not metrics and not errors:
                    item = await asyncio.wait_for(queue.get(), timeout_seconds)
                    if isinstance(item, Exception):
                        raise item
                    if item.get("type") == "error":
                        break
            await socket.send(json.dumps({"type": "stop"}))
            return {
                "connected_ms": (connected - started) * 1000,
                "session_ready_ms": (session_ready_at - started) * 1000,
                "completed_turns": len(metrics),
                "turn_total_ms": [float(item.get("total_ms", 0.0)) for item in metrics],
                "turn_metrics": metrics,
                "errors": errors,
            }
        finally:
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass


async def run_load(args: argparse.Namespace) -> dict[str, object]:
    audio = read_audio(args.audio_wav) if args.audio_wav else None
    started = time.perf_counter()
    sessions = await asyncio.gather(
        *(
            _one_session(args.url, audio, args.silence_seconds, args.end_silence_ms, args.timeout_seconds)
            for _ in range(args.concurrency)
        ),
        return_exceptions=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    failures = [str(item) for item in sessions if isinstance(item, Exception)]
    completed = [item for item in sessions if isinstance(item, dict)]
    connected = [float(item["connected_ms"]) for item in completed]
    ready = [float(item["session_ready_ms"]) for item in completed]
    turn_latencies = [latency for item in completed for latency in item["turn_total_ms"]]
    turn_metrics = [metric for item in completed for metric in item["turn_metrics"]]
    errors = [error for item in completed for error in item["errors"]]
    metric_names = (
        "vad_ms",
        "asr_ms",
        "retrieval_ms",
        "llm_ms",
        "tts_ms",
        "total_ms",
        "time_to_first_token_ms",
        "time_to_first_audio_ms",
    )
    component_metrics = {}
    for name in metric_names:
        values = [float(metric[name]) for metric in turn_metrics if metric.get(name) is not None]
        component_metrics[name] = {"p50": percentile(values, 0.50), "p95": percentile(values, 0.95)}
    return {
        "url": args.url,
        "concurrency": args.concurrency,
        "elapsed_ms": round(elapsed_ms, 2),
        "sessions_succeeded": len(completed),
        "sessions_failed": len(failures),
        "connection_ms": {"p50": percentile(connected, 0.50), "p95": percentile(connected, 0.95)},
        "session_ready_ms": {"p50": percentile(ready, 0.50), "p95": percentile(ready, 0.95)},
        "completed_turns": len(turn_latencies),
        "turn_total_ms": {
            "p50": percentile(turn_latencies, 0.50),
            "p95": percentile(turn_latencies, 0.95),
            "mean": statistics.fmean(turn_latencies) if turn_latencies else None,
        },
        "pipeline_metrics_ms": component_metrics,
        "errors": errors,
        "failures": failures,
        "audio_mode": "real_annotated_wav" if audio is not None else "silence_transport_only",
        "warning": "Measured against the running local server; audio mode may use CPU, cloud LLM and TTS resources.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/voice")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--silence-seconds", type=float, default=2.0, help="Transport/VAD-only mode")
    parser.add_argument("--audio-wav", type=Path, help="Optional mono PCM16 16 kHz audio to run real turns")
    parser.add_argument("--end-silence-ms", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.concurrency > 100
        or args.silence_seconds <= 0
        or args.end_silence_ms < 100
        or args.timeout_seconds <= 0
    ):
        parser.error("concurrency must be 1..100, times must be positive and end silence must be at least 100 ms")
    report = asyncio.run(run_load(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["sessions_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
