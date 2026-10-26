"""Measure intent/RAG quality and optional reference-based ASR/noise robustness."""

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

from app.config import Settings
from app.monitoring.evaluation import evaluate_text_cases
from app.rag.retriever import RAGRetriever
from app.speech.asr import ASRService
from app.speech.evaluation import add_white_noise_snr, character_error_rate, word_error_rate


def _read_cases(path: Path) -> list[dict[str, object]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(case, dict) or not case.get("expected_intent"):
            raise ValueError(f"Each benchmark row needs expected_intent ({path}:{line_number})")
        if not (case.get("query") or (case.get("audio_path") and case.get("reference_transcript"))):
            raise ValueError(f"Each row needs query or audio_path plus reference_transcript ({path}:{line_number})")
        cases.append(case)
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return cases


def _read_pcm16_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"ASR benchmark audio must be mono PCM16: {path}")
        if wav_file.getframerate() != 16_000:
            raise ValueError(f"ASR benchmark audio must be 16 kHz PCM16: {path}")
        pcm = wav_file.readframes(wav_file.getnframes())
        if not pcm:
            raise ValueError(f"ASR benchmark audio is empty: {path}")
        return pcm, wav_file.getframerate()


def _metric_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(rows),
        "wer_macro": sum(float(row["wer"]) for row in rows) / len(rows),
        "cer_macro": sum(float(row["cer"]) for row in rows) / len(rows),
        "real_time_factor_mean": sum(float(row["real_time_factor"]) for row in rows) / len(rows),
        "cases": rows,
    }


async def _evaluate_asr(
    cases: list[dict[str, object]],
    settings: Settings,
    base_dir: Path,
    noise_conditions: list[float],
    include_transcripts: bool,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    audio_cases = [case for case in cases if case.get("audio_path") and case.get("reference_transcript")]
    if not audio_cases:
        return {"case_count": 0, "note": "No annotated audio cases; WER/CER are not inferred."}, {}

    service = ASRService(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        cpu_threads=settings.whisper_cpu_threads,
        beam_size=settings.whisper_beam_size,
        initial_prompt=settings.whisper_initial_prompt,
    )
    model_load_started = time.perf_counter()
    await asyncio.to_thread(service.load_model)
    model_load_seconds = time.perf_counter() - model_load_started
    loaded: list[tuple[dict[str, object], Path, bytes, int]] = []
    for case in audio_cases:
        audio_path = Path(str(case["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = base_dir / audio_path
        pcm, rate = _read_pcm16_wav(audio_path)
        loaded.append((case, audio_path, pcm, rate))

    conditions = [("clean", None), *((f"white_noise_{snr:g}dB", snr) for snr in noise_conditions)]
    reports: dict[str, list[dict[str, object]]] = {}
    retrieval_cases: dict[str, list[dict[str, object]]] = {}
    for condition, snr_db in conditions:
        rows: list[dict[str, object]] = []
        downstream: list[dict[str, object]] = []
        for index, (case, audio_path, pcm, sample_rate) in enumerate(loaded):
            tested_pcm = add_white_noise_snr(pcm, snr_db, seed=index) if snr_db is not None else pcm
            result = await service.transcribe(tested_pcm, sample_rate)
            reference = str(case["reference_transcript"])
            row: dict[str, object] = {
                "id": str(case.get("id", audio_path.name)),
                "duration_seconds": result.duration,
                "processing_seconds": result.processing_time,
                "real_time_factor": result.processing_time / max(result.duration, 1e-9),
                "wer": word_error_rate(reference, result.text),
                "cer": character_error_rate(reference, result.text),
            }
            if include_transcripts:
                row.update({"reference": reference, "hypothesis": result.text})
            rows.append(row)
            downstream.append({**case, "query": result.text})
        reports[condition] = rows
        retrieval_cases[condition] = downstream

    return {
        "conditions": {name: _metric_summary(rows) for name, rows in reports.items()},
        "asr_model": settings.whisper_model,
        "model_load_seconds": model_load_seconds,
        "notes": [
            "WER/CER are computed only against user-supplied reference transcripts.",
            "Noise variants are generated in memory with a fixed seed; original WAV files are not modified.",
            "Macro-average weights each utterance equally; real-time factor is ASR processing time divided "
            "by audio duration.",
            "Model loading/download is timed separately; per-sample ASR times are warm inference measurements.",
        ],
    }, retrieval_cases


def _make_retriever(settings: Settings, args: argparse.Namespace, strategy: str) -> RAGRetriever:
    return RAGRetriever(
        knowledge_base=args.knowledge_base.resolve(),
        vector_store_dir=settings.vector_store_dir / f"eval-{strategy}",
        model_name=settings.embedding_model,
        top_k=args.top_k,
        min_score=settings.rag_min_score,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        strategy=strategy,
        reranker_model=args.reranker_model or "",
    )


def _retrieval_benchmarks(
    cases: list[dict[str, object]], settings: Settings, args: argparse.Namespace, strategies: list[str]
) -> dict[str, dict[str, object]]:
    reports = {}
    for strategy in strategies:
        retriever = _make_retriever(settings, args, strategy)
        report = evaluate_text_cases(cases, retriever)
        report["requested_strategy"] = strategy
        report["actual_modes"] = sorted({str(row["retrieval_mode"]) for row in report["cases"]})
        if args.reranker_model:
            report["reranker_model"] = args.reranker_model
        reports[strategy] = report
    return reports


def _run(args: argparse.Namespace) -> dict[str, object]:
    settings = Settings()
    dataset_path = args.dataset.resolve()
    cases = _read_cases(dataset_path)
    strategies = [item.strip().lower() for item in args.strategies.split(",") if item.strip()]
    if not strategies or any(item not in {"bm25", "dense", "hybrid"} for item in strategies):
        raise ValueError("--strategies must be a comma-separated subset of bm25,dense,hybrid")

    text_cases = [case for case in cases if case.get("query")]
    strategy_reports = _retrieval_benchmarks(text_cases, settings, args, strategies) if text_cases else {}
    audio_cases = [case for case in cases if case.get("audio_path") and case.get("reference_transcript")]
    if audio_cases:
        asr_report, recognition_by_condition = asyncio.run(
            _evaluate_asr(cases, settings, dataset_path.parent, args.noise_snr_db, args.include_transcripts)
        )
        audio_retrieval = {
            condition: _retrieval_benchmarks(rows, settings, args, strategies)
            for condition, rows in recognition_by_condition.items()
        }
    else:
        asr_report = {"case_count": 0, "note": "No annotated audio cases; WER/CER are not inferred."}
        audio_retrieval = {}

    primary = strategies[0]
    report: dict[str, object] = dict(strategy_reports.get(primary, {}))
    report.update(
        {
            "retrieval_comparison": strategy_reports,
            "asr": asr_report,
            "retrieval_from_recognized_audio": audio_retrieval,
            "dataset": str(dataset_path),
            "privacy_note": "Audio and transcripts are not written unless you explicitly request their inclusion "
            "in the report.",
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/cases.jsonl"))
    parser.add_argument("--knowledge-base", type=Path, default=Settings().knowledge_base_dir)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--strategies", default="bm25", help="Comma-separated bm25,dense,hybrid; dense may download weights"
    )
    parser.add_argument("--reranker-model", default="", help="Optional CrossEncoder name; may download weights")
    parser.add_argument(
        "--noise-snr-db", nargs="*", type=float, default=[], help="Optional in-memory noisy ASR variants, e.g. 20 10"
    )
    parser.add_argument(
        "--include-transcripts", action="store_true", help="Include reference/hypothesis text in the JSON report"
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    report = _run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
