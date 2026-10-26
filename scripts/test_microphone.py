"""Optional microphone check; reports levels and never records to disk."""

import sys


def main() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is optional. Install it with: pip install sounddevice", file=sys.stderr)
        raise SystemExit(2) from None

    import numpy as np

    sample_rate = 16_000
    print("Microphone check for 5 seconds; audio stays in memory. Press Ctrl+C to stop.")
    try:
        recording = sd.rec(int(5 * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        print(f"Could not access the microphone: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    rms = float(np.sqrt(np.mean(np.square(recording)))) if recording.size else 0.0
    print(f"Captured 5.0 s; RMS={rms:.5f}. No audio file was written.")


if __name__ == "__main__":
    main()
