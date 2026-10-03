"""Decode short recordings and transcribe them with a reusable Whisper pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from threading import Lock
from typing import Any

import numpy as np

MODEL_ID = "openai/whisper-tiny"
SAMPLE_RATE = 16_000
MAX_SECONDS = 30
MAX_BYTES = 20 * 1024 * 1024
LANGUAGES = {"Auto-detect": None, "English": "english", "German": "german"}


class TranscriptionError(Exception):
    """An error safe to display to a user."""


def decode_audio(audio_file: str | Path | None) -> np.ndarray:
    """Read at most 30.1 seconds as mono, 16 kHz float32 audio using FFmpeg."""
    if not audio_file:
        raise TranscriptionError("Upload an audio recording first.")
    path = Path(audio_file)
    if not path.is_file():
        raise TranscriptionError("The recording is missing. Please upload it again.")
    size = path.stat().st_size
    if size == 0:
        raise TranscriptionError("The uploaded file is empty.")
    if size > MAX_BYTES:
        raise TranscriptionError("Use a recording smaller than 20 MiB.")

    # Bound decoded audio as well as file size. Compressed files can be very long.
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-i", str(path.resolve()),
        "-t", str(MAX_SECONDS + 0.1), "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ]
    try:
        decoded = subprocess.run(command, capture_output=True, check=True, timeout=20)
    except FileNotFoundError as exc:
        raise TranscriptionError("FFmpeg is missing. Install it and restart the app.") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError("The audio took too long to read. Try a short WAV file.") from exc
    except subprocess.CalledProcessError as exc:
        raise TranscriptionError("Could not read that audio. Try a valid WAV or MP3 file.") from exc

    samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if samples.size == 0 or not np.isfinite(samples).all():
        raise TranscriptionError("The file does not contain readable audio.")
    if samples.size > MAX_SECONDS * SAMPLE_RATE:
        raise TranscriptionError("The recording is longer than 30 seconds. Trim it and try again.")
    if np.max(np.abs(samples)) < 0.0001:
        raise TranscriptionError("The recording is silent or too quiet. Try a louder recording.")
    return samples


def create_pipeline() -> Any:
    """Load on the requested device, or choose CUDA, MPS, then CPU."""
    import torch
    from transformers import pipeline

    device = os.environ.get("TRANSCRIPTION_DEVICE", "auto")
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise TranscriptionError("TRANSCRIPTION_DEVICE must be auto, cpu, mps, or cuda.")
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise TranscriptionError("The Apple GPU is unavailable. Set TRANSCRIPTION_DEVICE=cpu.")
    if device == "cuda" and not torch.cuda.is_available():
        raise TranscriptionError("CUDA is unavailable. Set TRANSCRIPTION_DEVICE=cpu.")
    return pipeline("automatic-speech-recognition", model=MODEL_ID, device=device)


class Transcriber:
    """Load once on the first valid request and serialize access to the model."""

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._lock = Lock()

    def transcribe(self, audio_file: str | Path | None, language: str = "Auto-detect") -> str:
        if language not in LANGUAGES:
            raise TranscriptionError("Choose Auto-detect, English, or German.")
        samples = decode_audio(audio_file)
        options = {"task": "transcribe", "language": LANGUAGES[language]}
        with self._lock:
            if self._pipeline is None:
                try:
                    self._pipeline = create_pipeline()
                except TranscriptionError:
                    raise
                except Exception as exc:
                    raise TranscriptionError(
                        "Could not load Whisper Tiny. The first transcription needs internet "
                        "access to download the model. Check your connection and available memory."
                    ) from exc
            try:
                result = self._pipeline(
                    {"raw": samples, "sampling_rate": SAMPLE_RATE},
                    generate_kwargs=options,
                )
            except Exception as exc:
                raise TranscriptionError(
                    "Transcription failed. Try a shorter recording. If it persists, "
                    "restart with TRANSCRIPTION_DEVICE=cpu."
                ) from exc
        text = result["text"].strip()
        if not text:
            raise TranscriptionError("No speech was transcribed. Try a clearer recording.")
        return text
