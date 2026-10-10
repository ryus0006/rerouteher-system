"""Local offline speech-to-text via onnx-asr/Parakeet (E7 AI Interview Coach).

Audio never leaves the machine and is never persisted: FFmpeg validates and
normalises the upload to mono 16 kHz WAV off the event loop, the Parakeet TDT 0.6B v2
ONNX model transcribes it, and only the transcript text is returned. A single
asyncio.Semaphore(1) still caps concurrency to 1, matched to the 1-2 vCPU deploy host
rather than any single-model-instance restriction (unlike whisper.cpp, onnxruntime
sessions tolerate concurrent calls; there just isn't spare CPU to make that useful
here). Loading is resilient: if the model files or the onnx-asr package are
unavailable, `load()` returns an adapter whose `available` is False so the app still
boots (app/main.py lifespan) and callers get `transcription_unavailable` at call time.
English-only model: there is no language to detect or force.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("rerouteher")

_ACCEPTED_CONTENT_TYPES = {
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "video/mp4": ".mp4",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
}
_ACCEPTED_SUFFIXES = {".webm", ".ogg", ".mp4", ".m4a", ".wav"}


class TranscriptionError(Exception):
    """code is one of the approved public error codes; never carries internals."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class AudioInput:
    content: bytes
    filename: str
    content_type: str


@dataclass
class TranscriptionResult:
    transcript: str
    detected_language: str
    duration_s: float


def _suffix_for(audio: AudioInput) -> str:
    mime = (audio.content_type or "").split(";")[0].strip().lower()
    suffix = _ACCEPTED_CONTENT_TYPES.get(mime)
    if suffix:
        return suffix
    ext = Path(audio.filename or "").suffix.lower()
    return ext if ext in _ACCEPTED_SUFFIXES else ""


def _run_ffmpeg(src_path: Path, wav_path: Path, timeout_s: float) -> None:
    """Validate and normalise to mono 16 kHz signed 16-bit PCM WAV.

    No stdin, error-only stderr. Never surfaces subprocess output to the caller.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y",
                "-i", str(src_path),
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                str(wav_path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise TranscriptionError("invalid_audio") from exc
    if proc.returncode != 0 or not wav_path.exists():
        raise TranscriptionError("invalid_audio")


def _wav_duration_seconds(wav_path: Path) -> float:
    """Read duration from the normalised WAV (16-bit mono PCM).

    Uses the wave module rather than fixed header offsets, because FFmpeg writes a
    LIST/INFO chunk before the data chunk.
    """
    try:
        with wave.open(str(wav_path), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
    except (wave.Error, EOFError) as exc:
        raise TranscriptionError("invalid_audio") from exc
    if rate <= 0:
        raise TranscriptionError("invalid_audio")
    return frames / rate


def _construct_model(model_path: str, threads: int):
    # imported lazily so the app boots without onnx-asr or the model files present,
    # and so a load failure is caught per-attempt by the caller.
    import onnxruntime as rt
    import onnx_asr

    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = threads

    return onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v2",
        path=model_path,
        quantization="int8",
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


class ParakeetTranscriber:
    def __init__(
        self,
        model,
        *,
        max_bytes: int,
        max_seconds: int,
        ffmpeg_timeout_s: float,
        available: bool = True,
    ) -> None:
        self._model = model
        self._max_bytes = max_bytes
        self._max_seconds = max_seconds
        self._ffmpeg_timeout_s = ffmpeg_timeout_s
        self._available = available
        self._lock = asyncio.Semaphore(1)

    @property
    def available(self) -> bool:
        return self._available

    @classmethod
    def load(
        cls,
        model_path: str,
        *,
        threads: int,
        max_bytes: int,
        max_seconds: int,
        ffmpeg_timeout_s: float,
    ) -> "ParakeetTranscriber":
        try:
            model = _construct_model(model_path, threads)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "parakeet model not loaded (%s) from %s; transcription disabled",
                type(exc).__name__, model_path,
            )
            return cls(
                None, max_bytes=max_bytes, max_seconds=max_seconds,
                ffmpeg_timeout_s=ffmpeg_timeout_s, available=False,
            )
        logger.info("parakeet model loaded: %s", model_path)
        return cls(
            model, max_bytes=max_bytes, max_seconds=max_seconds,
            ffmpeg_timeout_s=ffmpeg_timeout_s, available=True,
        )

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        if not self._available:
            raise TranscriptionError("transcription_unavailable")
        suffix = _suffix_for(audio)
        if not suffix:
            raise TranscriptionError("unsupported_audio_type")
        if not audio.content:
            raise TranscriptionError("invalid_audio")
        if len(audio.content) > self._max_bytes:
            raise TranscriptionError("recording_too_large")

        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, audio.content, suffix)

    def _transcribe_sync(self, content: bytes, suffix: str) -> TranscriptionResult:
        with tempfile.TemporaryDirectory(prefix="rerouteher-interview-") as tmp:
            tmp_dir = Path(tmp)
            src_path = tmp_dir / f"source{suffix}"
            wav_path = tmp_dir / "normalised.wav"
            try:
                src_path.write_bytes(content)
                _run_ffmpeg(src_path, wav_path, self._ffmpeg_timeout_s)
                duration_s = _wav_duration_seconds(wav_path)
                if duration_s > self._max_seconds:
                    raise TranscriptionError("recording_too_long")

                transcript = self._model.recognize(str(wav_path)).strip()

                if not any(ch.isalnum() for ch in transcript):
                    raise TranscriptionError("no_speech_detected")

                return TranscriptionResult(
                    transcript=transcript, detected_language="en", duration_s=duration_s,
                )
            finally:
                src_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
