"""Local offline speech-to-text via whisper.cpp (E7 AI Interview Coach).

Audio never leaves the machine and is never persisted: FFmpeg validates and
normalises the upload to mono 16 kHz WAV off the event loop, whisper.cpp transcribes
it, and only the transcript text is returned. A single asyncio.Semaphore(1) protects
the one loaded model, since one whisper.cpp context does not support concurrent
inference. Loading is resilient: if the model file or the pywhispercpp binding is
unavailable, `load()` returns an adapter whose `available` is False so the app still
boots (app/main.py lifespan) and callers get `transcription_unavailable` at call time.
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
    # imported lazily so the app boots without pywhispercpp or the model file present,
    # and so a load failure is caught per-attempt by the caller.
    from pywhispercpp.model import Model

    return Model(model_path, n_threads=threads, print_realtime=False, print_progress=False)


class WhisperTranscriber:
    def __init__(
        self,
        model,
        *,
        max_bytes: int,
        max_seconds: int,
        ffmpeg_timeout_s: float,
        language: str = "en",
        available: bool = True,
    ) -> None:
        self._model = model
        self._max_bytes = max_bytes
        self._max_seconds = max_seconds
        self._ffmpeg_timeout_s = ffmpeg_timeout_s
        self._language = language
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
        language: str = "en",
    ) -> "WhisperTranscriber":
        try:
            model = _construct_model(model_path, threads)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "whisper model not loaded (%s) from %s; transcription disabled",
                type(exc).__name__, model_path,
            )
            return cls(
                None, max_bytes=max_bytes, max_seconds=max_seconds,
                ffmpeg_timeout_s=ffmpeg_timeout_s, language=language, available=False,
            )
        logger.info("whisper model loaded: %s", model_path)
        return cls(
            model, max_bytes=max_bytes, max_seconds=max_seconds,
            ffmpeg_timeout_s=ffmpeg_timeout_s, language=language, available=True,
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

                if self._language == "auto":
                    (language, _prob), _all_probs = self._model.auto_detect_language(str(wav_path))
                else:
                    language = self._language
                segments = self._model.transcribe(str(wav_path), language=language, translate=False)
                transcript = " ".join(
                    seg.text.strip() for seg in segments if seg.text.strip()
                )

                if not any(ch.isalnum() for ch in transcript):
                    raise TranscriptionError("no_speech_detected")

                return TranscriptionResult(
                    transcript=transcript, detected_language=language, duration_s=duration_s,
                )
            finally:
                src_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
