"""Local whisper.cpp transcription adapter.

Whisper and FFmpeg are faked throughout: the real pywhispercpp/ffmpeg integration is
exercised separately by the marked `whisper_smoke` test only when the real model file
is present locally (never in the default suite).
"""
import asyncio
import wave

import pytest

import app.services.transcription as mod
from app.services.transcription import (
    AudioInput,
    TranscriptionError,
    TranscriptionResult,
    WhisperTranscriber,
)

def _wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    import io

    buf = io.BytesIO()
    n_frames = int(seconds * sample_rate)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


class FakeModel:
    def __init__(self, segments_text: str = "I led a small team and improved the process.",
                 language: str = "en"):
        self._segments_text = segments_text
        self._language = language
        self.transcribe_calls: list[dict] = []
        self.detect_calls: list[str] = []

    def auto_detect_language(self, media):
        self.detect_calls.append(media)
        return (self._language, 0.95), {self._language: 0.95}

    def transcribe(self, media, **params):
        self.transcribe_calls.append({"media": media, **params})

        class _Seg:
            def __init__(self, text):
                self.text = text

        return [_Seg(self._segments_text)]


def _make_transcriber(model=None, *, available=True, max_bytes=25 * 1024 * 1024,
                       max_seconds=300, ffmpeg_timeout_s=45.0, language="en"):
    return WhisperTranscriber(
        model, max_bytes=max_bytes, max_seconds=max_seconds,
        ffmpeg_timeout_s=ffmpeg_timeout_s, language=language, available=available,
    )


def _stub_ffmpeg(monkeypatch, duration_s: float = 3.0, fail: bool = False):
    def fake_run_ffmpeg(src_path, wav_path, timeout_s):
        if fail:
            raise TranscriptionError("invalid_audio")
        wav_path.write_bytes(_wav_bytes(duration_s))

    monkeypatch.setattr(mod, "_run_ffmpeg", fake_run_ffmpeg)


async def test_accepted_mime_and_extension_pairs(monkeypatch):
    model = FakeModel()
    transcriber = _make_transcriber(model)
    _stub_ffmpeg(monkeypatch, duration_s=2.0)

    for content_type, filename in [
        ("audio/webm", "answer.webm"),
        ("audio/ogg", "answer.ogg"),
        ("video/mp4", "answer.mp4"),
        ("audio/mp4", "answer.m4a"),
        ("audio/wav", "answer.wav"),
    ]:
        result = await transcriber.transcribe(AudioInput(b"fake-bytes", filename, content_type))
        assert isinstance(result, TranscriptionResult)
        assert result.transcript


async def test_unsupported_media_type_rejected(monkeypatch):
    transcriber = _make_transcriber(FakeModel())
    with pytest.raises(TranscriptionError) as exc:
        await transcriber.transcribe(AudioInput(b"data", "answer.txt", "text/plain"))
    assert exc.value.code == "unsupported_audio_type"


async def test_oversized_upload_rejected_before_processing(monkeypatch):
    model = FakeModel()
    transcriber = _make_transcriber(model, max_bytes=10)
    with pytest.raises(TranscriptionError) as exc:
        await transcriber.transcribe(AudioInput(b"x" * 11, "answer.wav", "audio/wav"))
    assert exc.value.code == "recording_too_large"
    assert model.detect_calls == []  # rejected before touching the model


async def test_malformed_audio_rejected(monkeypatch):
    def fake_run_ffmpeg(src_path, wav_path, timeout_s):
        raise TranscriptionError("invalid_audio")

    monkeypatch.setattr(mod, "_run_ffmpeg", fake_run_ffmpeg)
    transcriber = _make_transcriber(FakeModel())
    with pytest.raises(TranscriptionError) as exc:
        await transcriber.transcribe(AudioInput(b"not-real-audio", "answer.wav", "audio/wav"))
    assert exc.value.code == "invalid_audio"


async def test_too_long_recording_rejected(monkeypatch):
    _stub_ffmpeg(monkeypatch, duration_s=301.0)
    transcriber = _make_transcriber(FakeModel(), max_seconds=300)
    with pytest.raises(TranscriptionError) as exc:
        await transcriber.transcribe(AudioInput(b"data", "answer.wav", "audio/wav"))
    assert exc.value.code == "recording_too_long"


async def test_no_speech_detected_rejected(monkeypatch):
    _stub_ffmpeg(monkeypatch, duration_s=2.0)
    model = FakeModel(segments_text="   ")
    transcriber = _make_transcriber(model)
    with pytest.raises(TranscriptionError) as exc:
        await transcriber.transcribe(AudioInput(b"data", "answer.wav", "audio/wav"))
    assert exc.value.code == "no_speech_detected"


async def test_model_unavailable_raises_service_error(monkeypatch):
    transcriber = _make_transcriber(None, available=False)
    with pytest.raises(TranscriptionError) as exc:
        await transcriber.transcribe(AudioInput(b"data", "answer.wav", "audio/wav"))
    assert exc.value.code == "transcription_unavailable"


async def test_forces_configured_language_without_detection(monkeypatch):
    _stub_ffmpeg(monkeypatch, duration_s=2.0)
    model = FakeModel(language="ms")
    transcriber = _make_transcriber(model, language="en")
    result = await transcriber.transcribe(AudioInput(b"data", "answer.wav", "audio/wav"))

    assert result.detected_language == "en"
    assert len(model.detect_calls) == 0
    assert model.transcribe_calls[0]["language"] == "en"
    assert model.transcribe_calls[0]["translate"] is False


async def test_auto_language_uses_detection(monkeypatch):
    _stub_ffmpeg(monkeypatch, duration_s=2.0)
    model = FakeModel(language="ms")
    transcriber = _make_transcriber(model, language="auto")
    result = await transcriber.transcribe(AudioInput(b"data", "answer.wav", "audio/wav"))

    assert result.detected_language == "ms"
    assert len(model.detect_calls) == 1
    assert model.transcribe_calls[0]["language"] == "ms"
    assert model.transcribe_calls[0]["translate"] is False


async def test_concurrent_requests_never_enter_the_model_simultaneously(monkeypatch):
    _stub_ffmpeg(monkeypatch, duration_s=1.0)
    in_flight = 0
    max_in_flight = 0

    class SlowModel(FakeModel):
        def transcribe(self, media, **params):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                import time

                time.sleep(0.05)
                return super().transcribe(media, **params)
            finally:
                in_flight -= 1

    transcriber = _make_transcriber(SlowModel())
    await asyncio.gather(
        transcriber.transcribe(AudioInput(b"a", "a.wav", "audio/wav")),
        transcriber.transcribe(AudioInput(b"b", "b.wav", "audio/wav")),
        transcriber.transcribe(AudioInput(b"c", "c.wav", "audio/wav")),
    )
    assert max_in_flight == 1


async def test_blocking_work_is_offloaded_from_the_event_loop(monkeypatch):
    _stub_ffmpeg(monkeypatch, duration_s=1.0)

    class BlockingModel(FakeModel):
        def transcribe(self, media, **params):
            import time

            time.sleep(0.1)
            return super().transcribe(media, **params)

    transcriber = _make_transcriber(BlockingModel())
    ticked = False

    async def ticker():
        nonlocal ticked
        await asyncio.sleep(0.02)
        ticked = True

    await asyncio.gather(
        transcriber.transcribe(AudioInput(b"a", "a.wav", "audio/wav")),
        ticker(),
    )
    assert ticked  # the event loop ran other work while the "blocking" call was in flight


async def test_temporary_files_are_removed_after_success(monkeypatch, tmp_path):
    seen_paths: list = []

    def fake_run_ffmpeg(src_path, wav_path, timeout_s):
        seen_paths.extend([src_path, wav_path])
        wav_path.write_bytes(_wav_bytes(1.0))

    monkeypatch.setattr(mod, "_run_ffmpeg", fake_run_ffmpeg)
    transcriber = _make_transcriber(FakeModel())
    await transcriber.transcribe(AudioInput(b"data", "answer.wav", "audio/wav"))

    for p in seen_paths:
        assert not p.exists()


async def test_temporary_files_are_removed_after_failure(monkeypatch):
    seen_paths: list = []

    def fake_run_ffmpeg(src_path, wav_path, timeout_s):
        seen_paths.extend([src_path, wav_path])
        wav_path.write_bytes(_wav_bytes(1.0))
        raise TranscriptionError("invalid_audio")

    monkeypatch.setattr(mod, "_run_ffmpeg", fake_run_ffmpeg)
    transcriber = _make_transcriber(FakeModel())
    with pytest.raises(TranscriptionError):
        await transcriber.transcribe(AudioInput(b"data", "answer.wav", "audio/wav"))

    for p in seen_paths:
        assert not p.exists()


def test_wav_duration_reads_header():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "a.wav"
        wav_path.write_bytes(_wav_bytes(2.5))
        assert abs(mod._wav_duration_seconds(wav_path) - 2.5) < 0.05


def test_wav_duration_skips_ffmpeg_list_chunk():
    import struct
    import tempfile
    from pathlib import Path

    # FFmpeg puts a LIST/INFO chunk between fmt and data; duration must still be right.
    raw = _wav_bytes(3.0)
    info = b"INFOISFT\x0e\x00\x00\x00Lavf61.7.100\x00\x00"
    list_chunk = b"LIST" + struct.pack("<I", len(info)) + info
    body = raw[12:36] + list_chunk + raw[36:]
    ffmpeg_style = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "a.wav"
        wav_path.write_bytes(ffmpeg_style)
        assert abs(mod._wav_duration_seconds(wav_path) - 3.0) < 0.05


def test_wav_duration_rejects_non_wav():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "a.wav"
        wav_path.write_bytes(b"not a wav file at all")
        with pytest.raises(TranscriptionError) as exc:
            mod._wav_duration_seconds(wav_path)
        assert exc.value.code == "invalid_audio"


def test_load_returns_unavailable_adapter_when_model_construction_fails(monkeypatch):
    def boom(model_path, threads):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(mod, "_construct_model", boom)
    transcriber = WhisperTranscriber.load(
        "models/whisper/ggml-base.en.bin", threads=4, max_bytes=1000, max_seconds=60,
        ffmpeg_timeout_s=10.0,
    )
    assert transcriber.available is False


def test_load_returns_available_adapter_when_model_constructs(monkeypatch):
    sentinel = FakeModel()
    monkeypatch.setattr(mod, "_construct_model", lambda model_path, threads: sentinel)
    transcriber = WhisperTranscriber.load(
        "models/whisper/ggml-base.en.bin", threads=4, max_bytes=1000, max_seconds=60,
        ffmpeg_timeout_s=10.0,
    )
    assert transcriber.available is True


@pytest.mark.whisper_smoke
def test_real_model_transcribes_a_silent_clip():
    from pathlib import Path

    from app.config import get_settings

    settings = get_settings()
    model_path = Path(settings.whisper_model_path)
    if not model_path.exists():
        pytest.skip(f"real whisper model not present at {model_path}")

    transcriber = WhisperTranscriber.load(
        str(model_path), threads=settings.whisper_threads,
        max_bytes=settings.interview_max_audio_bytes,
        max_seconds=settings.interview_max_audio_seconds,
        ffmpeg_timeout_s=settings.interview_ffmpeg_timeout_s,
    )
    assert transcriber.available is True

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(_wav_bytes(2.0))
        f.flush()

        async def _run():
            return await transcriber.transcribe(
                AudioInput(open(f.name, "rb").read(), "clip.wav", "audio/wav")
            )

        # Silence legitimately triggers no_speech_detected; either outcome proves the
        # model initialised and ran inference, which is all this smoke test asserts.
        try:
            result = asyncio.run(_run())
        except TranscriptionError as exc:
            assert exc.code == "no_speech_detected"
        else:
            assert isinstance(result.detected_language, str)
