"""Reranker settings defaults and model-id parsing."""
from app.config import Settings


def test_rerank_defaults_and_model_list():
    s = Settings()
    assert s.rerank_enabled is True
    assert s.rerank_top_k == 3
    assert s.rerank_candidate_pool == 15
    ids = s.rerank_model_id_list
    assert ids == ["models/ms-marco-MiniLM-L6-v2"]  # vendored local path, loaded offline
    assert all(i.strip() == i and i for i in ids)


def test_session_cookie_defaults():
    s = Settings()
    assert s.session_secret  # non-empty
    assert s.session_https_only is False
    assert s.session_same_site == "lax"


def test_interview_transcription_defaults():
    s = Settings()
    assert s.whisper_model_path == "models/whisper/ggml-base.en.bin"
    assert s.whisper_threads == 2
    assert s.interview_max_audio_bytes == 25 * 1024 * 1024
    assert s.interview_max_audio_seconds == 300
    assert s.interview_ffmpeg_timeout_s == 45.0
    assert s.interview_content_retention_days == 30


def test_job_source_defaults():
    s = Settings()
    assert s.job_search_location == "Malaysia"
    assert s.jooble_api_key == ""
    assert s.jooble_api_key_2 == ""
    assert s.jooble_base_url == "https://my.jooble.org/api"
    assert s.foundit_base_url == "https://www.foundit.my"
