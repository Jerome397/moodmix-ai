"""Audio-reference ranking, privacy and end-to-end pipeline checks."""
from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.audio_recommendation import resolve_context
from backend.catalog import load_catalog
from backend.main import app
from backend.recommendation import retrieve, baseline
from backend.schemas import AudioReference, PlaylistRequest


def reference(**updates):
    data = {"bpm": 160, "duration_seconds": 20, "is_silent": False,
            "neural_status": "ok", "mood_scores": [{"label": "happy", "score": .8}],
            "genre_scores": [{"label": "jazz", "score": .9}]}
    return AudioReference(**(data | updates))


def test_audio_tempo_changes_ranking_and_can_be_overridden():
    tracks = [track.model_copy(update={"genre": "jazz", "energy": .5, "mood": "unknown", "bpm": bpm})
              for track, bpm in zip(load_catalog()[:3], [70, 110, 160])]
    manual = PlaylistRequest(bpm=70)
    audio = manual.model_copy(update={"audio_reference": reference(), "use_audio_genre": False})
    assert retrieve(tracks, manual)[0].bpm == 70
    assert retrieve(tracks, audio)[0].bpm == 160
    assert retrieve(tracks, audio.model_copy(update={"use_audio_tempo": False}))[0].bpm == 70


def test_genre_cues_are_soft_and_explicit_filter_wins():
    base = load_catalog()[0]
    tracks = [base.model_copy(update={"track_id": genre, "genre": genre, "energy": .5, "bpm": 110}) for genre in ["rock", "jazz"]]
    request = PlaylistRequest(audio_reference=reference(), use_audio_tempo=False)
    assert retrieve(tracks, request)[0].genre == "jazz"
    assert {track.genre for track in retrieve(tracks, request.model_copy(update={"genre": "rock"}))} == {"rock"}


def test_unsupported_genre_is_not_silently_mapped():
    request = PlaylistRequest(audio_reference=reference(genre_scores=[{"label": "funk / soul / R&B", "score": .99}]))
    context = resolve_context(request, load_catalog())
    assert not context.genre_affinity
    assert any("no supported catalog mapping" in warning for warning in context.warnings)


def test_out_of_range_tempo_uses_manual_without_clamping():
    context = resolve_context(PlaylistRequest(bpm=95, audio_reference=reference(bpm=300)), load_catalog())
    assert context.target_bpm == 95 and "estimated BPM" not in context.compared_audio_features
    assert any("outside" in warning for warning in context.warnings)


@pytest.mark.parametrize("updates", [
    {"waveform": []}, {"rms_amplitude": .5},
    {"mood_scores": [{"label": "happy", "score": 2}]},
    {"genre_scores": [{"label": "jazz", "score": .5}, {"label": "jazz", "score": .4}]},
    {"neural_status": "error"},
])
def test_reference_rejects_unsupported_data_or_invalid_scores(updates):
    with pytest.raises(ValidationError):
        reference(**updates)


def test_silent_or_empty_audio_reference_rejected_by_api():
    client = TestClient(app)
    for ref in [reference(is_silent=True), reference(bpm=None, mood_scores=[], genre_scores=[])]:
        response = client.post("/playlist", json={"audio_reference": ref.model_dump()})
        assert response.status_code == 422


def test_neural_summary_reaches_llm_with_effective_tempo(monkeypatch):
    from backend import generation
    tracks = load_catalog()
    request = PlaylistRequest(mode="llm", audio_reference=reference(), bpm=80)
    captured = {}
    config = generation.Settings(_env_file=None, gemini_api_key="offline-test-key")
    monkeypatch.setattr(generation, "Settings", lambda: config)
    def planner(settings, payload, correction):
        captured.update(payload)
        from backend.schemas import Track
        candidates = [next(t for t in tracks if t.track_id == item["track_id"]) for item in payload["candidates"]]
        return baseline(candidates, PlaylistRequest.model_validate(payload["request"]))
    monkeypatch.setattr(generation, "gemini_plan", planner)
    response = TestClient(app).post("/playlist", json=request.model_dump())
    assert response.status_code == 200 and response.json()["method"] == "llm"
    assert captured["request"]["bpm"] == 160
    assert captured["request"]["audio_reference"]["mood_scores"][0]["label"] == "happy"
    assert "waveform" not in captured["request"]["audio_reference"]
    assert "RMS" in " ".join(response.json()["warnings"])


def test_upload_analysis_to_baseline_playlist():
    import librosa
    wave = librosa.clicks(times=np.arange(.5, 12, .5), sr=22050, length=22050 * 12)
    buffer = BytesIO()
    sf.write(buffer, wave, 22050, format="WAV")
    client = TestClient(app)
    analysis = client.post("/audio/analyze", files={"file": ("clicks.wav", buffer.getvalue(), "audio/wav")}).json()
    assert analysis["bpm"] is not None
    ref = AudioReference(bpm=analysis["bpm"], duration_seconds=analysis["duration_seconds"], is_silent=analysis["is_silent"])
    result = client.post("/playlist", json={"audio_reference": ref.model_dump(), "duration_minutes": 10}).json()
    assert result["method"] == "baseline"
    assert result["retrieval_context"]["audio_reference_attached"]
    assert result["retrieval_context"]["target_bpm"] == pytest.approx(analysis["bpm"])
    assert 0 < result["duration_seconds"] <= 600
    assert all(track["track_id"].startswith("fma-") for track in result["tracks"])
