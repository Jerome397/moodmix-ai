"""Offline tests for preprocessing, label semantics, graceful failure and routing."""
from io import BytesIO
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from backend import music_model as model
from backend.main import app


def test_silence_and_short_input_skip_model_loading(monkeypatch):
    monkeypatch.setattr(model, "load_models", lambda: pytest.fail("Model must not load"))
    assert model.predict_music(np.zeros(16000 * 10), 16000, silent=True).status == "skipped_silence"
    assert model.predict_music(np.ones(16000 * 3), 16000).status == "skipped_short"


def test_missing_or_invalid_weights_do_not_break_signal_features(monkeypatch):
    def missing():
        raise FileNotFoundError()
    monkeypatch.setattr(model, "load_models", missing)
    assert model.predict_music(np.ones(16000 * 6), 16000).status == "unavailable"
    def invalid():
        raise ValueError("bad asset")
    monkeypatch.setattr(model, "load_models", invalid)
    assert model.predict_music(np.ones(16000 * 6), 16000).status == "error"


def test_patch_shape_dtype_and_limit():
    y = .1 * np.sin(2 * np.pi * 440 * np.arange(16000 * 35) / 16000)
    patches = model.musicnn_patches(y.astype(np.float32), 16000)
    assert patches.shape == (10, 187, 96) and patches.dtype == np.float32
    assert np.isfinite(patches).all() and patches.min() >= 0
    assert model.musicnn_patches(np.zeros(16000 * 6, dtype=np.float32), 16000).max() == 0


def test_score_semantics_do_not_normalize_independent_moods():
    result = model.summarize_scores({"happy": .8, "sad": .2, "relaxed": .7, "aggressive": .1}, ["jazz", "rock"], np.array([.6, .4]), 2)
    assert result.mood == "happy, relaxed" and result.genre == "jazz"
    assert sum(item.score for item in result.mood_scores) > 1
    assert "not calibrated" in result.warnings[0]


def test_positive_head_class_index_is_read_from_metadata(monkeypatch):
    class Encoder:
        def run(self, outputs, inputs):
            return [np.zeros((1, 200), dtype=np.float32)]
    class Head:
        def __init__(self, scores):
            self.scores = scores
        def run(self, outputs, inputs):
            return [np.tile(self.scores, (len(inputs["embeddings"]), 1))]
    sessions = {"msd-musicnn-1": Encoder()}
    labels = {}
    for mood in model.MOODS:
        name = f"mood_{mood}-msd-musicnn-1"
        sessions[name] = Head([.1, .9])
        labels[name] = ["negative", mood]
    sessions["genre_dortmund-msd-musicnn-1"] = Head([.7, .3])
    labels["genre_dortmund-msd-musicnn-1"] = ["jazz", "rock"]
    monkeypatch.setattr(model, "load_models", lambda: (sessions, labels))
    monkeypatch.setattr(model, "musicnn_patches", lambda *args: np.zeros((2, 187, 96), dtype=np.float32))
    result = model.predict_music(np.ones(16000 * 6), 16000)
    assert result.status == "ok" and all(item.score == pytest.approx(.9) for item in result.mood_scores)
    assert result.patches_analyzed == 2


def test_endpoint_model_option_handles_silence():
    buffer = BytesIO()
    sf.write(buffer, np.zeros(16000 * 6), 16000, format="WAV")
    response = TestClient(app).post("/audio/analyze", files={"file": ("silence.wav", buffer.getvalue(), "audio/wav")}, data={"include_models": "true"})
    assert response.status_code == 200
    assert response.json()["music_model"]["status"] == "skipped_silence"
