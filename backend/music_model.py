"""Local MusiCNN embeddings and independent MTG music classifiers.

Model output scores are not calibrated confidence, nor clinical emotion labels.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Literal

import librosa
import numpy as np
import onnxruntime as ort
from pydantic import Field
from backend.schemas import StrictModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/musicnn"
MANIFEST_PATH = ROOT / "data/audio-models.manifest.json"
MOODS = ("happy", "sad", "relaxed", "aggressive")
GENRE_NAMES = {"folkcountr": "folk / country", "funksoulrnb": "funk / soul / R&B", "raphiphop": "rap / hip-hop"}
INFERENCE_LOCK = Lock()


class LabelScore(StrictModel):
    label: str
    score: float = Field(ge=0, le=1)


class MusicPrediction(StrictModel):
    status: Literal["ok", "unavailable", "skipped_silence", "skipped_short", "error"]
    model: str = "MTG MSD MusiCNN + mood heads + genre Dortmund"
    mood: str = "not estimated"
    genre: str = "not estimated"
    mood_scores: list[LabelScore] = Field(default_factory=list)
    genre_scores: list[LabelScore] = Field(default_factory=list)
    patches_analyzed: int = 0
    warnings: list[str] = Field(default_factory=list)


def model_status() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {"installed": all((MODEL_DIR / item["filename"]).is_file() for item in manifest["files"]),
            "model": "MTG MSD MusiCNN", "license": manifest["license"],
            "minimum_seconds": 6, "max_patches": 10}


@lru_cache(maxsize=1)
def load_models():
    """Verify pinned bytes before loading CPU sessions; never auto-download."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    for item in manifest["files"]:
        path = MODEL_DIR / item["filename"]
        if not path.exists():
            raise FileNotFoundError("Run scripts/download_audio_models.py first.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Model asset checksum mismatch.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    sessions, labels = {}, {}
    for name in ["msd-musicnn-1", *[f"mood_{m}-msd-musicnn-1" for m in MOODS], "genre_dortmund-msd-musicnn-1"]:
        sessions[name] = ort.InferenceSession(str(MODEL_DIR / f"{name}.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        labels[name] = json.loads((MODEL_DIR / f"{name}.json").read_text())["classes"]
    return sessions, labels


def musicnn_patches(y: np.ndarray, sr: int) -> np.ndarray:
    """Port the published 16 kHz/512/256/96-band MusiCNN input recipe.

    Symmetric unnormalized Hann, Slaney unit-area power mel bands and
    log10(1 + 10000 * mel). Sample at most ten 187-frame patches over the clip.
    """
    audio = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    spectrum = np.abs(librosa.stft(audio, n_fft=512, hop_length=256, window=np.hanning(512), center=True, pad_mode="constant")) ** 2
    filters = librosa.filters.mel(sr=16000, n_fft=512, n_mels=96, fmin=0, fmax=8000, htk=False, norm="slaney")
    mel = np.log10(1 + 10000 * (filters @ spectrum)).T.astype(np.float32)
    if len(mel) < 187:
        raise ValueError("Too few mel frames for the model.")
    starts = np.arange(0, len(mel) - 186, 187)
    if len(starts) > 10:
        starts = starts[np.linspace(0, len(starts) - 1, 10, dtype=int)]
    return np.stack([mel[start:start + 187] for start in starts])


def summarize_scores(moods: dict[str, float], genre_labels: list[str], genres: np.ndarray, patches: int) -> MusicPrediction:
    """Respect head label order; mood heads are independent binary classifiers."""
    mood_scores = [LabelScore(label=label, score=float(score)) for label, score in moods.items()]
    # Explicit binary decision threshold, not a claim of calibrated confidence.
    positives = [item.label for item in mood_scores if item.score >= 0.5]
    genre_scores = sorted([LabelScore(label=GENRE_NAMES.get(label, label), score=float(score)) for label, score in zip(genre_labels, genres)], key=lambda item: item.score, reverse=True)
    return MusicPrediction(status="ok", mood=", ".join(positives) if positives else "no positive mood cue",
        genre=genre_scores[0].label, mood_scores=mood_scores, genre_scores=genre_scores,
        patches_analyzed=patches, warnings=[
            "Experimental music estimates: model scores are not calibrated confidence or ground truth.",
            "Mood attributes are independent and may overlap; 0.5 is the binary decision threshold.",
            "Genre is selected from nine training classes; speech, noise and missing genres can still receive a label.",
        ])


def predict_music(y: np.ndarray, sr: int, silent: bool = False) -> MusicPrediction:
    if silent:
        return MusicPrediction(status="skipped_silence", warnings=["Mood/genre estimation skipped for silence."])
    if len(y) / sr < 6:
        return MusicPrediction(status="skipped_short", warnings=["Use at least 6 seconds, preferably 15–30 seconds, for neural music estimation."])
    try:
        with INFERENCE_LOCK:
            sessions, labels = load_models()
            patches = musicnn_patches(y, sr)
            encoder = sessions["msd-musicnn-1"]
            # One patch at a time keeps peak memory bounded on student laptops.
            embeddings = np.concatenate([encoder.run(["embeddings"], {"melspectrogram": patch[None]})[0] for patch in patches])
            moods = {}
            for mood in MOODS:
                name = f"mood_{mood}-msd-musicnn-1"
                scores = sessions[name].run(["activations"], {"embeddings": embeddings})[0].mean(axis=0)
                moods[mood] = float(scores[labels[name].index(mood)])
            name = "genre_dortmund-msd-musicnn-1"
            genres = sessions[name].run(["activations"], {"embeddings": embeddings})[0].mean(axis=0)
            return summarize_scores(moods, labels[name], genres, len(patches))
    except FileNotFoundError:
        return MusicPrediction(status="unavailable", warnings=["Local model files are missing. Run: .venv\\Scripts\\python.exe scripts/download_audio_models.py"])
    except (ValueError, RuntimeError, ort.capi.onnxruntime_pybind11_state.Fail, ort.capi.onnxruntime_pybind11_state.InvalidArgument):
        return MusicPrediction(status="error", warnings=["Local model inference failed validation or could not run. Signal measurements remain available; check model files and dependencies."])
