"""Bounded local audio decoding and documented signal descriptors.

Optional neural mood/genre inference runs locally; no external API calls occur.
"""
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydantic import Field
from backend.schemas import StrictModel
from backend.music_model import MusicPrediction, predict_music

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_DURATION_SECONDS = 3
MAX_DURATION_SECONDS = 120
ANALYSIS_SAMPLE_RATE = 22050


class AudioInputError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class AudioAnalysis(StrictModel):
    duration_seconds: float
    source_sample_rate: int
    analysis_sample_rate: int
    channels: int
    decoded_format: str
    bpm: float | None
    beat_count: int
    rms_amplitude: float = Field(ge=0)
    rms_dbfs: float | None
    peak_amplitude: float = Field(ge=0)
    spectral_centroid_hz: float = Field(ge=0)
    spectral_bandwidth_hz: float = Field(ge=0)
    spectral_rolloff_hz: float = Field(ge=0)
    zero_crossing_rate: float = Field(ge=0, le=1)
    is_silent: bool
    mood: str = "not estimated"
    genre: str = "not estimated"
    music_model: MusicPrediction | None = None
    warnings: list[str]
    waveform: list[dict[str, float]]
    rms_timeline: list[dict[str, float]]


def analyze_audio(content: bytes, filename: str, include_models: bool = False) -> AudioAnalysis:
    """Validate and analyze a 3–120 second MP3/WAV without saving input audio."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp3", ".wav"}:
        raise AudioInputError("Upload an MP3 or WAV file.", 415)
    if not content:
        raise AudioInputError("The uploaded file is empty.", 400)
    if len(content) > MAX_UPLOAD_BYTES:
        raise AudioInputError("Audio must be 20 MiB or smaller.", 413)
    try:
        with sf.SoundFile(BytesIO(content)) as source:
            allowed = {".mp3": {"MP3"}, ".wav": {"WAV", "WAVEX"}}
            if source.format not in allowed[suffix]:
                raise AudioInputError("The decoded format does not match the file extension.", 415)
            sr, channels = source.samplerate, source.channels
            duration = source.frames / sr
            if not 8000 <= sr <= 96000 or channels not in (1, 2):
                raise AudioInputError("Use mono/stereo audio with a sample rate from 8 to 96 kHz.")
            if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
                raise AudioInputError("Use an audio excerpt between 3 and 120 seconds.")
            # Bounded decode even when a damaged file's header is inconsistent.
            samples = source.read(frames=int(sr * MAX_DURATION_SECONDS) + 1, dtype="float32", always_2d=True)
    except (sf.SoundFileError, RuntimeError, OSError) as exc:
        raise AudioInputError("Audio could not be decoded. The file may be damaged or unsupported.") from exc
    if len(samples) < sr * MIN_DURATION_SECONDS or len(samples) > sr * MAX_DURATION_SECONDS:
        raise AudioInputError("Decoded audio must be between 3 and 120 seconds.")
    if not np.isfinite(samples).all():
        raise AudioInputError("Audio contains non-finite sample values.")
    warnings = []
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak = float(np.max(np.abs(samples)))
    if peak > 1:
        warnings.append("Samples exceed digital full scale; this can occur with floating-point or decoded compressed audio.")
    mono = np.mean(samples, axis=1)
    if channels == 2 and float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) < rms * 0.1:
        mono = samples[:, int(np.argmax(np.mean(samples.astype(np.float64) ** 2, axis=0)))]
        warnings.append("Stereo cancellation detected; spectral/tempo analysis uses the stronger channel.")
    y = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SAMPLE_RATE) if sr != ANALYSIS_SAMPLE_RATE else mono
    silent = rms <= 1e-6
    bpm, beat_count = None, 0
    centroid = bandwidth = rolloff = crossings = 0.0
    if silent:
        warnings.append("Silent or near-silent input: no meaningful tempo or spectral interpretation is available.")
    else:
        spectrum = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        centroid = float(np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=ANALYSIS_SAMPLE_RATE)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=spectrum, sr=ANALYSIS_SAMPLE_RATE)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=spectrum, sr=ANALYSIS_SAMPLE_RATE, roll_percent=0.85)))
        crossings = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(spectrum, ref=np.max), sr=ANALYSIS_SAMPLE_RATE)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=ANALYSIS_SAMPLE_RATE, hop_length=512)
        beat_count = int(len(beats))
        value = float(np.asarray(tempo).reshape(-1)[0])
        if beat_count >= 3 and np.isfinite(value) and value > 0:
            bpm = value
        else:
            warnings.append("Too few stable beat events to report a tempo.")
        warnings.append("BPM is an estimate and may follow half/double tempo, especially for weak or changing rhythms.")
    if duration < 10:
        warnings.append("Short excerpts provide limited evidence for tempo estimation.")
    # Peak-preserving min/max envelope instead of decimation that can hide transients.
    bounds = np.linspace(0, len(y), min(500, len(y)) + 1, dtype=int)
    waveform, timeline = [], []
    for start, end in zip(bounds[:-1], bounds[1:]):
        chunk = y[start:end]
        time = float((start + end) / 2 / ANALYSIS_SAMPLE_RATE)
        waveform.append({"time_seconds": time, "minimum": float(chunk.min()), "maximum": float(chunk.max())})
        timeline.append({"time_seconds": time, "rms_amplitude": float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))})
    prediction = predict_music(mono, sr, silent) if include_models else None
    return AudioAnalysis(duration_seconds=len(samples) / sr, source_sample_rate=sr,
        analysis_sample_rate=ANALYSIS_SAMPLE_RATE, channels=channels, decoded_format="MP3" if suffix == ".mp3" else "WAV",
        bpm=bpm, beat_count=beat_count, rms_amplitude=rms, rms_dbfs=20 * float(np.log10(rms)) if rms > 0 else None,
        peak_amplitude=peak, spectral_centroid_hz=centroid, spectral_bandwidth_hz=bandwidth,
        spectral_rolloff_hz=rolloff, zero_crossing_rate=crossings, is_silent=silent,
        mood=prediction.mood if prediction else "not estimated", genre=prediction.genre if prediction else "not estimated",
        music_model=prediction, warnings=warnings, waveform=waveform, rms_timeline=timeline)
