"""Signal-based checks using generated recordings, not private user audio."""
from io import BytesIO
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.audio import analyze_audio, AudioInputError, MAX_UPLOAD_BYTES
from backend.main import app


def encoded(samples, sr=22050, format="WAV", subtype=None):
    buffer = BytesIO()
    sf.write(buffer, samples, sr, format=format, subtype=subtype)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def tone():
    time = np.arange(22050 * 4) / 22050
    return encoded(0.2 * np.sin(2 * np.pi * 440 * time))


def test_known_tone_features(tone):
    result = analyze_audio(tone, "tone.wav")
    assert result.duration_seconds == 4
    assert result.rms_amplitude == pytest.approx(.2 / np.sqrt(2), abs=0.001)
    assert result.rms_dbfs == pytest.approx(-16.99, abs=.1)
    assert result.spectral_centroid_hz == pytest.approx(440, abs=20)
    assert result.zero_crossing_rate == pytest.approx(2 * 440 / 22050, abs=.003)
    assert not result.is_silent and result.mood == "not estimated"
    assert len(result.waveform) <= 500
    json.dumps(result.model_dump(), allow_nan=False)


def test_silence_has_no_tempo_or_infinite_dbfs():
    result = analyze_audio(encoded(np.zeros(22050 * 3)), "silence.wav")
    assert result.is_silent and result.bpm is None and result.rms_dbfs is None
    assert result.spectral_centroid_hz == 0 and result.rms_amplitude == 0
    json.dumps(result.model_dump(), allow_nan=False)


def test_click_track_tempo():
    import librosa
    clicks = librosa.clicks(times=np.arange(.5, 20, .5), sr=22050, length=22050 * 20)
    result = analyze_audio(encoded(clicks), "clicks.wav")
    assert result.bpm == pytest.approx(120, abs=5)
    assert result.beat_count >= 25


def test_antiphase_stereo_is_not_reported_as_silence():
    time = np.arange(44100 * 3) / 44100
    wave = .2 * np.sin(2 * np.pi * 440 * time)
    result = analyze_audio(encoded(np.column_stack([wave, -wave]), 44100), "stereo.wav")
    assert result.channels == 2 and result.analysis_sample_rate == 22050
    assert not result.is_silent and result.spectral_centroid_hz > 400
    assert any("cancellation" in warning for warning in result.warnings)


def test_mp3_roundtrip(tone):
    samples, sr = sf.read(BytesIO(tone))
    content = encoded(samples, sr, "MP3")
    result = analyze_audio(content, "sample.MP3")
    assert result.decoded_format == "MP3"
    assert result.duration_seconds == pytest.approx(4, abs=.15)
    assert result.spectral_centroid_hz == pytest.approx(440, abs=30)


@pytest.mark.parametrize("content,name,status", [
    (b"", "empty.wav", 400), (b"not audio", "bad.wav", 422),
    (b"not audio", "bad.exe", 415), (b"x" * (MAX_UPLOAD_BYTES + 1), "big.wav", 413),
])
def test_invalid_inputs(content, name, status):
    with pytest.raises(AudioInputError) as error:
        analyze_audio(content, name)
    assert error.value.status_code == status


@pytest.mark.parametrize("duration", [1])
def test_duration_limits(duration):
    with pytest.raises(AudioInputError, match="at least 3"):
        analyze_audio(encoded(np.zeros(8000 * duration), sr=8000), "duration.wav")


@pytest.mark.parametrize('duration', [30, 31, 121])
def test_long_upload_analyzes_only_first_30_seconds(duration):
    samples = np.zeros(8000 * duration, dtype=np.float32)
    samples[8000 * 30:] = .8
    result = analyze_audio(encoded(samples, 8000), 'song.wav')
    assert result.duration_seconds == 30
    assert result.original_duration_seconds == duration
    assert result.trimmed == (duration > 30)
    assert result.is_silent  # Loud samples after the cut must not affect features.
    assert result.waveform[-1]['time_seconds'] <= 30


def test_large_wav_over_previous_limit_is_trimmed():
    content = encoded(np.zeros((96000 * 60, 2), dtype=np.float32), 96000)
    assert len(content) > 20 * 1024 * 1024
    response = TestClient(app).post('/audio/analyze', files={'file': ('large.wav', content, 'audio/wav')})
    assert response.status_code == 200
    assert response.json()['trimmed'] and response.json()['duration_seconds'] == 30


def test_reject_renamed_format(tone):
    with pytest.raises(AudioInputError, match="does not match"):
        analyze_audio(tone, "renamed.mp3")


def test_reject_nonfinite_float_audio():
    samples = np.zeros(24000, dtype=np.float32)
    samples[10] = np.nan
    with pytest.raises(AudioInputError, match="non-finite"):
        analyze_audio(encoded(samples, 8000, subtype="FLOAT"), "nonfinite.wav")


def test_audio_http_endpoint(tone):
    with TestClient(app) as client:
        response = client.post("/audio/analyze", files={"file": ("tone.wav", tone, "audio/wav")})
        assert response.status_code == 200
        assert response.json()["rms_amplitude"] > .1
        assert client.post("/audio/analyze", files={"file": ("bad.wav", b"bad", "audio/wav")}).status_code == 422
        assert client.post("/audio/analyze").status_code == 422
