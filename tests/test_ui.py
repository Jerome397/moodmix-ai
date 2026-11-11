"""Exercise the Streamlit form against the real local backend in-process."""
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from backend.main import app
from pathlib import Path


def test_streamlit_baseline_flow(monkeypatch):
    import httpx
    from backend.catalog import load_catalog
    fixture = Path(__file__).resolve().parents[1] / "data" / "development_catalog.csv"
    monkeypatch.setattr("backend.main.load_catalog", lambda: load_catalog(fixture))
    client = TestClient(app)
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: client.post("/playlist", json=json))
    monkeypatch.setattr(httpx, "get", lambda url, timeout: client.get("/llm/status" if url.endswith("/llm/status") else "/health"))
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=20).run()
    ui.sidebar.radio[0].set_value("Playlist Generator").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert len(ui.dataframe) == 2
    assert len(ui.metric) == 3


def test_audio_page_upload_and_measurements(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    import httpx
    import numpy as np
    import soundfile as sf
    import streamlit as st
    buffer = BytesIO()
    samples = .2 * np.sin(2 * np.pi * 440 * np.arange(22050 * 4) / 22050)
    sf.write(buffer, samples, 22050, format="WAV")
    content = buffer.getvalue()
    upload = SimpleNamespace(name="test.wav", type="audio/wav", getvalue=lambda: content)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    client = TestClient(app)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: client.get("/llm/status" if url.endswith("/llm/status") else "/health"))
    monkeypatch.setattr(httpx, "post", lambda url, files, timeout, **kwargs: client.post("/audio/analyze", files=files, data=kwargs.get("data")))
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=120).run()
    ui.sidebar.radio[0].set_value("Music Analysis").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert ui.session_state["audio_analysis"]["rms_amplitude"] > .1
    assert len(ui.metric) == 4 and len(ui.dataframe) == 1


def test_audio_upload_to_playlist_navigation(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    import httpx
    import librosa
    import numpy as np
    import soundfile as sf
    import streamlit as st
    buffer = BytesIO()
    samples = librosa.clicks(times=np.arange(.5, 12, .5), sr=22050, length=22050 * 12)
    sf.write(buffer, samples, 22050, format="WAV")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: SimpleNamespace(name="clicks.wav", type="audio/wav", getvalue=buffer.getvalue))
    client = TestClient(app)
    monkeypatch.setattr(httpx, "get", lambda url, timeout: client.get("/llm/status" if url.endswith("/llm/status") else "/health"))
    def post(url, timeout, **kwargs):
        return client.post("/audio/analyze" if url.endswith("/audio/analyze") else "/playlist", **kwargs)
    monkeypatch.setattr(httpx, "post", post)
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=120).run()
    ui.sidebar.radio[0].set_value("Music Analysis").run()
    ui.checkbox[0].set_value(False).run()
    next(b for b in ui.button if b.label == "Analyze audio").click().run()
    next(b for b in ui.button if b.label == "Use this audio for a playlist").click().run()
    assert not ui.exception
    assert ui.sidebar.radio[0].value == "Playlist Generator"
    next(b for b in ui.button if b.label == "Generate playlist").click().run()
    assert not ui.exception
    assert ui.session_state["playlist"]["retrieval_context"]["audio_reference_attached"]
    next(c for c in ui.checkbox if c.label == "Use analyzed audio as reference").set_value(False).run()
    next(b for b in ui.button if b.label == "Generate playlist").click().run()
    assert not ui.session_state["playlist"]["retrieval_context"]["audio_reference_attached"]
