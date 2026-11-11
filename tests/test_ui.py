"""Exercise the Streamlit form against the real local backend in-process."""
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
from backend.main import app
from pathlib import Path


def test_streamlit_baseline_flow(monkeypatch):
    import httpx
    client = TestClient(app)
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: client.post("/playlist", json=json))
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py", default_timeout=20).run()
    ui.sidebar.radio[0].set_value("Playlist Generator").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert len(ui.dataframe) == 1
    assert len(ui.metric) == 3
