"""Offline tests: constraints and failure handling, no paid inference."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.catalog import load_catalog as load_catalog_file
from pathlib import Path
from backend.schemas import PlaylistRequest, Plan, PlannedTrack
from backend.recommendation import retrieve, baseline, validate_plan
from backend.generation import generate

client = TestClient(app)


def load_catalog():
    return load_catalog_file(Path(__file__).resolve().parents[1] / "data" / "development_catalog.csv")


@pytest.fixture(autouse=True)
def use_development_catalog(monkeypatch):
    monkeypatch.setattr("backend.main.load_catalog", load_catalog)


@pytest.mark.parametrize("progression", ["steady", "rising", "falling"])
def test_baseline_constraints(progression):
    request = PlaylistRequest(progression=progression)
    candidates = retrieve(load_catalog(), request)
    tracks = validate_plan(baseline(candidates, request), candidates, request)
    assert 0 < sum(t.duration_seconds for t in tracks) <= 900
    assert len(tracks) == len({t.track_id for t in tracks})


@pytest.mark.parametrize("ids", [["unknown"], ["dev-01", "dev-01"]])
def test_reject_invented_or_duplicate_ids(ids):
    with pytest.raises(ValueError):
        validate_plan(Plan(title="Invalid", tracks=[PlannedTrack(track_id=i, explanation="test") for i in ids]), load_catalog(), PlaylistRequest())


def test_http_validation_and_missing_genre():
    assert client.post("/playlist", json={"duration_minutes": -1}).status_code == 422
    assert client.post("/playlist", json={"genre": "nonexistent"}).status_code == 422


def test_api_uses_canonical_metadata():
    response = client.post("/playlist", json={"progression": "rising"})
    assert response.status_code == 200
    result = response.json()
    assert result["method"] == "baseline"
    assert result["duration_seconds"] == sum(t["duration_seconds"] for t in result["tracks"])
    assert result["catalog_kind"] == "synthetic development data"


def test_missing_credentials_is_labelled_fallback(monkeypatch):
    from backend import generation
    monkeypatch.setattr(generation, "Settings", lambda: type("Config", (), {"openai_api_key": "", "openai_model": ""})())
    _, method, warnings = generate(load_catalog(), PlaylistRequest(mode="llm"))
    assert method == "fallback" and warnings


def test_invalid_llm_output_retries_then_falls_back(monkeypatch):
    from backend import generation
    calls = []
    class FakeClient:
        responses = None
        def __init__(self, **kwargs):
            self.responses = self
        def parse(self, **kwargs):
            calls.append(1)
            return type("Response", (), {"output_parsed": Plan(title="Bad", tracks=[PlannedTrack(track_id="invented", explanation="bad")])})()
    monkeypatch.setattr(generation, "Settings", lambda: type("Config", (), {"openai_api_key": "test", "openai_model": "test"})())
    monkeypatch.setattr(generation, "OpenAI", FakeClient)
    _, method, warnings = generate(load_catalog(), PlaylistRequest(mode="llm"))
    assert len(calls) == 2 and method == "fallback" and warnings


def test_reject_duration_overflow_and_wrong_order():
    catalog = load_catalog()
    with pytest.raises(ValueError, match="duration"):
        validate_plan(Plan(title="Too long", tracks=[PlannedTrack(track_id=t.track_id, explanation="test") for t in catalog]), catalog, PlaylistRequest(duration_minutes=3))
    request = PlaylistRequest(progression="rising")
    plan = baseline(catalog, request)
    plan.tracks.reverse()
    with pytest.raises(ValueError, match="Energy"):
        validate_plan(plan, catalog, request)


def test_valid_llm_plan_is_preserved(monkeypatch):
    from backend import generation
    catalog = load_catalog()
    request = PlaylistRequest(mode="llm")
    plan = baseline(catalog, request)
    plan.title = "Model-selected title"
    class FakeClient:
        def __init__(self, **kwargs):
            self.responses = self
        def parse(self, **kwargs):
            return type("Response", (), {"output_parsed": plan})()
    monkeypatch.setattr(generation, "Settings", lambda: type("Config", (), {"openai_api_key": "test", "openai_model": "test"})())
    monkeypatch.setattr(generation, "OpenAI", FakeClient)
    result, method, warnings = generate(catalog, request)
    assert result.title == "Model-selected title" and method == "llm" and not warnings
