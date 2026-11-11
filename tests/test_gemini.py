"""Exercise Gemini REST without credentials or external requests."""
import json
import httpx
import pytest

from backend import generation
from backend.catalog import load_catalog
from backend.recommendation import baseline
from backend.schemas import PlaylistRequest


@pytest.fixture
def settings(monkeypatch):
    config = generation.Settings(_env_file=None, llm_provider="gemini", gemini_api_key="private-test-key")
    monkeypatch.setattr(generation, "Settings", lambda: config)
    return config


def mock_transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(generation.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def envelope(plan):
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": plan.model_dump_json()}]}}]}


def test_gemini_validated_plan_and_private_header(settings, monkeypatch):
    tracks = load_catalog()
    request = PlaylistRequest(mode="llm", instructions="Build toward a lively finale.")
    plan = baseline(tracks, request)
    plan.title = "Gemini test title"
    calls = []
    def handle(req):
        calls.append(req)
        assert req.headers["x-goog-api-key"] == "private-test-key"
        assert "private-test-key" not in str(req.url)
        body = json.loads(req.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert "responseJsonSchema" in body["generationConfig"]
        assert body["generationConfig"]["responseJsonSchema"]["properties"]["tracks"]["items"]["properties"]["track_id"]["enum"] == [t.track_id for t in tracks]
        assert "tools" not in body
        assert request.instructions in body["contents"][0]["parts"][0]["text"]
        return httpx.Response(200, json=envelope(plan))
    mock_transport(monkeypatch, handle)
    actual, method, warnings = generation.generate(tracks, request)
    assert actual.title == "Gemini test title" and method == "llm" and not warnings
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_provider_error_falls_back_without_retry_or_secrets(settings, monkeypatch, status):
    calls = []
    def handle(req):
        calls.append(req)
        return httpx.Response(status, json={"error": "private-test-key"})
    mock_transport(monkeypatch, handle)
    _, method, warnings = generation.generate(load_catalog(), PlaylistRequest(mode="llm"))
    assert method == "fallback" and len(calls) == 1
    assert "private-test-key" not in str(warnings)
    if status == 429:
        assert "quota" in warnings[0]


@pytest.mark.parametrize("response", [
    {"candidates": []},
    {"candidates": [{"finishReason": "MAX_TOKENS"}]},
    {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "invalid JSON"}]}}]},
    {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"title":"bad","tracks":[{"track_id":"invented","explanation":"bad"}]}'}]}}]},
])
def test_bad_outputs_have_limited_retry(settings, monkeypatch, response):
    calls = []
    def handle(req):
        calls.append(req)
        return httpx.Response(200, json=response)
    mock_transport(monkeypatch, handle)
    _, method, _ = generation.generate(load_catalog(), PlaylistRequest(mode="llm"))
    assert method == "fallback" and len(calls) == 2
    assert "Previous output was invalid" in calls[1].content.decode()


def test_missing_key_never_calls_network(settings, monkeypatch):
    settings.gemini_api_key = ""
    monkeypatch.setattr(generation, "gemini_plan", lambda *args: pytest.fail("Unexpected network call"))
    _, method, warnings = generation.generate(load_catalog(), PlaylistRequest(mode="llm"))
    assert method == "fallback" and "not configured" in warnings[0]


def test_status_does_not_expose_key(settings):
    status = generation.llm_status()
    assert status == {"provider": "gemini", "model": settings.gemini_model, "configured": True}
    assert "private-test-key" not in str(status)


def test_portable_schema_preserves_title_property_and_inlines_references():
    schema = generation.gemini_schema()
    assert "title" in schema["properties"]
    assert all(name in schema["properties"] for name in schema["required"])
    assert schema["properties"]["tracks"]["items"]["type"] == "object"
    assert "$ref" not in json.dumps(schema) and "$defs" not in schema


def test_retry_supplies_failed_plan_and_duration_feedback(settings, monkeypatch):
    from backend.schemas import Plan, PlannedTrack
    tracks = load_catalog()
    request = PlaylistRequest(mode="llm", duration_minutes=10)
    valid = baseline(tracks, request)
    too_short = Plan(title="Under coverage", tracks=[PlannedTrack(track_id=tracks[0].track_id, explanation="test")])
    calls = []
    def handle(req):
        body = json.loads(req.content)
        calls.append(body)
        return httpx.Response(200, json=envelope(too_short if len(calls) == 1 else valid))
    mock_transport(monkeypatch, handle)
    _, method, _ = generation.generate(tracks, request)
    assert method == "llm" and len(calls) == 2
    second = calls[1]["contents"][0]["parts"][0]["text"]
    assert "previous_duration_seconds" in second and "validation_feedback" in second


def test_constraint_repair_never_adds_tracks_or_accepts_unknown_ids():
    from backend.schemas import Plan, PlannedTrack
    from backend.recommendation import validate_plan
    tracks = load_catalog()
    request = PlaylistRequest(duration_minutes=10, progression="rising")
    proposal = Plan(title="Model plan", tracks=[PlannedTrack(track_id=t.track_id, explanation="Model reason") for t in tracks[:60]])
    fixed = generation.repair_constraints(proposal, tracks, request)
    validate_plan(fixed, tracks, request)
    assert {p.track_id for p in fixed.tracks} <= {p.track_id for p in proposal.tracks}
    assert all(p.explanation == "Model reason" for p in fixed.tracks)
    proposal.tracks[0].track_id = "invented"
    with pytest.raises(ValueError):
        generation.repair_constraints(proposal, tracks, request)


def test_repaired_gemini_plan_is_explicitly_labelled(settings, monkeypatch):
    from backend.schemas import Plan, PlannedTrack
    tracks = load_catalog()
    proposal = Plan(title="Model proposal", tracks=[PlannedTrack(track_id=t.track_id, explanation="Model reason") for t in tracks[:60]])
    mock_transport(monkeypatch, lambda req: httpx.Response(200, json=envelope(proposal)))
    _, method, warnings = generation.generate(tracks, PlaylistRequest(mode="llm", duration_minutes=10))
    assert method == "llm_repaired" and "only the LLM-proposed tracks" in warnings[0]
