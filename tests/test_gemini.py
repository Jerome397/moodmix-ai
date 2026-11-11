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
