"""Exercise constrained Gemini decisions with offline HTTP transport."""
import json
import httpx
import pytest
from backend import generation
from backend.catalog import load_catalog
from backend.schemas import PlaylistRequest, PlaylistDecision, PlannedTrack


@pytest.fixture
def settings(monkeypatch):
    config = generation.Settings(_env_file=None, llm_provider='gemini', gemini_api_key='private-test-key')
    monkeypatch.setattr(generation, 'Settings', lambda: config)
    return config


def mock_transport(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(generation.httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def payload_from(request):
    body = json.loads(request.content)
    return json.JSONDecoder().raw_decode(body['contents'][0]['parts'][0]['text'])[0]


def decision_for(payload, index=0, title='Gemini test title'):
    option = payload['feasible_options'][index]
    return PlaylistDecision(option_id=option['option_id'], title=title, rationale='Fits the requested musical progression using the supplied metadata.',
        track_reasons=[PlannedTrack(track_id=i, explanation='A suitable tempo and energy for this mix.') for i in option['track_ids']])


def envelope(decision):
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': decision.model_dump_json()}]}}]}


def test_gemini_validated_decision_and_private_header(settings, monkeypatch):
    calls = []
    selected = []
    request = PlaylistRequest(mode='llm', duration_minutes=10, genre='jazz', instructions='Build toward a lively finale.')
    def handle(req):
        calls.append(req)
        assert req.headers['x-goog-api-key'] == 'private-test-key'
        assert 'private-test-key' not in str(req.url)
        body = json.loads(req.content)
        payload = payload_from(req)
        schema = body['generationConfig']['responseJsonSchema']
        assert schema['properties']['option_id']['enum'] == [o['option_id'] for o in payload['feasible_options']]
        assert schema['properties']['track_reasons']['items']['properties']['track_id']['enum'] == [t['track_id'] for t in payload['candidates']]
        assert 'tools' not in body and request.instructions in body['contents'][0]['parts'][0]['text']
        decision = decision_for(payload, index=1)
        selected.extend(payload['feasible_options'][1]['track_ids'])
        return httpx.Response(200, json=envelope(decision))
    mock_transport(monkeypatch, handle)
    from backend.recommendation import retrieve
    candidates = retrieve(load_catalog(), request)
    audit = {}
    actual, method, warnings = generation.generate(candidates, request, audit)
    assert actual.title == 'Gemini test title' and method == 'llm' and not warnings
    assert [t.track_id for t in actual.tracks] == selected
    assert audit['membership_changed_from_baseline'] and audit['provider_calls'] == 1
    assert len(calls) == 1


@pytest.mark.parametrize('status', [400, 401, 403, 404, 429, 500])
def test_provider_error_falls_back_without_retry_or_secrets(settings, monkeypatch, status):
    calls = []
    def handle(req):
        calls.append(req)
        return httpx.Response(status, json={'error': 'private-test-key'})
    mock_transport(monkeypatch, handle)
    audit = {}
    _, method, warnings = generation.generate(load_catalog(), PlaylistRequest(mode='llm'), audit)
    assert method == 'fallback' and len(calls) == 1
    assert 'private-test-key' not in str(warnings)
    assert audit['failure_kind'] == 'provider_error' and audit['selected_option_id'] is None
    if status == 429:
        assert 'quota' in warnings[0]


@pytest.mark.parametrize('response', [
    {'candidates': []},
    {'candidates': [{'finishReason': 'MAX_TOKENS'}]},
    {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': 'invalid JSON'}]}}]},
    {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{"title":"bad","tracks":[{"track_id":"invented","explanation":"bad"}]}'}]}}]},
])
def test_bad_outputs_have_limited_retry(settings, monkeypatch, response):
    calls = []
    def handle(req):
        calls.append(req)
        return httpx.Response(200, json=response)
    mock_transport(monkeypatch, handle)
    audit = {}
    _, method, _ = generation.generate(load_catalog(), PlaylistRequest(mode='llm'), audit)
    assert method == 'fallback' and len(calls) == 2
    assert len(audit['validation_failures']) == 2
    assert 'Previous output was invalid' in calls[1].content.decode()


def test_missing_key_never_calls_network(settings, monkeypatch):
    settings.gemini_api_key = ''
    monkeypatch.setattr(generation, 'gemini_plan', lambda *args: pytest.fail('Unexpected network call'))
    _, method, warnings = generation.generate(load_catalog(), PlaylistRequest(mode='llm'))
    assert method == 'fallback' and 'not configured' in warnings[0]


def test_status_does_not_expose_key(settings):
    assert generation.llm_status() == {'provider': 'gemini', 'model': settings.gemini_model, 'configured': True}


def test_portable_schema_preserves_title_and_inlines_references():
    schema = generation.gemini_schema()
    assert 'title' in schema['properties']
    assert all(name in schema['properties'] for name in schema['required'])
    assert schema['properties']['track_reasons']['items']['type'] == 'object'
    assert '$ref' not in json.dumps(schema) and '$defs' not in schema


def test_invalid_option_is_corrected_with_one_retry(settings, monkeypatch):
    calls = []
    def handle(req):
        calls.append(req)
        decision = decision_for(payload_from(req))
        if len(calls) == 1:
            decision.option_id = 'invented-option'
        return httpx.Response(200, json=envelope(decision))
    mock_transport(monkeypatch, handle)
    audit = {}
    _, method, _ = generation.generate(load_catalog(), PlaylistRequest(mode='llm', duration_minutes=10), audit)
    assert method == 'llm' and len(calls) == 2 and len(audit['validation_failures']) == 1
    assert 'one exact option_id' in calls[1].content.decode()


def test_missing_individual_notes_are_labelled_as_playlist_rationale(settings, monkeypatch):
    def handle(req):
        decision = decision_for(payload_from(req))
        decision.track_reasons = []
        return httpx.Response(200, json=envelope(decision))
    mock_transport(monkeypatch, handle)
    actual, method, warnings = generation.generate(load_catalog(), PlaylistRequest(mode='llm'))
    assert method == 'llm' and any('individual notes' in warning for warning in warnings)
    assert all(t.explanation.startswith('Playlist rationale:') for t in actual.tracks)
