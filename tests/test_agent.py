import pytest
from backend import agent
from backend.schemas import PlaylistRequest, Plan, PlannedTrack


def brief():
    return agent.Brief(summary='Gentle jazz', preferences=PlaylistRequest(genre='jazz', duration_minutes=10),
                       phases=[agent.Phase(name='Opening', minutes=4, energy=.3), agent.Phase(name='Build', minutes=6, energy=.6)])


def test_agent_validates_repairs_and_reviews(monkeypatch):
    calls = []
    def ask(model, system, payload):
        calls.append(payload)
        if model == agent.Brief:
            return brief()
        if len(calls) == 2:
            return agent.Action(action='validate_playlist', critique='Initial attempt', playlist=Plan(title='Bad', tracks=[PlannedTrack(track_id='invented', explanation='No')]))
        if len(calls) == 3:
            assert payload['tool_feedback']['valid'] is False
            return agent.Action(action='validate_playlist', critique='Correct the invalid IDs', playlist=Plan.model_validate(calls[1]['tool_feedback']['seed_playlist']))
        assert payload['tool_feedback']['valid'] is True
        return agent.Action(action='finish', critique='The validated mix fits the metadata preferences.')
    monkeypatch.setattr(agent, 'ask', ask)
    result = agent.create_mix('10 minutes of jazz')
    assert result['method'] == 'agent' and result['critique_completed']
    assert result['provider_calls'] == 4
    assert result['events'][1]['valid'] is False
    assert all(t['track_id'] != 'invented' for t in result['tracks'])
    calls.clear()
    agent.create_mix('Keep this but make the beginning gentler', result)
    assert calls[0]['previous_mix']['tracks'] == result['tracks']


def test_agent_bounds_calls_and_labels_fallback(monkeypatch):
    count = 0
    def ask(model, system, payload):
        nonlocal count
        count += 1
        if model == agent.Brief:
            return brief()
        return agent.Action(action='finish', critique='Cannot finish without validation.')
    monkeypatch.setattr(agent, 'ask', ask)
    result = agent.create_mix('Jazz please')
    assert count == 5 and result['provider_calls'] == 5
    assert result['method'] == 'baseline_fallback'
    assert not result['critique_completed']


def test_agent_rejects_invalid_phase_budgets(monkeypatch):
    item = brief()
    item.phases[0].minutes = 3
    monkeypatch.setattr(agent, 'ask', lambda *args: item)
    with pytest.raises(ValueError, match='phase budgets'):
        agent.create_mix('Jazz please')


def test_agent_explicit_fitting_tool(monkeypatch):
    def ask(model, system, payload):
        if model == agent.Brief:
            return brief()
        if payload['validated_playlist']:
            return agent.Action(action='finish', critique='Reviewed the fitted result.')
        seed = Plan.model_validate(payload['valid_seed'])
        return agent.Action(action='fit_playlist', critique='Fit my preferred opening track.', playlist=Plan(title='Fitted mix', tracks=seed.tracks[:1]))
    monkeypatch.setattr(agent, 'ask', ask)
    result = agent.create_mix('Jazz please')
    assert result['method'] == 'agent' and result['critique_completed']
    assert any(e['tool'] == 'fit_playlist' and e['valid'] for e in result['events'])
    assert result['duration_seconds'] >= 540


def test_studio_keeps_last_mix_on_failure(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from backend.generation import ProviderFailure
    monkeypatch.setattr(agent, 'create_mix', lambda *args: (_ for _ in ()).throw(ProviderFailure('Quota reached')))
    ui = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app/agent_app.py').run()
    ui.chat_input[0].set_value('Jazz please').run()
    assert not ui.exception
    assert ui.error[0].value == 'Quota reached'
