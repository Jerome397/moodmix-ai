from backend.catalog import load_catalog
from backend.evaluation import CASES, catalog_digest, evaluate_case
from backend import generation
from backend.schemas import PlaylistDecision, PlannedTrack


def test_paired_metrics_and_fallback_are_distinct(monkeypatch):
    tracks = load_catalog()
    baseline = evaluate_case(tracks, CASES[0], "baseline")
    config = generation.Settings(_env_file=None, llm_provider="gemini", gemini_api_key="test")
    monkeypatch.setattr(generation, "Settings", lambda: config)
    def choose(settings, payload, correction):
        option = payload['feasible_options'][-1]
        return PlaylistDecision(option_id=option['option_id'], title="Test", rationale="Test rationale",
            track_reasons=[PlannedTrack(track_id=i, explanation="Test note") for i in option['track_ids']])
    monkeypatch.setattr(generation, "gemini_plan", choose)
    ai = evaluate_case(tracks, CASES[0], "llm")
    assert baseline['actual_method'] == 'baseline' and ai['actual_method'] == 'llm'
    for row in [baseline, ai]:
        assert row['final_valid']
        assert row['duration_seconds'] + row['duration_error_seconds'] == 600
        assert row['artist_count'] == len({t['artist'] for t in row['tracks']})
        assert 0 < row['artist_diversity'] <= 1
    monkeypatch.setattr(generation, "gemini_plan", lambda *args: (_ for _ in ()).throw(generation.ProviderFailure('offline')))
    fallback = evaluate_case(tracks, CASES[0], 'llm')
    assert fallback['actual_method'] == 'fallback' and fallback['final_valid']
    assert catalog_digest(tracks) == catalog_digest(list(tracks))
