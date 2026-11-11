"""Constraint validity and meaningful selection across alternative mixes."""
import pytest
from backend.catalog import load_catalog
from backend.planning import build_options, materialize_decision
from backend.recommendation import retrieve, baseline, validate_plan
from backend.schemas import PlaylistRequest, PlaylistDecision, PlannedTrack


@pytest.mark.parametrize("minutes", [3, 10, 45])
@pytest.mark.parametrize("progression", ["steady", "rising", "falling"])
def test_every_option_preserves_existing_hard_constraints(minutes, progression):
    request = PlaylistRequest(duration_minutes=minutes, progression=progression)
    candidates = retrieve(load_catalog(), request)
    options = build_options(candidates, request)
    assert 1 <= len(options) <= 8
    assert [t.track_id for t in options[0].tracks] == [t.track_id for t in baseline(candidates, request).tracks]
    assert len({tuple(t.track_id for t in option.tracks) for option in options}) == len(options)
    for option in options:
        decision = PlaylistDecision(option_id=option.option_id, title="Test", rationale="Fits", track_reasons=[])
        plan, _ = materialize_decision(decision, options, candidates, request)
        validate_plan(plan, candidates, request)
        assert len({t.track_id for t in plan.tracks}) == len(plan.tracks)


def test_option_choice_changes_real_membership():
    request = PlaylistRequest(duration_minutes=10, genre="jazz", progression="rising")
    candidates = retrieve(load_catalog(), request)
    options = build_options(candidates, request)
    assert len({frozenset(t.track_id for t in o.tracks) for o in options}) > 1
    decision = PlaylistDecision(option_id=options[1].option_id, title="Different mix", rationale="A distinct selection.", track_reasons=[])
    plan, _ = materialize_decision(decision, options, candidates, request)
    assert {t.track_id for t in plan.tracks} != {t.track_id for t in baseline(candidates, request).tracks}


@pytest.mark.parametrize("notes", [
    [PlannedTrack(track_id="invented", explanation="Bad")],
    [PlannedTrack(track_id="duplicate", explanation="Bad")] * 2,
])
def test_invalid_track_notes_rejected(notes):
    request = PlaylistRequest(duration_minutes=10)
    candidates = retrieve(load_catalog(), request)
    options = build_options(candidates, request)
    decision = PlaylistDecision(option_id=options[0].option_id, title="Bad", rationale="Bad", track_reasons=notes)
    with pytest.raises(ValueError):
        materialize_decision(decision, options, candidates, request)


def test_one_feasible_option_is_disclosed():
    candidate = load_catalog()[0].model_copy(update={"duration_seconds": 180})
    request = PlaylistRequest(duration_minutes=3)
    options = build_options([candidate], request)
    assert len(options) == 1
    decision = PlaylistDecision(option_id=options[0].option_id, title="One", rationale="Only feasible mix.", track_reasons=[])
    _, warnings = materialize_decision(decision, options, [candidate], request)
    assert any("Only one" in warning for warning in warnings)
