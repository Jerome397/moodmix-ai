"""Reproducible paired development evaluation; never invent human ratings."""
import hashlib
from time import perf_counter

from backend.audio_recommendation import resolve_context
from backend.generation import generate
from backend.recommendation import retrieve, validate_plan
from backend.schemas import PlaylistRequest


CASES = [
    {"case_id": "gentle_jazz", "duration_minutes": 10, "genre": "jazz", "energy": .3, "bpm": 90, "instructions": "Prefer a gentle mix with lower average energy and several artists."},
    {"case_id": "electronic_build", "duration_minutes": 15, "genre": "electronic", "progression": "rising", "energy": .6, "bpm": 125, "instructions": "Build energy gradually and prefer several artists."},
    {"case_id": "rock_cooldown", "duration_minutes": 10, "genre": "rock", "progression": "falling", "energy": .6, "bpm": 115, "instructions": "Finish gently; prefer a varied artist lineup."},
    {"case_id": "mixed_artists", "duration_minutes": 15, "instructions": "Maximize artist variety while keeping a coherent flow."},
    {"case_id": "short_pop", "duration_minutes": 5, "genre": "pop", "energy": .7, "bpm": 120, "instructions": "Choose an upbeat short break with distinct artists if practical."},
    {"case_id": "long_variety", "duration_minutes": 30, "progression": "rising", "instructions": "Prefer many different artists and genres, building toward a lively ending."},
]


def evaluate_case(tracks, case, mode):
    request = PlaylistRequest(**{k: v for k, v in case.items() if k != "case_id"}, mode=mode)
    start = perf_counter()
    context = resolve_context(request, tracks)
    candidates = retrieve(tracks, request, context)
    resolved = request.model_copy(update={"bpm": context.target_bpm})
    audit = {}
    plan, method, warnings = generate(candidates, resolved, audit)
    selected = validate_plan(plan, candidates, resolved)
    duration = sum(t.duration_seconds for t in selected)
    return {"case_id": case["case_id"], "requested_mode": mode, "actual_method": method,
            "request": request.model_dump(), "final_valid": True,
            "duration_seconds": duration, "duration_error_seconds": request.duration_minutes * 60 - duration,
            "artist_count": len({t.artist for t in selected}), "genre_count": len({t.genre for t in selected}),
            "artist_diversity": len({t.artist for t in selected}) / len(selected),
            "mean_energy_distance": sum(abs(t.energy - request.energy) for t in selected) / len(selected),
            "mean_bpm_distance": sum(abs(t.bpm - context.target_bpm) for t in selected) / len(selected),
            "latency_seconds": round(perf_counter() - start, 4), "diagnostics": audit, "warnings": warnings,
            "title": plan.title, "tracks": [t.model_dump() for t in selected],
            "notes": [t.model_dump() for t in plan.tracks]}


def catalog_digest(tracks):
    import json
    return hashlib.sha256(json.dumps([t.model_dump() for t in tracks], sort_keys=True).encode()).hexdigest()
