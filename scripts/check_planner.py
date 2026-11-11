"""Reproduce targeted planner smoke checks. Network use requires --live."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.audio_recommendation import resolve_context
from backend.catalog import load_catalog
from backend.generation import generate, llm_status
from backend.planning import build_options
from backend.recommendation import retrieve, validate_plan
from backend.schemas import PlaylistRequest


def run_checks(live: bool) -> dict:
    reference = json.loads((ROOT / "evaluation/audio-pipeline-smoke.json").read_text())["audio_reference"]
    cases = [
        ("audio_jazz_rising_10", PlaylistRequest(mode="llm", mood="any", duration_minutes=10, genre="jazz", progression="rising", audio_reference=reference,
            instructions="Use the upload as a loose reference. Prefer a varied lineup with as many distinct artists as practical, building to a lively finish.")),
        ("electronic_rising_10", PlaylistRequest(mode="llm", duration_minutes=10, genre="electronic", progression="rising",
            instructions="Start gently, then build energy. Prefer a lower average energy over a relentless high-energy set.")),
        ("variety_steady_15", PlaylistRequest(mode="llm", duration_minutes=15,
            instructions="Choose a varied lineup with several artists and genres. Keep the flow coherent; avoid relying on just one artist.")),
    ]
    catalog = load_catalog()
    rows = []
    for case_id, request in cases:
        context = resolve_context(request, catalog)
        resolved = request.model_copy(update={"bpm": context.target_bpm})
        candidates = retrieve(catalog, request, context)
        start = perf_counter()
        if live:
            audit = {}
            plan, method, warnings = generate(candidates, resolved, audit)
            selected = validate_plan(plan, candidates, resolved)
            row = {"case_id": case_id, "request": request.model_dump(), "method": method,
                   "duration_seconds": sum(t.duration_seconds for t in selected),
                   "duration_error_seconds": request.duration_minutes * 60 - sum(t.duration_seconds for t in selected),
                   "latency_seconds": round(perf_counter() - start, 3), "diagnostics": audit, "warnings": warnings,
                   "title": plan.title, "tracks": [{"track_id": t.track_id, "title": t.title, "artist": t.artist, "explanation": note.explanation} for t, note in zip(selected, plan.tracks)]}
            print(case_id, method, row["duration_seconds"], "seconds", "provider calls:", audit["provider_calls"])
        else:
            options = build_options(candidates, resolved)
            row = {"case_id": case_id, "request": request.model_dump(), "option_count": len(options),
                   "distinct_track_sets": len({frozenset(t.track_id for t in o.tracks) for o in options}),
                   "options": [option.summary() for option in options]}
            print(case_id, "offline", len(options), "valid options")
        rows.append(row)
    return {"purpose": "Targeted development smoke checks, not a held-out recommendation-quality benchmark",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "live": live,
            "strategy": "feasible_options_v1", "provider": llm_status() if live else None, "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use configured provider; up to two calls per case, quotas/charges depend on your project")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/planner-checks.json")
    args = parser.parse_args()
    result = run_checks(args.live)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
