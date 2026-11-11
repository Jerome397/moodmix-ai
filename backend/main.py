"""HTTP interface for the Phase 1 playlist prototype."""
from time import perf_counter
from fastapi import FastAPI, HTTPException
from backend.catalog import load_catalog, catalog_info
from backend.generation import generate
from backend.recommendation import retrieve, validate_plan
from backend.schemas import PlaylistRequest

app = FastAPI(title="MoodMix AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", **catalog_info(load_catalog())}


@app.post("/playlist")
def playlist(request: PlaylistRequest):
    start = perf_counter()
    try:
        catalog = load_catalog()
        candidates = retrieve(catalog, request)
        plan, method, warnings = generate(candidates, request)
        tracks = validate_plan(plan, candidates, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total = sum(t.duration_seconds for t in tracks)
    return {"title": plan.title, "method": method, "warnings": warnings,
            **catalog_info(catalog), "duration_seconds": total,
            "duration_error_seconds": request.duration_minutes * 60 - total,
            "latency_seconds": perf_counter() - start,
            "tracks": [{**t.model_dump(), "explanation": item.explanation} for t, item in zip(tracks, plan.tracks)]}
