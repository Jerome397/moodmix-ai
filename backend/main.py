"""HTTP interface for the Phase 1 playlist prototype."""
from time import perf_counter
from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.concurrency import run_in_threadpool
from backend.audio import AudioAnalysis, AudioInputError, MAX_UPLOAD_BYTES, analyze_audio
from backend.catalog import load_catalog, catalog_info
from backend.generation import generate, llm_status
from backend.recommendation import retrieve, validate_plan
from backend.schemas import PlaylistRequest

app = FastAPI(title="MoodMix AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", **catalog_info(load_catalog())}


@app.get("/llm/status")
def model_status():
    return llm_status()


@app.post("/audio/analyze", response_model=AudioAnalysis)
async def audio_analysis(file: UploadFile = File(...)):
    """Run CPU work off the event loop; close uploads on every outcome."""
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        return await run_in_threadpool(analyze_audio, content, file.filename or "")
    except AudioInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        await file.close()


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
