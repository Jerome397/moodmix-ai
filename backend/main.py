"""HTTP interface for the Phase 1 playlist prototype."""
from time import perf_counter
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool
from backend.audio import AudioAnalysis, AudioInputError, MAX_UPLOAD_BYTES, analyze_audio
from backend.music_model import model_status as audio_model_status
from backend.catalog import load_catalog, catalog_info
from backend.generation import generate, llm_status
from backend.recommendation import retrieve, validate_plan
from backend.schemas import PlaylistRequest
from backend.audio_recommendation import resolve_context
from dataclasses import asdict

app = FastAPI(title="MoodMix AI", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", **catalog_info(load_catalog())}


@app.get("/llm/status")
def model_status():
    return llm_status()


@app.post("/audio/analyze", response_model=AudioAnalysis)
async def audio_analysis(file: UploadFile = File(...), include_models: bool = Form(False)):
    """Run CPU work off the event loop; close uploads on every outcome."""
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        return await run_in_threadpool(analyze_audio, content, file.filename or "", include_models)
    except AudioInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        await file.close()


@app.get("/audio/model-status")
def music_model_status():
    return audio_model_status()


@app.post("/playlist")
def playlist(request: PlaylistRequest):
    start = perf_counter()
    try:
        catalog = load_catalog()
        context = resolve_context(request, catalog)
        candidates = retrieve(catalog, request, context)
        resolved_request = request.model_copy(update={"bpm": context.target_bpm})
        plan, method, warnings = generate(candidates, resolved_request)
        tracks = validate_plan(plan, candidates, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total = sum(t.duration_seconds for t in tracks)
    return {"title": plan.title, "method": method, "warnings": context.warnings + warnings,
            "retrieval_context": asdict(context),
            **catalog_info(catalog), "duration_seconds": total,
            "duration_error_seconds": request.duration_minutes * 60 - total,
            "latency_seconds": perf_counter() - start,
            "tracks": [{**t.model_dump(), "explanation": item.explanation} for t, item in zip(tracks, plan.tracks)]}
