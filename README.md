# MoodMix AI

Phase 1: a local text-to-playlist prototype with content-based retrieval, a configurable Gemini/OpenAI Transformer planner, backend validation, and a Streamlit interface.

The default catalog contains **240 real FMA tracks**, with published artist/title/duration/genre metadata and Echo Nest tempo/energy attributes. Mood is explicitly unknown. No recordings or audio playback are bundled. No GenAI superiority claims have been established. The fictional development catalog remains only for isolated tests. See `data/ATTRIBUTION.md` and `data/catalog.manifest.json` for licensing, provenance, and selection rules.

## Windows quick start

Python 3.13.7 was found locally. The text prototype uses this interpreter. Reassess audio-model compatibility before adding the Phase 2 dependencies; use Python 3.11/3.12 if the selected model requires it.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

For the exact versions verified in the initial Windows setup, install from `requirements-lock.txt` instead of `requirements.txt`. The verified suite has 48 tests; live real-catalog baseline generation and both server health endpoints were also checked. Gemini inference is verified on a live smoke request with labelled constraint repair; MP3/WAV signal analysis is implemented and verified on generated audio fixtures; neural mood classification remains upcoming.

In a second terminal at the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Visit http://localhost:8501. API documentation: http://127.0.0.1:8000/docs. Activation is optional, so PowerShell execution-policy changes are unnecessary. VS Code's default Python interpreter points to `.venv`; if you selected another interpreter previously, use **Python: Select Interpreter** to choose `.venv\Scripts\python.exe`.

## Optional LLM mode

Gemini is the default provider. Follow `docs/gemini-setup.md` to create a free-tier Google AI Studio key and place it privately in `.env`:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_private_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep paid billing disabled in your Google project. Free-tier quotas and project eligibility apply; the app cannot inspect the project's billing tier. No automatic billing upgrade, model switching, search grounding or provider switching is performed. Free-tier request content may be used by Google to improve its products. No hosted request is made during automated tests.

The backend sends a Pydantic JSON schema through Gemini's REST API using the existing `httpx` dependency, then validates both JSON and playlist constraints. Status is available at `/llm/status`, exposing provider/model/readiness only. Optional OpenAI support remains available with `LLM_PROVIDER=openai`, `OPENAI_API_KEY` and `OPENAI_MODEL`. The sidebar displays the selected provider. Never paste keys into chat or commit them.

Official documentation: https://ai.google.dev/api/generate-content and https://ai.google.dev/gemini-api/docs/structured-output . Free-tier model pricing: https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite . A live Gemini smoke request succeeded with `llm_repaired`; see `evaluation/gemini-live-smoke.json`. This single request establishes connectivity and a validated repaired output, not recommendation quality or reliability across requests.

## Audio analysis

Open **Music Analysis**, upload an MP3/WAV excerpt (3–120 seconds, up to 20 MiB), then click **Analyze audio**. Playback, estimated BPM, RMS signal level, spectral features, waveform envelope, RMS timeline and JSON export are available. The first analysis may initialize compiled signal routines and take longer.

The analysis runs locally via `POST /audio/analyze`; no audio is sent to Gemini. Mood and genre are explicitly **not estimated** yet. Upload features are not connected to recommendation retrieval at this milestone: catalog Echo Nest energy is not interchangeable with RMS. See `docs/audio-analysis.md` for formulas, privacy, format limits and test methodology.

## How it works

Streamlit → FastAPI → catalog validation → feature-distance retrieval → baseline or LLM planner → constraint validation → canonical metadata response.

- Retrieval ranks up to 60 candidates by squared energy distance, squared scaled BPM distance, and a categorical mood mismatch penalty. Genre is an exact filter; mood is a soft preference.
- The baseline uses a 0/1 duration knapsack, preferring earlier similarity-ranked candidates on ties, then sorts energy when requested. Duration coverage is its primary optimization; it is a simple comparison baseline, not an optimal music recommender. Free text is not interpreted by the baseline.
- The Transformer receives candidate metadata plus the complete musical request and decides track membership, order, title, and explanations. It does not merely annotate a preselected playlist.
- Backend code rejects unknown/duplicate IDs, duration overflow, duration more than 60 seconds below the best feasible candidate fit, and nonmonotonic requested energy progression. Titles, artists and durations come from the catalog, never from generated text.
- If an otherwise valid model proposal violates duration or energy order, deterministic repair may fit only the model-proposed tracks. It never introduces baseline candidates, and the result is labelled `llm_repaired`. Unknown/duplicate IDs remain rejected. Transition explanations may refer to a prior ordering.
- At most two provider requests occur per generation (transport/SDK retries disabled). Access, quota and network failures fall back immediately; invalid output gets one correction attempt. Failure uses a clearly labelled deterministic fallback. Natural-language instruction satisfaction remains a soft goal for later human evaluation.

## Milestones

0. Environment, configuration and tests.
1. Text-based vertical slice and licensed real catalog (current); verify one authorized live model call next.
2. Audio upload/validation and librosa signal features (current); pretrained mood model next.
3. Match audio and catalog feature definitions and normalization.
4. Improve prompts and planning using observed failures.
5. Integrate and polish the complete audio flow.
6. Held-out baseline versus LLM evaluation with measured results.
7. Documentation and presentation preparation; no presentation file yet.

## Evaluation plan

Record requested constraints, selected IDs, actual planner (including fallback), latency and source catalog version for each trial. Split development prompts from held-out tests. Report raw LLM validity before fallback separately from final output validity. Duration error is absolute target-minus-actual seconds. Mood consistency is a label-based proxy, not emotional ground truth. Diversity can use distinct artists/genres; instruction satisfaction and relevance require a documented human rubric and preferably blinded reviews. Do not report fallback runs as successful GenAI outputs. Test general behavior beyond the narrow task. The dashboard currently contains no experimental results.

## Team ownership

For five students: audio processing; catalog/retrieval; GenAI planning; backend/frontend; evaluation/documentation. For four, combine application integration and documentation. Agree shared Pydantic contracts before parallel implementation.

## Rebuild the real catalog

Download the official 342 MiB FMA metadata ZIP into the ignored `tmp/` directory, then run:

```powershell
curl.exe -L --fail -o tmp/fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
.\.venv\Scripts\python.exe scripts/import_fma.py tmp/fma_metadata.zip
```

The importer verifies the official SHA-1 before parsing and writes only the selected metadata and manifest. No audio archive is required. Records are selected round-robin by top-level genre from ascending FMA IDs after filtering for complete attributes. This is a development catalog, not a representative random sample or a held-out evaluation set. Echo Nest energy must not be directly equated with uploaded audio RMS. Mood preferences reach the LLM, but cannot currently be scored against mood ground truth.

## Troubleshooting

- Backend unreachable: start Uvicorn at the project root on port 8000.
- Port already used: choose another port and set `MOODMIX_API_URL` in the frontend terminal.
- No matching genre: choose `any` or a genre present in the catalog.
- LLM fallback: check local `.env`, account/model access and network. Errors shown to users intentionally omit provider details to protect secrets.
- Duration shortfall: the catalog is small and tracks cannot repeat. Expand the sourced catalog rather than inventing tracks.

Repository: https://github.com/Jerome397/moodmix-ai . Tested development milestones are committed and pushed with the project owner's authorization. Secrets, environments, model weights and private audio stay excluded.
