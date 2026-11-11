# MoodMix AI

Phase 1: a local text-to-playlist prototype with content-based retrieval, an optional OpenAI Transformer planner, backend validation, and a Streamlit interface.

**Current catalog is entirely synthetic development data.** Track names, artists, duration, BPM, energy, genre and mood are invented fixtures, not real music metadata or measured features. No audio playback or real-track evaluation is supported yet. No GenAI superiority claims have been established.

## Windows quick start

Python 3.13.7 was found locally. The text prototype uses this interpreter. Reassess audio-model compatibility before adding the Phase 2 dependencies; use Python 3.11/3.12 if the selected model requires it.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

For the exact versions verified in the initial Windows setup, install from `requirements-lock.txt` instead of `requirements.txt`. The verified suite has 12 tests; live baseline generation and both server health endpoints were also checked. Hosted inference and real audio remain unverified.

In a second terminal at the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Visit http://localhost:8501. API documentation: http://127.0.0.1:8000/docs. Activation is optional, so PowerShell execution-policy changes are unnecessary. VS Code's default Python interpreter points to `.venv`; if you selected another interpreter previously, use **Python: Select Interpreter** to choose `.venv\Scripts\python.exe`.

## Optional LLM mode

Copy `.env.example` to `.env`, then privately set `OPENAI_API_KEY` and `OPENAI_MODEL` to a model available in your account supporting structured outputs. Restart the backend. Never paste keys into chat or commit them. Selecting LLM mode in the application initiates potentially billed API calls; baseline mode is entirely local. No hosted request is made during automated tests.

The SDK uses Pydantic structured output via `responses.parse`: https://developers.openai.com/api/docs/guides/structured-outputs . Model access and billing must be checked in your own account. Credentials are not supplied by this project.

## How it works

Streamlit → FastAPI → catalog validation → feature-distance retrieval → baseline or LLM planner → constraint validation → canonical metadata response.

- Retrieval ranks up to 60 candidates by squared energy distance, squared scaled BPM distance, and a categorical mood mismatch penalty. Genre is an exact filter; mood is a soft preference.
- The baseline uses a 0/1 duration knapsack, preferring earlier similarity-ranked candidates on ties, then sorts energy when requested. Duration coverage is its primary optimization; it is a simple comparison baseline, not an optimal music recommender. Free text is not interpreted by the baseline.
- The Transformer receives candidate metadata plus the complete musical request and decides track membership, order, title, and explanations. It does not merely annotate a preselected playlist.
- Backend code rejects unknown/duplicate IDs, duration overflow, duration more than 60 seconds below the best feasible candidate fit, and nonmonotonic requested energy progression. Titles, artists and durations come from the catalog, never from generated text.
- At most two SDK requests occur per generation (SDK retries disabled). Failure uses a clearly labelled deterministic fallback. Natural-language instruction satisfaction remains a soft goal for later human evaluation.

## Milestones

0. Environment, configuration and tests.
1. Text-based vertical slice (current), then source a licensed real catalog and verify one authorized live model call.
2. Audio validation, librosa features, and a suitable pretrained audio model.
3. Match audio and catalog feature definitions and normalization.
4. Improve prompts and planning using observed failures.
5. Integrate and polish the complete audio flow.
6. Held-out baseline versus LLM evaluation with measured results.
7. Documentation and presentation preparation; no presentation file yet.

## Evaluation plan

Record requested constraints, selected IDs, actual planner (including fallback), latency and source catalog version for each trial. Split development prompts from held-out tests. Report raw LLM validity before fallback separately from final output validity. Duration error is absolute target-minus-actual seconds. Mood consistency is a label-based proxy, not emotional ground truth. Diversity can use distinct artists/genres; instruction satisfaction and relevance require a documented human rubric and preferably blinded reviews. Do not report fallback runs as successful GenAI outputs. Test general behavior beyond the narrow task. The dashboard currently contains no experimental results.

## Team ownership

For five students: audio processing; catalog/retrieval; GenAI planning; backend/frontend; evaluation/documentation. For four, combine application integration and documentation. Agree shared Pydantic contracts before parallel implementation.

## Troubleshooting

- Backend unreachable: start Uvicorn at the project root on port 8000.
- Port already used: choose another port and set `MOODMIX_API_URL` in the frontend terminal.
- No matching genre: choose `any` or a genre present in the catalog.
- LLM fallback: check local `.env`, account/model access and network. Errors shown to users intentionally omit provider details to protect secrets.
- Duration shortfall: the catalog is small and tracks cannot repeat. Expand the sourced catalog rather than inventing tracks.

Repository: https://github.com/Jerome397/moodmix-ai . Tested development milestones are committed and pushed with the project owner's authorization. Secrets, environments, model weights and private audio stay excluded.
