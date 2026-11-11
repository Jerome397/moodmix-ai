## Development evaluation dashboard — 2026-11-11

Added six reproducible paired cases, catalog fingerprint, numeric preference/diversity/latency measurements and Streamlit results inspection. Live run: 5/6 LLM successes, one HTTP 503 fallback; all 12 final outputs validated. Human ratings remain uncollected; documented a blinded review protocol. Full suite: 83 passed in 16.17 seconds.

# Initial setup verification

## Constraint-safe LLM planning: 2026-11-11

- Replaced free-form duration arithmetic with up to eight distinct, prevalidated mix options. The model's decision changes actual membership/order; diagnostics separately report selection and membership changes relative to the baseline.
- Added strict option/track-note validation, provider-call counts, raw decision failure counts, rationale display and reproducible offline/live development checks. Original hard constraints and the existing 60-second coverage tolerance are unchanged.
- **81 offline tests passed in 18.36 seconds**, including constraint coverage at multiple durations/progressions, real membership changes, invalid decisions, retries, provider failures and UI rationale display.
- Three targeted live Gemini cases succeeded with one call each, no raw validation failures and no repair/fallback: audio-guided jazz 577/600 seconds, electronic 599/600 seconds, and mixed-style 900/900 seconds. Each offered eight distinct track sets. One model choice changed membership from the baseline; two chose the baseline option.
- Records: `evaluation/planner-live-checks.json`. These development checks establish integration success for these cases, not general reliability, model arithmetic ability or better music recommendations. Blinded/held-out quality evaluation remains next.

## Audio-to-playlist integration: 2026-11-11

- Added a typed, bounded audio-reference summary, optional BPM targeting and soft genre-cue ranking, while keeping RMS and catalog energy separate.
- Music Analysis can open Playlist Generator with its current analysis attached. Reference controls allow manual override, detachment and clearing. New uploads invalidate prior references.
- Mood/genre estimates travel to the LLM as uncalibrated context; raw audio, waveforms and filenames do not.
- **68 offline tests passed in 41.84 seconds**, including real endpoint and Streamlit upload-to-playlist flow, cue ranking, privacy/validation and mocked LLM context propagation. No recommendation-quality improvement is claimed.
- Live upload-to-analysis-to-baseline UI check passed with the licensed music example; source BPM became the retrieval target. Gemini received the summary in live requests but returned provider/invalid-duration failures and labelled baseline fallback. These attempts are preserved in `evaluation/audio-pipeline-smoke.json`; no new successful audio-guided LLM plan is claimed.
- Added candidate-ID enums to Gemini output schemas and excluded catalog records explicitly titled `interview` from recommendations. Final live baseline and frontend health checks passed.

## Local neural music milestone: 2026-11-11

- Downloaded and checksum-pinned official MTG MusiCNN encoder and four mood plus one genre head, running locally through CPU ONNX Runtime. Weights are ignored by Git; setup script and manifest are included.
- Added optional mood/genre inference to the upload endpoint and UI; independent mood scores and top genre alternatives are explicitly uncalibrated estimates. Missing files, silence and short excerpts retain usable signal results.
- **54 offline tests passed in 10.95 seconds** on final verification; dependency check passed.
- Official models also ran on the first 30 seconds of the licensed Kevin MacLeod/librosa Vibe Ace example; output and measured processing time are in `evaluation/audio-model-smoke.json`. This is an integration check, not an accuracy benchmark.
- No Gemini calls, billing changes or uploads to external inference services occurred. Upload-to-recommendation integration remains upcoming.
- Live Streamlit -> HTTP backend -> official local ONNX weights verification passed on the licensed 30-second music excerpt, returning four mood scores and nine genre scores. Model status reports installed; frontend health passed.

## Audio signal milestone: 2026-11-11

- Added local MP3/WAV upload endpoint and Streamlit playback/analysis page, with input size, decoded format, sample rate, channel and duration validation.
- Measures tempo, beat count, RMS/dBFS, peak and spectral descriptors; displays waveform/RMS envelopes and exports JSON. Silence and phase-cancelling stereo have explicit handling.
- Installed librosa 0.11.0 and SoundFile 0.13.1; dependency consistency check passed and lockfile updated.
- **48 offline tests passed in 48.95 seconds**, including 440 Hz/RMS checks, 120 BPM pulses, stereo resampling, MP3 roundtrip, invalid/non-finite audio, endpoint validation and Streamlit form-to-backend analysis. The initial run includes compiled routine initialization.
- Live verification also passed: Streamlit analysis form called the running HTTP endpoint with a generated WAV; measured RMS 0.14142 and centroid 447.96 Hz for a 440 Hz reference tone. MP3 HTTP analysis and Streamlit health passed. No hosted API call occurred.
- Neural mood/genre prediction and upload-driven retrieval remain upcoming. No recordings were sent to Gemini, and no private audio is checked into Git.

## Live Gemini verification: 2026-11-11

- Private key detected without displaying it. Read-only model discovery confirmed access to the configured Gemini model.
- Earlier live attempts exposed request-schema incompatibility and duration-planning failures. The wire schema now inlines references and leaves full length limits to local Pydantic validation.
- Retrieval excludes individually overlong tracks. Correction requests include failed plan, duration total and validation feedback.
- Constraint repair fits only LLM-proposed tracks, cannot repair invented/duplicate IDs, and is explicitly labelled `llm_repaired`.
- Live electronic rising-energy request returned a validated 556-second playlist for a 600-second cap, with Gemini-generated title and explanations, using constraint repair. Recorded in `evaluation/gemini-live-smoke.json`. This is one smoke check, not a benchmark or a billing-tier verification.
- **33 offline regression tests passed in 5.51 seconds.** The private key remains ignored by Git. Google billing settings were not changed.

Earlier milestone entries below are historical.

## Gemini adapter milestone: verified 2026-11-11

- Gemini is the default provider; Google AI Studio free-tier setup documented in `docs/gemini-setup.md`.
- Uses existing HTTP dependency, JSON-schema structured output and Pydantic/backend constraints. API key is sent in a header and never returned by the status endpoint.
- **28 offline tests passed in 5.44 seconds**, including Gemini success, incomplete output, invented IDs, invalid JSON, quota/access/model failures, retry bounds and secret-safe error messages.
- No live Google call or account billing verification has been performed. Private key setup remains required. Configured readiness does not prove live success or free-tier eligibility.

## Real catalog milestone: verified 2026-11-11

- Official FMA metadata archive downloaded; SHA-1 matches `f0df49ffe5f2a6008d7dc83c6915b31835dfe733`.
- Imported 240 real tracks across 12 top-level genres with published Echo Nest tempo/energy, metadata attribution, and a reproducible selection manifest.
- Full automated suite: **15 passed in 4.25 seconds**; dependency check passed.
- Live Streamlit form exercised against the actual HTTP backend after replacing stale server processes: four real tracks, 900 seconds, baseline mode. Backend reports 240 tracks; Streamlit health returns `ok`.
- No audio, mood classifier, hosted LLM call, or comparative model-quality experiment is claimed. Mood is unknown in the current catalog.

## Resumed setup: verified 2026-11-11

- Dependency installation succeeded using `--no-cache-dir`; `pip check` reports no broken requirements.
- Exact installed versions recorded in `requirements-lock.txt`.
- Full automated suite: **12 passed in 4.32 seconds**, including a Streamlit form-to-backend integration test. The UI test uses an absolute entrypoint path and a 20-second startup timeout.
- Live FastAPI health endpoint returned `ok` and 24 synthetic catalog tracks.
- Live playlist request returned four tracks totalling 900 seconds for a 15-minute rising-energy request, using baseline mode.
- Live Streamlit health endpoint returned `ok` on port 8501.
- Servers started in hidden background processes. Logs are in `tmp/backend.*.log` and `tmp/frontend.*.log`.
- Hosted LLM calls remain unverified and no paid inference was performed. Real catalog sourcing and audio analysis remain future milestones.

The entries below describe the earlier disk-space blocker, which is now resolved.

- Workspace inspected before authoring: no source project existed.
- Python 3.13.7 and Git 2.53.0.windows.1 detected.
- Local `.venv` created, VS Code interpreter path configured, Git initialized.
- Source compilation completed successfully.
- Dependency installation failed with `OSError: [Errno 28] No space left on device`. C: had approximately 40 MB free at inspection. Installation is incomplete; `.venv` contains only pip.
- Full pytest run attempted and blocked: `No module named pytest`.
- Bundled-runtime checks exercise catalog/schema/recommendation logic only. They do not establish backend, UI, or provider integration success.
- No paid inference, API credential setup, commit, push, or remote service change occurred.

## Resume setup

Free at least 1-2 GB of space on C: (more will be needed for audio models), then run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

After tests pass, launch the backend and frontend using the README commands. Record exact resolved dependency versions after successful installation. The authored tests include offline provider substitutes; a live model test requires private credentials and authorization to incur API charges.
