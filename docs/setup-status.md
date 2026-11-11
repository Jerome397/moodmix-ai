# Initial setup verification

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
