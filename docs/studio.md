# MoodMix Studio: separate conversational agent

The original application remains on port 8501 with its existing code and one-time musical-instructions field. The new app is launched separately:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/agent_app.py --server.address 127.0.0.1 --server.port 8502
```

Studio reads the existing private GEMINI_API_KEY and GEMINI_MODEL from .env. It needs no new credentials. It does not modify the original server, planner, evaluation records or UI. Conversation/session data stays in Streamlit memory until restart/new session, with optional JSON export. A previous structured plan and selected track metadata are sent with each refinement; raw recordings are never sent. Public catalog metadata and submitted chat/preferences go to Gemini. Each message uses at most five provider requests without automatic retries, and the existing project's quota/billing settings apply.

The first structured generation creates preferences, summary and phase budgets. Phase minutes must sum to the requested cap. A deterministic retrieval and valid baseline seed supply initial evidence. Gemini then selects actions from a strict JSON action protocol: search_catalog, validate_playlist, finish. This is application-dispatched tool use, not a claim of native Gemini function calling. Searches may vary energy/tempo ranking while preserving original genre and duration constraints. Gemini can propose any ordered subset of the returned canonical IDs, unlike the original eight-option planner. Validation enforces real IDs, no duplicates, duration fit and requested monotonic energy. Validation feedback enables bounded correction. Finish requires a validated proposal; the final critique is shown. If the budget expires, the app discloses incomplete critique. No valid proposal produces an explicit baseline fallback. Provider/response errors preserve the last successful session result.

Phases are soft intended journeys. Exact per-phase track allocation, listening quality, mood agreement and superior recommendation quality have not been established. Catalog recordings are not hosted. Existing evaluation results apply to the original planner, not Studio; separate evaluation is required for this experimental agent.

The fit_playlist tool explicitly fits the agent's proposed priority order using the existing duration solver, filling from remaining candidates. Its membership and generic tool notes can differ from the proposal; the agent sees and critiques the actual fitted result. Three iterative live development trials are retained in evaluation/studio-live-checks.json: two earlier free-form runs fell back; the final tool-enabled run completed in four calls with a validated 600-second playlist. These are integration checks, not comparative quality evidence.

