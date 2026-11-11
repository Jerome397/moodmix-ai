# Constraint-safe LLM playlist selection

## Problem and resulting behavior

Earlier free-form generation asked the LLM to select track IDs while calculating an aggregate duration. Recorded audio-guided requests exceeded the cap or failed minimum coverage. Trimming could not always repair the proposed subset; the application correctly fell back to its baseline. `evaluation/audio-pipeline-smoke.json` preserves those earlier failures.

The current `feasible_options_v1` strategy separates constraint construction from semantic selection:

1. Retrieval produces the same candidate pool for baseline and LLM workflows.
2. A deterministic 0/1 duration solver constructs up to eight distinct, validated ordered playlists. The first is the original baseline.
3. Gemini sees their actual track IDs, duration, energy sequence, mean tempo, artist count and genre set, plus track metadata and user/audio preferences.
4. The LLM chooses one option, generates its title and rationale, and supplies track-specific notes. Its chosen option determines actual membership and order.
5. The backend materializes exactly that option, hydrates canonical metadata, and revalidates hard constraints. It never silently changes the chosen option to another one.

The choice is bounded: the LLM cannot invent an arbitrary new playlist outside the offered alternatives. This trades open-ended combination generation for reliable constraint handling. Semantic interpretation and selection remain model decisions; deterministic code does the arithmetic.

## How alternatives are constructed

Candidate orderings explore the original/reversed similarity ranking, lower/higher energy, tempo proximity, and interleaved artists/genres. For each ordering the knapsack retains one combination for each reachable duration. Different reachable totals within the **existing** 60-second tolerance below the best feasible duration are explored. The hard cap, unique IDs, genre filter and monotonic requested energy progression are unchanged.

Actual ordered sequences are deduplicated. A summary describes measured properties of each resulting sequence; it does not assume that a seed ordering guarantees better diversity or relevance. Very small or constrained catalogs can offer only one valid option. That case is explicitly disclosed because the LLM then cannot change track selection. Diagnostics separately count options and distinct membership sets, since different orders can share the same tracks.

## Structured decision and validation

`PlaylistDecision` contains `option_id`, `title`, `rationale`, and `track_reasons`. Gemini's schema enumerates offered option IDs and track-note IDs. The backend also rejects unknown options, notes for tracks outside the selected option, duplicate note IDs, and malformed/overlong strings. Missing individual notes use a clearly labelled playlist-level rationale and a warning; they do not change the music selection or invent track-specific facts.

The default planner never trims or reorders a chosen option. The earlier `repair_constraints` helper and `llm_repaired` records remain historical; new outcomes are `llm` for a validated option decision, `fallback` for an unsuccessful provider/decision, and `baseline` for manual baseline mode.

There are at most two provider calls. An invalid decision gets one correction attempt; provider/access/quota/network errors immediately return labelled fallback. Raw provider response bodies and API keys are not logged or exposed. Candidate/model context remains the same small metadata/analysis summary; recordings and waveforms stay local.

## Auditable results

`planning_diagnostics` reports strategy, provider calls, number of alternatives, distinct track sets, selected option, rationale, raw decision-validation failures, fallback reason, and whether selection/membership changed from the baseline. The frontend displays the rationale and a concise planning explanation.

Run offline construction checks without external requests:

```powershell
.\.venv\Scripts\python.exe scripts/check_planner.py
```

Explicit live verification uses your configured provider and its quota/billing terms:

```powershell
.\.venv\Scripts\python.exe scripts/check_planner.py --live --output evaluation/planner-live-checks.json
```

The recorded three development cases succeeded as `llm` with one provider call each and no raw validation failures: audio-guided jazz 577/600 seconds, electronic 599/600 seconds, and varied music 900/900 seconds. Each had eight distinct membership alternatives. Jazz selected different tracks from the baseline; the other two selected the baseline option.

These are targeted integration checks, not a statistically representative held-out evaluation. Constraint satisfaction here primarily demonstrates deterministic guardrails, not the LLM's arithmetic skill. Choosing a different mix does not establish better relevance, diversity or instruction satisfaction. The next evaluation should measure those properties with explicit criteria and blinded human reviews, and distinguish selecting the baseline option from changing it.
