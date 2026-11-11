# Development evaluation

Run `python scripts/evaluate.py` for six offline baseline cases. Add `--live` for paired baseline/LLM cases using the configured provider (at most twelve provider calls). The dashboard reads `evaluation/results.json`; it never calls Gemini just to display results. Use `--output` to preserve separate runs. Offline execution replaces the default report with baseline-only measurements.

The fixed development-v1 suite covers jazz, electronic, rock, pop and unrestricted requests at 5–30 minutes, with steady/rising/falling energy. These prompts are public development cases, not a held-out test set. Catalog SHA-256, exact requests, provider configuration without secrets, selected tracks, diagnostics, UTC timestamp and measured latency are recorded. Latency includes local retrieval/planning/validation; methods run sequentially, with no repeated-trial uncertainty estimate.

The first paired run produced 12 validated final playlists: six baseline, five successful LLM choices and one LLM attempt falling back after HTTP 503. All six hosted attempts made one provider call. Preserve this failure rather than rerunning until the report appears perfect. Final validity includes fallback and benefits from deterministic construction and validation; it is not raw model validity. Diagnostics retain invalid-decision counts separately from provider failures.

Duration error is cap minus actual duration. Artist diversity is distinct artists divided by track count. Genre count measures variety. Energy/BPM distances are mean absolute distances from numeric targets; they are preference proxies, not musical-quality scores. Larger variety is not always desirable. The same retrieved candidate set and numeric constraints are used for each pair. Model choices are bounded by the feasible alternative space. No results establish that Gemini is a better recommender.

## Human review protocol for the next study

Freeze a new prompt set before tuning against its results. Use different listeners from the developers when practical, and randomize baseline/LLM presentation order. Hide method labels and generated explanations during track-list relevance review; retain a private mapping to restore labels after collection. Metadata-only review cannot establish listening quality or transition smoothness.

Collect independent 1–5 ratings for instruction satisfaction (1: contradicts the request; 3: partly satisfies it; 5: fully satisfies supported preferences) and metadata relevance (1: poor fit; 3: mixed; 5: strong fit). Record case ID, anonymous reviewer ID, displayed option ID and a short reason. Allow “cannot assess” rather than forcing ratings. Evaluate explanation grounding separately against catalog facts, with method labels still hidden where possible. Do not invent mood truth for this catalog.

Report reviewer count, missing ratings, paired differences and disagreement. Keep fallback trials identified when revealing results. Obtain lawful recordings and a listening protocol before claiming auditory quality. Human ratings remain uncollected in the current report.
