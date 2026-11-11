# Architecture and course connections

## Phase 1 contracts

`PlaylistRequest` captures validated explicit preferences plus free text. `Track` holds canonical metadata and provenance. `Plan` contains only IDs and explanations, so model-written titles or artist names cannot overwrite catalog metadata. FastAPI hydrates IDs after semantic validation.

The current retrieval distance is `energy_difference² + (BPM_difference / 180)² + 0.25 * mood_mismatch`. This fixed scaling is documented for development only. Phase 3 must compare consistently extracted features from catalog audio and uploads; dataset-specific energy scales must not be silently mixed.

## GenAI course connections

- Neural networks: explain learned audio representations once the audio model is chosen, distinguishing predictions from signal measurements.
- Transformers: a pretrained language model conditions on instructions and retrieved candidates to generate a structured sequence of playlist decisions. This is inference, not training or fine-tuning.
- Evaluation: choose multiple criteria tied to user goals, maintain separate development and held-out requests, evaluate human judges, and record failures before fallback.
- Retrieval versus fine-tuning: supplying catalog entries at inference supplies information without changing weights. Our current feature retrieval is deterministic; it is not a learned embedding system.
- Optional adaptation: consider SFT plus LoRA only after baseline comparisons identify a behavior gap and suitable training examples exist.
- Diffusion is outside the initial scope because this project arranges music rather than synthesizing original audio.

These are project design choices informed by the course topics, not claimed instructor requirements.

## Current boundaries

A genuine 240-track FMA metadata catalog is available with published Echo Nest attributes and CC BY 4.0 attribution. No audio analysis, pretrained audio classifier, experimental comparison, or live provider verification is complete. Synthetic fixtures test software behavior only. Tests substituting a fake SDK or HTTP transport verify retries and validation, not Transformer quality or provider access.

The real catalog has unknown mood; no mood penalty is applied during baseline retrieval. The UI obtains genre options and provenance from the backend health response. Metadata is historical; source links refer to the dataset when a track-specific URL is unavailable. No recordings are redistributed.

LLM network work runs in a synchronous FastAPI handler, with a 30-second provider timeout and up to two requests. It is appropriate for a local student prototype, not a production multi-user workload. The browser never receives API keys.

## Provider adapter

Gemini is the default. It uses the existing HTTP client with a key in the `x-goog-api-key` header, not URL parameters. A JSON schema derived from `Plan` constrains response structure; backend validation independently enforces music constraints. Incomplete/blocked responses and invalid plans never count as successful LLM generation. Only two calls are allowed and quotas/access failures immediately trigger a labelled baseline fallback. No free-text provider errors or response bodies are exposed. OpenAI remains an explicit alternative, never an automatic fallback.
