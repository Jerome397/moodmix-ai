# From uploaded audio to a playlist

**Use this audio for a playlist** fills the visible form from the new analysis, replacing values from a previous visit: supported BPM (40–220), strongest mood attribute scoring at least 0.5 (sad → melancholic, aggressive → energetic), and the strongest genre when its explicit mapping exists in the catalog. Missing mood cues and unsupported genres select `any`. These are editable starting preferences. Duration, catalog energy and progression remain user choices. Tempo matching uses the exact uploaded BPM in the backend; the slider displays its rounded value. Mood cues inform LLM planning because catalog moods are unannotated.

## User workflow

1. On **Music Analysis**, upload an MP3/WAV excerpt and click **Analyze audio**. Local mood/genre inference is optional; 15–30 seconds is recommended when enabled.
2. Click **Use this audio for a playlist**. This opens Playlist Generator with an analysis summary attached. Silence or a recording without usable tempo/neural cues cannot be attached.
3. Keep or disable **Match upload tempo** and **Prefer upload genre cues**. Set desired duration, genre, energy progression and instructions. The energy slider is a separate user preference.
4. Choose baseline or LLM and generate. **How your upload influenced this playlist** reports the effective BPM, compared cues and limitations. The JSON export includes the same retrieval context.
5. Disable or clear the reference to return to manual generation. Selecting a different recording or analysis option clears the previous reference so an unrelated upload is not reused silently.

## Feature compatibility

This milestone compares only quantities supported on both sides:

- **BPM:** optional upload estimate replaces the manual target, retaining the original `(BPM difference / 180)²` scale. BPM is comparable as a unit, but the upload and catalog estimators differ and may have octave errors. Estimates outside 40–220 BPM leave the manual target active, with an explanation; they are not clamped.
- **Genre cues:** a soft ranking term `0.15 × (1 − matching genre score)` is added when compatible cues exist. Raw genre model scores remain uncalibrated; 0.15 is an engineering default, not an experimentally optimized weight. Scores are not renormalized after unsupported classes are omitted.
- **Taxonomy mapping:** blues, electronic, jazz, pop and rock match directly. `folk / country` maps coarsely to catalog `folk`; `rap / hip-hop` maps to `hip-hop`. Alternative and funk/soul/R&B have no assumed mapping. Only mappings available in the actual catalog are applied. This is a coarse heuristic, not evidence that the two datasets share identical genre definitions.
- **User genre filter:** remains a hard filter even when it conflicts with estimated genre. Audio genre preference affects ranking within the allowed tracks.
- **Mood:** the real catalog has no validated mood annotations, so neural mood scores become soft LLM planning context. They do not establish track-level mood matches. A desired mood explicitly selected by the user takes priority; following uploaded mood cues is offered only when cues exist. The baseline does not infer missing catalog moods.
- **RMS/spectral descriptors:** not compared with Echo Nest energy or used to invent missing acoustic features. Catalog energy remains controlled by the independent user slider. Learned audio embeddings are not compared because catalog embeddings are unavailable.

Retrieval returns up to 60 candidates using feature distance. The baseline still prioritizes duration coverage with similarity-ranked tie breaking, so improved candidate ranking does not guarantee improved playlist relevance. Gemini chooses among constraint-valid alternative mixes using the same strategy as text-only generation. No performance improvement claim is made without held-out evaluation.

Catalog entries explicitly titled `interview` are excluded from music recommendations while preserved in the sourced metadata. This conservative filter was added after a live check surfaced an interview segment; it is not a comprehensive speech detector or a complete catalog audit.

## Data flow and privacy

`PlaylistRequest.audio_reference` accepts only estimated BPM, excerpt duration, silence status, neural status, and bounded labelled mood/genre scores. Scores must be in [0,1], labels must be unique, and cues require a successful neural result. Raw recording bytes, waveform arrays, filenames, RMS and spectral data are not fields in this contract and are rejected if supplied.

The summary is retained in the Streamlit session and sent to the backend with the generation request. In LLM mode it also reaches the configured provider alongside preferences and candidate metadata. The recording and waveform remain local. This endpoint accepts client-supplied analysis summaries; it does not cryptographically certify that an external caller ran the analysis pipeline.

The response's `retrieval_context` records whether a reference was attached, the effective tempo target, compatible genre affinities, cues compared and warnings. Current outcomes are `baseline`, `llm` and `fallback`; `llm_repaired` describes historical free-form generation.

## Verification

Automated tests check ranking changes, explicit overrides, unsupported mappings, out-of-range tempo, invalid/duplicate scores, rejection of waveform/RMS fields, silent references, end-to-end upload/analysis/retrieval, Streamlit navigation and detachment, and the exact summary delivered to a mocked LLM. Live smoke checks use the previously licensed Vibe Ace example. Neither smoke results nor synthetic fixtures constitute a recommendation-quality benchmark.

The initial live baseline UI check succeeded. Subsequent live Gemini requests received the summary but failed provider or duration validation and returned labelled baseline fallback; those attempts are retained in `evaluation/audio-pipeline-smoke.json`. Candidate IDs are now also enumerated in Gemini's output schema. Those records describe the earlier free-form strategy. The later feasible-option strategy passed a targeted live audio-guided case; see `evaluation/planner-live-checks.json` and `planner-reliability.md`.
