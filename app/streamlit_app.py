"""Local Streamlit client; start FastAPI separately."""
import os
import httpx
import pandas as pd
import streamlit as st


def use_analyzed_audio():
    """Carry only the small analysis summary to the playlist page."""
    analysis = st.session_state["audio_analysis"]
    prediction = analysis.get("music_model") or {}
    successful = prediction.get("status") == "ok"
    bpm = analysis.get("bpm")
    st.session_state["audio_reference"] = {
        "bpm": bpm if bpm is not None and 0 < bpm <= 400 else None,
        "duration_seconds": analysis["duration_seconds"], "is_silent": analysis["is_silent"],
        "neural_status": prediction.get("status", "not_requested"),
        "mood_scores": prediction.get("mood_scores", []) if successful else [],
        "genre_scores": prediction.get("genre_scores", []) if successful else [],
    }
    st.session_state.pop("use_audio_reference", None)
    st.session_state.pop("playlist", None)
    st.session_state["navigation"] = "Playlist Generator"


def clear_audio_reference():
    st.session_state.pop("audio_reference", None)
    st.session_state.pop("playlist", None)


st.set_page_config(page_title="MoodMix AI", page_icon="🎵", layout="wide")
st.title("🎵 MoodMix AI")
st.caption("From your musical intent to an explainable playlist")
page = st.sidebar.radio("Explore", ["Home", "Music Analysis", "Playlist Generator", "Evaluation"], key="navigation")
api_url = os.getenv("MOODMIX_API_URL", "http://127.0.0.1:8000")
try:
    health_response = httpx.get(api_url + "/health", timeout=10)
    health_response.raise_for_status()
    catalog = health_response.json()
except (httpx.HTTPError, ValueError):
    st.error("Cannot load catalog information. Start the FastAPI backend on port 8000.")
    st.stop()
st.sidebar.info(f"{catalog['catalog_tracks']} tracks · {catalog['catalog_kind']}")
st.sidebar.caption("Metadata only: recordings are not hosted in this application.")
try:
    status_response = httpx.get(api_url + "/llm/status", timeout=10)
    status_response.raise_for_status()
    llm = status_response.json()
    st.sidebar.caption(f"LLM: {llm['provider']} · {llm['model']}")
    if not llm["configured"]:
        st.sidebar.warning("Add your private Gemini API key to .env to enable live planning.")
except (httpx.HTTPError, ValueError, KeyError):
    st.sidebar.caption("LLM status unavailable; restart the backend after configuration changes.")
previous = st.session_state.get("playlist")
if previous and previous.get("catalog_kind") != catalog["catalog_kind"]:
    st.session_state.pop("playlist", None)
if page == "Home":
    st.header("A playlist with a plan")
    st.write("Compare content-based recommendations with a Transformer planner that selects and orders catalog tracks from your instructions.")
    st.info(f"Current catalog: {catalog['catalog_kind']}. Upload an excerpt on Music Analysis to inspect measured audio features.")
    st.write("Choose Playlist Generator to explore real track metadata, published audio features, and source links.")
elif page == "Music Analysis":
    st.header("Understand your audio")
    st.caption("MP3/WAV · 3–120 seconds · up to 20 MiB · mono/stereo · 8–96 kHz")
    st.info("Analysis and pretrained mood/genre estimates run locally; no audio is sent to Gemini. Use 15–30 seconds of music for a more useful estimate.")
    upload = st.file_uploader("Choose an audio excerpt", type=["mp3", "wav"])
    include_models = st.checkbox("Estimate mood and genre locally", value=True)
    if upload is not None:
        import hashlib
        raw = upload.getvalue()
        identity = hashlib.sha256(raw + upload.name.encode() + str(include_models).encode()).hexdigest()
        if st.session_state.get("audio_identity") != identity:
            st.session_state.pop("audio_analysis", None)
            if st.session_state.get("audio_reference"):
                clear_audio_reference()
        st.audio(raw, format="audio/mpeg" if upload.name.lower().endswith(".mp3") else "audio/wav")
        if st.button("Analyze audio", type="primary"):
            st.session_state.pop("audio_analysis", None)
            if len(raw) > 20 * 1024 * 1024:
                st.error("Audio must be 20 MiB or smaller.")
            else:
                try:
                    with st.spinner("Measuring audio features… First use may take longer while signal-processing functions initialize."):
                        response = httpx.post(api_url + "/audio/analyze", files={"file": (upload.name, raw, upload.type)}, data={"include_models": str(include_models).lower()}, timeout=180)
                        response.raise_for_status()
                        st.session_state["audio_analysis"] = response.json()
                        st.session_state["audio_identity"] = identity
                except httpx.HTTPStatusError as exc:
                    st.error(f"Audio rejected: {exc.response.json().get('detail', 'Invalid audio')}")
                except httpx.RequestError:
                    st.error("Audio analysis could not finish. Check the backend and try a shorter excerpt.")
        if analysis := st.session_state.get("audio_analysis"):
            a, b, c, d = st.columns(4)
            a.metric("Estimated BPM", f"{analysis['bpm']:.1f}" if analysis["bpm"] is not None else "Unavailable")
            b.metric("RMS amplitude", f"{analysis['rms_amplitude']:.4f}")
            c.metric("RMS level", f"{analysis['rms_dbfs']:.1f} dBFS" if analysis["rms_dbfs"] is not None else "Silence")
            d.metric("Duration", f"{analysis['duration_seconds']:.1f} s")
            st.caption("RMS measures digital signal level, not perceived loudness or the catalog's Echo Nest energy. It is not used as a substitute in recommendations.")
            for warning in analysis["warnings"]:
                st.warning(warning)
            st.subheader("Waveform envelope")
            st.line_chart(pd.DataFrame(analysis["waveform"]).set_index("time_seconds"), x_label="Time (s)", y_label="Amplitude (full scale)")
            st.subheader("Signal level over time")
            st.line_chart(pd.DataFrame(analysis["rms_timeline"]).set_index("time_seconds"), x_label="Time (s)", y_label="RMS amplitude")
            st.dataframe(pd.DataFrame({"Feature": ["Spectral centroid (Hz)", "Spectral bandwidth (Hz)", "85% spectral rolloff (Hz)", "Zero-crossing rate", "Detected beats", "Mood", "Genre"], "Value": [str(round(analysis["spectral_centroid_hz"], 1)), str(round(analysis["spectral_bandwidth_hz"], 1)), str(round(analysis["spectral_rolloff_hz"], 1)), str(round(analysis["zero_crossing_rate"], 4)), str(analysis["beat_count"]), analysis["mood"], analysis["genre"]]}), hide_index=True)
            if prediction := analysis.get("music_model"):
                st.subheader("Pretrained music estimates")
                st.caption(f"{prediction['model']} · {prediction['status']}")
                for warning in prediction["warnings"]:
                    st.warning(warning)
                if prediction["status"] == "ok":
                    st.write(f"**Mood cues:** {prediction['mood']}")
                    st.write(f"**Estimated genre:** {prediction['genre']}")
                    st.bar_chart(pd.DataFrame(prediction["mood_scores"]).set_index("label"), y_label="Positive-class model score (not confidence)")
                    st.dataframe(pd.DataFrame(prediction["genre_scores"][:3]), hide_index=True)
                    st.caption(f"{prediction['patches_analyzed']} sampled ~3-second patches. Mood heads are independent; genre is limited to nine training classes. Scores are not calibrated confidence.")
            st.download_button("Download analysis JSON", data=__import__("json").dumps(analysis, indent=2), file_name="moodmix-analysis.json", mime="application/json")
            usable = not analysis["is_silent"] and (analysis["bpm"] is not None or (analysis.get("music_model") or {}).get("status") == "ok")
            st.button("Use this audio for a playlist", on_click=use_analyzed_audio, disabled=not usable, type="primary")
            if not usable:
                st.caption("Analyze a longer, non-silent music excerpt with usable tempo or neural cues before using it as a playlist reference.")
elif page == "Evaluation":
    import json
    from pathlib import Path
    st.header("Planner evaluation")
    path = Path(__file__).resolve().parents[1] / "evaluation/results.json"
    if not path.exists():
        st.info("Run scripts/evaluate.py to record development measurements.")
    else:
        report = json.loads(path.read_text(encoding="utf-8"))
        st.caption(f"{report['suite']} · {report['created_at_utc']}")
        st.warning("Development cases, not held-out quality evidence. No human relevance or instruction-satisfaction ratings have been collected.")
        rows = report["rows"]
        llm_rows = [r for r in rows if r['requested_mode'] == 'llm']
        a, b, c = st.columns(3)
        a.metric("Validated outputs", f"{sum(r['final_valid'] for r in rows)}/{len(rows)}")
        b.metric("LLM outputs / attempts", f"{sum(r['actual_method'] == 'llm' for r in llm_rows)}/{len(llm_rows)}")
        c.metric("Fallbacks", sum(r['actual_method'] == 'fallback' for r in llm_rows))
        columns = ['case_id', 'requested_mode', 'actual_method', 'duration_error_seconds', 'artist_count', 'genre_count', 'artist_diversity', 'mean_energy_distance', 'mean_bpm_distance', 'latency_seconds']
        frame = pd.DataFrame(rows)[columns]
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption("Energy/BPM distances measure proximity to numeric preferences; artist/genre counts measure variety. None establish perceived musical quality. Final validity includes fallback; LLM success is reported separately.")
        st.subheader("Latency by case")
        st.bar_chart(frame.pivot(index='case_id', columns='requested_mode', values='latency_seconds'), y_label="Seconds")
        selected_case = st.selectbox("Inspect case", list(dict.fromkeys(r['case_id'] for r in rows)))
        for row in rows:
            if row['case_id'] == selected_case:
                with st.expander(f"{row['requested_mode']} → {row['actual_method']}"):
                    st.write(row['request']['instructions'])
                    st.dataframe(pd.DataFrame(row['tracks'])[['title', 'artist', 'genre', 'duration_seconds', 'energy', 'bpm']], hide_index=True)
                    st.json(row['diagnostics'])
        st.download_button("Download evaluation JSON", path.read_text(encoding="utf-8"), "moodmix-evaluation.json", "application/json")
else:
    active_reference = None
    use_audio_tempo = use_audio_genre = False
    if reference := st.session_state.get("audio_reference"):
        if st.checkbox("Use analyzed audio as reference", value=True, key="use_audio_reference"):
            active_reference = reference
            st.success("Your analyzed upload is attached. You can change its influence below.")
            st.caption(f"Upload BPM: {reference['bpm']:.1f}" if reference["bpm"] is not None else "Upload BPM unavailable; use a manual tempo target.")
            a, b = st.columns(2)
            tempo_available = reference["bpm"] is not None and 40 <= reference["bpm"] <= 220
            use_audio_tempo = a.checkbox("Match upload tempo", value=True, disabled=not tempo_available) and tempo_available
            use_audio_genre = b.checkbox("Prefer upload genre cues", value=True, disabled=not reference["genre_scores"]) and bool(reference["genre_scores"])
            st.caption("Genre cues affect ranking softly. Your chosen genre filter and desired mood remain your preferences.")
        st.button("Clear audio reference", on_click=clear_audio_reference)
    if not catalog["mood_available"]:
        st.info("This catalog has no annotated moods. Mood is passed to the LLM as a preference; baseline retrieval uses tempo, energy and genre.")
    with st.form("preferences"):
        left, right = st.columns(2)
        mood_options = (["follow uploaded mood cues"] if active_reference and active_reference["mood_scores"] else []) + ["relaxed", "happy", "energetic", "melancholic"]
        mood = left.selectbox("Mood", mood_options)
        duration = right.slider("Duration cap (minutes)", 3, 60, 15)
        genre = left.selectbox("Genre", ["any", *catalog["genres"]])
        progression = right.selectbox("Energy progression", ["steady", "rising", "falling"])
        energy = left.slider("Target energy (published catalog scale)", 0.0, 1.0, 0.5)
        left.caption("Echo Nest energy is a model-derived attribute, not RMS loudness or a mood confidence score.")
        bpm = right.slider("Target BPM", 40, 220, round(active_reference["bpm"]) if use_audio_tempo else 110, disabled=use_audio_tempo)
        instructions = st.text_area("Musical instructions", placeholder="Start relaxed, then build toward a lively finish.")
        mode = st.radio("Planner", ["baseline", "llm"], horizontal=True)
        st.caption("LLM mode sends preferences, candidate metadata and the attached audio's tempo/model-score summary to the configured provider. Recordings and waveforms stay local. Keep Google on the free tier; quotas apply. Baseline does not interpret free text.")
        submitted = st.form_submit_button("Generate playlist", type="primary")
    if submitted:
        try:
            with st.spinner("Planning your mix…"):
                response = httpx.post(api_url + "/playlist", json=dict(mood="any" if mood == "follow uploaded mood cues" else mood, duration_minutes=duration, genre=genre, progression=progression, energy=energy, bpm=bpm, instructions=instructions, mode=mode, audio_reference=active_reference, use_audio_tempo=use_audio_tempo, use_audio_genre=use_audio_genre), timeout=75)
                response.raise_for_status()
                st.session_state["playlist"] = response.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"Backend rejected the request: {exc.response.json().get('detail', 'Unknown error')}")
        except httpx.RequestError:
            st.error("Cannot reach the backend. Start FastAPI on port 8000 and try again.")
    if result := st.session_state.get("playlist"):
        st.subheader(result["title"])
        st.caption(f"Actual planner: {result['method']} · {result['catalog_kind']}")
        diagnostics = result.get("planning_diagnostics", {})
        if diagnostics.get("outcome") == "llm":
            st.write(diagnostics["rationale"])
            with st.expander("How this mix was planned"):
                provider_name = {"gemini": "Gemini", "openai": "OpenAI"}.get(diagnostics.get("provider"), "The model")
                st.caption(f"{provider_name} chose among {diagnostics['option_count']} duration-checked ordered mixes ({diagnostics['distinct_track_sets']} distinct track sets).")
                st.caption("Track selection " + ("differs from" if diagnostics["membership_changed_from_baseline"] else "matches") + " the content-based baseline for this request.")
        context = result.get("retrieval_context", {})
        for warning in result["warnings"]:
            if warning not in context.get("warnings", []):
                st.warning(warning)
        if context.get("audio_reference_attached"):
            with st.expander("How your upload influenced this playlist", expanded=True):
                st.write(f"Effective tempo target: {context['target_bpm']:.1f} BPM")
                st.write("Compared audio cues: " + (", ".join(context["compared_audio_features"]) or "none; estimates supplied as planning context only"))
                for note in context["warnings"]:
                    st.caption(note)
        a, b, c = st.columns(3)
        a.metric("Duration", f"{result['duration_seconds'] // 60}:{result['duration_seconds'] % 60:02d}")
        b.metric("Below requested cap", f"{result['duration_error_seconds']} s")
        c.metric("Latency", f"{result['latency_seconds']:.2f} s")
        frame = pd.DataFrame(result["tracks"])
        st.dataframe(frame[["title", "artist", "duration_seconds", "genre", "energy", "explanation"]], hide_index=True, width="stretch")
        st.line_chart(frame["energy"])
        with st.expander("Track sources and feature provenance"):
            st.dataframe(frame[["title", "source_url", "source", "metadata_license", "energy_source", "mood_source"]], hide_index=True, width="stretch")
        st.download_button("Download playlist JSON", data=__import__("json").dumps(result, indent=2), file_name="moodmix-playlist.json", mime="application/json")
