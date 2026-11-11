"""Local Streamlit client; start FastAPI separately."""
import os
import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MoodMix AI", page_icon="🎵", layout="wide")
st.title("🎵 MoodMix AI")
st.caption("From your musical intent to an explainable playlist")
page = st.sidebar.radio("Explore", ["Home", "Music Analysis", "Playlist Generator", "Evaluation"])
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
elif page == "Evaluation":
    st.info("No comparative experiments have been recorded yet. No model-quality claims are available.")
else:
    if not catalog["mood_available"]:
        st.info("This catalog has no annotated moods. Mood is passed to the LLM as a preference; baseline retrieval uses tempo, energy and genre.")
    with st.form("preferences"):
        left, right = st.columns(2)
        mood = left.selectbox("Mood", ["relaxed", "happy", "energetic", "melancholic"])
        duration = right.slider("Duration cap (minutes)", 3, 60, 15)
        genre = left.selectbox("Genre", ["any", *catalog["genres"]])
        progression = right.selectbox("Energy progression", ["steady", "rising", "falling"])
        energy = left.slider("Target energy (published catalog scale)", 0.0, 1.0, 0.5)
        left.caption("Echo Nest energy is a model-derived attribute, not RMS loudness or a mood confidence score.")
        bpm = right.slider("Target BPM", 40, 220, 110)
        instructions = st.text_area("Musical instructions", placeholder="Start relaxed, then build toward a lively finish.")
        mode = st.radio("Planner", ["baseline", "llm"], horizontal=True)
        st.caption("LLM mode sends preferences and candidate metadata to the configured provider (Gemini by default). Keep the Google project on the free tier; quotas apply. Baseline uses explicit controls and does not interpret free text.")
        submitted = st.form_submit_button("Generate playlist", type="primary")
    if submitted:
        try:
            with st.spinner("Planning your mix…"):
                response = httpx.post(os.getenv("MOODMIX_API_URL", "http://127.0.0.1:8000") + "/playlist", json=dict(mood=mood, duration_minutes=duration, genre=genre, progression=progression, energy=energy, bpm=bpm, instructions=instructions, mode=mode), timeout=75)
                response.raise_for_status()
                st.session_state["playlist"] = response.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"Backend rejected the request: {exc.response.json().get('detail', 'Unknown error')}")
        except httpx.RequestError:
            st.error("Cannot reach the backend. Start FastAPI on port 8000 and try again.")
    if result := st.session_state.get("playlist"):
        st.subheader(result["title"])
        st.caption(f"Actual planner: {result['method']} · {result['catalog_kind']}")
        for warning in result["warnings"]:
            st.warning(warning)
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
