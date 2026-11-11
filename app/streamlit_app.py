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
    st.info(f"Current catalog: {catalog['catalog_kind']}. Audio upload and model-based mood understanding come next.")
    st.write("Choose Playlist Generator to explore real track metadata, published audio features, and source links.")
elif page == "Music Analysis":
    st.info("Audio upload and pretrained audio understanding are scheduled for Phase 2.")
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
