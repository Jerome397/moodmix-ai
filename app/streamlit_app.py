"""Local Streamlit client; start FastAPI separately."""
import os
import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MoodMix AI", page_icon="🎵", layout="wide")
st.title("🎵 MoodMix AI")
st.caption("From your musical intent to an explainable playlist")
page = st.sidebar.radio("Explore", ["Home", "Music Analysis", "Playlist Generator", "Evaluation"])
st.sidebar.warning("Phase 1 • synthetic catalog • no playable recordings")
if page == "Home":
    st.header("A playlist with a plan")
    st.write("Compare content-based recommendations with a Transformer planner that selects and orders catalog tracks from your instructions.")
    st.info("This milestone tests the pipeline using clearly fictional tracks. Real music data and audio understanding come next.")
elif page == "Music Analysis":
    st.info("Audio upload and pretrained audio understanding are scheduled for Phase 2.")
elif page == "Evaluation":
    st.info("No comparative experiments have been recorded yet. No model-quality claims are available.")
else:
    with st.form("preferences"):
        left, right = st.columns(2)
        mood = left.selectbox("Mood", ["relaxed", "happy", "energetic", "melancholic"])
        duration = right.slider("Duration cap (minutes)", 3, 60, 15)
        genre = left.selectbox("Genre", ["any", "funk", "jazz", "electronic", "ambient"])
        progression = right.selectbox("Energy progression", ["steady", "rising", "falling"])
        energy = left.slider("Target energy (development scale)", 0.0, 1.0, 0.5)
        bpm = right.slider("Target BPM", 40, 220, 110)
        instructions = st.text_area("Musical instructions", placeholder="Start relaxed, then build toward a lively finish.")
        mode = st.radio("Planner", ["baseline", "llm"], horizontal=True)
        st.caption("LLM mode sends preferences and candidate metadata to OpenAI and may incur API charges. Baseline uses the explicit controls; it does not interpret free text.")
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
        st.caption(f"Actual planner: {result['method']} · synthetic development data")
        for warning in result["warnings"]:
            st.warning(warning)
        a, b, c = st.columns(3)
        a.metric("Duration", f"{result['duration_seconds'] // 60}:{result['duration_seconds'] % 60:02d}")
        b.metric("Below requested cap", f"{result['duration_error_seconds']} s")
        c.metric("Latency", f"{result['latency_seconds']:.2f} s")
        frame = pd.DataFrame(result["tracks"])
        st.dataframe(frame[["title", "artist", "duration_seconds", "genre", "energy", "explanation"]], hide_index=True, width="stretch")
        st.line_chart(frame["energy"])
        st.download_button("Download playlist JSON", data=__import__("json").dumps(result, indent=2), file_name="moodmix-playlist.json", mime="application/json")
