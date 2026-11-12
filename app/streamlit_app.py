"""Local Streamlit client; start FastAPI separately."""
import os
import httpx
import pandas as pd
import streamlit as st
from pathlib import Path
from html import escape
from backend.audio_recommendation import GENRE_TO_CATALOG


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
    st.session_state["apply_audio_preferences"] = True
    st.session_state["match_audio_tempo"] = bpm is not None and 40 <= bpm <= 220
    st.session_state["match_audio_genre"] = successful and bool(prediction.get("genre_scores"))
    st.session_state.pop("playlist", None)
    st.session_state["navigation"] = "Playlist Generator"


def clear_audio_reference():
    st.session_state.pop("audio_reference", None)
    st.session_state.pop("playlist", None)


def open_audio_page():
    st.session_state['navigation'] = 'Music Analysis'


st.set_page_config(page_title="MoodMix AI", page_icon="🎵", layout="wide")
st.html('<style>' + (Path(__file__).parent / "style.css").read_text(encoding="utf-8") + '</style>')
st.title("MoodMix AI")
st.caption("YOUR SOUND. YOUR MOOD. YOUR MIX.")
st.sidebar.markdown("### ◉ MoodMix\nYour personal music lab")
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
    st.html('<div class="hero"><span class="eyebrow">A little science. A lot of sound.</span><h2>Find the next track<br>for your state of mind.</h2><p>Start with a song or a feeling. Explore your audio, shape the energy, and create an explainable mix from real music metadata.</p></div>')
    for col, icon, title, description in zip(st.columns(3), ['01 / LISTEN', '02 / CREATE', '03 / COMPARE'], ['Discover your sound', 'Shape your next mix', 'Explore the results'], ['Upload an MP3 or WAV. We analyze the first 30 seconds locally for tempo, mood and genre cues.', 'Choose a mood, set the pace, and let your preferences guide a playlist with a reason behind every track.', 'Compare baseline and AI planning with recorded measurements and transparent results.']):
        with col:
            st.html(f'<div class="feature-card"><span class="eyebrow">{icon}</span><h3>{title}</h3><p>{description}</p></div>')
    st.write("")
    st.button("Start with your audio →", type="primary", on_click=open_audio_page)
    st.caption(f"{catalog['catalog_tracks']} real catalog tracks · Local audio analysis · Explainable recommendations")
elif page == "Music Analysis":
    st.header("Understand your audio")
    st.caption("Drop in a song. Discover its tempo, texture, and musical character.")
    st.info("Full songs welcome: uploads longer than 30 seconds are automatically analyzed using only the first 30 seconds. MP3/WAV · up to 200 MiB · at least 3 seconds. Audio analysis stays local.")
    upload = st.file_uploader("Choose your audio", type=["mp3", "wav"])
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
            if len(raw) > 200 * 1024 * 1024:
                st.error("Audio must be 200 MiB or smaller.")
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
            if analysis.get('trimmed'):
                st.success(f"Ready: analyzed the first 30 seconds of your {analysis['original_duration_seconds'] / 60:.1f}-minute upload.")
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
    # Explicit widget keys prevent an earlier visit's form state from masking
    # newly attached audio. Apply once so later user edits remain editable.
    if st.session_state.pop("apply_audio_preferences", False):
        reference = st.session_state["audio_reference"]
        bpm = reference['bpm']
        if bpm is not None and 40 <= bpm <= 220:
            st.session_state['playlist_bpm'] = round(bpm)
        moods = reference['mood_scores']
        mood_map = {'happy': 'happy', 'sad': 'melancholic', 'relaxed': 'relaxed', 'aggressive': 'energetic'}
        strongest_mood = max(moods, key=lambda cue: cue['score'], default=None)
        st.session_state['playlist_mood'] = mood_map[strongest_mood['label']] if strongest_mood and strongest_mood['score'] >= .5 else 'any'
        strongest_genre = max(reference['genre_scores'], key=lambda cue: cue['score'], default=None)
        mapped = GENRE_TO_CATALOG.get(strongest_genre['label']) if strongest_genre else None
        st.session_state['playlist_genre'] = mapped if mapped in catalog['genres'] else 'any'
    active_reference = None
    use_audio_tempo = use_audio_genre = False
    if reference := st.session_state.get("audio_reference"):
        if st.checkbox("Use analyzed audio as reference", value=True, key="use_audio_reference"):
            active_reference = reference
            st.success("Your analyzed upload is attached. You can change its influence below.")
            mood_cue = max(reference['mood_scores'], key=lambda cue: cue['score'], default=None)
            genre_cue = max(reference['genre_scores'], key=lambda cue: cue['score'], default=None)
            st.caption("Detected cues · Mood: " + (mood_cue['label'] if mood_cue and mood_cue['score'] >= .5 else 'unavailable') + " · Genre: " + (genre_cue['label'] if genre_cue else 'unavailable'))
            if genre_cue and GENRE_TO_CATALOG.get(genre_cue['label']) not in catalog['genres']:
                st.caption("The detected genre has no matching catalog category, so the genre filter starts at any.")
            st.caption(f"Upload BPM: {reference['bpm']:.1f}" if reference["bpm"] is not None else "Upload BPM unavailable; use a manual tempo target.")
            a, b = st.columns(2)
            tempo_available = reference["bpm"] is not None and 40 <= reference["bpm"] <= 220
            st.session_state.setdefault('match_audio_tempo', True)
            st.session_state.setdefault('match_audio_genre', True)
            use_audio_tempo = a.checkbox("Match upload tempo", key="match_audio_tempo", disabled=not tempo_available) and tempo_available
            use_audio_genre = b.checkbox("Prefer upload genre cues", key="match_audio_genre", disabled=not reference["genre_scores"]) and bool(reference["genre_scores"])
            st.caption("Genre cues affect ranking softly. Your chosen genre filter and desired mood remain your preferences.")
        st.button("Clear audio reference", on_click=clear_audio_reference)
    if not catalog["mood_available"]:
        st.info("This catalog has no annotated moods. Mood is passed to the LLM as a preference; baseline retrieval uses tempo, energy and genre.")
    with st.form("preferences"):
        left, right = st.columns(2)
        mood_options = ["relaxed", "happy", "energetic", "melancholic", "any"]
        mood = left.selectbox("Mood", mood_options, key="playlist_mood")
        duration = right.slider("Duration cap (minutes)", 3, 60, 15)
        genre = left.selectbox("Genre", ["any", *catalog["genres"]], key="playlist_genre")
        progression = right.selectbox("Energy progression", ["steady", "rising", "falling"])
        energy = left.slider("Target energy (published catalog scale)", 0.0, 1.0, 0.5)
        left.caption("Echo Nest energy is a model-derived attribute, not RMS loudness or a mood confidence score.")
        if use_audio_tempo:
            st.session_state['playlist_bpm'] = round(active_reference['bpm'])
        st.session_state.setdefault('playlist_bpm', 110)
        bpm = right.slider("Target BPM", 40, 220, key="playlist_bpm", disabled=use_audio_tempo)
        if active_reference:
            right.caption("Tempo, mood and supported genre were filled from your analysis. Duration, energy and progression remain your choices; energy cannot be inferred from RMS. Choose LLM to use mood cues in planning.")
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
        for number, track in enumerate(result['tracks'], 1):
            st.html(f'<div class="track-card"><div class="track-number">{number:02}</div><div><strong>{escape(track["title"])}</strong><p>{escape(track["artist"])} · {escape(track["genre"])} · {track["duration_seconds"] // 60}:{track["duration_seconds"] % 60:02d}</p><p>{escape(track["explanation"])}</p></div></div>')
        st.dataframe(frame[["title", "artist", "duration_seconds", "genre", "energy", "explanation"]], hide_index=True, width="stretch")
        st.line_chart(frame["energy"])
        with st.expander("Track sources and feature provenance"):
            st.dataframe(frame[["title", "source_url", "source", "metadata_license", "energy_source", "mood_source"]], hide_index=True, width="stretch")
        st.download_button("Download playlist JSON", data=__import__("json").dumps(result, indent=2), file_name="moodmix-playlist.json", mime="application/json")
