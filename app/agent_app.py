"""Independent conversational app: streamlit run app/agent_app.py --server.port 8502."""
from pathlib import Path
import sys
import json
from html import escape
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
from backend.agent import create_mix
from backend.audio import analyze_audio, AudioInputError
from backend.generation import ProviderFailure

st.set_page_config(page_title='MoodMix Studio', page_icon='◉', layout='wide')
st.html('<style>' + (Path(__file__).parent / 'style.css').read_text(encoding='utf-8') + '</style>')
st.title('MoodMix Studio')
st.caption('CONVERSATIONAL MUSIC AGENT · EXPERIMENTAL')
st.sidebar.link_button('Open original MoodMix ↗', 'http://127.0.0.1:8501')
st.sidebar.caption('Uses your existing Gemini key. Up to five Gemini requests per message. No automatic retries. Audio analysis runs locally.')
if st.sidebar.button('Start a new mix'):
    for key in ['studio_mix', 'studio_chat', 'studio_audio']:
        st.session_state.pop(key, None)
    st.rerun()
with st.sidebar.expander('Optional audio inspiration'):
    upload = st.file_uploader('MP3 or WAV', type=['mp3', 'wav'], key='studio_upload')
    if upload and st.button('Analyze opening 30 seconds'):
        try:
            with st.spinner('Analyzing locally…'):
                analysis = analyze_audio(upload.getvalue(), upload.name, include_models=True)
            if analysis.is_silent:
                st.warning('Choose non-silent music.')
            else:
                prediction = analysis.music_model
                st.session_state['studio_audio'] = {'bpm': analysis.bpm, 'duration_seconds': analysis.duration_seconds,
                    'is_silent': False, 'neural_status': prediction.status if prediction else 'not_requested',
                    'mood_scores': [c.model_dump() for c in prediction.mood_scores] if prediction and prediction.status == 'ok' else [],
                    'genre_scores': [c.model_dump() for c in prediction.genre_scores] if prediction and prediction.status == 'ok' else []}
        except AudioInputError as exc:
            st.error(str(exc))
    if st.session_state.get('studio_audio'):
        st.success('Audio summary attached')
        if st.button('Remove audio inspiration'):
            st.session_state.pop('studio_audio', None)
            st.rerun()
st.html('<div class="hero"><span class="eyebrow">Tell us where the music should take you</span><h2>A mix that evolves<br>with your conversation.</h2><p>Describe a listening session. Your agent creates a musical plan, searches real catalog metadata, validates its playlist, and critiques the result. Ask for changes to refine it.</p></div>')
for entry in st.session_state.get('studio_chat', []):
    with st.chat_message(entry['role']):
        st.write(entry['content'])
message = st.chat_input('Try: 15 minutes of gentle jazz, building toward a lively finish', max_chars=2000)
if message:
    try:
        with st.spinner('Planning, searching, validating and reviewing your mix…'):
            result = create_mix(message, st.session_state.get('studio_mix'), st.session_state.get('studio_audio'))
        st.session_state['studio_mix'] = result
        history = st.session_state.setdefault('studio_chat', [])
        history.extend([{'role': 'user', 'content': message}, {'role': 'assistant', 'content': result['brief']['summary']}])
        st.session_state['studio_chat'] = history[-12:]
        st.rerun()
    except (ProviderFailure, ValueError) as exc:
        st.error(str(exc))
if result := st.session_state.get('studio_mix'):
    st.subheader(result['plan']['title'])
    st.caption(f"Actual method: {result['method']} · {result['provider_calls']} Gemini calls · {result['duration_seconds'] // 60}:{result['duration_seconds'] % 60:02d}")
    if result['method'] != 'agent':
        st.warning('The agent did not produce a valid playlist within its call budget. This is a deterministic baseline fallback.')
    if not result['critique_completed']:
        st.warning('Call budget reached before the agent finished its critique. The displayed playlist passed code validation.')
    st.subheader('Your musical plan')
    st.dataframe(pd.DataFrame(result['brief']['phases']), hide_index=True)
    st.caption('Phase budgets describe the intended journey. Exact song boundaries and perceived moods are not guaranteed.')
    notes = {t['track_id']: t['explanation'] for t in result['plan']['tracks']}
    for index, track in enumerate(result['tracks'], 1):
        st.html(f'<div class="track-card"><div class="track-number">{index:02}</div><div><strong>{escape(track["title"])}</strong><p>{escape(track["artist"])} · {escape(track["genre"])}</p><p>{escape(notes[track["track_id"]])}</p></div></div>')
    st.line_chart(pd.DataFrame(result['tracks'])['energy'])
    with st.expander('Agent review and tool activity'):
        st.write(result['critique'])
        st.json(result['events'])
    st.download_button('Download this session result', json.dumps(result, indent=2), 'moodmix-studio.json', 'application/json')
    st.caption('Refine it below: “Less rock”, “make it 20 minutes”, or “more different artists”. Catalog recordings are not hosted.')
