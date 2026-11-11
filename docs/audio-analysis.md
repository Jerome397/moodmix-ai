# Audio signal analysis: Phase 2, first milestone

The Music Analysis page sends an MP3/WAV excerpt to the local FastAPI endpoint `POST /audio/analyze` as multipart field `file`. Analysis does not call Gemini or download neural model weights. Optional local pretrained inference is now available via multipart `include_models=true`; see `docs/audio-models.md`. Without that option, mood/genre remain `not estimated`.

## Input limits

- File size up to 20 MiB; decoded duration 3–120 seconds.
- Mono or stereo; source sample rate 8–96 kHz.
- SoundFile must decode a WAV/WAVEX container for `.wav`, or an MP3 stream for `.mp3`. Renamed files are rejected.
- Non-finite sample values, malformed data, invalid channel/sample-rate metadata, and unsupported formats are rejected with useful HTTP errors.
- Validation checks duration before decoding, bounds decoded frames, and checks the actual decoded duration again. The endpoint reads at most the upload limit plus one byte.

The browser holds the selected recording for playback. Framework multipart handling can temporarily spool an upload to disk; the endpoint always closes it afterward. The backend does not create an audio library or persist uploaded recordings. The JSON response contains summary features and a small waveform envelope, not raw audio. Use only audio you are entitled to analyze.

## Feature definitions

| Feature | Definition and interpretation |
| --- | --- |
| Duration | Decoded frame count divided by source sample rate |
| RMS amplitude | Square root of mean squared samples, across the original decoded channels, without gain normalization |
| RMS dBFS | `20 log10(RMS)` relative to amplitude 1; exact digital silence returns null rather than negative infinity |
| Peak | Maximum absolute source sample amplitude |
| Tempo | librosa dynamic-programming beat tracking from a spectral onset envelope; reported only when at least three beat events are found |
| Spectral centroid | Mean magnitude-weighted frequency, in Hz; a brightness descriptor |
| Spectral bandwidth | Mean spectral spread around the centroid, in Hz |
| Spectral rolloff | Mean frequency below which 85% of spectral magnitude lies |
| Zero-crossing rate | Mean fraction of sign changes in analysis frames |

Tempo and spectral features use a mono signal resampled to 22,050 Hz, FFT size 2048 and hop 512. Stereo is averaged; if strong phase cancellation suppresses that mix, the higher-RMS channel is used with a warning. RMS and peak remain calculated from original decoded channels. Silent/near-silent recordings (RMS ≤ 1e-6) return no tempo and zero spectral summary values with an explicit warning. These zeros are conventions for silence, not evidence of musical meaning.

The waveform is a peak-preserving min/max envelope of at most 500 time bins. A separate timeline shows RMS per display bin. Neither is a full-resolution exported recording.

## Interpretation limits

BPM can have half/double-tempo errors and can be unreliable on weak, nonmusical, short or changing rhythms. Three detected beats do not constitute a calibrated confidence score. No mood/genre confidence is fabricated. RMS measures digital signal level; it is neither perceptual loudness (LUFS) nor the FMA catalog's Echo Nest energy attribute. Upload BPM and compatible genre scores can guide playlist retrieval; RMS and spectral descriptors remain excluded because compatible catalog values are unavailable. See `audio-to-playlist.md`.

## Verification

Tests generate their own recordings in memory: a known-frequency sine wave for spectral/RMS checks, a 120 BPM click track for tempo, silence, opposite-phase stereo, WAV and MP3 roundtrips, and invalid uploads. These demonstrate signal-processing correctness on controlled fixtures; they do not establish accuracy on a diverse music dataset. First use can take longer because librosa/Numba initialize compiled routines.

Dependencies: `librosa` 0.11.x, NumPy, SoundFile 0.13.x, and python-multipart. SoundFile's Windows wheel bundles libsndfile. The dependency lock records the installed transitive packages. No system FFmpeg installation is required for formats supported by the bundled decoder.

References:

- https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html
- https://python-soundfile.readthedocs.io/en/latest/
