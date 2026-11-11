# Local pretrained mood and genre estimates

## What the application does

Music Analysis can run **MTG MSD MusiCNN** locally to obtain 200-dimensional audio embeddings, followed by pretrained MTG classification heads. This is neural inference with published weights, not rules that turn BPM or loudness into emotion. The embedding model is a CNN; Gemini remains the project's Transformer-based generative component.

- Four independent binary mood attributes: happy, sad, relaxed, aggressive.
- A nine-class genre model: alternative, blues, electronic, folk/country, funk/soul/R&B, jazz, pop, rap/hip-hop, rock.
- Independent mood attributes can overlap. A score ≥ 0.5 is treated as the positive binary decision; if no head crosses that threshold, the result is `no positive mood cue`.
- Genre is the highest-scoring class. The interface shows the top three; JSON exports all nine.
- Outputs are **uncalibrated model scores**, not confidence percentages or emotional ground truth.

The supplied mood vocabulary does not contain `energetic` or every possible emotion. The genre model omits styles including classical and ambient; it can misclassify them into one of its available classes. It can also label speech, tones and noise. This milestone does not add a reliable out-of-distribution detector.

## Installation and privacy

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_audio_models.py
```

The download script retrieves about 4 MB of official model/metadata files to ignored `models/musicnn/`, checking pinned size and SHA-256 hashes from `data/audio-models.manifest.json`. It preserves already verified files and uses temporary files for replacement. Inference never silently downloads weights. Missing/corrupt assets produce an explicit unavailable/error status while signal measurements remain usable. Restart the backend after installing/replacing model files.

CPU ONNX Runtime runs one patch at a time with two inference threads; at most ten patches are processed. A roughly 16 GB RAM Windows laptop was used for integration verification. No GPU, paid API, or recording upload to a third party is needed. Only model download requests leave the machine.

## Preprocessing and aggregation

The NumPy/librosa implementation follows the published `TensorflowInputMusiCNN` parameters: mono 16 kHz audio, 512-sample symmetric unnormalized Hann window, 256-sample hop, 96 Slaney mel power bands with unit-area normalization, then `log10(1 + 10000 × mel)` compression. STFT framing is centered with zero padding. Magnitudes do not require the reference implementation's zero-phase rotation.

Each model patch is 187 frames, about three seconds. Non-overlapping full patches are formed; at most ten are evenly selected across the clip. Incomplete trailing patches are discarded. Embeddings from selected patches go through each head; class activations are averaged across patches. Positive-class indices are looked up in each head's original metadata, since label order differs between heads.

The model branch resamples directly from the decoded mono signal, independently of the 22.05 kHz signal-analysis branch. It skips silence and clips shorter than six seconds. Prefer 15–30 seconds of representative music. Predictions characterize the sampled excerpt, not necessarily the complete song.

This is a port of the published preprocessing recipe; bit-identical equivalence to Essentia across platforms has not been established. That and excerpt aggregation are limitations to assess in future evaluation.

## License and attribution

Model weights: **Creative Commons Attribution–NonCommercial–ShareAlike 4.0**, published by the Music Technology Group, Universitat Pompeu Fabra (MTG-UPF). Model metadata credits Pablo Alonso. No model weights are redistributed in this repository. The project does not claim ownership of those models. Commercial use needs a compatible license from the authors.

- Model catalog and license statement: https://essentia.upf.edu/models.html
- License terms: https://creativecommons.org/licenses/by-nc-sa/4.0/
- Feature extractor: https://essentia.upf.edu/models/feature-extractors/musicnn/
- Mood heads: https://essentia.upf.edu/models/classification-heads/
- Genre model card: https://essentia.upf.edu/models/classification-heads/genre_dortmund/genre_dortmund-msd-musicnn-1.json
- Preprocessing source: https://github.com/MTG/essentia/blob/master/src/algorithms/spectral/tensorflowinputmusicnn.cpp
- Jordi Pons and Xavier Serra. *musicnn: Pre-trained convolutional neural networks for music audio tagging.* 2019. https://arxiv.org/abs/1909.06654
- Pablo Alonso-Jiménez, Dmitry Bogdanov, Jordi Pons and Xavier Serra. *TensorFlow Audio Models in Essentia.* ICASSP 2020. https://arxiv.org/abs/2003.07393

## Verification and next evaluation

Offline tests exercise patch shape/limits, label ordering, independent mood semantics, missing/corrupt model handling, skipped inputs and multipart routing. Actual official weights were run locally on generated audio and a real-music example. `evaluation/audio-model-smoke.json` records the latter's measured output and runtime, not an accuracy result.

Example attribution: **Vibe Ace**, Kevin MacLeod, CC BY 3.0, https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace . First 30 seconds converted from the checksum-verified official librosa OGG example to WAV for testing. Audio remains in ignored temporary files and is not redistributed. No ground-truth mood annotation was used.

We have not reproduced the authors' reported benchmark metrics or measured accuracy on our users' music. Before using estimates as hard recommendation filters, evaluate held-out music, inspect model failures and consider human mood disagreement. Uploaded BPM and compatible genre scores can now guide retrieval; mood scores become LLM context. Their effect is optional and reported in the result. See `audio-to-playlist.md`.
