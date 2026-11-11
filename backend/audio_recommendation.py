"""Resolve comparable uploaded-audio cues without mixing incompatible scales."""
from dataclasses import dataclass, field
from backend.schemas import PlaylistRequest, Track

# Explicit coarse taxonomy mapping. Unsupported categories stay unsupported.
GENRE_TO_CATALOG = {"blues": "blues", "electronic": "electronic", "folk / country": "folk",
                    "jazz": "jazz", "pop": "pop", "rap / hip-hop": "hip-hop", "rock": "rock"}


@dataclass
class RetrievalContext:
    target_bpm: float
    audio_reference_attached: bool = False
    genre_affinity: dict[str, float] = field(default_factory=dict)
    compared_audio_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def resolve_context(request: PlaylistRequest, catalog: list[Track]) -> RetrievalContext:
    context = RetrievalContext(target_bpm=request.bpm)
    reference = request.audio_reference
    if reference is None:
        return context
    if reference.is_silent:
        raise ValueError("Silent audio cannot be a playlist reference. Disable the audio reference or analyze another recording.")
    if reference.bpm is None and not reference.mood_scores and not reference.genre_scores:
        raise ValueError("Audio has no usable tempo or neural cues. Analyze a longer music excerpt or disable the audio reference.")
    context.audio_reference_attached = True
    if request.use_audio_tempo:
        if reference.bpm is not None and 40 <= reference.bpm <= 220:
            context.target_bpm = reference.bpm
            context.compared_audio_features.append("estimated BPM")
            context.warnings.append("Upload and catalog BPM are estimates from different methods; half/double-tempo errors remain possible.")
        else:
            context.warnings.append("Upload BPM is unavailable or outside 40–220; the manual target BPM is used.")
    if request.use_audio_genre:
        available = {track.genre for track in catalog}
        for cue in reference.genre_scores:
            mapped = GENRE_TO_CATALOG.get(cue.label)
            if mapped in available and cue.score > 0:
                context.genre_affinity[mapped] = cue.score
        if context.genre_affinity:
            context.compared_audio_features.append("soft genre cues")
            context.warnings.append("Genre estimates provide a soft preference; your selected genre filter takes priority.")
        elif reference.genre_scores:
            context.warnings.append("The estimated genre has no supported catalog mapping; no audio genre preference is applied.")
    context.warnings.append("Upload RMS and spectral features are not compared with Echo Nest energy. The energy slider remains your separate preference.")
    context.warnings.append("The catalog has no verified mood labels. Upload mood scores are planning context, not guaranteed track-mood matches.")
    return context
