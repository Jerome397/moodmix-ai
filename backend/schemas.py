"""Shared request, catalog, and generated-output contracts."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Track(StrictModel):
    track_id: str
    title: str
    artist: str
    duration_seconds: int = Field(gt=0)
    genre: str
    bpm: float = Field(gt=0, le=300)
    energy: float = Field(ge=0, le=1)
    mood: str
    source: str
    source_url: str = ""
    metadata_license: str = ""
    energy_source: str = "synthetic fixture"
    mood_source: str = "synthetic fixture"


class MoodCue(StrictModel):
    label: Literal["happy", "sad", "relaxed", "aggressive"]
    score: float = Field(ge=0, le=1)


class GenreCue(StrictModel):
    label: Literal["alternative", "blues", "electronic", "folk / country", "funk / soul / R&B", "jazz", "pop", "rap / hip-hop", "rock"]
    score: float = Field(ge=0, le=1)


class AudioReference(StrictModel):
    """Small client-supplied analysis summary; never raw audio or waveform data."""
    bpm: float | None = Field(default=None, gt=0, le=400)
    duration_seconds: float = Field(ge=3, le=120)
    is_silent: bool
    neural_status: Literal["ok", "not_requested", "unavailable", "skipped_silence", "skipped_short", "error"] = "not_requested"
    mood_scores: list[MoodCue] = Field(default_factory=list, max_length=4)
    genre_scores: list[GenreCue] = Field(default_factory=list, max_length=9)

    @model_validator(mode="after")
    def validate_cues(self):
        for scores in (self.mood_scores, self.genre_scores):
            if len({item.label for item in scores}) != len(scores):
                raise ValueError("Audio cue labels must be unique.")
        if self.neural_status != "ok" and (self.mood_scores or self.genre_scores):
            raise ValueError("Neural cues require a successful model result.")
        return self


class PlaylistRequest(StrictModel):
    mood: str = Field(default="relaxed", max_length=80)
    duration_minutes: int = Field(default=15, ge=3, le=120)
    genre: str = Field(default="any", max_length=80)
    energy: float = Field(default=0.5, ge=0, le=1)
    bpm: float = Field(default=110, ge=40, le=220)
    progression: Literal["steady", "rising", "falling"] = "steady"
    instructions: str = Field(default="", max_length=2000)
    mode: Literal["baseline", "llm"] = "baseline"
    audio_reference: AudioReference | None = None
    use_audio_tempo: bool = True
    use_audio_genre: bool = True


class PlannedTrack(StrictModel):
    track_id: str
    explanation: str = Field(min_length=1, max_length=600)


class Plan(StrictModel):
    title: str = Field(min_length=1, max_length=150)
    tracks: list[PlannedTrack] = Field(min_length=1, max_length=100)


class PlaylistDecision(StrictModel):
    """LLM chooses a prevalidated sequence rather than doing duration arithmetic."""
    option_id: str
    title: str = Field(min_length=1, max_length=150)
    rationale: str = Field(min_length=1, max_length=1000)
    track_reasons: list[PlannedTrack] = Field(max_length=100)
