"""Shared request, catalog, and generated-output contracts."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


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


class PlaylistRequest(StrictModel):
    mood: str = Field(default="relaxed", max_length=80)
    duration_minutes: int = Field(default=15, ge=3, le=120)
    genre: str = Field(default="any", max_length=80)
    energy: float = Field(default=0.5, ge=0, le=1)
    bpm: float = Field(default=110, ge=40, le=220)
    progression: Literal["steady", "rising", "falling"] = "steady"
    instructions: str = Field(default="", max_length=2000)
    mode: Literal["baseline", "llm"] = "baseline"


class PlannedTrack(StrictModel):
    track_id: str
    explanation: str = Field(min_length=1, max_length=600)


class Plan(StrictModel):
    title: str = Field(min_length=1, max_length=150)
    tracks: list[PlannedTrack] = Field(min_length=1, max_length=100)
