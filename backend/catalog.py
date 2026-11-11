"""Load and validate a local catalog; provenance is mandatory."""
import csv
from pathlib import Path
from backend.schemas import Track

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.csv"


def load_catalog(path: Path = CATALOG_PATH) -> list[Track]:
    with path.open(encoding="utf-8", newline="") as handle:
        tracks = [Track.model_validate(row) for row in csv.DictReader(handle)]
    if not tracks or len({t.track_id for t in tracks}) != len(tracks):
        raise ValueError("Catalog must contain tracks with unique IDs.")
    return tracks


def catalog_info(tracks: list[Track]) -> dict:
    """Describe the actual loaded data instead of hardcoding a UI label."""
    synthetic = all(t.source.startswith("synthetic") for t in tracks)
    return {"catalog_tracks": len(tracks),
            "catalog_kind": "synthetic development data" if synthetic else "FMA real-track metadata",
            "genres": sorted({t.genre for t in tracks}),
            "mood_available": any(t.mood != "unknown" for t in tracks),
            "audio_available": False}
