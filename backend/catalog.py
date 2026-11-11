"""Load and validate a local catalog; provenance is mandatory."""
import csv
from pathlib import Path
from backend.schemas import Track

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "development_catalog.csv"


def load_catalog(path: Path = CATALOG_PATH) -> list[Track]:
    with path.open(encoding="utf-8", newline="") as handle:
        tracks = [Track.model_validate(row) for row in csv.DictReader(handle)]
    if not tracks or len({t.track_id for t in tracks}) != len(tracks):
        raise ValueError("Catalog must contain tracks with unique IDs.")
    return tracks
