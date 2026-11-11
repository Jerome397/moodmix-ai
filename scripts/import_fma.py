"""Build a compact, reproducible FMA catalog from the official metadata ZIP.

No audio is extracted or redistributed. Input integrity is verified before parsing.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

ARCHIVE_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
EXPECTED_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
ROOT = Path(__file__).resolve().parents[1]


def build_catalog(archive: Path, output: Path, limit: int = 240) -> dict:
    """Join real metadata and acoustic attributes, balancing top-level genres."""
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha1").hexdigest()
    if digest != EXPECTED_SHA1:
        raise ValueError("Archive checksum differs from the official FMA release.")
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open("fma_metadata/tracks.csv") as handle:
            tracks = pd.read_csv(handle, index_col=0, header=[0, 1], low_memory=False)
        with bundle.open("fma_metadata/echonest.csv") as handle:
            features = pd.read_csv(handle, index_col=0, header=[0, 1, 2], low_memory=False)
    rows = []
    for track_id in sorted(set(tracks.index) & set(features.index)):
        meta, acoustic = tracks.loc[track_id], features.loc[track_id]
        title, artist, genre = [meta[key] for key in [("track", "title"), ("artist", "name"), ("track", "genre_top")]]
        duration = meta[("track", "duration")]
        bpm = acoustic[("echonest", "audio_features", "tempo")]
        energy = acoustic[("echonest", "audio_features", "energy")]
        if any(pd.isna(v) for v in [title, artist, genre, duration, bpm, energy]):
            continue
        if not (60 <= duration <= 900 and 40 <= bpm <= 220 and 0 <= energy <= 1):
            continue
        url = meta.get(("track", "url"), "")
        rows.append(dict(track_id=f"fma-{track_id:06d}", title=str(title), artist=str(artist),
                         duration_seconds=int(duration), genre=str(genre).lower(), bpm=float(bpm),
                         energy=float(energy), mood="unknown", source="FMA 2017 metadata + Echo Nest audio_features",
                         source_url=str(url) if pd.notna(url) and str(url).startswith("https://") else "https://github.com/mdeff/fma", metadata_license="CC BY 4.0",
                         energy_source="Echo Nest energy; not RMS loudness", mood_source="not annotated"))
    groups = {}
    for row in rows:
        groups.setdefault(row["genre"], []).append(row)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for genre in sorted(groups):
            if groups[genre] and len(selected) < limit:
                selected.append(groups[genre].pop(0))
    if len(selected) != limit:
        raise ValueError(f"Only {len(selected)} eligible tracks; requested {limit}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    manifest = {"dataset": "FMA (ISMIR 2017)", "archive_url": ARCHIVE_URL,
                "archive_sha1": digest, "metadata_license": "CC BY 4.0",
                "attribution": "Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson. FMA: A Dataset For Music Analysis. ISMIR 2017.",
                "reference": "https://github.com/mdeff/fma", "tracks": len(selected),
                "selection": "Ascending FMA ID, round-robin over genre_top; complete metadata and Echo Nest features; 60–900 s duration, 40–220 BPM, energy in [0,1]",
                "mood": "Unknown: no mood ground truth imported", "recordings_redistributed": False,
                "genres": {g: sum(r["genre"] == g for r in selected) for g in sorted({r["genre"] for r in selected})}}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "catalog.csv")
    parser.add_argument("--limit", type=int, default=240)
    args = parser.parse_args()
    print(json.dumps(build_catalog(args.archive, args.output, args.limit), indent=2))
