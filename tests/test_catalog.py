"""Real catalog integrity and importer failure checks."""
from pathlib import Path
import pytest
from backend.catalog import load_catalog, catalog_info
from backend.recommendation import retrieve, baseline, validate_plan
from backend.schemas import PlaylistRequest
from scripts.import_fma import build_catalog


def test_real_catalog_provenance_and_planning():
    tracks = load_catalog()
    assert len(tracks) == 240
    assert all(t.track_id.startswith("fma-") and t.metadata_license == "CC BY 4.0" for t in tracks)
    assert all(t.mood == "unknown" and t.energy_source.startswith("Echo Nest") for t in tracks)
    assert not catalog_info(tracks)["mood_available"]
    for genre in catalog_info(tracks)["genres"]:
        request = PlaylistRequest(genre=genre, progression="rising")
        candidates = retrieve(tracks, request)
        selected = validate_plan(baseline(candidates, request), candidates, request)
        assert all(t.genre == genre for t in selected)


def test_retrieval_excludes_tracks_longer_than_entire_playlist():
    request = PlaylistRequest(duration_minutes=3)
    candidates = retrieve(load_catalog(), request)
    assert candidates and all(t.duration_seconds <= 180 for t in candidates)


def test_retrieval_excludes_catalog_interview_segments():
    tracks = load_catalog()
    assert any(t.title.casefold() == "interview" for t in tracks)
    candidates = retrieve(tracks, PlaylistRequest(genre="jazz"))
    assert candidates and all(t.title.casefold() != "interview" for t in candidates)


def test_import_rejects_unverified_archive(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"not an official archive")
    with pytest.raises(ValueError, match="checksum"):
        build_catalog(archive, tmp_path / "catalog.csv")
    assert not (tmp_path / "catalog.csv").exists()


def test_import_joins_features_and_balances_genres(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    import pandas as pd
    from scripts import import_fma
    metadata = pd.DataFrame([
        ["One", "Artist A", "Rock", 180],
        ["Two", "Artist B", "Rock", 200],
        ["Three", "Artist C", "Jazz", 240],
        ["Invalid", "Artist D", "Jazz", 0],
    ], index=[1, 2, 3, 4], columns=pd.MultiIndex.from_tuples([
        ("track", "title"), ("artist", "name"), ("track", "genre_top"), ("track", "duration")]))
    features = pd.DataFrame([[100, .2], [110, .3], [120, .4], [130, .5]], index=[1, 2, 3, 4],
        columns=pd.MultiIndex.from_tuples([("echonest", "audio_features", "tempo"), ("echonest", "audio_features", "energy")]))
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fma_metadata/tracks.csv", metadata.to_csv())
        bundle.writestr("fma_metadata/echonest.csv", features.to_csv())
    # Only this isolated test substitutes the release checksum for its tiny fixture.
    monkeypatch.setattr(import_fma, "EXPECTED_SHA1", hashlib.sha1(archive.read_bytes()).hexdigest())
    output = tmp_path / "catalog.csv"
    manifest = build_catalog(archive, output, limit=2)
    tracks = load_catalog(output)
    assert manifest["genres"] == {"jazz": 1, "rock": 1}
    assert {t.track_id for t in tracks} == {"fma-000001", "fma-000003"}
    assert next(t.energy for t in tracks if t.track_id == "fma-000003") == .4
    assert all(t.mood == "unknown" for t in tracks)
