"""Feature-distance retrieval and deterministic duration-constrained baseline."""
from backend.schemas import Track, PlaylistRequest, Plan, PlannedTrack


def retrieve(tracks: list[Track], request: PlaylistRequest) -> list[Track]:
    eligible = [t for t in tracks if request.genre.lower() == "any" or t.genre.lower() == request.genre.lower()]
    if not eligible:
        raise ValueError("No tracks match the requested genre.")
    def distance(track: Track) -> float:
        # BPM is scaled by 180 BPM; energy already lies in [0, 1].
        mood_penalty = 0.25 * (track.mood.lower() != request.mood.lower()) if track.mood != "unknown" and request.mood != "any" else 0
        return (track.energy - request.energy) ** 2 + ((track.bpm - request.bpm) / 180) ** 2 + mood_penalty
    return sorted(eligible, key=lambda t: (distance(t), t.track_id))[:60]


def baseline(candidates: list[Track], request: PlaylistRequest) -> Plan:
    # 0/1 knapsack: maximize duration without exceeding the requested cap.
    # Candidate order resolves ties in favor of more similar tracks.
    cap = request.duration_minutes * 60
    reachable: dict[int, list[Track]] = {0: []}
    for track in candidates:
        for seconds, chosen in list(reachable.items()):
            total = seconds + track.duration_seconds
            if total <= cap and total not in reachable:
                reachable[total] = chosen + [track]
    chosen = reachable[max(reachable)]
    if not chosen:
        raise ValueError("No candidate fits the requested duration.")
    if request.progression != "steady":
        chosen = sorted(chosen, key=lambda t: t.energy, reverse=request.progression == "falling")
    return Plan(title="Content-based mix", tracks=[PlannedTrack(track_id=t.track_id, explanation="Selected by feature similarity and duration fitting.") for t in chosen])


def validate_plan(plan: Plan, candidates: list[Track], request: PlaylistRequest) -> list[Track]:
    lookup = {t.track_id: t for t in candidates}
    ids = [item.track_id for item in plan.tracks]
    if len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
        raise ValueError("Playlist contains duplicate or unknown track IDs.")
    tracks = [lookup[i] for i in ids]
    total = sum(t.duration_seconds for t in tracks)
    cap = request.duration_minutes * 60
    best = baseline(candidates, request)
    best_duration = sum(lookup[t.track_id].duration_seconds for t in best.tracks)
    # Permit at most one minute less coverage than the best available fit.
    if total > cap or total < best_duration - 60:
        raise ValueError("Playlist duration violates the available-duration tolerance.")
    energies = [t.energy for t in tracks]
    if request.progression == "rising" and energies != sorted(energies):
        raise ValueError("Energy must rise monotonically.")
    if request.progression == "falling" and energies != sorted(energies, reverse=True):
        raise ValueError("Energy must fall monotonically.")
    return tracks
