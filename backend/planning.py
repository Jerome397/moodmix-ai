"""Construct a bounded, diverse menu of constraint-valid playlists."""
from collections import defaultdict, deque
from dataclasses import dataclass

from backend.recommendation import baseline, validate_plan, duration_combinations
from backend.schemas import Plan, PlaylistDecision, PlaylistRequest, PlannedTrack, Track


@dataclass
class PlaylistOption:
    option_id: str
    tracks: list[Track]

    def summary(self) -> dict:
        return {"option_id": self.option_id, "track_ids": [t.track_id for t in self.tracks],
                "duration_seconds": sum(t.duration_seconds for t in self.tracks),
                "mean_energy": round(sum(t.energy for t in self.tracks) / len(self.tracks), 4),
                "energy_sequence": [t.energy for t in self.tracks],
                "mean_bpm": round(sum(t.bpm for t in self.tracks) / len(self.tracks), 2),
                "distinct_artists": len({t.artist for t in self.tracks}),
                "genres": sorted({t.genre for t in self.tracks})}


def interleave(tracks: list[Track], key) -> list[Track]:
    groups = defaultdict(deque)
    for track in tracks:
        groups[key(track)].append(track)
    result = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                result.append(group.popleft())
    return result


def build_options(candidates: list[Track], request: PlaylistRequest) -> list[PlaylistOption]:
    """Change knapsack tie preferences to create up to eight valid sequences.

    The first option is the unchanged content-based baseline. Other orderings
    explore energy, tempo and artist/genre diversity; facts in summaries describe
    actual results rather than claiming a seed guarantees a particular quality.
    """
    if len({t.track_id for t in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique.")
    seeds = [candidates, list(reversed(candidates)), sorted(candidates, key=lambda t: t.energy),
             sorted(candidates, key=lambda t: -t.energy), sorted(candidates, key=lambda t: abs(t.bpm - request.bpm)),
             interleave(candidates, lambda t: t.artist), interleave(candidates, lambda t: t.genre),
             candidates[len(candidates) // 3:] + candidates[:len(candidates) // 3]]
    original = baseline(candidates, request)
    lookup = {t.track_id: t for t in candidates}
    best_duration = sum(lookup[item.track_id].duration_seconds for item in original.tracks)
    minimum_duration = max(1, best_duration - 60)
    result, seen = [], set()
    for index, seed in enumerate(seeds):
        reachable = duration_combinations(seed, request.duration_minutes * 60)
        # Explore the already permitted 60-second tolerance instead of returning
        # the same unique exact-duration subset for every ordering.
        target = best_duration - index * 8
        totals = sorted((t for t in reachable if t >= minimum_duration), key=lambda t: (abs(t - target), -t))
        for total in totals:
            chosen = reachable[total]
            if request.progression != "steady":
                chosen = sorted(chosen, key=lambda t: t.energy, reverse=request.progression == "falling")
            key = tuple(t.track_id for t in chosen)
            if key in seen:
                continue
            plan = Plan(title="Checked mix", tracks=[PlannedTrack(track_id=i, explanation="Constraint-checked candidate.") for i in key])
            validate_plan(plan, candidates, request)
            seen.add(key)
            result.append(PlaylistOption(option_id=f"mix-{len(result) + 1:02d}", tracks=chosen))
            break
    return result


def materialize_decision(decision: PlaylistDecision, options: list[PlaylistOption], candidates: list[Track], request: PlaylistRequest) -> tuple[Plan, list[str]]:
    """Preserve exactly the chosen option; never quietly switch its membership."""
    option = next((item for item in options if item.option_id == decision.option_id), None)
    if option is None:
        raise ValueError("Unknown playlist option.")
    reasons = {item.track_id: item.explanation for item in decision.track_reasons}
    allowed = {t.track_id for t in option.tracks}
    if len(reasons) != len(decision.track_reasons) or not set(reasons) <= allowed:
        raise ValueError("Track notes contain duplicate IDs or IDs outside the chosen option.")
    missing = allowed - set(reasons)
    plan = Plan(title=decision.title, tracks=[PlannedTrack(track_id=t.track_id,
        explanation=reasons.get(t.track_id, "Playlist rationale: " + decision.rationale[:570])) for t in option.tracks])
    validate_plan(plan, candidates, request)
    warnings = ["Some tracks use the model's playlist-level rationale because individual notes were omitted."] if missing else []
    if len(options) == 1:
        warnings.append("Only one feasible playlist option was available; the model could not change its track selection.")
    return plan, warnings
