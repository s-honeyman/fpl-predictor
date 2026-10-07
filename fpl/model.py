"""Expected-points projection model.

Combines FPL's own next-gameweek prediction (`ep_next`, which already
accounts for that fixture's difficulty and current club news) with a
fixture-adjusted season baseline (`points_per_game`) for the gameweeks
beyond that, scaled by each player's availability.

Early in a season `points_per_game` is still last season's final figure
(FPL resets it to reflect only the new season as games are played), so the
model naturally shifts from "last season's rate" to "this season's actual
rate" as more matches happen - no separate blending logic needed since FPL
maintains that field for us.

VALIDATED, AND FIXED, against real history: backtesting this model against
two completed seasons (see ../README.md "Backtested against two real
completed seasons") found it over-projects by ~35-40%, because the ILP
optimizer picks players by highest projected points - which means it
disproportionately selects players whose raw points_per_game was inflated
by variance (a purple patch, a soft run of fixtures), since those are
exactly the players a naive optimizer finds most attractive. `points_per_game`
is now shrunk toward the positional mean before use, weighted by how many
starts it's actually based on (empirical-Bayes style) - see `_shrink_ppg`.
Re-run `scripts/backtest_season.py` after changing SHRINKAGE_K to confirm
it still helps rather than just moving the bias around.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import team_strength

# Position id -> short name, and squad/formation rules (from bootstrap's element_types).
POSITIONS = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

FORMATION_RULES = {
    1: {"squad_select": 2, "min_play": 1, "max_play": 1},
    2: {"squad_select": 5, "min_play": 3, "max_play": 5},
    3: {"squad_select": 5, "min_play": 2, "max_play": 5},
    4: {"squad_select": 3, "min_play": 1, "max_play": 3},
}

SQUAD_SIZE = 15
STARTING_XI = 11
BUDGET_TENTHS = 1000  # £100.0m, in FPL's 0.1m units
MAX_PER_TEAM = 3

# "Prior strength" in starts-equivalent for points_per_game shrinkage - at
# `starts == SHRINKAGE_K`, a player's raw PPG and the positional mean are
# weighted equally; below that, the positional mean dominates. Not fitted to
# data (unlike pl-club-forecast's match model) - a documented judgment call,
# validated only in the sense that the backtest confirms shrinkage helps at
# all, not that this specific K is optimal. Re-run the backtest if you tune it.
SHRINKAGE_K = 8.0


def _availability_multiplier(player: dict) -> float:
    chance = player.get("chance_of_playing_next_round")
    if chance is not None:
        return chance / 100.0
    status = player.get("status")
    if status == "a":
        return 1.0
    if status == "d":
        return 0.75
    return 0.0  # injured, suspended, unavailable, or not in squad


def _later_gameweek_availability(player: dict) -> float:
    """FPL's chance_of_playing_next_round is about ONE round only, so a
    'doubtful' knock must not discount every later gameweek in the horizon -
    that was turning a 75% one-week doubt into a ~25% haircut on five weeks
    and wrongly dumping players like Gakpo. Doubtful players are assumed back
    after the next round; injured/suspended/unavailable keep their discount
    since their return date is genuinely unknown."""
    if player.get("status") == "d":
        return 1.0
    return _availability_multiplier(player)


def _fixture_difficulty_multiplier(difficulty: int) -> float:
    """Map FPL's 1 (easy) - 5 (hard) FDR to a multiplier centred on 1.0."""
    return 1.375 - 0.125 * difficulty


# Fixture-strength alternative to FDR (see fpl/team_strength.py for the
# scoring). NOT empirically fitted - a documented starting judgment call,
# same status as pl-club-forecast's own strength blend. Validate against
# scripts/backtest_season.py before trusting it over the FDR-based approach.
FIXTURE_STRENGTH_HOME_BONUS = 0.15
FIXTURE_STRENGTH_SCALE = 0.15


def _fixture_strength_multiplier(own_z: float, opp_z: float, is_home: bool) -> float:
    """Alternative to _fixture_difficulty_multiplier: uses the *actual* strength
    gap between a player's team and their specific opponent (via team_strength.py)
    instead of FPL's own subjective 1-5 FDR integer. Floored to avoid a
    degenerate/negative multiplier against extreme mismatches."""
    diff = (own_z - opp_z) + (FIXTURE_STRENGTH_HOME_BONUS if is_home else -FIXTURE_STRENGTH_HOME_BONUS)
    return max(0.5, 1.0 + FIXTURE_STRENGTH_SCALE * diff)


def build_team_fixture_lookup(fixtures: list[dict]) -> dict[int, dict[int, list[tuple[int, bool, int]]]]:
    """team_id -> gameweek(event) -> list of (difficulty, is_home, opponent_team_id)
    for that team's fixtures that week."""
    lookup: dict[int, dict[int, list[tuple[int, bool, int]]]] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None:
            continue  # not yet scheduled (some late-season fixtures start unscheduled)
        for team_key, opp_key, diff_key, is_home in (
            ("team_h", "team_a", "team_h_difficulty", True),
            ("team_a", "team_h", "team_a_difficulty", False),
        ):
            team_id = fixture[team_key]
            opponent_id = fixture[opp_key]
            difficulty = fixture[diff_key]
            lookup.setdefault(team_id, {}).setdefault(event, []).append((difficulty, is_home, opponent_id))
    return lookup


def _positional_mean_ppg(bootstrap: dict) -> dict[int, float]:
    """Starts-weighted mean points_per_game per position - regulars who
    actually played a lot define the "true" positional baseline, rather than
    an unweighted average being dragged down by fringe players who barely
    featured."""
    weighted_sum: dict[int, float] = {}
    weight_total: dict[int, float] = {}
    for player in bootstrap["elements"]:
        pos = player["element_type"]
        starts = float(player.get("starts") or 0)
        ppg = float(player.get("points_per_game") or 0.0)
        weighted_sum[pos] = weighted_sum.get(pos, 0.0) + ppg * starts
        weight_total[pos] = weight_total.get(pos, 0.0) + starts

    means = {}
    for pos in POSITIONS:
        total_weight = weight_total.get(pos, 0.0)
        means[pos] = (weighted_sum.get(pos, 0.0) / total_weight) if total_weight > 0 else 0.0
    return means


def _shrink_ppg(raw_ppg: float, starts: int, positional_mean_ppg: float) -> float:
    """Empirical-Bayes-style shrinkage toward the positional mean, weighted by
    how many starts the raw points-per-game figure is actually based on - see
    module docstring for why this was added (backtested overprojection fix)."""
    weight = starts / (starts + SHRINKAGE_K)
    return positional_mean_ppg + weight * (raw_ppg - positional_mean_ppg)


# Per-fixture points bonus for nominated set-piece takers - a structural
# signal `points_per_game` can't see until it's already shown up in results
# (e.g. a player who just became the #1 penalty taker gets no credit for that
# in their PPG until they've actually scored some). NOT empirically fitted -
# a documented judgment call, same status as SHRINKAGE_K before it was
# validated. Order 1 = primary taker, 2 = backup, 3+ treated as negligible.
# Penalties are worth the most (a scored penalty is a near-certain goal, ~4-6
# pts depending on position); direct free-kicks and corners/indirect
# free-kicks are lower-probability goal/assist chances.
SET_PIECE_BONUS = {
    "penalties_order": {1: 0.3, 2: 0.05},
    "direct_freekicks_order": {1: 0.08, 2: 0.02},
    "corners_and_indirect_freekicks_order": {1: 0.08, 2: 0.02},
}


def _set_piece_bonus(player: dict) -> float:
    """Sum of per-gameweek bonuses for every set-piece duty this player holds."""
    total = 0.0
    for field, tiers in SET_PIECE_BONUS.items():
        order = player.get(field)
        if order is not None:
            total += tiers.get(int(order), 0.0)
    return total


def _gameweek_effectively_finished(gw: int, event: dict, fixtures: list[dict]) -> bool:
    """FPL's own `event['finished']` flag lags real completion by up to a day
    or two - it only flips once bonus points are OFFICIALLY confirmed, not
    once matches actually end. Relying on it alone means the model can keep
    treating an already-fully-played gameweek as "upcoming" for a while
    after every match has finished, which corrupts `ep_next` (meaningless
    for a gameweek that's already been played) and the whole projection
    window built on top of it - this caused real, wrong transfer/captaincy
    output once already. `finished_provisional` flips per-fixture as soon as
    a match ends (before bonus is confirmed) and is a much more reliable
    "has this actually been played" signal - treat the gameweek as finished
    once every one of its fixtures is at least provisionally done."""
    if event["finished"]:
        return True
    gw_fixtures = [f for f in fixtures if f.get("event") == gw]
    return bool(gw_fixtures) and all(f.get("finished_provisional") for f in gw_fixtures)


def next_gameweeks(bootstrap: dict, fixtures: list[dict], horizon: int) -> list[int]:
    events = bootstrap["events"]
    upcoming = [e["id"] for e in events if not _gameweek_effectively_finished(e["id"], e, fixtures)]
    upcoming.sort()
    return upcoming[:horizon]


def project_players(
    bootstrap: dict, fixtures: list[dict], horizon: int = 5,
    fixture_mode: str = "fdr", set_piece_bonus: bool = False, ep_next_weight: float = 1.0,
) -> pd.DataFrame:
    """Return a DataFrame with one row per player and a `predicted_points` column
    summing projected points across the next `horizon` gameweeks.

    `fixture_mode`: "fdr" (default) uses FPL's own 1-5 difficulty rating;
    "strength" uses the data-driven team-strength differential instead (see
    fpl/team_strength.py). "strength" is a candidate replacement, not yet
    proven better - compare both via scripts/backtest_season.py --fixture-mode
    before switching the default.

    `set_piece_bonus`: adds SET_PIECE_BONUS per fixture for nominated
    penalty/free-kick/corner takers. Off by default - not yet proven to help,
    same status as fixture_mode="strength". Validate before switching it on.

    `ep_next_weight`: how much of the FIRST gameweek's estimate comes from
    FPL's `ep_next` (1.0 = all of it, today's default) vs this model's own
    shrunk-PPG x fixture-difficulty figure (0.0). In live data ep_next equals
    FPL's `form` (a ~2-game average mid-season), so it's un-shrunk recent
    form - see scripts/backtest_season.py --ep-weight for how it was tested."""
    if fixture_mode not in ("fdr", "strength"):
        raise ValueError(f"fixture_mode must be 'fdr' or 'strength', got {fixture_mode!r}")

    teams = {t["id"]: t for t in bootstrap["teams"]}
    fixture_lookup = build_team_fixture_lookup(fixtures)
    gameweeks = next_gameweeks(bootstrap, fixtures, horizon)
    if not gameweeks:
        raise RuntimeError("No upcoming gameweeks found - season may be over.")
    next_gw = gameweeks[0]
    positional_means = _positional_mean_ppg(bootstrap)
    strength_z = team_strength.compute_team_strength_z(bootstrap) if fixture_mode == "strength" else {}

    rows = []
    for player in bootstrap["elements"]:
        team_id = player["team"]
        team_fixtures = fixture_lookup.get(team_id, {})
        availability = _availability_multiplier(player)
        later_availability = _later_gameweek_availability(player)
        baseline_ppg = float(player.get("points_per_game") or 0.0)
        starts = int(player.get("starts") or 0)
        shrunk_ppg = _shrink_ppg(baseline_ppg, starts, positional_means[player["element_type"]])

        bonus_per_fixture = _set_piece_bonus(player) if set_piece_bonus else 0.0

        def fixture_points(gw: int, avail: float) -> float:
            total = 0.0
            for difficulty, is_home, opponent_id in team_fixtures.get(gw, []):
                if fixture_mode == "strength":
                    multiplier = _fixture_strength_multiplier(strength_z[team_id], strength_z[opponent_id], is_home)
                else:
                    multiplier = _fixture_difficulty_multiplier(difficulty)
                total += (shrunk_ppg * multiplier + bonus_per_fixture) * avail
            return total

        ep_term = float(player.get("ep_next") or 0.0) * availability
        if ep_next_weight >= 1.0:
            predicted = ep_term
        else:
            predicted = ep_next_weight * ep_term + (1 - ep_next_weight) * fixture_points(gameweeks[0], availability)

        for gw in gameweeks[1:]:
            predicted += fixture_points(gw, later_availability)

        num_fixtures_in_horizon = sum(len(team_fixtures.get(gw, [])) for gw in gameweeks)

        rows.append(
            {
                "id": player["id"],
                "web_name": player["web_name"],
                "full_name": f"{player['first_name']} {player['second_name']}",
                "team": team_id,
                "team_short": teams[team_id]["short_name"],
                "position_id": player["element_type"],
                "position": POSITIONS[player["element_type"]],
                "price": player["now_cost"] / 10.0,
                "now_cost": player["now_cost"],
                "status": player["status"],
                "availability": availability,
                "selected_by_percent": float(player.get("selected_by_percent") or 0.0),
                "total_points_last": player.get("total_points", 0),
                "points_per_game": baseline_ppg,
                "shrunk_points_per_game": round(shrunk_ppg, 3),
                "starts": starts,
                "ep_next": float(player.get("ep_next") or 0.0),
                "fixtures_in_horizon": num_fixtures_in_horizon,
                "predicted_points": round(predicted, 2),
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["gameweeks"] = gameweeks
    df.attrs["next_gw"] = next_gw
    return df
