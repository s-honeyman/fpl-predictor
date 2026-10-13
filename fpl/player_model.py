"""Points-decomposition projection: minutes x what each player scores when he plays.

The original model multiplies FPL's `points_per_game` by a fixture factor.
Two structural problems with that, both visible in real decisions:
  - `points_per_game` is per APPEARANCE, not per gameweek, so a rotation
    player (2 apps, 20 pts -> "10 a game") looks like a nailed-on star;
  - one fixture factor is applied to every point a player scores, when a
    tough fixture hits a defender's clean sheet and a forward's goals in
    completely different ways.

Here each player-fixture is projected as
    P(60+ mins) x [2 + goals + assists + clean sheet - goals-conceded + other]
  + P(1-59 mins) x (typical points for a short appearance)
where
  - P(60+), P(1-59): recency-weighted share of his team's recent matches,
    shrunk toward last season's share (or a low prior for unknowns);
  - goals/assists: his expected-goals/assists rate per 90 (this season,
    shrunk toward last season's rate), scaled by how many goals his team is
    expected to score in THIS fixture (fpl/team_ratings.py);
  - clean sheet / goals conceded: that fixture's clean-sheet probability and
    expected goals against, at FPL's per-position scoring;
  - other: his average leftover per 60+ appearance once appearance, goals,
    assists, clean sheets and conceded have been removed - bonus, defensive
    contribution, saves, cards - shrunk toward the position average.

Constants are documented judgements except the minutes prior (tuned on
one season, confirmed on another); whether this
beats the original is decided by scripts/backtest_player_model.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
MINUTES_HALF_LIFE = 4.0  # gameweeks
MINUTES_PRIOR_MATCHES = 1.0  # tuned on 2024-25, confirmed on 2025-26 (backtest_player_model)
RATE_PRIOR_MINUTES = 900.0
OTHER_PRIOR_APPS = 5.0
UNKNOWN_P60, UNKNOWN_PSUB = 0.4, 0.15  # newcomers: tuned with the line above

HISTORY_COLS = ["gw", "player_id", "team_id", "et", "minutes", "points", "goals", "assists",
                "xg", "xa", "cs", "conceded", "nfix"]


def history_from_live(bootstrap: dict, live_by_gw: dict[int, dict], fixtures: list[dict]) -> pd.DataFrame:
    """One row per player per finished gameweek from FPL's event/{gw}/live data."""
    el = {p["id"]: p for p in bootstrap["elements"]}
    nfix = _fixtures_per_team_gw(fixtures)
    rows = []
    for gw, live in live_by_gw.items():
        for e in live["elements"]:
            p = el.get(e["id"])
            if p is None:
                continue
            s = e["stats"]
            rows.append((gw, e["id"], p["team"], p["element_type"], s["minutes"], s["total_points"],
                         s["goals_scored"], s["assists"], float(s["expected_goals"] or 0),
                         float(s["expected_assists"] or 0), s["clean_sheets"], s["goals_conceded"],
                         nfix.get((p["team"], gw), 0)))
    return pd.DataFrame(rows, columns=HISTORY_COLS)


def _fixtures_per_team_gw(fixtures: list[dict]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for f in fixtures:
        if f.get("event") is None:
            continue
        for t in (f["team_h"], f["team_a"]):
            out[(t, f["event"])] = out.get((t, f["event"]), 0) + 1
    return out


def _expected_conceded_penalty(lam: float) -> float:
    """E[floor(G/2)] for G ~ Poisson(lam): defenders/keepers lose 1 pt per 2 conceded."""
    total, p = 0.0, math.exp(-lam)
    for g in range(1, 15):
        p *= lam / g
        total += (g // 2) * p
    return total


def _player_profiles(history: pd.DataFrame, as_of_gw: int, priors: dict[int, dict],
                     pos_rates: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """Per-player minutes probabilities, attacking rates and leftover points."""
    h = history[(history["gw"] < as_of_gw) & (history["nfix"] > 0)].copy()
    h["w"] = 0.5 ** ((as_of_gw - 1 - h["gw"]) / MINUTES_HALF_LIFE)
    h["full"] = (h["minutes"] >= 60).astype(float)
    h["sub"] = ((h["minutes"] > 0) & (h["minutes"] < 60)).astype(float)

    g = h.groupby("player_id")
    prof = pd.DataFrame({
        "W": g["w"].sum(),
        "x60": (h["w"] * h["full"]).groupby(h["player_id"]).sum(),
        "xsub": (h["w"] * h["sub"]).groupby(h["player_id"]).sum(),
        "mins": g["minutes"].sum(),
        "xg": g["xg"].sum(),
        "xa": g["xa"].sum(),
    })

    # Leftover per 60+ appearance (bonus, defensive contribution, saves, cards...),
    # single-fixture gameweeks only so the appearance arithmetic is exact.
    f = h[(h["full"] == 1) & (h["nfix"] == 1)].copy()
    f["base"] = (2 + f["goals"] * f["et"].map(GOAL_PTS) + f["assists"] * 3
                 + f["cs"] * f["et"].map(CS_PTS) - np.where(f["et"] <= 2, f["conceded"] // 2, 0))
    f["other"] = f["points"] - f["base"]
    prof = prof.join(f.groupby("player_id")["other"].agg(other_sum="sum", other_n="size"), how="left")
    prof[["other_sum", "other_n"]] = prof[["other_sum", "other_n"]].fillna(0)

    et_of = history.drop_duplicates("player_id").set_index("player_id")["et"]
    prof["et"] = et_of.reindex(prof.index)
    pos_other = (f.groupby("et")["other"].mean()).to_dict()
    pos_sub = h[h["sub"] == 1].groupby("et")["points"].mean().to_dict()

    p60, psub, xg90, xa90, other = [], [], [], [], []
    for pid, r in prof.iterrows():
        pr = priors.get(pid, {})
        k = MINUTES_PRIOR_MATCHES
        p60_0 = pr.get("p60", UNKNOWN_P60)
        psub_0 = pr.get("psub", UNKNOWN_PSUB)
        p60.append((r.x60 + k * p60_0) / (r.W + k))
        psub.append((r.xsub + k * psub_0) / (r.W + k))
        m0 = RATE_PRIOR_MINUTES
        pg, pa = pos_rates.get(int(r.et), (0.0, 0.0))
        xg90.append(90 * (r.xg + pr.get("xg90", pg) / 90 * m0) / (r.mins + m0))
        xa90.append(90 * (r.xa + pr.get("xa90", pa) / 90 * m0) / (r.mins + m0))
        po = pos_other.get(r.et, 0.5)
        other.append((r.other_sum + OTHER_PRIOR_APPS * po) / (r.other_n + OTHER_PRIOR_APPS))
    prof["p60"], prof["psub"], prof["xg90"], prof["xa90"], prof["other"] = p60, psub, xg90, xa90, other
    prof["sub_pts"] = prof["et"].map(pos_sub).fillna(1.0)
    return prof


def project(elements: list[dict], history: pd.DataFrame, fixture_lookup: dict, gameweeks: list[int],
            ratings, priors: dict[int, dict], pos_rates: dict[int, tuple[float, float]],
            availability_now: dict[int, float] | None = None,
            availability_later: dict[int, float] | None = None) -> dict[int, tuple[float, float]]:
    """Returns player_id -> (points over `gameweeks`, points in the first gameweek)."""
    prof = _player_profiles(history, gameweeks[0], priors, pos_rates)
    avail_now = availability_now or {}
    avail_later = availability_later or {}
    out = {}
    for p in elements:
        pid, team, et = p["id"], p["team"], p["element_type"]
        if pid in prof.index:
            r = prof.loc[pid]
            p60, psub, xg90, xa90, other, sub_pts = r.p60, r.psub, r.xg90, r.xa90, r.other, r.sub_pts
        else:
            pr = priors.get(pid, {})
            p60, psub = pr.get("p60", UNKNOWN_P60), pr.get("psub", UNKNOWN_PSUB)
            pg, pa = pos_rates.get(et, (0.0, 0.0))
            xg90, xa90, other, sub_pts = pr.get("xg90", pg), pr.get("xa90", pa), 0.5, 1.0
        total = first = 0.0
        for i, gw in enumerate(gameweeks):
            a = avail_now.get(pid, 1.0) if i == 0 else avail_later.get(pid, 1.0)
            for _, is_home, opp in fixture_lookup.get(team, {}).get(gw, []):
                gf, ga = ratings.expected_goals(team, opp, is_home)
                r_att, _ = ratings.ratios(team, opp, is_home)
                full = (2 + GOAL_PTS[et] * xg90 * 0.95 * r_att + 3 * xa90 * 0.95 * r_att
                        + CS_PTS[et] * math.exp(-ga)
                        - (_expected_conceded_penalty(ga) if et <= 2 else 0.0)
                        + other)
                pts = a * (p60 * full + psub * sub_pts)
                total += pts
                if i == 0:
                    first += pts
        out[pid] = (total, first)
    return out


def priors_from_previous_season(prev_players: pd.DataFrame, key_of_prev, current_ids_by_key: dict) -> dict[int, dict]:
    """Last season's minutes share and attacking rates, keyed to this season's
    player ids by name. Players without last-season data get the defaults."""
    out = {}
    for _, r in prev_players.iterrows():
        pid = current_ids_by_key.get(key_of_prev(r))
        if pid is None:
            continue
        mins = float(r.get("minutes", 0) or 0)
        starts = float(r.get("starts", 0) or 0)
        prior = {"p60": float(np.clip(starts / 38, 0.05, 0.92)), "psub": 0.12}
        if mins >= 450:  # otherwise fall back to the position average
            prior["xg90"] = 90 * float(r.get("expected_goals", 0) or 0) / mins
            prior["xa90"] = 90 * float(r.get("expected_assists", 0) or 0) / mins
        out[pid] = prior
    return out


def position_rates(prev_players: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Average expected goals / assists per 90 by position among last season's
    regulars (900+ minutes) - the default for anyone without their own record."""
    regular = prev_players[prev_players["minutes"] >= 900]
    out = {}
    for et, g in regular.groupby("element_type"):
        m = g["minutes"].sum()
        out[int(et)] = (90 * g["expected_goals"].astype(float).sum() / m, 90 * g["expected_assists"].astype(float).sum() / m)
    return out


def live_inputs(bootstrap: dict, fixtures: list[dict]):
    """Everything the decomposition needs for the live season: history from
    FPL's per-gameweek live data, last season's priors/position rates from the
    archive, the goals model, and the fixture lookup."""
    from . import api, model, team_ratings

    finished = [e["id"] for e in bootstrap["events"] if model._gameweek_effectively_finished(e["id"], e, fixtures)]
    hist = history_from_live(bootstrap, {gw: api.get_gameweek_live(gw) for gw in finished}, fixtures)
    start_year = int(bootstrap["events"][0]["deadline_time"][:4])
    prev = f"{start_year - 1}-{str(start_year)[2:]}"
    prev_players = pd.read_csv(api.get_archive_csv(f"{prev}/players_raw.csv", f"players_{prev}.csv"))
    ids_by_name = {f"{p['first_name']} {p['second_name']}": p["id"] for p in bootstrap["elements"]}
    priors = priors_from_previous_season(prev_players, lambda r: f"{r['first_name']} {r['second_name']}", ids_by_name)
    ratings, _ = team_ratings.live_ratings(bootstrap, fixtures)
    return hist, model.build_team_fixture_lookup(fixtures), ratings, priors, position_rates(prev_players)


def live_projection(bootstrap: dict, fixtures: list[dict], gameweeks: list[int],
                    availability_now: dict[int, float], availability_later: dict[int, float]) -> dict[int, tuple[float, float]]:
    """Returns player_id -> (points over gameweeks, first gameweek) for the live season."""
    hist, lookup, ratings, priors, rates = live_inputs(bootstrap, fixtures)
    return project(bootstrap["elements"], hist, lookup, gameweeks, ratings, priors, rates,
                   availability_now, availability_later)


def fixture_params(elements: list[dict], history: pd.DataFrame, fixture_lookup: dict, gameweeks: list[int],
                   ratings, priors: dict[int, dict], pos_rates: dict[int, tuple[float, float]],
                   availability_now: dict[int, float] | None = None,
                   availability_later: dict[int, float] | None = None) -> pd.DataFrame:
    """One row per player-fixture with every ingredient of the projection, so
    callers can take expected points per gameweek or simulate a distribution
    (captaincy ceilings) from the same numbers `project` uses."""
    prof = _player_profiles(history, gameweeks[0], priors, pos_rates)
    avail_now, avail_later = availability_now or {}, availability_later or {}
    rows = []
    for p in elements:
        pid, team, et = p["id"], p["team"], p["element_type"]
        if pid in prof.index:
            r = prof.loc[pid]
            p60, psub, xg90, xa90, other, sub_pts = r.p60, r.psub, r.xg90, r.xa90, r.other, r.sub_pts
        else:
            pr = priors.get(pid, {})
            pg, pa = pos_rates.get(et, (0.0, 0.0))
            p60, psub = pr.get("p60", UNKNOWN_P60), pr.get("psub", UNKNOWN_PSUB)
            xg90, xa90, other, sub_pts = pr.get("xg90", pg), pr.get("xa90", pa), 0.5, 1.0
        for i, gw in enumerate(gameweeks):
            a = avail_now.get(pid, 1.0) if i == 0 else avail_later.get(pid, 1.0)
            for _, is_home, opp in fixture_lookup.get(team, {}).get(gw, []):
                gf, ga = ratings.expected_goals(team, opp, is_home)
                r_att, _ = ratings.ratios(team, opp, is_home)
                lam_g, lam_a = xg90 * 0.95 * r_att, xa90 * 0.95 * r_att
                full = (2 + GOAL_PTS[et] * lam_g + 3 * lam_a + CS_PTS[et] * math.exp(-ga)
                        - (_expected_conceded_penalty(ga) if et <= 2 else 0.0) + other)
                rows.append((pid, gw, et, a * p60, a * psub, sub_pts, lam_g, lam_a, ga, other,
                             a * (p60 * full + psub * sub_pts)))
    return pd.DataFrame(rows, columns=["player_id", "gw", "et", "p60", "psub", "sub_pts", "lam_goals",
                                       "lam_assists", "lam_conceded", "other", "xpts"])


def by_gameweek(params: pd.DataFrame, gameweeks: list[int]) -> dict[int, np.ndarray]:
    """player_id -> expected points for each gameweek in `gameweeks` (doubles summed, blanks 0)."""
    piv = params.pivot_table(index="player_id", columns="gw", values="xpts", aggfunc="sum").reindex(columns=gameweeks).fillna(0.0)
    return {int(pid): np.array(row, dtype=float) for pid, row in piv.iterrows()}


def simulate(params: pd.DataFrame, n: int = 20000, seed: int = 11) -> np.ndarray:
    """Simulated FPL points for one player's fixtures in one gameweek (rows of
    `params` for that player and gameweek). Goals, assists and goals conceded
    are Poisson; clean sheet follows from conceded; bonus and other extras are
    held at his average, so ceilings are if anything slightly understated."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    for r in params.itertuples():
        u = rng.random(n)
        full = u < r.p60
        sub = (~full) & (u < r.p60 + r.psub)
        goals = rng.poisson(r.lam_goals, n)
        assists = rng.poisson(r.lam_assists, n)
        conceded = rng.poisson(r.lam_conceded, n)
        pts = (2 + GOAL_PTS[r.et] * goals + 3 * assists + CS_PTS[r.et] * (conceded == 0)
               - (conceded // 2 if r.et <= 2 else 0) + r.other)
        total += np.where(full, pts, 0) + np.where(sub, r.sub_pts, 0)
    return total
