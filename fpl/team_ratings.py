"""Team attack/defence ratings fitted from real scorelines - a goals-based
alternative to FPL's 1-5 fixture difficulty rating (FDR).

Model (standard independent-Poisson football model):
    home goals ~ Poisson(exp(mu + home + att[h] - dfn[a]))
    away goals ~ Poisson(exp(mu        + att[a] - dfn[h]))
Higher `dfn` = concedes fewer. Fitted by ridge-penalised Newton/IRLS, so
teams with few matches are pulled toward a prior instead of overfitting a
handful of results. Last season's matches count at a reduced weight;
promoted teams (no top-flight data last season) get a prior centred on last
season's relegated sides rather than on the league average.

What it gives the projection model, per fixture: each side's expected goals
and the clean-sheet probability exp(-expected goals conceded). The earlier
team-strength experiment (fpl/team_strength.py) rated teams from FPL points
and prices and lost to FDR in backtest; this one uses actual goals for and
against, which is what clean sheets and attacking returns depend on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PREV_SEASON_WEIGHT = 0.5
RIDGE = 4.0  # penalty strength, roughly "this many matches' worth" of prior
CAP = (0.4, 2.5)  # clamp fixture-vs-typical ratios so tiny samples can't explode


@dataclass
class Match:
    home: str
    away: str
    home_goals: int
    away_goals: int
    weight: float = 1.0


def _fit(matches: list[Match], teams: list[str], prior_att: dict[str, float], prior_dfn: dict[str, float]):
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    p = 2 + 2 * n  # mu, home, att..., dfn...
    rows, y, w = [], [], []
    for m in matches:
        if m.home not in idx or m.away not in idx:
            continue
        h, a = idx[m.home], idx[m.away]
        r = np.zeros(p); r[0] = 1; r[1] = 1; r[2 + h] = 1; r[2 + n + a] = -1
        rows.append(r); y.append(m.home_goals); w.append(m.weight)
        r = np.zeros(p); r[0] = 1; r[2 + a] = 1; r[2 + n + h] = -1
        rows.append(r); y.append(m.away_goals); w.append(m.weight)
    X, y, w = np.array(rows), np.array(y, float), np.array(w, float)

    centre = np.zeros(p)
    for t, i in idx.items():
        centre[2 + i] = prior_att.get(t, 0.0)
        centre[2 + n + i] = prior_dfn.get(t, 0.0)
    pen = np.full(p, RIDGE); pen[:2] = 0.0

    theta = centre.copy()
    theta[0] = np.log(max(y.mean(), 0.1))
    for _ in range(50):
        eta = np.clip(X @ theta, -5, 5)
        lam = np.exp(eta)
        grad = X.T @ (w * (y - lam)) - pen * (theta - centre)
        hess = (X * (w * lam)[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(p)
        step = np.linalg.solve(hess, grad)
        theta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    att = {t: theta[2 + i] for t, i in idx.items()}
    dfn = {t: theta[2 + n + i] for t, i in idx.items()}
    return theta[0], theta[1], att, dfn


class TeamRatings:
    def __init__(self, mu: float, home: float, att: dict, dfn: dict, team_names: dict[int, str]):
        self.mu, self.home, self.att, self.dfn = mu, home, att, dfn
        self.name = team_names  # current-season team id -> name
        ids = list(team_names)
        # A team's "typical" fixture: its average over playing every other
        # current team once home and once away - the baseline a player's
        # points_per_game already reflects.
        self._typ_for, self._typ_cs = {}, {}
        for t in ids:
            lf, cs = [], []
            for o in ids:
                if o == t:
                    continue
                for home in (True, False):
                    f, a = self.expected_goals(t, o, home)
                    lf.append(f); cs.append(np.exp(-a))
            self._typ_for[t], self._typ_cs[t] = float(np.mean(lf)), float(np.mean(cs))

    def expected_goals(self, team_id: int, opp_id: int, is_home: bool) -> tuple[float, float]:
        t, o = self.name[team_id], self.name[opp_id]
        h = self.home if is_home else 0.0
        h_opp = 0.0 if is_home else self.home
        gf = np.exp(self.mu + h + self.att.get(t, 0.0) - self.dfn.get(o, 0.0))
        ga = np.exp(self.mu + h_opp + self.att.get(o, 0.0) - self.dfn.get(t, 0.0))
        return float(gf), float(ga)

    def clean_sheet_prob(self, team_id: int, opp_id: int, is_home: bool) -> float:
        return float(np.exp(-self.expected_goals(team_id, opp_id, is_home)[1]))

    def ratios(self, team_id: int, opp_id: int, is_home: bool) -> tuple[float, float]:
        """(attack ratio, clean-sheet ratio) of this fixture vs the team's typical fixture."""
        gf, ga = self.expected_goals(team_id, opp_id, is_home)
        r_att = np.clip(gf / self._typ_for[team_id], *CAP)
        r_cs = np.clip(np.exp(-ga) / self._typ_cs[team_id], *CAP)
        return float(r_att), float(r_cs)


def fit_ratings(
    current: list[Match],
    previous: list[Match],
    current_team_names: dict[int, str],
    relegated_last_season: list[str] | None = None,
) -> TeamRatings:
    """`current` = this season's played matches (weight 1); `previous` = last
    season's (down-weighted). Team identity is by name so last season's
    results line up with this season's teams."""
    prev = [Match(m.home, m.away, m.home_goals, m.away_goals, PREV_SEASON_WEIGHT * m.weight) for m in previous]
    teams = sorted({m.home for m in current + prev} | {m.away for m in current + prev} | set(current_team_names.values()))

    # Pass 1: plain fit to learn what last season's relegated teams looked like.
    prior_att, prior_dfn = {}, {}
    prev_teams = {m.home for m in previous} | {m.away for m in previous}
    promoted = [t for t in current_team_names.values() if t not in prev_teams]
    if previous and relegated_last_season and promoted:
        _, _, a0, d0 = _fit(prev, sorted(prev_teams), {}, {})
        rel = [t for t in relegated_last_season if t in a0]
        if rel:
            pa, pd_ = float(np.mean([a0[t] for t in rel])), float(np.mean([d0[t] for t in rel]))
            for t in promoted:
                prior_att[t], prior_dfn[t] = pa, pd_

    mu, home, att, dfn = _fit(current + prev, teams, prior_att, prior_dfn)
    return TeamRatings(mu, home, att, dfn, current_team_names)


def live_ratings(bootstrap: dict, fixtures: list[dict]) -> tuple[TeamRatings, dict]:
    """Ratings for the live season: this season's played matches plus last
    season's from the archive. Returns (ratings, position weights)."""
    import pandas as pd

    from . import api

    start_year = int(bootstrap["events"][0]["deadline_time"][:4])
    prev = f"{start_year - 1}-{str(start_year)[2:]}"
    names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    current = [
        Match(names[f["team_h"]], names[f["team_a"]], f["team_h_score"], f["team_a_score"])
        for f in fixtures if f.get("finished_provisional") and f.get("team_h_score") is not None
    ]
    pf = pd.read_csv(api.get_archive_csv(f"{prev}/fixtures.csv", f"fixtures_{prev}.csv"))
    pt = pd.read_csv(api.get_archive_csv(f"{prev}/teams.csv", f"teams_{prev}.csv"))
    pm = pd.read_csv(api.get_archive_csv(f"{prev}/gws/merged_gw.csv", f"merged_gw_{prev}.csv"))
    pn = dict(zip(pt["id"], pt["name"]))
    previous = [Match(pn[r.team_h], pn[r.team_a], int(r.team_h_score), int(r.team_a_score))
                for r in pf.itertuples() if r.finished and pd.notna(r.team_h_score)]
    relegated = sorted(set(pn.values()) - set(names.values()))
    return fit_ratings(current, previous, names, relegated), position_weights(pm)


def position_weights(merged_gw) -> dict[int, tuple[float, float]]:
    """Share of each position's FPL points that comes from attacking returns
    and from clean sheets, measured from a season's real per-match data.
    Returns element_type -> (attack_share, clean_sheet_share)."""
    pos_map = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    goal_pts = {1: 10, 2: 6, 3: 5, 4: 4}
    cs_pts = {1: 4, 2: 4, 3: 1, 4: 0}
    out = {}
    df = merged_gw[merged_gw["position"].isin(pos_map)].copy()
    df["et"] = df["position"].map(pos_map)
    for et, g in df.groupby("et"):
        total = g["total_points"].clip(lower=0).sum()
        att = (g["goals_scored"] * goal_pts[et] + g["assists"] * 3).sum()
        cs = (g["clean_sheets"] * cs_pts[et]).sum()
        out[int(et)] = (float(att / total), float(cs / total))
    return out


def fixture_multiplier(ratings: TeamRatings, weights: dict[int, tuple[float, float]],
                       team_id: int, opp_id: int, is_home: bool, element_type: int) -> float:
    """Scale a player's typical points for this specific fixture: only the
    attacking slice of his points moves with expected goals for, only the
    clean-sheet slice moves with clean-sheet probability."""
    w_att, w_cs = weights.get(element_type, (0.0, 0.0))
    r_att, r_cs = ratings.ratios(team_id, opp_id, is_home)
    return 1.0 + w_att * (r_att - 1.0) + w_cs * (r_cs - 1.0)
