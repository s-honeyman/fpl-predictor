"""Bookmaker-implied expected goals per match, as a drop-in replacement for the
goals model's fixture estimates (fpl/team_ratings.py).

Source: football-data.co.uk's free season files, averaged across bookmakers
(AvgH/AvgD/AvgA and Avg>2.5/Avg<2.5). Those are collected before the
weekend's games (Friday for weekend rounds) - roughly what's known at the FPL
deadline - not closing odds, so a backtest using them has no look-ahead.

Conversion: with independent Poisson goals, total goals ~ Poisson(lh + la), so
the over/under 2.5 price fixes lh + la exactly; the home/away win prices then
fix how that total splits. Both solved by bisection after removing the
bookmaker margin.

The point: odds already price in team news, form and tactics that a model fed
only on past scorelines can't see. Whether that improves FPL projections is
decided by scripts/backtest_market_odds.py, not assumed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ODDS_DIR = Path(__file__).resolve().parent.parent / "data" / "odds"
BASE_URL = "https://football-data.co.uk/mmz4281"
TYPICAL_PRIOR_MATCHES = 3.0

# football-data.co.uk name -> FPL name, where they differ
ALIASES = {
    "Man United": "Man Utd", "Tottenham": "Spurs", "Sheffield United": "Sheffield Utd",
    "Coventry": "Coventry City", "Hull": "Hull City", "Ipswich": "Ipswich Town",
}


def _norm(name: str) -> str:
    return name.lower().replace("'", "").replace(".", "").strip()


def match_team(fd_name: str, fpl_names: dict[int, str]) -> int | None:
    """football-data team name -> FPL team id, tolerating 'Ipswich' vs 'Ipswich Town'."""
    target = _norm(ALIASES.get(fd_name, fd_name))
    by_norm = {_norm(n): i for i, n in fpl_names.items()}
    if target in by_norm:
        return by_norm[target]
    short = _norm(fd_name)
    hits = [i for n, i in by_norm.items() if n.startswith(short) or short.startswith(n)]
    return hits[0] if len(hits) == 1 else None


def season_file(season: str, refresh: bool = False) -> Path:
    """'2025-26' -> cached E0 csv for that season (current season re-fetched on refresh)."""
    code = season[2:4] + season[5:7]
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    path = ODDS_DIR / f"E0_{code}.csv"
    if refresh or not path.exists():
        r = requests.get(f"{BASE_URL}/{code}/E0.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def _p_over(total: float, line: float = 2.5) -> float:
    k = int(math.floor(line))
    return 1 - sum(math.exp(-total) * total ** i / math.factorial(i) for i in range(k + 1))


def _home_away_probs(lh: float, la: float, max_goals: int = 12) -> tuple[float, float]:
    ph = np.array([math.exp(-lh) * lh ** i / math.factorial(i) for i in range(max_goals)])
    pa = np.array([math.exp(-la) * la ** i / math.factorial(i) for i in range(max_goals)])
    grid = np.outer(ph, pa)
    return float(np.tril(grid, -1).sum()), float(np.triu(grid, 1).sum())


def implied_goals(odds_h: float, odds_d: float, odds_a: float, odds_over: float, odds_under: float) -> tuple[float, float]:
    """(home expected goals, away expected goals) implied by 1X2 and O/U 2.5 prices."""
    inv = np.array([1 / odds_h, 1 / odds_d, 1 / odds_a])
    p_h, _, p_a = inv / inv.sum()
    p_over = (1 / odds_over) / (1 / odds_over + 1 / odds_under)

    lo, hi = 0.2, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _p_over(mid) < p_over else (lo, mid)
    total = (lo + hi) / 2

    target = p_h / (p_h + p_a)  # share of decisive results going to the home side
    lo, hi = 0.02, 0.98
    for _ in range(50):
        s = (lo + hi) / 2
        wh, wa = _home_away_probs(total * s, total * (1 - s))
        lo, hi = (s, hi) if wh / (wh + wa) < target else (lo, s)
    s = (lo + hi) / 2
    return total * s, total * (1 - s)


def load_match_goals(path: Path, fpl_names: dict[int, str]) -> dict[tuple[int, int], tuple[float, float]]:
    """(home_id, away_id) -> (home xG, away xG) for every match in a season file
    with complete odds. Raises if a team can't be matched, rather than silently
    dropping its matches."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = {}
    cols = ["AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"]
    for r in df.itertuples(index=False):
        row = r._asdict()
        h, a = match_team(row["HomeTeam"], fpl_names), match_team(row["AwayTeam"], fpl_names)
        if h is None or a is None:
            raise ValueError(f"Can't match odds team names {row['HomeTeam']!r}/{row['AwayTeam']!r} to FPL teams")
        vals = [r[df.columns.get_loc(c)] for c in cols]
        if any(pd.isna(v) for v in vals):
            continue
        out[(h, a)] = implied_goals(*vals)
    return out


class OddsAdjustedRatings:
    """Same interface as team_ratings.TeamRatings. Matches with odds use the
    market's expected goals; anything else falls back to the goals model.
    `ratios` compares a fixture with the team's typical one, where 'typical'
    is its average market expected goals over matches already played (the
    same matches its players' rates were measured on), shrunk toward the
    goals model's typical figure early on."""

    def __init__(self, base, match_goals: dict[tuple[int, int], tuple[float, float]],
                 match_gw: dict[tuple[int, int], int], as_of_gw: int):
        self.base = base
        self.goals = match_goals
        self._typ_for, self._typ_cs = {}, {}
        sums: dict[int, list] = {}
        for (h, a), gw in match_gw.items():
            if gw >= as_of_gw or (h, a) not in match_goals:
                continue
            lh, la = match_goals[(h, a)]
            sums.setdefault(h, []).append((lh, math.exp(-la)))
            sums.setdefault(a, []).append((la, math.exp(-lh)))
        k = TYPICAL_PRIOR_MATCHES
        for t in base._typ_for:
            seen = sums.get(t, [])
            n = len(seen)
            self._typ_for[t] = (sum(x for x, _ in seen) + k * base._typ_for[t]) / (n + k)
            self._typ_cs[t] = (sum(c for _, c in seen) + k * base._typ_cs[t]) / (n + k)

    def expected_goals(self, team_id: int, opp_id: int, is_home: bool) -> tuple[float, float]:
        key = (team_id, opp_id) if is_home else (opp_id, team_id)
        if key in self.goals:
            lh, la = self.goals[key]
            return (lh, la) if is_home else (la, lh)
        return self.base.expected_goals(team_id, opp_id, is_home)

    def clean_sheet_prob(self, team_id: int, opp_id: int, is_home: bool) -> float:
        return float(math.exp(-self.expected_goals(team_id, opp_id, is_home)[1]))

    def ratios(self, team_id: int, opp_id: int, is_home: bool) -> tuple[float, float]:
        from .team_ratings import CAP

        key = (team_id, opp_id) if is_home else (opp_id, team_id)
        if key not in self.goals:  # no odds for this fixture: stay entirely on the goals model's scale
            return self.base.ratios(team_id, opp_id, is_home)
        gf, ga = self.expected_goals(team_id, opp_id, is_home)
        return (float(np.clip(gf / self._typ_for[team_id], *CAP)),
                float(np.clip(math.exp(-ga) / self._typ_cs[team_id], *CAP)))


def upcoming_file(refresh: bool = False) -> Path:
    """football-data.co.uk's fixtures.csv: odds for the next round, published
    a few days before it (often only from midweek)."""
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    path = ODDS_DIR / "fixtures.csv"
    if refresh or not path.exists():
        r = requests.get("https://football-data.co.uk/fixtures.csv", headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def live_match_goals(bootstrap: dict, refresh: bool = False) -> dict[tuple[int, int], tuple[float, float]]:
    """Market expected goals for this season's played matches plus any
    upcoming ones football-data has priced. Empty if the files aren't reachable."""
    names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    start_year = int(bootstrap["events"][0]["deadline_time"][:4])
    season = f"{start_year}-{str(start_year + 1)[2:]}"
    out = {}
    try:
        out.update(load_match_goals(season_file(season, refresh), names))
        up = upcoming_file(refresh)
        df = pd.read_csv(up, encoding="utf-8-sig")
        df = df[df["Div"] == "E0"]
        if len(df):
            tmp = ODDS_DIR / "upcoming_E0.csv"
            df.to_csv(tmp, index=False)
            out.update(load_match_goals(tmp, names))
    except (requests.RequestException, OSError, pd.errors.EmptyDataError):
        pass
    return out
