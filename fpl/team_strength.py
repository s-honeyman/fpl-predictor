"""Per-team strength score for fixture difficulty - a data-driven alternative
to FPL's own subjective 1-5 FDR rating.

Ported from pl-club-forecast's compute_team_strengths (see
../../pl-club-forecast/pl_club_forecast/strength.py) rather than imported
across repos, since the two projects are independent uv-managed packages and
this is the only piece fpl-predictor needs.

Blends two per-team signals, each summed over the team's top-K players:
  - last season's total_points, credited to whichever club a player is
    registered at NOW (rewards proven output; 0 for brand-new-to-the-league
    signings)
  - current transfer-market price, i.e. now_cost (captures new-signing
    reputation/quality that total_points can't see yet)

Equal-weight blend, standardized to a league-wide z-score. NOT empirically
fitted - same documented limitation as the pl-club-forecast original. Any
change here should be validated against scripts/backtest_season.py before
being trusted, exactly like SHRINKAGE_K in fpl/model.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_K_PLAYERS = 18
OUTPUT_WEIGHT = 0.5


def _top_k_sum(values: list[float], k: int = TOP_K_PLAYERS) -> float:
    return float(sum(sorted(values, reverse=True)[:k]))


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


def compute_team_strength_z(bootstrap: dict) -> dict[int, float]:
    """team_id -> standardized strength z-score (mean 0, std 1 across the league)."""
    players = pd.DataFrame(bootstrap["elements"])

    rows = []
    for team in bootstrap["teams"]:
        squad = players[players["team"] == team["id"]]
        output_score = _top_k_sum(squad["total_points"].tolist())
        value_score = _top_k_sum((squad["now_cost"] / 10.0).tolist())
        rows.append({"team_id": team["id"], "output_score": output_score, "value_score": value_score})

    df = pd.DataFrame(rows)
    df["strength_z"] = OUTPUT_WEIGHT * _zscore(df["output_score"]) + (1 - OUTPUT_WEIGHT) * _zscore(df["value_score"])
    return dict(zip(df["team_id"], df["strength_z"]))
