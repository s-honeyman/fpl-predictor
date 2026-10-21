"""Does replacing the goals model's fixture estimates with bookmaker-implied
expected goals (fpl/market_odds.py) improve the decomposed projections?

Same two tests as scripts/backtest_player_model.py, on the same seasons, no
lookahead: per-gameweek prediction quality over ~17,000 player-gameweeks,
and the 5-weekly season simulation. Only the fixture source changes.

Usage: uv run python scripts/backtest_market_odds.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import pandas as pd

import backtest_goals_model as bgm
import backtest_player_model as bpm
from fpl import market_odds as mo


def odds_wrapper(current: str, teams: pd.DataFrame, fl: list[dict]):
    names = dict(zip(teams["id"], teams["name"]))
    goals = mo.load_match_goals(mo.season_file(current), names)
    match_gw = {(f["team_h"], f["team_a"]): f["event"] for f in fl if f.get("event")}
    covered = sum(1 for k in match_gw if k in goals) / max(len(match_gw), 1)
    print(f"   odds found for {covered:.0%} of {current} fixtures")

    def wrap(base, gw):
        return mo.OddsAdjustedRatings(base, goals, match_gw, gw)
    return wrap


def run(prior, current):
    _, teams, _, fl, *_ = bpm.season(prior, current)
    wrap = odds_wrapper(current, teams, fl)
    original = bgm.goals_model_as_of
    results = {}
    for label, use_odds in (("goals model", False), ("market odds", True)):
        if use_odds:
            def patched(gw, *args, _orig=original):
                base, w = _orig(gw, *args)
                return wrap(base, gw), w
            bgm.goals_model_as_of = patched
        try:
            q, n, _ = bpm.prediction_quality(prior, current)
            sim = bpm.dynamic(prior, current)
        finally:
            bgm.goals_model_as_of = original
        results[label] = (q.loc["decomposed"], sim, n)
    return results


def main():
    for prior, current in bpm.SEASONS:
        print(f"\n=== {current} ===")
        res = run(prior, current)
        rows = {k: {**v[0].to_dict(), "season_sim": v[1]} for k, v in res.items()}
        print(f"Decomposed model, two fixture sources ({next(iter(res.values()))[2]:,} player-gameweeks):")
        print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
