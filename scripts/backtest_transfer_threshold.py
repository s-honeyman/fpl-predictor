"""Does a minimum-gain bar on transfers score more than acting on every
projected gain? Weekly replay of two past seasons, no lookahead.

Each week the planner (fpl/planner.py) finds its best 5-week plan. Holding is
valued the same way: this week's XI from the current squad, then the best
plan over the remaining weeks with the rolled transfer. This week's moves are
made only if the plan beats holding by at least the bar. Every bar runs on
the same projections in one pass, each with its own squad, bank and free
transfers, scored on real points (XI and captain chosen on projections).

Live this season: 8 transfers before the GW9 Wildcard netted 0 points against
holding (after a -4); the question is whether a bar would have filtered the
churn without losing the moves that matter.

Usage: uv run python scripts/backtest_transfer_threshold.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest_goals_model as bgm
import backtest_player_model as bpm
from backtest_planner import H, choose_xi
from fpl import model, optimizer, planner, player_model as pmod

BARS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]


def hold_value(squad, bank, ft, proj, pos, team, price, n):
    """This week's XI from the current squad, then the best plan from next week."""
    now = planner.best_xi(squad, 0, proj, pos)
    if n == 1:
        return now
    shifted = {i: a[1:] for i, a in proj.items()}
    rest = planner.plan_transfers(squad, bank, min(planner.MAX_FREE, ft + 1), shifted, pos, team, price, n - 1, beam=20, k=6)
    return now + rest.net


def run(prior, current):
    merged, teams, fixtures, fl, hist, priors, pos_rates, gm_inputs = bpm.season(prior, current)
    lookup = model.build_team_fixture_lookup(fl)
    actual = merged.groupby(["round", "element"])["total_points"].sum()
    rounds = sorted(merged["round"].unique())
    last_price, last_team, pos = {}, {}, {}
    state, totals, hits, moves_made = {}, {b: 0 for b in BARS}, {b: 0 for b in BARS}, {b: 0 for b in BARS}
    for t in rounds:
        now = merged[merged["round"] == t].groupby("element").agg(team=("team_id", "first"), et=("et", "first"), price=("value", "first"))
        for i, r in now.iterrows():
            last_price[i], last_team[i], pos[i] = int(r.price), int(r.team), int(r.et)
        gws = [g for g in range(t, t + H) if g <= 38]
        ratings, _ = bgm.goals_model_as_of(t, *gm_inputs)
        elements = [{"id": i, "team": int(r.team), "element_type": int(r.et)} for i, r in now.iterrows()]
        params = pmod.fixture_params(elements, hist, lookup, gws, ratings, priors, pos_rates)
        proj = pmod.by_gameweek(params, gws)
        for s in state.values():
            for i in s["squad"]:
                proj.setdefault(i, np.zeros(len(gws)))

        if not state:
            df = pd.DataFrame([{"id": i, "predicted_points": proj[i].sum(), "now_cost": last_price[i], "position_id": pos[i],
                                "team": last_team[i], "availability": 1.0} for i in now.index])
            r = optimizer.optimize_squad(df, budget=model.BUDGET_TENTHS)
            for b in BARS:
                state[b] = {"squad": set(r.squad_ids), "bank": model.BUDGET_TENTHS - r.total_cost, "ft": 1}
        else:
            cache = {}
            for b, s in state.items():
                key = (frozenset(s["squad"]), s["bank"], s["ft"])
                if key not in cache:
                    pl = planner.plan_transfers(s["squad"], s["bank"], s["ft"], proj, pos, last_team, last_price, len(gws), beam=20, k=6)
                    gain = pl.net - hold_value(s["squad"], s["bank"], s["ft"], proj, pos, last_team, last_price, len(gws)) if pl.moves[0] else 0.0
                    cache[key] = (pl.moves[0], gain)
                mv, gain = cache[key]
                if mv and gain < b:
                    mv = []
                for o, i in mv:
                    s["squad"] = (s["squad"] - {o}) | {i}
                    s["bank"] += last_price[o] - last_price[i]
                hits[b] += planner.HIT * max(0, len(mv) - s["ft"])
                moves_made[b] += len(mv)
                s["ft"] = min(planner.MAX_FREE, max(0, s["ft"] - len(mv)) + 1)

        for b, s in state.items():
            xi, c = choose_xi(s["squad"], proj, pos)
            totals[b] += sum(actual.get((t, i), 0) for i in xi) + actual.get((t, c), 0)
    return totals, hits, moves_made


def main():
    for prior, current in bpm.SEASONS:
        totals, hits, moves = run(prior, current)
        print(f"\n=== {current} ===")
        print(f"{'bar':>5} {'net pts':>8} {'hits':>5} {'transfers':>10}")
        for b in BARS:
            print(f"{b:5.1f} {totals[b] - hits[b]:8d} {hits[b]:5d} {moves[b]:10d}")


if __name__ == "__main__":
    main()
