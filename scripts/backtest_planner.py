"""Weekly season replay testing the multi-week planner (fpl/planner.py) and
captaincy rules, on real past seasons with no lookahead.

Every gameweek, using only data from before it: project the next 5 weeks
with the decomposed model, then manage the squad two ways, each under real
FPL rules (1 free transfer a week, roll over to 5, -4 per extra):
  one-shot - today's `fpl transfers` logic: best k transfers now for the
             5-week total, k up to free transfers + 1;
  planner  - plan the whole 5-week sequence, make only this week's moves.
Both start from the same GW1 squad and score real points from the XI and
captain chosen on projections (no hindsight, no auto-subs).

Captaincy is tested separately each week on the 15 highest-projected players:
captain by highest average vs by highest chance of 10+ points vs by 90th
percentile, using the model's own simulated distribution.

Usage: uv run python scripts/backtest_planner.py
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
from fpl import model, optimizer, planner, player_model as pmod

H = 5


def choose_xi(squad, proj, pos):
    vals = {i: proj.get(i, np.zeros(1))[0] for i in squad}
    by_pos = {p: sorted([i for i in squad if pos[i] == p], key=lambda i: -vals[i]) for p in (1, 2, 3, 4)}
    xi, rest = [], []
    for p in (1, 2, 3, 4):
        xi += by_pos[p][:planner.MIN_PLAY[p]]
        rest += by_pos[p][planner.MIN_PLAY[p]:planner.MAX_PLAY[p]]
    xi += sorted(rest, key=lambda i: -vals[i])[: 11 - len(xi)]
    return xi, max(xi, key=lambda i: vals[i])


def run(prior, current):
    merged, teams, fixtures, fl, hist, priors, pos_rates, gm_inputs = bpm.season(prior, current)
    lookup = model.build_team_fixture_lookup(fl)
    actual = merged.groupby(["round", "element"])["total_points"].sum()
    rounds = sorted(merged["round"].unique())
    last_price, last_team, pos = {}, {}, {}

    state = {}
    totals = {"one-shot": 0, "planner": 0}
    hits = {"one-shot": 0, "planner": 0}
    cap = {"mean": 0, "p10": 0, "p90": 0}
    for t in rounds:
        now = merged[merged["round"] == t].groupby("element").agg(team=("team_id", "first"), et=("et", "first"), price=("value", "first"))
        for i, r in now.iterrows():
            last_price[i], last_team[i], pos[i] = int(r.price), int(r.team), int(r.et)
        gws = [g for g in range(t, t + H) if g <= 38]
        ratings, _ = bgm.goals_model_as_of(t, *gm_inputs)
        elements = [{"id": i, "team": int(r.team), "element_type": int(r.et)} for i, r in now.iterrows()]
        params = pmod.fixture_params(elements, hist, lookup, gws, ratings, priors, pos_rates)
        proj = pmod.by_gameweek(params, gws)
        for i in set().union(*[s["squad"] for s in state.values()]) if state else set():
            proj.setdefault(i, np.zeros(len(gws)))  # owned but gone from the league: scores 0

        def df_for(ids):
            return pd.DataFrame([{"id": i, "predicted_points": proj[i].sum(), "now_cost": last_price[i],
                                  "position_id": pos[i], "team": last_team[i], "availability": 1.0} for i in ids])

        if not state:
            r = optimizer.optimize_squad(df_for(list(now.index)), budget=model.BUDGET_TENTHS)
            for k in totals:
                state[k] = {"squad": set(r.squad_ids), "bank": model.BUDGET_TENTHS - r.total_cost, "ft": 1}
        else:
            # one-shot
            s = state["one-shot"]
            ids = set(now.index) | s["squad"]
            df = df_for(ids)
            budget = sum(last_price[i] for i in s["squad"]) + s["bank"]
            best, best_net, best_k = None, None, 0
            for k in range(0, s["ft"] + 2):
                try:
                    c = optimizer.optimize_squad(df, budget=budget, old_squad_ids=list(s["squad"]), max_transfers=k)
                except RuntimeError:
                    continue
                net = c.predicted_starting_points - planner.HIT * max(0, k - s["ft"])
                if best_net is None or net > best_net + 1e-9:
                    best, best_net, best_k = c, net, k
            made = len(set(best.squad_ids) - s["squad"])
            hits["one-shot"] += planner.HIT * max(0, made - s["ft"])
            s.update(squad=set(best.squad_ids), bank=budget - best.total_cost,
                     ft=min(planner.MAX_FREE, max(0, s["ft"] - made) + 1))
            # planner
            s = state["planner"]
            pl = planner.plan_transfers(s["squad"], s["bank"], s["ft"], proj, pos, last_team, last_price, len(gws), beam=20, k=6)
            moves = pl.moves[0]
            for o, i in moves:
                s["squad"] = (s["squad"] - {o}) | {i}
                s["bank"] += last_price[o] - last_price[i]
            hits["planner"] += planner.HIT * max(0, len(moves) - s["ft"])
            s["ft"] = min(planner.MAX_FREE, max(0, s["ft"] - len(moves)) + 1)

        for k, s in state.items():
            xi, c = choose_xi(s["squad"], proj, pos)
            totals[k] += sum(actual.get((t, i), 0) for i in xi) + actual.get((t, c), 0)

        # captaincy rules on the 15 highest projected players this week
        first = params[params["gw"] == t]
        top = first.groupby("player_id")["xpts"].sum().nlargest(15).index
        stats = {}
        for pid in top:
            sim = pmod.simulate(first[first["player_id"] == pid], n=4000, seed=pid)
            stats[pid] = (sim.mean(), (sim >= 10).mean(), np.percentile(sim, 90))
        for rule, idx in (("mean", 0), ("p10", 1), ("p90", 2)):
            pick = max(stats, key=lambda p: (stats[p][idx], stats[p][0]))
            cap[rule] += actual.get((t, pick), 0)
    return {k: totals[k] - hits[k] for k in totals}, hits, cap


def main():
    for prior, current in bpm.SEASONS:
        net, hits, cap = run(prior, current)
        print(f"\n=== {current} ===")
        print(f"Transfers (season points after hits): one-shot {net['one-shot']} (hits -{hits['one-shot']})  |  "
              f"planner {net['planner']} (hits -{hits['planner']})")
        print(f"Captain's raw points over the season: by average {cap['mean']}  |  by chance of 10+ {cap['p10']}  |  "
              f"by 90th percentile {cap['p90']}")


if __name__ == "__main__":
    main()
