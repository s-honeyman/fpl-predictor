"""Backtests the points-decomposition model (fpl/player_model.py) against the
current model (points-per-game x FDR) and FPL's own per-gameweek estimate
(`xP` in the archive), on real past seasons with no lookahead.

Usage: uv run python scripts/backtest_player_model.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest_goals_model as bgm
import backtest_season as bt
from fpl import model, optimizer, player_model as pmod

SEASONS = bgm.SEASONS
POS = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


def season(prior: str, current: str):
    prev_players = pd.read_csv(bt._fetch(f"{prior}/players_raw.csv", f"players_{prior}.csv"))
    _, merged, teams, fixtures = bt.load_data(prior, current)
    merged = merged[merged["position"].isin(POS)].copy()
    team_id = dict(zip(teams["name"], teams["id"]))
    fl = bt.fixtures_as_list(fixtures)
    nfix = pmod._fixtures_per_team_gw(fl)
    merged["team_id"] = merged["team"].map(team_id)
    merged["et"] = merged["position"].map(POS)
    agg = merged.groupby(["element", "round"]).agg(
        team_id=("team_id", "first"), et=("et", "first"), minutes=("minutes", "sum"),
        points=("total_points", "sum"), goals=("goals_scored", "sum"), assists=("assists", "sum"),
        xg=("expected_goals", "sum"), xa=("expected_assists", "sum"), cs=("clean_sheets", "sum"),
        conceded=("goals_conceded", "sum")).reset_index()
    agg["nfix"] = [nfix.get((t, g), 0) for t, g in zip(agg["team_id"], agg["round"])]
    hist = agg.rename(columns={"element": "player_id", "round": "gw"})[pmod.HISTORY_COLS]

    ids_by_name = dict(zip(merged["name"], merged["element"]))
    priors = pmod.priors_from_previous_season(prev_players, lambda r: f"{r['first_name']} {r['second_name']}", ids_by_name)
    pos_rates = pmod.position_rates(prev_players)
    gm_inputs = bgm.season_inputs(prior, current)
    return merged, teams, fixtures, fl, hist, priors, pos_rates, gm_inputs


def prediction_quality(prior, current):
    merged, teams, fixtures, fl, hist, priors, pos_rates, gm_inputs = season(prior, current)
    lookup = model.build_team_fixture_lookup(fl)
    rows = []
    for t in range(4, 39):
        now = merged[merged["round"] == t]
        if now.empty:
            continue
        before = hist[hist["gw"] < t]
        apps = before[before["minutes"] > 0].groupby("player_id").agg(pts=("points", "sum"), n=("points", "size"))
        starts = before.groupby("player_id")["minutes"].apply(lambda m: (m >= 60).sum())
        ratings, _ = bgm.goals_model_as_of(t, *gm_inputs)
        cur = now.groupby("element").agg(team_id=("team_id", "first"), et=("et", "first"),
                                          actual=("total_points", "sum"), xP=("xP", "sum"))
        cur = cur[cur.index.isin(apps.index)]  # pool: anyone who's played this season
        elements = [{"id": i, "team": r.team_id, "element_type": r.et} for i, r in cur.iterrows()]
        v2 = pmod.project(elements, hist, lookup, [t], ratings, priors, pos_rates)
        info = cur.join(apps).join(starts.rename("starts")).fillna({"starts": 0})
        info["ppg"] = info["pts"] / info["n"]
        pm = {e: np.average(g.ppg, weights=g.starts + 1e-9) for e, g in info.groupby("et")}
        for pid, r in info.iterrows():
            fl_t = lookup.get(r.team_id, {}).get(t, [])
            if not fl_t:
                continue
            shrunk = pm[r.et] + r.starts / (r.starts + model.SHRINKAGE_K) * (r.ppg - pm[r.et])
            rows.append(dict(gw=t, actual=r.actual, fpl_xP=r.xP,
                             current=shrunk * sum(model._fixture_difficulty_multiplier(d) for d, _, _ in fl_t),
                             decomposed=v2[pid][0]))
    d = pd.DataFrame(rows)
    out = {}
    for m in ["current", "decomposed", "fpl_xP"]:
        out[m] = dict(
            spearman=d.groupby("gw").apply(lambda g: g[[m, "actual"]].corr(method="spearman").iloc[0, 1]).mean(),
            top30=d.groupby("gw").apply(lambda g: g.nlargest(30, m)["actual"].mean()).mean(),
            top100=d.groupby("gw").apply(lambda g: g.nlargest(100, m)["actual"].mean()).mean(),
            mae=(d[m] - d["actual"]).abs().mean(),
            bias=(d[m] - d["actual"]).mean())
    return pd.DataFrame(out).T.round(3), len(d), d


def dynamic(prior, current):
    merged, teams, fixtures, fl, hist, priors, pos_rates, gm_inputs = season(prior, current)
    prior_players, merged_full, teams, fixtures = bt.load_data(prior, current)
    lookup_prior = bt.build_prior_season_lookup(prior_players)
    lookup = model.build_team_fixture_lookup(fl)
    total, old, budget = 0, None, model.BUDGET_TENTHS
    for gw in range(1, 39, bt.RECHECK_EVERY_GW):
        ppg = lookup_prior if gw == 1 else bt.running_ppg_lookup(merged_full, gw)
        with contextlib.redirect_stdout(io.StringIO()):
            boot = bt.build_bootstrap(merged_full, teams, ppg, as_of_gw=gw)
        end = min(gw + bt.RECHECK_EVERY_GW, 39)
        ratings, _ = bgm.goals_model_as_of(gw, *gm_inputs)
        proj = pmod.project(boot["elements"], hist, lookup, list(range(gw, end)), ratings, priors, pos_rates)
        df = pd.DataFrame([{"id": e["id"], "predicted_points": proj[e["id"]][0], "now_cost": e["now_cost"],
                            "position_id": e["element_type"], "team": e["team"], "availability": 1.0}
                           for e in boot["elements"]])
        keep = [i for i in (old or []) if i in set(df["id"])]
        try:
            r = optimizer.optimize_squad(df, budget=budget, old_squad_ids=keep or None,
                                         max_transfers=bt.FREE_TRANSFERS_PER_CHECKPOINT if old else None)
        except RuntimeError:
            r = optimizer.optimize_squad(df, budget=model.BUDGET_TENTHS)
        total += bt.score_window(r.starting_ids, r.captain_id, merged_full, range(gw, end))
        old, budget = r.squad_ids, r.total_cost
    return total


def main():
    for prior, current in SEASONS:
        print(f"\n=== {current} (priors from {prior}) ===")
        q, n, _ = prediction_quality(prior, current)
        print(f"1. Per-gameweek prediction quality, GW4-38, {n:,} player-gameweeks")
        print("   (spearman/top-N higher = better; mae lower = better; bias ~0 = calibrated)")
        print(q.to_string())
        _, merged, teams, fixtures = bt.load_data(prior, current)
        with contextlib.redirect_stdout(io.StringIO()):
            base = bt.run_dynamic_test(merged, teams, bt.fixtures_as_list(fixtures),
                                       bt.build_prior_season_lookup(bt.load_data(prior, current)[0]))
        print(f"2. Season simulation: current model {base}  vs  decomposed {dynamic(prior, current)}")


if __name__ == "__main__":
    main()
