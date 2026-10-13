"""Backtests the goals-based fixture model (fpl/team_ratings.py) against FPL's
FDR on real past seasons, with no lookahead: at gameweek t the team ratings
only see last season plus this season's results before t.

Two tests:
  1. Prediction quality (large sample): every player, every gameweek -
     does (running points-per-game x fixture multiplier) rank actual points
     better with the goals model or with FDR? Measured by rank correlation
     and by the actual points of the top-N projected players.
  2. Season simulation: the same dynamic re-optimise-every-5-weeks test as
     scripts/backtest_season.py, swapping only the fixture method.

Usage: uv run python scripts/backtest_goals_model.py
"""

from __future__ import annotations

import io
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import backtest_season as bt
from fpl import model, optimizer, team_ratings as tr

SEASONS = [("2023-24", "2024-25"), ("2024-25", "2025-26")]


def season_inputs(prior: str, current: str):
    prior_fx = pd.read_csv(bt._fetch(f"{prior}/fixtures.csv", f"fixtures_{prior}.csv"))
    prior_tm = pd.read_csv(bt._fetch(f"{prior}/teams.csv", f"teams_{prior}.csv"))
    prior_mg = pd.read_csv(bt._fetch(f"{prior}/gws/merged_gw.csv", f"merged_gw_{prior}.csv"))
    cur_fx = pd.read_csv(bt._fetch(f"{current}/fixtures.csv", f"fixtures_{current}.csv"))
    cur_tm = pd.read_csv(bt._fetch(f"{current}/teams.csv", f"teams_{current}.csv"))
    pn, cn = dict(zip(prior_tm.id, prior_tm.name)), dict(zip(cur_tm.id, cur_tm.name))
    prev = [tr.Match(pn[r.team_h], pn[r.team_a], int(r.team_h_score), int(r.team_a_score))
            for r in prior_fx.itertuples() if r.finished and pd.notna(r.team_h_score)]
    relegated = sorted(set(pn.values()) - set(cn.values()))
    weights = tr.position_weights(prior_mg)
    return prev, relegated, weights, cur_fx, cn


def goals_model_as_of(gw, prev, relegated, weights, cur_fx, cn):
    cur = [tr.Match(cn[r.team_h], cn[r.team_a], int(r.team_h_score), int(r.team_a_score))
           for r in cur_fx.itertuples() if pd.notna(r.event) and r.event < gw and pd.notna(r.team_h_score)]
    return tr.fit_ratings(cur, prev, cn, relegated), weights


def prediction_quality(prior, current):
    prev, relegated, weights, cur_fx, cn = season_inputs(prior, current)
    _, merged, teams, fixtures = bt.load_data(prior, current)
    merged = merged.drop_duplicates(subset=["element", "round", "fixture"]) if "fixture" in merged else merged
    team_id = dict(zip(teams["name"], teams["id"]))
    pos_map = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    lookup = model.build_team_fixture_lookup(bt.fixtures_as_list(fixtures))
    rows = []
    for t in range(4, 39):
        hist = merged[merged["round"] < t]
        apps = hist[hist["minutes"] > 0].groupby("element").agg(pts=("total_points", "sum"), n=("total_points", "size"))
        starts = hist.groupby("element")["starts"].sum() if "starts" in hist else hist.groupby("element")["minutes"].apply(lambda m: (m >= 60).sum())
        now = merged[merged["round"] == t]
        if now.empty:
            continue
        R, W = goals_model_as_of(t, prev, relegated, weights, cur_fx, cn)
        info = now.groupby("element").agg(actual=("total_points", "sum"), team=("team", "first"), pos=("position", "first"))
        info = info[info["pos"].isin(pos_map)].join(apps, how="inner").join(starts.rename("starts"), how="left").fillna({"starts": 0})
        if info.empty:
            continue
        info["ppg"] = info["pts"] / info["n"]
        info["et"] = info["pos"].map(pos_map)
        pm = {e: np.average(g.ppg, weights=g.starts + 1e-9) for e, g in info.groupby("et")}
        info["shrunk"] = [pm[e] + s / (s + model.SHRINKAGE_K) * (p - pm[e]) for e, s, p in zip(info.et, info.starts, info.ppg)]
        for el, r in info.iterrows():
            tid = team_id[r["team"]]
            fl = lookup.get(tid, {}).get(t, [])
            if not fl:
                continue
            fdr = sum(model._fixture_difficulty_multiplier(d) for d, _, _ in fl)
            gm = sum(tr.fixture_multiplier(R, W, tid, o, h, r["et"]) for _, h, o in fl)
            rows.append(dict(gw=t, actual=r["actual"], shrunk=r["shrunk"], nfix=len(fl), fdr_m=fdr, goals_m=gm,
                             none=r["shrunk"] * len(fl), fdr=r["shrunk"] * fdr, goals=r["shrunk"] * gm))
    d = pd.DataFrame(rows)
    out = {}
    for m in ["none", "fdr", "goals"]:
        sp = d.groupby("gw").apply(lambda g: g[[m, "actual"]].corr(method="spearman").iloc[0, 1]).mean()
        top30 = d.groupby("gw").apply(lambda g: g.nlargest(30, m)["actual"].mean()).mean()
        top100 = d.groupby("gw").apply(lambda g: g.nlargest(100, m)["actual"].mean()).mean()
        mae = (d[m] - d["actual"]).abs().mean()
        out[m] = dict(spearman=round(sp, 4), top30=round(top30, 3), top100=round(top100, 3), mae=round(mae, 3))
    return pd.DataFrame(out).T, len(d), d


def dynamic_goals(prior, current):
    prev, relegated, weights, cur_fx, cn = season_inputs(prior, current)
    prior_players, merged, teams, fixtures = bt.load_data(prior, current)
    prior_lookup = bt.build_prior_season_lookup(prior_players)
    fl = bt.fixtures_as_list(fixtures)
    total, old, budget = 0, None, model.BUDGET_TENTHS
    for gw in range(1, 39, bt.RECHECK_EVERY_GW):
        lookup = prior_lookup if gw == 1 else bt.running_ppg_lookup(merged, gw)
        with contextlib.redirect_stdout(io.StringIO()):
            boot = bt.build_bootstrap(merged, teams, lookup, as_of_gw=gw)
        end = min(gw + bt.RECHECK_EVERY_GW, 39)
        gm = goals_model_as_of(gw, prev, relegated, weights, cur_fx, cn)
        df = model.project_players(boot, fl, horizon=end - gw, fixture_mode="goals", goals_model=gm)
        keep = [i for i in (old or []) if i in set(df["id"])]
        try:
            r = optimizer.optimize_squad(df, budget=budget, old_squad_ids=keep or None,
                                         max_transfers=bt.FREE_TRANSFERS_PER_CHECKPOINT if old else None)
        except RuntimeError:
            r = optimizer.optimize_squad(df, budget=model.BUDGET_TENTHS)
        total += bt.score_window(r.starting_ids, r.captain_id, merged, range(gw, end))
        old, budget = r.squad_ids, r.total_cost
    return total


def main():
    for prior, current in SEASONS:
        print(f"\n=== {current} (ratings start from {prior}) ===")
        q, n, _ = prediction_quality(prior, current)
        print(f"1. Prediction quality, GW4-38, {n:,} player-gameweeks (higher spearman/top-N, lower MAE = better):")
        print(q.to_string())
        _, merged, teams, fixtures = bt.load_data(prior, current)
        with contextlib.redirect_stdout(io.StringIO()):
            fdr_total = bt.run_dynamic_test(merged, teams, bt.fixtures_as_list(fixtures),
                                            bt.build_prior_season_lookup(bt.load_data(prior, current)[0]))
        print(f"2. Season simulation: FDR {fdr_total}  vs  goals model {dynamic_goals(prior, current)}")


if __name__ == "__main__":
    main()
