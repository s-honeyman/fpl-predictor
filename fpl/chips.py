"""Chip outlook: what each remaining chip is worth in each upcoming gameweek,
from the same per-gameweek projections as everything else.

This season chips come in two sets: one of each for GW1-19 and a fresh set
for GW20-38 (read live from FPL's bootstrap, not assumed). An unused
first-half chip is simply lost after GW19.

These are projections of value, not backtested timing rules: chip value
depends heavily on blank and double gameweeks, which FPL only schedules as
cup fixtures get rearranged. Re-run as the fixture list changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import planner
from .optimizer import optimize_squad


def chip_windows(bootstrap: dict, used: list[dict]) -> list[dict]:
    """Chips still available, with the gameweeks they can be played in."""
    used_by_name: dict[str, list[int]] = {}
    for u in used:
        used_by_name.setdefault(u["name"], []).append(u["event"])
    out = []
    for c in bootstrap.get("chips", []):
        played = any(c["start_event"] <= e <= c["stop_event"] for e in used_by_name.get(c["name"], []))
        if not played:
            out.append({"name": c["name"], "start": c["start_event"], "stop": c["stop_event"]})
    return out


def bench_boost_values(squad, proj, pos, n_weeks: int) -> np.ndarray:
    return np.array([planner.best_xi(squad, w, proj, pos, bench_boost=True) - planner.best_xi(squad, w, proj, pos)
                     for w in range(n_weeks)])


def free_hit_values(squad, budget: int, proj, pos, team, price, n_weeks: int) -> np.ndarray:
    """Best possible one-week squad on your budget, minus your own XI that week."""
    vals = []
    for w in range(n_weeks):
        df = pd.DataFrame([{"id": i, "predicted_points": a[w], "now_cost": price[i], "position_id": pos[i],
                            "team": team[i], "availability": 1.0} for i, a in proj.items() if i in price])
        best = optimize_squad(df, budget=budget).predicted_starting_points
        vals.append(best - planner.best_xi(squad, w, proj, pos))
    return np.array(vals)


def wildcard_value(squad, bank: int, ft: int, proj, pos, team, price, n_weeks: int) -> tuple[float, float]:
    """(points from the best unconstrained squad over the window, points from
    the best normal-transfer plan over the same window). Both count only
    starting XI + captain; the gap is roughly what a Wildcard now buys."""
    budget = sum(price[i] for i in squad) + bank
    df = pd.DataFrame([{"id": i, "predicted_points": a[:n_weeks].sum(), "now_cost": price[i], "position_id": pos[i],
                        "team": team[i], "availability": 1.0} for i, a in proj.items() if i in price])
    wc_squad = optimize_squad(df, budget=budget).squad_ids
    wc_pts = sum(planner.best_xi(wc_squad, w, proj, pos) for w in range(n_weeks))
    normal = planner.plan_transfers(squad, bank, ft, proj, pos, team, price, n_weeks, beam=20, k=6)
    return wc_pts, normal.net
