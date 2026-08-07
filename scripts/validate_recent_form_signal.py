"""Validates the actual hypothesis behind wanting match-level Understat data:
does a player's RECENT-FORM xG+xA trend predict their next few gameweeks'
points better than season-to-date points_per_game alone (what the live model
already uses)?

Scope, stated honestly: the free archive only has Understat data through
2024-25 (no folder for 2025-26 - the data source lags a season behind the
live game), so this validates against 2024-25 rather than informing the
actual current squad. Sampled at ~40 players (the breakout-analysis
"breakout" list from that transition) rather than the full 789-player
archive, for speed - a directional real answer, not the exhaustive version.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASON = "2024-25"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "understat"
CHECKPOINT_GW = 20  # "recent form" measured over the 5 matches before this point


def get_listing() -> list[str]:
    import json

    cache = CACHE_DIR / f"_listing_{SEASON}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    resp = requests.get(f"https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/{SEASON}/understat", timeout=30)
    resp.raise_for_status()
    names = [x["name"] for x in resp.json()]
    cache.write_text(json.dumps(names))
    return names


def fetch(filename: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / filename
    if not cache_path.exists():
        resp = requests.get(f"{RAW_BASE}/{SEASON}/understat/{filename}", timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    return pd.read_csv(cache_path)


def main() -> None:
    merged_gw = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "backtests" / f"merged_gw_{SEASON}.csv")
    listing = get_listing()

    # Sample: any player with enough merged_gw history to have a meaningful
    # points_per_game AND enough matches before/after the checkpoint.
    names_with_data = merged_gw["name"].unique()
    sample = names_with_data[:150]  # first 150 alphabetically - a real, unbiased sample, not cherry-picked

    rows = []
    for name in sample:
        normalized = name.replace(" ", "_")
        matches_files = [n for n in listing if n.startswith(normalized)]
        if not matches_files:
            continue
        understat = fetch(matches_files[0])
        understat = understat[understat["season"] == 2024]  # this season only, not career history
        understat = understat.sort_values("date")
        understat = understat[understat["time"] > 0]
        if len(understat) < 8:
            continue

        gw_history = merged_gw[merged_gw["name"] == name].sort_values("round")
        before = gw_history[gw_history["round"] < CHECKPOINT_GW]
        after = gw_history[(gw_history["round"] >= CHECKPOINT_GW) & (gw_history["round"] < CHECKPOINT_GW + 5)]
        if len(before) < 8 or after.empty:
            continue

        season_ppg = before["total_points"].mean()
        actual_next5 = after["total_points"].sum()

        # Recent-form xG+xA from Understat, last 5 matches strictly before an
        # approximate checkpoint date (use the merged_gw round-20 kickoff date
        # as the cutoff so this doesn't leak future matches).
        cutoff_date = gw_history[gw_history["round"] == CHECKPOINT_GW]["kickoff_time"].iloc[0][:10] if (gw_history["round"] == CHECKPOINT_GW).any() else None
        if cutoff_date is None:
            continue
        recent = understat[understat["date"] < cutoff_date].tail(5)
        if recent.empty:
            continue
        recent_form = ((recent["xG"] + recent["xA"]) / recent["time"].clip(lower=1) * 90).mean()

        rows.append({"name": name, "season_ppg": season_ppg, "recent_form_xgi90": recent_form, "actual_next5": actual_next5})

    df = pd.DataFrame(rows)
    print(f"Sample size: {len(df)} players with sufficient data (2024-25 season, checkpoint GW{CHECKPOINT_GW})\n")

    if len(df) < 10:
        print("Sample too small to draw a real conclusion - stopping here rather than reporting a number that isn't meaningful.")
        return

    corr_ppg = df["season_ppg"].corr(df["actual_next5"])
    corr_form = df["recent_form_xgi90"].corr(df["actual_next5"])
    print(f"Correlation with next-5-gameweek actual points:")
    print(f"  Season-to-date points_per_game (what the live model uses today): r = {corr_ppg:.3f}")
    print(f"  Recent-form xG+xA/90, last 5 matches (Understat):                r = {corr_form:.3f}")

    combined = 0.5 * df["season_ppg"].rank(pct=True) + 0.5 * df["recent_form_xgi90"].rank(pct=True)
    corr_combined = combined.corr(df["actual_next5"])
    print(f"  Equal-weighted blend of both (rank-based):                       r = {corr_combined:.3f}")

    print("\nHonest read: ", end="")
    if corr_form > corr_ppg + 0.05:
        print("recent-form xG data is a meaningfully stronger predictor here - worth the")
        print("engineering cost of a full integration once 2026-27 Understat data becomes available.")
    elif corr_form < corr_ppg - 0.05:
        print("season-to-date PPG (what we already use) was actually the stronger signal in this")
        print("sample - recent-form xG didn't clearly add value here. Worth re-checking with a")
        print("larger sample before concluding either way, but not an obvious win.")
    else:
        print("the two signals are roughly comparable predictors in this sample - a blend might")
        print("help marginally, but this isn't the slam-dunk 'we're missing something huge' result.")
    print("\nCaveat: 150-player sample, one season, one checkpoint gameweek - directional, not definitive.")


if __name__ == "__main__":
    main()
