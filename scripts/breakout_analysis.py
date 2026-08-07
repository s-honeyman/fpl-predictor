"""Moneyball-for-FPL: finds the pre-breakout ATTRIBUTES of players who were cheap
and low-expectation at a season's start but massively outperformed - then screens
this year's player pool for the same profile.

The Moneyball parallel, specifically: the market prices players mostly on
reputation and TOTAL prior output (goals, points, minutes) - exactly the stats
that are visible and already priced in. The analogous "on-base percentage" signal
here is per-90 underlying efficiency (xG/90, xA/90, ICT/90) - if a player was
genuinely efficient in limited minutes, that's a real signal the market
underweights relative to "did they actually score enough goals to be famous".

Method (3 season transitions: 2022-23->2023-24, 2023-24->2024-25, 2024-25->2025-26):
  1. From the PRIOR season: each player's total output (what the market saw) and
     per-90 underlying efficiency (what a Moneyball scout would look at).
  2. From the CURRENT season: starting price (GW1 value - what the market believed
     going in) and total realized points (what actually happened).
  3. "Breakout" = cheap at the current season's start (<=£5.5m) AND finished in the
     top quartile of points for their position that season.
  4. Compare breakout players' PRIOR per-90 efficiency against a control group -
     similarly cheap, similarly low-output players who did NOT break out - to see
     whether per-90 efficiency actually distinguished them, or whether it's noise.
  5. Apply the resulting profile to the CURRENT (2026-27) live player pool.

Honest caveat up front: breakout players are rare (roughly 10-20/season by this
definition), so 3 seasons gives a few dozen examples - enough to see a real
pattern if one exists, not enough to treat any specific threshold as precisely
calibrated. Report exact numbers, don't round them into false confidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from fpl import api

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "backtests"

TRANSITIONS = [
    ("2022-23", "2023-24"),
    ("2023-24", "2024-25"),
    ("2024-25", "2025-26"),
]

CHEAP_PRICE_THRESHOLD = 55  # £5.5m, in FPL's 0.1m units
LOW_PRIOR_POINTS_THRESHOLD = 50  # "nobody expected them to do anything" - roughly a fringe/unproven player
BREAKOUT_PERCENTILE = 0.75  # top quartile of points for their position that season, among the low-prior-output pool


def load_prior(season: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"players_{season}.csv")
    df["name"] = df["first_name"] + " " + df["second_name"]
    per90 = 90.0 / df["minutes"].clip(lower=1)
    df["xg_per90_prior"] = df["expected_goals"] * per90
    df["xa_per90_prior"] = df["expected_assists"] * per90
    df["ict_per90_prior"] = df["ict_index"] * per90
    return df[
        ["name", "element_type", "minutes", "starts", "total_points", "points_per_game",
         "xg_per90_prior", "xa_per90_prior", "ict_per90_prior"]
    ].rename(columns={"element_type": "position_id"})


def load_current_outcomes(season: str) -> pd.DataFrame:
    gw = pd.read_csv(DATA_DIR / f"merged_gw_{season}.csv")
    gw1 = gw[gw["round"] == 1][["name", "value", "position"]].drop_duplicates(subset="name")
    totals = gw.groupby("name")["total_points"].sum().rename("season_total")
    out = gw1.join(totals, on="name")
    pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    out["position_id"] = out["position"].map(pos_map)
    return out[["name", "value", "position_id", "season_total"]].dropna()


def find_breakouts_one_transition(prior_season: str, current_season: str) -> pd.DataFrame:
    prior = load_prior(prior_season)
    current = load_current_outcomes(current_season)
    merged = current.merge(prior, on="name", suffixes=("", "_prior"), how="left")

    # Restrict to players who were BOTH cheap AND had genuinely low prior output -
    # "cheap" alone isn't "nobody expected them to do anything" (a lot of cheap
    # defenders are cheap because of position pricing, not low expectations, and
    # were already productive - e.g. Saliba, Rice, Tarkowski show up as false
    # positives without this second filter). NaN (never played in the PL before)
    # counts as low prior output too.
    merged["total_points"] = merged["total_points"].fillna(0)
    cheap = merged[
        (merged["value"] <= CHEAP_PRICE_THRESHOLD) & (merged["total_points"] <= LOW_PRIOR_POINTS_THRESHOLD)
    ].copy()
    cheap["pos_percentile"] = cheap.groupby("position_id")["season_total"].rank(pct=True)

    breakouts = cheap[cheap["pos_percentile"] >= BREAKOUT_PERCENTILE].copy()
    non_breakouts = cheap[cheap["pos_percentile"] < BREAKOUT_PERCENTILE].copy()
    breakouts["transition"] = f"{prior_season}->{current_season}"
    non_breakouts["transition"] = f"{prior_season}->{current_season}"
    return breakouts, non_breakouts


def main() -> None:
    all_breakouts, all_controls = [], []
    for prior_season, current_season in TRANSITIONS:
        breakouts, controls = find_breakouts_one_transition(prior_season, current_season)
        all_breakouts.append(breakouts)
        all_controls.append(controls)
        print(f"{prior_season} -> {current_season}: {len(breakouts)} breakout candidates "
              f"(cheap + top-quartile-for-position), {len(controls)} cheap non-breakouts as control")

    breakouts_df = pd.concat(all_breakouts, ignore_index=True)
    controls_df = pd.concat(all_controls, ignore_index=True)

    print(f"\nTotal breakout examples across 3 seasons: {len(breakouts_df)}")
    print("\nBreakout players found (name, transition, starting price, season total, prior total_points):")
    for _, row in breakouts_df.sort_values("season_total", ascending=False).iterrows():
        prior_pts = row["total_points"] if pd.notna(row["total_points"]) else 0
        print(f"  {row['name']:<25} {row['transition']:<14} started £{row['value']/10:.1f}m -> "
              f"{row['season_total']:.0f} pts (prior season: {prior_pts:.0f} pts)")

    print("\n--- Pre-breakout attribute comparison: breakouts vs. cheap non-breakouts ---")
    for col, label in [
        ("xg_per90_prior", "Prior xG/90"),
        ("xa_per90_prior", "Prior xA/90"),
        ("ict_per90_prior", "Prior ICT index/90"),
        ("minutes", "Prior season minutes"),
        ("starts", "Prior season starts"),
    ]:
        b_mean = breakouts_df[col].mean()
        c_mean = controls_df[col].mean()
        b_median = breakouts_df[col].median()
        c_median = controls_df[col].median()
        print(f"{label:<22} breakouts: mean={b_mean:.2f} median={b_median:.2f}  |  "
              f"non-breakouts: mean={c_mean:.2f} median={c_median:.2f}")

    # Simple, honest significance check - not claiming causation, just "is this
    # distinguishable from noise given the sample size we actually have".
    from math import sqrt
    for col, label in [
        ("xg_per90_prior", "xG/90"), ("xa_per90_prior", "xA/90"), ("ict_per90_prior", "ICT/90"),
        ("minutes", "prior minutes"), ("starts", "prior starts"),
    ]:
        b, c = breakouts_df[col].dropna(), controls_df[col].dropna()
        pooled_std = sqrt((b.var() + c.var()) / 2)
        cohens_d = (b.mean() - c.mean()) / pooled_std if pooled_std > 0 else float("nan")
        print(f"Effect size (Cohen's d) for {label}: {cohens_d:.2f} "
              f"({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} by convention)")

    breakouts_df.to_csv(DATA_DIR / "breakout_examples.csv", index=False)
    print(f"\nSaved full breakout example list to {DATA_DIR / 'breakout_examples.csv'}")

    screen_current_pool()


def screen_current_pool() -> None:
    """Applies the empirical finding to THIS season's live player pool: prior
    minutes/starts is the strong signal, per-90 efficiency a weak secondary one.
    Score = mostly "did they already play a lot despite a low price", with a
    small nudge for underlying efficiency as a tiebreaker - weighted roughly by
    the effect sizes actually found above, not just given equal billing."""
    bootstrap = api.get_bootstrap(force_refresh=False)
    players = pd.DataFrame(bootstrap["elements"])
    players = players[(players["now_cost"] <= CHEAP_PRICE_THRESHOLD) & (players["total_points"] <= LOW_PRIOR_POINTS_THRESHOLD)].copy()

    per90 = 90.0 / players["minutes"].clip(lower=1)
    players["xg_per90"] = players["expected_goals"].astype(float) * per90
    players["xa_per90"] = players["expected_assists"].astype(float) * per90

    # Normalize each signal 0-1 within this pool, then weight roughly by the
    # Cohen's d values found (minutes/starts dominant, per-90 stats minor).
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    players["breakout_score"] = (
        0.45 * norm(players["minutes"])
        + 0.35 * norm(players["starts"])
        + 0.10 * norm(players["xg_per90"])
        + 0.10 * norm(players["xa_per90"])
    )

    positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    players["position"] = players["element_type"].map(positions)

    print("\n" + "=" * 70)
    print("SCREEN: current 2026-27 pool, cheap + low last-season FPL points,")
    print("ranked by the empirically-weighted breakout score above")
    print("=" * 70)
    top = players.sort_values("breakout_score", ascending=False).head(15)
    for _, row in top.iterrows():
        print(f"  {row['web_name']:<18} {row['position']:<4} £{row['now_cost']/10:.1f}m  "
              f"last season: {int(row['minutes'])} mins, {int(row['starts'])} starts, {int(row['total_points'])} pts  "
              f"(xG/90={row['xg_per90']:.2f}, xA/90={row['xa_per90']:.2f})")

    print("\nRead this as: cheap players who already had the manager's trust for regular minutes")
    print("last season despite modest returns - the strongest empirical breakout signal found above -")
    print("not a guarantee any specific name repeats a Cole Palmer/Morgan Rogers story.")


if __name__ == "__main__":
    main()
