"""Backtests fpl-predictor against a REAL completed season: reconstructs exactly
what data would have been available at the start of that season (no lookahead),
runs the actual shipped `model.project_players` + `optimizer.optimize_squad` code
UNCHANGED, then scores the resulting squad against REAL results.

Three tiers, from least to most representative of actual tool usage:
  1. Static GW1-5: zero transfers, matches the horizon we've been quoting all night.
  2. Static full season: same static squad held for all 38 gameweeks (a strawman -
     nobody would actually do this, but it's the honest "if you did nothing" floor).
  3. Dynamic: re-optimizes every 5 gameweeks using only REALIZED results up to that
     point (a trailing-form signal, no lookahead) - the fair test of how the tool is
     actually meant to be used.

Data: vaastav/Fantasy-Premier-League GitHub repo (free, no auth, cached locally).

Usage: uv run python scripts/backtest_season.py --prior 2024-25 --current 2025-26
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from fpl import model, optimizer

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "backtests"

POSITION_TO_ID = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
RECHECK_EVERY_GW = 5
FREE_TRANSFERS_PER_CHECKPOINT = 5  # ~1/gameweek banked, a reasonable proxy


def _fetch(path: str, cache_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if not cache_path.exists():
        resp = requests.get(f"{RAW_BASE}/{path}", timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    return cache_path


def load_data(prior_season: str, current_season: str):
    prior_players = pd.read_csv(_fetch(f"{prior_season}/players_raw.csv", f"players_{prior_season}.csv"))
    merged_gw = pd.read_csv(_fetch(f"{current_season}/gws/merged_gw.csv", f"merged_gw_{current_season}.csv"))
    teams = pd.read_csv(_fetch(f"{current_season}/teams.csv", f"teams_{current_season}.csv"))
    fixtures = pd.read_csv(_fetch(f"{current_season}/fixtures.csv", f"fixtures_{current_season}.csv"))
    return prior_players, merged_gw, teams, fixtures


def build_prior_season_lookup(prior_players: pd.DataFrame) -> dict[str, dict]:
    lookup = {}
    for _, row in prior_players.iterrows():
        name = f"{row['first_name']} {row['second_name']}"
        lookup[name] = {
            "points_per_game": float(row["points_per_game"]),
            "total_points": int(row["total_points"]),
            "starts": int(row["starts"]),
            "penalties_order": None if pd.isna(row.get("penalties_order")) else int(row["penalties_order"]),
            "direct_freekicks_order": None if pd.isna(row.get("direct_freekicks_order")) else int(row["direct_freekicks_order"]),
            "corners_and_indirect_freekicks_order": (
                None if pd.isna(row.get("corners_and_indirect_freekicks_order")) else int(row["corners_and_indirect_freekicks_order"])
            ),
        }
    return lookup


def build_bootstrap(merged_gw: pd.DataFrame, teams: pd.DataFrame, points_per_game_lookup: dict, as_of_gw: int) -> dict:
    """`points_per_game_lookup`: name -> {points_per_game}. For GW1 this is the prior
    season's PPG; for later checkpoints, pass a lookup built from realized results so
    far (see running_ppg_lookup) - either way, no lookahead into the future."""
    team_id_by_name = dict(zip(teams["name"], teams["id"]))
    snapshot = merged_gw[merged_gw["round"] == as_of_gw].drop_duplicates(subset="element")

    elements = []
    skipped = 0
    for _, row in snapshot.iterrows():
        if row["position"] not in POSITION_TO_ID:
            skipped += 1  # data quirk in the source (e.g. a loan/unattached snapshot) - drop, don't crash
            continue
        prior = points_per_game_lookup.get(row["name"], {"points_per_game": 0.0, "total_points": 0, "starts": 0})
        name_parts = row["name"].rsplit(" ", 1)
        first, second = (name_parts[0], name_parts[1]) if len(name_parts) == 2 else (row["name"], "")
        elements.append(
            {
                "id": int(row["element"]),
                "web_name": row["name"].split(" ")[-1],
                "first_name": first,
                "second_name": second,
                "team": int(team_id_by_name[row["team"]]),
                "element_type": POSITION_TO_ID[row["position"]],
                "now_cost": int(row["value"]),
                "status": "a",
                "chance_of_playing_next_round": None,
                "selected_by_percent": "0",
                "total_points": prior.get("total_points", 0),
                "points_per_game": prior["points_per_game"],
                "starts": prior.get("starts", 0),
                "ep_next": float(row["xP"]) if pd.notna(row["xP"]) else prior["points_per_game"],
                "penalties_order": prior.get("penalties_order"),
                "direct_freekicks_order": prior.get("direct_freekicks_order"),
                "corners_and_indirect_freekicks_order": prior.get("corners_and_indirect_freekicks_order"),
            }
        )

    if skipped:
        print(f"  (skipped {skipped} row(s) with an unrecognized position code at GW{as_of_gw})")

    teams_list = [{"id": int(r["id"]), "name": r["name"], "short_name": r["name"][:3].upper()} for _, r in teams.iterrows()]
    events = [{"id": i, "finished": i < as_of_gw} for i in range(1, 39)]
    return {"elements": elements, "teams": teams_list, "events": events}


def running_ppg_lookup(merged_gw: pd.DataFrame, up_to_gw_exclusive: int) -> dict[str, dict]:
    """Trailing points-per-game (and starts) from REALIZED gameweeks only
    (< up_to_gw_exclusive) - the in-season equivalent of "last season's
    output", no lookahead."""
    history = merged_gw[merged_gw["round"] < up_to_gw_exclusive]
    grouped = history.groupby("name").agg(mean_pts=("total_points", "mean"), sum_pts=("total_points", "sum"), starts=("starts", "sum"))
    return {
        name: {"points_per_game": row["mean_pts"], "total_points": int(row["sum_pts"]), "starts": int(row["starts"])}
        for name, row in grouped.iterrows()
    }


def fixtures_as_list(fixtures: pd.DataFrame) -> list[dict]:
    return [
        {
            "event": int(row["event"]) if pd.notna(row["event"]) else None,
            "team_h": int(row["team_h"]),
            "team_a": int(row["team_a"]),
            "team_h_difficulty": int(row["team_h_difficulty"]),
            "team_a_difficulty": int(row["team_a_difficulty"]),
        }
        for _, row in fixtures.iterrows()
    ]


def score_window(starting_ids, captain_id, merged_gw, gw_range) -> int:
    relevant = merged_gw[merged_gw["round"].isin(gw_range) & merged_gw["element"].isin(starting_ids)]
    total = int(relevant["total_points"].sum())
    captain_points = int(merged_gw[merged_gw["round"].isin(gw_range) & (merged_gw["element"] == captain_id)]["total_points"].sum())
    return total + captain_points


def run_static_test(bootstrap, fixtures_list, merged_gw, horizon_gws, label, fixture_mode="fdr", set_piece_bonus=False, ep_next_weight=1.0):
    df = model.project_players(
        bootstrap, fixtures_list, horizon=len(horizon_gws), fixture_mode=fixture_mode, set_piece_bonus=set_piece_bonus,
        ep_next_weight=ep_next_weight,
    )
    result = optimizer.optimize_squad(df, budget=model.BUDGET_TENTHS)
    real_score = score_window(result.starting_ids, result.captain_id, merged_gw, horizon_gws)
    print(f"\n{label}")
    print(f"  Squad cost: £{result.total_cost/10:.1f}m | Model's own projection: {result.predicted_starting_points:.1f} | "
          f"ACTUAL realized: {real_score} "
          f"({'over' if result.predicted_starting_points > real_score else 'under'}-projected by "
          f"{abs(result.predicted_starting_points - real_score):.1f})")
    return result, real_score


def run_dynamic_test(merged_gw, teams, fixtures_list, prior_lookup, fixture_mode="fdr", ep_next_weight=1.0):
    print(f"\nDYNAMIC TEST ({fixture_mode}): re-optimizes every 5 gameweeks using only realized results so far")
    checkpoints = list(range(1, 39, RECHECK_EVERY_GW))
    old_squad_ids = None
    total_score = 0
    budget = model.BUDGET_TENTHS

    for i, gw in enumerate(checkpoints):
        ppg_lookup = prior_lookup if gw == 1 else running_ppg_lookup(merged_gw, gw)
        bootstrap = build_bootstrap(merged_gw, teams, ppg_lookup, as_of_gw=gw)
        window_end = min(gw + RECHECK_EVERY_GW, 39)
        horizon = window_end - gw
        if horizon <= 0:
            break
        df = model.project_players(bootstrap, fixtures_list, horizon=horizon, fixture_mode=fixture_mode, ep_next_weight=ep_next_weight)
        # Only players who actually appear at this checkpoint's snapshot are eligible.
        eligible_ids = set(df["id"])
        constrained_old = [i for i in (old_squad_ids or []) if i in eligible_ids]

        try:
            result = optimizer.optimize_squad(
                df, budget=budget, old_squad_ids=constrained_old or None,
                max_transfers=FREE_TRANSFERS_PER_CHECKPOINT if old_squad_ids else None,
            )
        except RuntimeError:
            # The transfer-count constraint occasionally combines with the budget/position
            # constraints into an infeasible ILP (e.g. too many kept players share a
            # position slot that's now oversubscribed). A real manager wouldn't be stuck -
            # they'd just make more changes. Fall back to an unconstrained rebuild for this
            # checkpoint rather than crash the whole backtest.
            print(f"  GW{gw}: transfer-constrained solve was infeasible - falling back to a full rebuild this checkpoint")
            result = optimizer.optimize_squad(df, budget=model.BUDGET_TENTHS)
        window_gws = range(gw, window_end)
        window_score = score_window(result.starting_ids, result.captain_id, merged_gw, window_gws)
        total_score += window_score
        transfers_made = len(set(result.squad_ids) - set(old_squad_ids)) if old_squad_ids else 0
        print(f"  GW{gw}-{window_end-1}: {transfers_made} transfer(s), squad £{result.total_cost/10:.1f}m, "
              f"realized {window_score} pts (running total {total_score})")

        old_squad_ids = result.squad_ids
        budget = result.total_cost  # roughly track squad value going forward (simplification: ignores price rises)

    print(f"\n  DYNAMIC TOTAL, full season: {total_score} points")
    return total_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", default="2024-25")
    parser.add_argument("--current", default="2025-26")
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument(
        "--fixture-mode", choices=["fdr", "strength", "both"], default="both",
        help="Fixture-difficulty method to test: FPL's FDR, the team-strength model, or both side by side (default).",
    )
    args = parser.parse_args()

    print(f"Backtesting: prior season {args.prior} (data signal) -> current season {args.current} (real outcomes)")
    prior_players, merged_gw, teams, fixtures = load_data(args.prior, args.current)
    prior_lookup = build_prior_season_lookup(prior_players)
    fixtures_list = fixtures_as_list(fixtures)

    bootstrap_gw1 = build_bootstrap(merged_gw, teams, prior_lookup, as_of_gw=1)
    print(f"Reconstructed {len(bootstrap_gw1['elements'])} players active at the start of {args.current}.")

    modes = ["fdr", "strength"] if args.fixture_mode == "both" else [args.fixture_mode]
    for mode in modes:
        run_static_test(
            bootstrap_gw1, fixtures_list, merged_gw, range(1, 6),
            f"STATIC, GW1-5, fixture_mode={mode}", fixture_mode=mode,
        )
        run_static_test(
            bootstrap_gw1, fixtures_list, merged_gw, range(1, 39),
            f"STATIC, full season, ZERO transfers, fixture_mode={mode}", fixture_mode=mode,
        )
        if not args.skip_dynamic:
            run_dynamic_test(merged_gw, teams, fixtures_list, prior_lookup, fixture_mode=mode)

    # set_piece_bonus only tested on the GW1 static snapshot: order data isn't
    # tracked per-gameweek in the free archive, only as a season-level snapshot
    # carried forward from the prior season (same limitation as PPG at GW1,
    # but there's no in-season equivalent of running_ppg_lookup for it).
    run_static_test(
        bootstrap_gw1, fixtures_list, merged_gw, range(1, 6),
        "STATIC, GW1-5, set_piece_bonus=True", set_piece_bonus=True,
    )


if __name__ == "__main__":
    main()
