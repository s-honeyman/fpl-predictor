"""Walks forward through gameweeks 1-5 individually, rolling the 5-gameweek
projection horizon forward one week at a time, and reports whether a transfer
is worth making at each point - respecting the real 1-free-transfer-per-week
rule.

IMPORTANT, stated plainly: this is a PRE-SEASON plan. There are no real results
yet for 2026-27, so this cannot react to actual form, injuries, or team news -
only to how each club's fixture DIFFICULTY changes as the horizon window rolls
forward (a team with a rough GW1-5 run can have a much better GW3-7 run, and
vice versa - that's a real, known-in-advance signal). Once real gameweeks
start, `fpl transfers` (which reacts to live data) supersedes this - this
script's job is done once GW1 kicks off.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpl import api, model, optimizer, squad_state

HORIZON = 5


def bootstrap_as_of(bootstrap: dict, gw: int) -> dict:
    """Returns a copy of bootstrap with events overridden so `next_gameweeks`
    treats `gw` as the next upcoming gameweek - lets us roll the horizon
    forward without pretending any results have actually happened."""
    b = copy.deepcopy(bootstrap)
    b["events"] = [{"id": i, "finished": i < gw} for i in range(1, 39)]
    return b


def describe_transfer(df, out_id: int, in_id: int, window_label: str) -> str:
    out_row = df.loc[df["id"] == out_id].iloc[0]
    in_row = df.loc[df["id"] == in_id].iloc[0]
    return (
        f"  OUT: {out_row['web_name']:<16} ({window_label} xPts {out_row['predicted_points']:>5.1f})"
        f"   ->   IN: {in_row['web_name']:<16} ({window_label} xPts {in_row['predicted_points']:>5.1f})"
    )


def main() -> None:
    bootstrap = api.get_bootstrap()
    fixtures = api.get_fixtures()

    saved = squad_state.load_squad()
    if saved is None:
        print("No saved squad found - run `fpl build` first.")
        return
    current_squad_ids = saved["squad_ids"]
    budget = saved["total_cost"] + saved.get("bank_tenths", 0)

    print(f"Walking forward through GW1-{HORIZON}, one free transfer/week, fixture-swing only (no live data yet).\n")

    for gw in range(1, HORIZON + 1):
        b = bootstrap_as_of(bootstrap, gw)
        df = model.project_players(b, fixtures, horizon=HORIZON)
        window_label = f"GW{gw}-{gw+HORIZON-1}"

        hold_result = optimizer.optimize_squad(df, budget=budget, old_squad_ids=current_squad_ids, max_transfers=0)
        transfer_result = optimizer.optimize_squad(df, budget=budget, old_squad_ids=current_squad_ids, max_transfers=1)

        gain = transfer_result.predicted_starting_points - hold_result.predicted_starting_points
        print(f"{window_label}:")
        if gain > 1.0:  # small threshold - not worth swapping for noise-level gains
            out_ids = set(current_squad_ids) - set(transfer_result.squad_ids)
            in_ids = set(transfer_result.squad_ids) - set(current_squad_ids)
            print(f"  Transfer recommended (+{gain:.1f} projected pts over {window_label}):")
            for out_id, in_id in zip(out_ids, in_ids):
                print(describe_transfer(df, out_id, in_id, window_label))
            current_squad_ids = transfer_result.squad_ids
            budget = transfer_result.total_cost
        else:
            print(f"  Hold - no transfer clears the +1.0 pt bar this week (best available gain: {gain:+.1f}).")
        print()

    print("Reminder: re-run this (or switch to `fpl transfers`) once GW1 actually kicks off -")
    print("from that point real results exist and reacting to them beats a pre-season fixture plan.")


if __name__ == "__main__":
    main()
