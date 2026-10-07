"""Chip-timing analysis: Bench Boost and Triple Captain candidates are derived
from real fixture data for your saved squad; Wildcard timing is stated as
general, evidence-informed convention (not derived from our own data - it's
inherently reactive to how the season actually unfolds, which doesn't exist
yet - see the honest caveat in each section).

Method for BB/TC: reuses model.py's own fixture-difficulty machinery
(_shrink_ppg, _fixture_difficulty_multiplier, build_team_fixture_lookup)
directly, NOT `ep_next` - ep_next only means anything for the true next real
gameweek, not for hypothetically evaluating gameweek 12 in isolation. This
avoids quietly misapplying a "next gameweek" figure to the wrong gameweek.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpl import api, model, squad_state

LOOKAHEAD_GWS = range(2, 16)  # GW1 is imminent/already locked in - look from GW2 onward


def find_double_and_blank_gws(fixtures: list[dict], team_ids: set[int]) -> tuple[dict, dict]:
    from collections import defaultdict

    fixture_count = defaultdict(lambda: defaultdict(int))
    for f in fixtures:
        gw = f.get("event")
        if gw is None:
            continue
        fixture_count[gw][f["team_h"]] += 1
        fixture_count[gw][f["team_a"]] += 1

    doubles, blanks = {}, {}
    for gw, counts in fixture_count.items():
        doubled = [t for t in team_ids if counts.get(t, 0) >= 2]
        blanked = [t for t in team_ids if counts.get(t, 0) == 0]
        if doubled:
            doubles[gw] = doubled
        if blanked:
            blanks[gw] = blanked
    return doubles, blanks


def main() -> None:
    bootstrap = api.get_bootstrap()
    fixtures = api.get_fixtures()
    fixture_lookup = model.build_team_fixture_lookup(fixtures)
    positional_means = model._positional_mean_ppg(bootstrap)

    saved = squad_state.load_squad()
    if saved is None:
        print("No saved squad found - run `fpl build` first.")
        return

    players_by_id = {p["id"]: p for p in bootstrap["elements"]}
    squad_players = [players_by_id[i] for i in saved["squad_ids"] if i in players_by_id]
    captain = players_by_id.get(saved["captain_id"])
    squad_team_ids = {p["team"] for p in squad_players}

    def points_at(player, gw):
        avail = model._availability_multiplier(player)
        shrunk = model._shrink_ppg(
            float(player.get("points_per_game") or 0.0), int(player.get("starts") or 0),
            positional_means[player["element_type"]],
        )
        return sum(
            shrunk * model._fixture_difficulty_multiplier(d) * avail
            for d, _home, _opp in fixture_lookup.get(player["team"], {}).get(gw, [])
        )

    print("=" * 60)
    print("BENCH BOOST candidates (full 15-man squad's combined fixture value)")
    print("=" * 60)
    bb_scores = {gw: sum(points_at(p, gw) for p in squad_players) for gw in LOOKAHEAD_GWS}
    for gw, score in sorted(bb_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        print(f"  GW{gw}: full-squad projected total {score:.1f} pts")

    print("\n" + "=" * 60)
    print(f"TRIPLE CAPTAIN candidates ({captain['web_name'] if captain else 'captain'}'s best single fixture)")
    print("=" * 60)
    if captain:
        tc_scores = {gw: points_at(captain, gw) for gw in LOOKAHEAD_GWS}
        for gw, score in sorted(tc_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            print(f"  GW{gw}: {captain['web_name']} projected {score:.1f} pts that gameweek")

    print("\n" + "=" * 60)
    print("Double/blank gameweeks affecting your current squad (from the published fixture list)")
    print("=" * 60)
    doubles, blanks = find_double_and_blank_gws(fixtures, squad_team_ids)
    if doubles:
        for gw, teams in sorted(doubles.items()):
            names = [bootstrap["teams"][[t["id"] for t in bootstrap["teams"]].index(tid)]["short_name"] for tid in teams]
            print(f"  GW{gw}: double gameweek for {', '.join(names)} - strong BB/TC candidate if it holds")
    else:
        print("  None currently scheduled - this early in the season, doubles/blanks are usually created later")
        print("  (postponed cup fixtures get rescheduled into gaps). Re-run this periodically, not just now.")
    if blanks:
        for gw, teams in sorted(blanks.items()):
            names = [bootstrap["teams"][[t["id"] for t in bootstrap["teams"]].index(tid)]["short_name"] for tid in teams]
            print(f"  GW{gw}: blank gameweek for {', '.join(names)} - Free Hit candidate if several squad players affected")

    print("\n" + "=" * 60)
    print("WILDCARD timing - general convention, NOT derived from our data")
    print("=" * 60)
    print("  This is inherently reactive to how the season actually unfolds (injuries, form")
    print("  surprises, price changes) - there's no real in-season data yet to base a specific")
    print("  gameweek on. Commonly-cited FPL community convention, stated as convention, not a")
    print("  data-backed finding from this tool:")
    print("    - First Wildcard: often used around GW6-10, once early-season surprises (both")
    print("      good and bad) are visible but before missing out on too many transfers.")
    print("    - Second Wildcard: often saved for GW20-32, timed to a good run-in of fixtures")
    print("      once the run-in itself is known.")
    print("  Re-run the fixture-based sections above periodically - THAT part of this analysis")
    print("  does get more accurate as more of the season's fixture list solidifies.")


if __name__ == "__main__":
    main()
