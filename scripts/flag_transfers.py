"""Flags squad players whose club has changed since last season (or who are
brand new to the PL) - the group the model structurally can't judge well.

Why this exists: `_shrink_ppg` in fpl/model.py projects a player's points using
their OWN historical points_per_game, regressed toward the positional mean.
That's fine for a player staying at the same club, but it's the wrong prior
for someone who just transferred - their history was earned in a different
team, system, and role. Neither the live FPL API nor the vaastav archive
tracks preseason friendlies, so there's no dataset-based fix for this; the
best we can do is flag who's affected so a human can sanity-check their
likely starting-XI role (e.g. via recent preseason friendly team news) before
locking in a squad.

Compares each squad member's current club (live bootstrap) against their club
in the most recently completed season (vaastav archive, cached under
data/backtests/ by scripts/backtest_season.py). A name not found in last
season's data at all is flagged as new to the PL rather than transferred.

Usage: uv run python scripts/flag_transfers.py --prior-season 2025-26
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from fpl import api, squad_state

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "backtests"


def _fetch(path: str, cache_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if not cache_path.exists():
        resp = requests.get(f"{RAW_BASE}/{path}", timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    return cache_path


def build_prior_team_lookup(prior_season: str) -> dict[str, str]:
    players = pd.read_csv(_fetch(f"{prior_season}/players_raw.csv", f"players_{prior_season}.csv"))
    teams = pd.read_csv(_fetch(f"{prior_season}/teams.csv", f"teams_{prior_season}.csv"))
    team_name_by_id = dict(zip(teams["id"], teams["name"]))
    return {
        f"{row['first_name']} {row['second_name']}": team_name_by_id[row["team"]]
        for _, row in players.iterrows()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-season", default="2025-26", help="Most recently completed season (default 2025-26).")
    parser.add_argument(
        "--all-players", action="store_true",
        help="Check every player in the live pool instead of just your saved squad.",
    )
    args = parser.parse_args()

    bootstrap = api.get_bootstrap()
    team_name_by_id = {t["id"]: t["name"] for t in bootstrap["teams"]}
    prior_team = build_prior_team_lookup(args.prior_season)

    if args.all_players:
        elements = bootstrap["elements"]
    else:
        saved = squad_state.load_squad()
        if saved is None:
            print("No saved squad found (data/my_squad.json) - run `fpl build` first, or pass --all-players.")
            return
        squad_ids = set(saved["squad_ids"])
        elements = [e for e in bootstrap["elements"] if e["id"] in squad_ids]

    transferred, new_to_pl = [], []
    for p in elements:
        name = f"{p['first_name']} {p['second_name']}"
        current_team = team_name_by_id[p["team"]]
        if name not in prior_team:
            new_to_pl.append((name, current_team))
        elif prior_team[name] != current_team:
            transferred.append((name, prior_team[name], current_team))

    print(f"Checked {len(elements)} player(s) against {args.prior_season} club data.\n")

    if transferred:
        print(f"TRANSFERRED SINCE {args.prior_season} ({len(transferred)}):")
        for name, old, new in transferred:
            print(f"  {name}: {old} -> {new}")
    else:
        print("No transferred players found.")

    print()
    if new_to_pl:
        print(f"NEW TO THE PL / NOT IN {args.prior_season} DATA ({len(new_to_pl)}):")
        for name, team in new_to_pl:
            print(f"  {name} ({team})")
    else:
        print("No brand-new players found.")

    if transferred or new_to_pl:
        print(
            "\nFor each name above, the model's projection is built on stats earned "
            "elsewhere (or nowhere, for new arrivals) - it has no signal on their role "
            "at their new club. Worth a manual check of recent preseason friendly team "
            "news for each before treating them as a confirmed starter."
        )


if __name__ == "__main__":
    main()
