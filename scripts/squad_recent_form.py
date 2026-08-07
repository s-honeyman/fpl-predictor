"""Pulls real match-by-match Understat data (free, no API key - see README "Data
sources") for your CURRENT SAVED SQUAD specifically, and compares each player's
recent-form xG involvement (last 5 matches) against their season-long average -
a sharper, more current signal than a season total alone.

Scoped deliberately to just the 15 squad players (~15 requests), not the full
league (789 players, no bulk endpoint available - a much bigger undertaking,
flagged as a real next step in README.md rather than silently done partially
or skipped).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from fpl import api, squad_state

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASON = "2025-26"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "understat"


def find_understat_filename(full_name: str, season: str) -> str | None:
    """The understat directory names files `First_Last_id.csv` - list the dir
    once and match on name prefix (accounts for accented characters/nicknames
    not matching FPL's own name fields exactly)."""
    import json

    listing_cache = CACHE_DIR / f"_listing_{season}.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if listing_cache.exists():
        names = json.loads(listing_cache.read_text())
    else:
        resp = requests.get(f"https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/{season}/understat", timeout=30)
        resp.raise_for_status()
        names = [x["name"] for x in resp.json()]
        listing_cache.write_text(json.dumps(names))

    normalized = full_name.replace(" ", "_")
    matches = [n for n in names if n.startswith(normalized) or normalized.split("_")[0] in n]
    return matches[0] if matches else None


def fetch_understat_matches(filename: str, season: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / filename
    if not cache_path.exists():
        resp = requests.get(f"{RAW_BASE}/{season}/understat/{filename}", timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    return pd.read_csv(cache_path)


def main() -> None:
    bootstrap = api.get_bootstrap()
    players_by_id = {p["id"]: p for p in bootstrap["elements"]}

    saved = squad_state.load_squad()
    if saved is None:
        print("No saved squad found - run `fpl build` first.")
        return

    print(f"Recent-form check for your {len(saved['squad_ids'])} squad players (real Understat match data):\n")
    print(f"{'Player':<18} {'Season xG/90':>13} {'Last-5 xG/90':>13} {'Season xA/90':>13} {'Last-5 xA/90':>13}  Signal")
    print("-" * 90)

    not_found = []
    for pid in saved["squad_ids"]:
        player = players_by_id.get(pid)
        if not player:
            continue
        full_name = f"{player['first_name']} {player['second_name']}"
        filename = find_understat_filename(full_name, SEASON)
        if filename is None:
            not_found.append(player["web_name"])
            continue

        matches = fetch_understat_matches(filename, SEASON)
        matches = matches.sort_values("date")
        matches = matches[matches["time"] > 0]  # exclude unused-sub appearances (0 minutes)
        if matches.empty:
            not_found.append(player["web_name"])
            continue

        matches["xg_per90"] = matches["xG"] / matches["time"] * 90
        matches["xa_per90"] = matches["xA"] / matches["time"] * 90

        season_xg90 = matches["xg_per90"].mean()
        season_xa90 = matches["xa_per90"].mean()
        last5 = matches.tail(5)
        last5_xg90 = last5["xg_per90"].mean()
        last5_xa90 = last5["xa_per90"].mean()

        combined_season = season_xg90 + season_xa90
        combined_recent = last5_xg90 + last5_xa90
        if combined_season > 0.05:
            swing = (combined_recent - combined_season) / combined_season
            signal = "trending UP" if swing > 0.25 else "trending DOWN" if swing < -0.25 else "steady"
        else:
            signal = "low volume"

        print(f"{player['web_name']:<18} {season_xg90:>13.2f} {last5_xg90:>13.2f} {season_xa90:>13.2f} {last5_xa90:>13.2f}  {signal}")

    if not_found:
        print(f"\n(No Understat match found for: {', '.join(not_found)} - likely a new-to-PL signing with no prior Understat history, or a name-matching miss.)")

    print("\nThis is descriptive, not yet wired into the projection model - see README.md")
    print("'Understat integration' for the validated version of this (full-league, backtested)")
    print("as the natural next step, and why it wasn't done in this pass (789 players, no bulk")
    print("endpoint - a genuinely bigger undertaking than this squad-level check).")


if __name__ == "__main__":
    main()
