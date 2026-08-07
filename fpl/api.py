"""Thin client for the official Fantasy Premier League API, with local caching."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://fantasy.premierleague.com/api"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_TTL_SECONDS = 60 * 60  # 1 hour - prices/form don't change faster than this


def _cached_get(url: str, cache_name: str, force_refresh: bool = False) -> Any:
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / cache_name

    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return json.loads(cache_path.read_text())

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    cache_path.write_text(json.dumps(payload))
    return payload


def get_bootstrap(force_refresh: bool = False) -> dict:
    """Players, teams, positions, and gameweek metadata."""
    return _cached_get(f"{BASE_URL}/bootstrap-static/", "bootstrap.json", force_refresh)


def get_fixtures(force_refresh: bool = False) -> list[dict]:
    """All fixtures for the season, including future ones with FDR ratings."""
    return _cached_get(f"{BASE_URL}/fixtures/", "fixtures.json", force_refresh)


def get_gameweek_live(gw: int, force_refresh: bool = False) -> dict:
    """Every player's actual stats for one specific (finished or in-progress)
    gameweek - one call for the whole league, not one per player. Used to
    fill in real actuals in the prediction log."""
    return _cached_get(f"{BASE_URL}/event/{gw}/live/", f"event_{gw}_live.json", force_refresh)
