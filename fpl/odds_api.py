"""Player goalscorer/assist odds and match odds from The Odds API, logged in
shadow mode: fetched before each deadline, stored, and scored against what
actually happened - but given NO weight in projections until the log shows
they beat the model. Historical player odds aren't free, so this can't be
backtested; the live log is the test.

The key lives in data/odds_api_key (git-ignored). It's sent only as a query
parameter to the API and never printed - request errors are re-raised
without the URL, since the URL contains the key.

Cost: 1 credit per market per region per match. A gameweek of anytime
goalscorer + assists (UK and US bookmakers) plus match odds is ~32 credits;
the free plan has 500 a month.

Converting a player's anytime-goalscorer price to a goal rate: median price
across bookmakers -> probability p -> Poisson rate -ln(1-p). Bookmakers price
these as "if he plays" (bets on non-players are void), and the rates still
carry the bookmaker margin. `rates_if_playing` removes the margin per team:
it scales the rates so that, weighted by each player's chance of playing
(from the minutes model), they add up to the team's market expected goals
less ~3% for own goals.
"""

from __future__ import annotations

import json
import math
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from . import market_odds

BASE = "https://api.the-odds-api.com/v4/sports/soccer_epl"
DATA = Path(__file__).resolve().parent.parent / "data"
KEY_PATH = DATA / "odds_api_key"
LOG_DIR = DATA / "odds_log"
PLAYER_MARKETS = ["player_goal_scorer_anytime", "player_assists"]
REGIONS = "uk,us"
OWN_GOAL_SHARE = 0.03

TEAM_ALIASES = {
    "brighton and hove albion": "Brighton", "tottenham hotspur": "Spurs", "manchester united": "Man Utd",
    "manchester city": "Man City", "nottingham forest": "Nott'm Forest", "wolverhampton wanderers": "Wolves",
    "newcastle united": "Newcastle", "west ham united": "West Ham", "leeds united": "Leeds",
    "afc bournemouth": "Bournemouth",
}


class OddsAPIError(RuntimeError):
    pass


def _key() -> str:
    if not KEY_PATH.exists() or not KEY_PATH.read_text().strip():
        raise OddsAPIError(f"No Odds API key - paste it into {KEY_PATH}")
    return KEY_PATH.read_text().strip()


def _get(path: str, **params) -> tuple[object, dict]:
    try:
        r = requests.get(f"{BASE}{path}", params={"apiKey": _key(), **params}, timeout=30)
    except requests.RequestException as exc:
        raise OddsAPIError(f"Odds API request failed ({type(exc).__name__})") from None
    if not r.ok:
        raise OddsAPIError(f"Odds API returned {r.status_code}: {r.text[:120]}")
    usage = {k: r.headers.get(f"x-requests-{k}") for k in ("remaining", "used", "last")}
    return r.json(), usage


def _fold(s: str) -> str:
    for a, b in (("ß", "ss"), ("ø", "o"), ("Ø", "O"), ("æ", "ae"), ("ł", "l"), ("ı", "i"), ("İ", "I"), ("đ", "d")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() else " " for c in s.lower())


def _tokens(s: str) -> set[str]:
    return {t for t in _fold(s).split() if len(t) > 1}


def team_id(api_name: str, fpl_names: dict[int, str]) -> int | None:
    return market_odds.match_team(TEAM_ALIASES.get(api_name.lower(), api_name), fpl_names)


def match_player(name: str, candidates: list[dict]) -> dict | None:
    """Bookmaker name -> one FPL player from the two teams in that match, or
    None if it isn't clear-cut. Rules, strictest first:
      - every word matches (one may be missing for 3-word names); or
      - the surname matches and the first name is a variant (same first two
        letters: Ben/Benjamin, Joe/Joseph, Kostas/Konstantinos), plus at most
        one unmatched middle name for 3+ word names; or
      - for 4+ word names, at least two words match including a surname word.
    A single shared word is never enough: 'Martin Odegaard' must not match
    'David Raya Martin'. Ties are refused, not guessed."""
    want = _tokens(name)
    if not want:
        return None
    scored = []
    for p in candidates:
        first = _tokens(p["first_name"])
        surname = _tokens(p["second_name"]) | _tokens(p["web_name"])
        surname |= {"".join(_fold(p["second_name"]).split())}  # 'Gannon-Doak' == 'Gannondoak'
        full = first | surname
        overlap = want & full
        missing = want - full
        candidates_scores = []
        if len(missing) <= (1 if len(want) >= 3 else 0):
            candidates_scores.append(3 + len(overlap))
        if want & surname and missing:
            # first-name variants are forgiven; for 3+ word names one extra
            # unmatched middle name is too ('Joshua Kofi Acheampong')
            residual = {m for m in missing if not any(m[:2] == f[:2] for f in first)}
            if len(residual) < len(missing) and len(residual) <= (1 if len(want) >= 3 else 0):
                candidates_scores.append(2 + len(overlap))
        if len(want) >= 4 and len(overlap) >= 2 and want & surname:
            candidates_scores.append(1 + len(overlap))
        if candidates_scores:
            scored.append((max(candidates_scores), p))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _median_yes_prices(event_odds: dict, market: str) -> dict[str, tuple[float, int]]:
    prices: dict[str, list[float]] = {}
    for book in event_odds.get("bookmakers", []):
        for m in book.get("markets", []):
            if m["key"] != market:
                continue
            for o in m["outcomes"]:
                if o.get("name") in ("Yes", "Over") and o.get("price", 0) > 1:
                    prices.setdefault(o.get("description") or o["name"], []).append(o["price"])
    return {n: (float(np.median(v)), len(v)) for n, v in prices.items()}


def _match_goals(match_odds: list[dict], fpl_names: dict[int, str]) -> dict[tuple[int, int], tuple[float, float]]:
    """Expected goals per side from bulk h2h + totals(2.5) odds, median across bookmakers."""
    out = {}
    for ev in match_odds:
        h, a = team_id(ev["home_team"], fpl_names), team_id(ev["away_team"], fpl_names)
        hs, ds, as_, ov, un = [], [], [], [], []
        for book in ev.get("bookmakers", []):
            for m in book["markets"]:
                if m["key"] == "h2h":
                    o = {x["name"]: x["price"] for x in m["outcomes"]}
                    if {ev["home_team"], ev["away_team"], "Draw"} <= o.keys():
                        hs.append(o[ev["home_team"]]); as_.append(o[ev["away_team"]]); ds.append(o["Draw"])
                if m["key"] == "totals":
                    o = {x["name"]: x["price"] for x in m["outcomes"] if x.get("point") == 2.5}
                    if {"Over", "Under"} <= o.keys():
                        ov.append(o["Over"]); un.append(o["Under"])
        if h and a and hs and ov:
            out[(h, a)] = market_odds.implied_goals(*(float(np.median(x)) for x in (hs, ds, as_, ov, un)))
    return out


def fetch_gameweek(bootstrap: dict, fixtures: list[dict], gw: int) -> dict:
    """Fetch match odds and player odds for every fixture in `gw`, convert to
    per-player goal/assist rates keyed by FPL id, save to data/odds_log/gw{gw}.json."""
    names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    gw_pairs = {(f["team_h"], f["team_a"]) for f in fixtures if f.get("event") == gw}
    events, _ = _get("/events")
    events = [e for e in events
              if (team_id(e["home_team"], names), team_id(e["away_team"], names)) in gw_pairs]
    if not events:
        raise OddsAPIError(f"No GW{gw} matches listed by the Odds API yet")

    match_odds, usage = _get("/odds", regions="uk", markets="h2h,totals", oddsFormat="decimal",
                             eventIds=",".join(e["id"] for e in events))
    raw = {"match_odds": match_odds, "events": {}}
    for e in events:
        raw["events"][e["id"]], usage = _get(f"/events/{e['id']}/odds", regions=REGIONS,
                                              markets=",".join(PLAYER_MARKETS), oddsFormat="decimal")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"gw{gw}_raw.json").write_text(json.dumps(raw))
    record = process(bootstrap, gw, raw, events)
    record["credits_remaining"] = usage.get("remaining")
    (LOG_DIR / f"gw{gw}.json").write_text(json.dumps(record, indent=1))
    return record


def process(bootstrap: dict, gw: int, raw: dict, events: list[dict] | None = None) -> dict:
    """Turn saved raw API responses into per-player rates (re-runnable, no credits)."""
    names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    team_xg = _match_goals(raw["match_odds"], names)
    events = events or [{"id": eid, "home_team": d["home_team"], "away_team": d["away_team"]}
                        for eid, d in raw["events"].items()]
    by_team: dict[int, list[dict]] = {}
    for p in bootstrap["elements"]:
        by_team.setdefault(p["team"], []).append(p)
    players, unmatched = {}, []
    for e in events:
        h, a = team_id(e["home_team"], names), team_id(e["away_team"], names)
        odds = raw["events"][e["id"]]
        cands = by_team.get(h, []) + by_team.get(a, [])
        for market, field in (("player_goal_scorer_anytime", "goals"), ("player_assists", "assists")):
            for name, (price, n_books) in _median_yes_prices(odds, market).items():
                if name.lower() in ("no scorer", "no goalscorer"):
                    continue
                p = match_player(name, cands)
                if p is None:
                    unmatched.append(name)
                    continue
                row = players.setdefault(str(p["id"]), {"web_name": p["web_name"], "team": p["team"]})
                row[f"p_{field}"] = 1 / price
                row[f"raw_rate_{field}"] = -math.log(max(1e-6, 1 - 1 / price))
                row[f"books_{field}"] = n_books

    return {
        "gw": gw,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "team_xg": {f"{h}-{a}": v for (h, a), v in team_xg.items()},
        "players": players,
        "unmatched_names": sorted(set(unmatched)),
    }


def load(gw: int) -> dict | None:
    path = LOG_DIR / f"gw{gw}.json"
    return json.loads(path.read_text()) if path.exists() else None


def rates_if_playing(record: dict, p_play: dict[int, float]) -> dict[int, float]:
    """FPL id -> market goal rate per match IF he plays, bookmaker margin removed."""
    players = {int(k): v for k, v in record["players"].items() if "raw_rate_goals" in v}
    out = {}
    for key, (lh, la) in record["team_xg"].items():
        h, a = (int(x) for x in key.split("-"))
        for team, target in ((h, lh), (a, la)):
            rows = {pid: r for pid, r in players.items() if r["team"] == team}
            weighted = sum(p_play.get(pid, 0.0) * r["raw_rate_goals"] for pid, r in rows.items())
            if weighted <= 0:
                continue
            scale = target * (1 - OWN_GOAL_SHARE) / weighted
            for pid, r in rows.items():
                out[pid] = r["raw_rate_goals"] * scale
    return out
