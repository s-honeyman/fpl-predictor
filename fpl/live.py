"""Live sync with your real FPL team and mini-league via FPL's public API.

Your entry id and league id live in data/league.json (git-ignored, because
the repo is public and those ids resolve to real names). Set them once with
`fpl sync --entry <id> --league <id>`.

Limitation worth knowing: FPL only publishes a team's picks as of the last
deadline. Transfers you've made for the upcoming gameweek are invisible until
that deadline passes, so pass them with --pending "Out>In".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import api

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "league.json"
MAX_BANKED_FREE_TRANSFERS = 5


def load_config() -> dict | None:
    if not CONFIG_PATH.exists():
        return None
    return json.loads(CONFIG_PATH.read_text())


def save_config(entry_id: int, league_id: int | None) -> None:
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({"entry_id": entry_id, "league_id": league_id}, indent=2))


def _get(path: str, cache: str, refresh: bool) -> dict:
    return api._cached_get(f"{api.BASE_URL}/{path}", cache, refresh)


def entry(entry_id: int, refresh: bool = False) -> dict:
    return _get(f"entry/{entry_id}/", f"entry_{entry_id}.json", refresh)


def history(entry_id: int, refresh: bool = False) -> dict:
    return _get(f"entry/{entry_id}/history/", f"entry_{entry_id}_history.json", refresh)


def picks(entry_id: int, gw: int, refresh: bool = False) -> dict:
    return _get(f"entry/{entry_id}/event/{gw}/picks/", f"entry_{entry_id}_gw{gw}_picks.json", refresh)


def league_standings(league_id: int, refresh: bool = False) -> dict:
    return _get(f"leagues-classic/{league_id}/standings/", f"league_{league_id}.json", refresh)


def free_transfers(hist: dict) -> int:
    """Free transfers available for the next deadline, rebuilt from the
    season's transfer history under the current rules: one per gameweek,
    unused ones roll over up to a cap of five. Wildcard and Free Hit keep the
    banked count; that rule is applied as documented by FPL, but this account
    hasn't played either chip yet, so it's untested against real data."""
    chips = {c["event"]: c["name"] for c in hist.get("chips", [])}
    rows = sorted(hist["current"], key=lambda e: e["event"])
    if not rows:
        return 1
    ft = 1
    for e in rows:
        if e["event"] == rows[0]["event"]:
            continue  # first gameweek: squad built freely, transfers don't count
        if chips.get(e["event"]) in ("wildcard", "freehit"):
            continue
        ft = min(MAX_BANKED_FREE_TRANSFERS, max(0, ft - e["event_transfers"]) + 1)
    if len(rows) == 1:
        return 1
    return ft


@dataclass
class LiveTeam:
    entry_id: int
    gameweek: int  # the deadline these picks are from
    squad_ids: list[int]
    starting_ids: list[int]
    bench_ids: list[int]
    captain_id: int
    vice_captain_id: int
    bank_tenths: int
    free_transfers: int
    chips_used: list[str] = field(default_factory=list)
    pending: list[tuple[int, int]] = field(default_factory=list)


def live_team(entry_id: int, refresh: bool = False) -> LiveTeam:
    e = entry(entry_id, refresh)
    gw = e["current_event"]
    p = picks(entry_id, gw, refresh)
    h = history(entry_id, refresh)
    pk = p["picks"]
    return LiveTeam(
        entry_id=entry_id,
        gameweek=gw,
        squad_ids=[x["element"] for x in pk],
        starting_ids=[x["element"] for x in pk if x["position"] <= 11],
        bench_ids=[x["element"] for x in sorted(pk, key=lambda x: x["position"]) if x["position"] > 11],
        captain_id=next(x["element"] for x in pk if x["is_captain"]),
        vice_captain_id=next(x["element"] for x in pk if x["is_vice_captain"]),
        bank_tenths=p["entry_history"]["bank"],
        free_transfers=free_transfers(h),
        chips_used=[c["name"] for c in h.get("chips", [])],
    )


def resolve_player(bootstrap: dict, name: str, among: set[int] | None = None) -> dict:
    """'Ajayi' or 'Wilson:LEE' -> player dict. Restricting to `among` (e.g. your
    squad for an outgoing player) avoids clashes like the three Wilsons."""
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    web, _, team = name.partition(":")
    hits = [
        p for p in bootstrap["elements"]
        if p["web_name"].lower() == web.strip().lower()
        and (not team or teams[p["team"]] == team.strip().upper())
        and (among is None or p["id"] in among)
    ]
    if len(hits) != 1:
        options = ", ".join(f"{p['web_name']}:{teams[p['team']]}" for p in hits) or "no match"
        raise ValueError(f"Can't pin down player {name!r} ({options}) - use Name:TEAM")
    return hits[0]


def apply_pending(team: LiveTeam, bootstrap: dict, pending: list[str]) -> LiveTeam:
    """Apply transfers made since the last deadline ('Out>In', web names)."""
    by_id = {p["id"]: p for p in bootstrap["elements"]}
    for move in pending:
        out_name, _, in_name = move.partition(">")
        out_p = resolve_player(bootstrap, out_name, set(team.squad_ids))
        in_p = resolve_player(bootstrap, in_name)
        if out_p["element_type"] != in_p["element_type"]:
            raise ValueError(f"{out_p['web_name']} and {in_p['web_name']} play different positions")
        team.bank_tenths += out_p["now_cost"] - in_p["now_cost"]
        for lst in (team.squad_ids, team.starting_ids, team.bench_ids):
            if out_p["id"] in lst:
                lst[lst.index(out_p["id"])] = in_p["id"]
        if team.captain_id == out_p["id"]:
            team.captain_id = in_p["id"]
        if team.vice_captain_id == out_p["id"]:
            team.vice_captain_id = in_p["id"]
        team.free_transfers = max(0, team.free_transfers - 1)
        team.pending.append((out_p["id"], in_p["id"]))
    if team.bank_tenths < 0:
        names = ", ".join(by_id[i]["web_name"] for _, i in team.pending)
        raise ValueError(f"Pending transfers ({names}) would leave the bank negative - check the moves")
    return team
