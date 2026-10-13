"""Mini-league awareness: how your squad stacks up against your actual rivals.

One thing this deliberately does NOT do is pretend owning a rival's players
raises your expected lead. Rivals' teams are fixed when you choose, so your
expected gap moves only with your own expected points. What rival ownership
changes is RISK:
  - a player both rivals own and you don't is a liability you're short on
    (their haul costs you twice, their blank gains you twice);
  - a player neither rival owns is pure leverage (every point moves the gap).
When behind, leaning toward leverage among near-equal options raises your
chance of overtaking; when ahead, covering their players protects the lead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import live
from .optimizer import optimize_squad

TOTAL_GAMEWEEKS = 38


@dataclass
class Manager:
    entry_id: int
    name: str
    team_name: str
    total: int
    squad_ids: list[int]
    captain_id: int
    weekly: list[int]
    is_you: bool


def load_managers(league_id: int, my_entry: int, refresh: bool = False) -> list[Manager]:
    st = live.league_standings(league_id, refresh)
    out = []
    for r in st["standings"]["results"]:
        eid = r["entry"]
        e = live.entry(eid, refresh)
        p = live.picks(eid, e["current_event"], refresh)
        h = live.history(eid, refresh)
        out.append(Manager(
            entry_id=eid,
            name=r["player_name"].split()[0],
            team_name=r["entry_name"],
            total=r["total"],
            squad_ids=[x["element"] for x in p["picks"]],
            captain_id=next(x["element"] for x in p["picks"] if x["is_captain"]),
            weekly=[x["points"] - x["event_transfers_cost"] for x in sorted(h["current"], key=lambda x: x["event"])],
            is_you=eid == my_entry,
        ))
    return out


def squad_strength(df: pd.DataFrame, squad_ids: list[int]) -> float:
    """Best XI + captain this model would pick from that exact 15, projected
    over the same horizon as `df`. Comparable across managers."""
    sub = df[df["id"].isin(squad_ids)]
    r = optimize_squad(sub, budget=int(sub["now_cost"].sum()), old_squad_ids=list(squad_ids), max_transfers=0)
    return r.predicted_starting_points


def win_probability(managers: list[Manager], weekly_edge: dict[int, float] | None = None,
                    edge_weeks: int | None = None, sims: int = 20000, seed: int = 7) -> dict[int, float]:
    """Rough chance each manager finishes top, simulating the remaining
    gameweeks with each manager's weekly score as normal noise around a mean.
    Spread is taken from how much your weekly scores have actually differed
    this season, so it's crude with only a handful of gameweeks - read it as
    'how open is this', not a precise forecast. `weekly_edge` (points per
    gameweek above/below the others) applies for `edge_weeks` only, since a
    squad edge today won't survive everyone's future transfers."""
    played = min(len(m.weekly) for m in managers)
    left = TOTAL_GAMEWEEKS - played
    diffs = [a - b for i, x in enumerate(managers) for y in managers[i + 1:] for a, b in zip(x.weekly, y.weekly)]
    sd_each = max(np.std(diffs, ddof=1) / np.sqrt(2), 8.0) if len(diffs) > 2 else 15.0
    rng = np.random.default_rng(seed)
    edge = weekly_edge or {}
    totals = np.array([m.total for m in managers], float)
    means = np.array([edge.get(m.entry_id, 0.0) for m in managers])
    drift_weeks = left if edge_weeks is None else min(edge_weeks, left)
    sims_tot = totals + rng.normal(means * drift_weeks, sd_each * np.sqrt(left), size=(sims, len(managers)))
    wins = np.bincount(sims_tot.argmax(axis=1), minlength=len(managers)) / sims
    return {m.entry_id: float(w) for m, w in zip(managers, wins)}


def ownership_table(df: pd.DataFrame, managers: list[Manager]) -> pd.DataFrame:
    you = next(m for m in managers if m.is_you)
    rivals = [m for m in managers if not m.is_you]
    ids = set().union(*(m.squad_ids for m in managers))
    rows = []
    for pid in ids:
        r = df[df["id"] == pid]
        if r.empty:
            continue
        r = r.iloc[0]
        owners = [m.name for m in rivals if pid in m.squad_ids]
        rows.append({
            "id": pid, "player": r["web_name"], "team": r["team_short"], "pos": r["position"],
            "xpts": r["predicted_points"], "you": pid in you.squad_ids,
            "rivals_owning": len(owners), "rival_names": "+".join(owners),
            "rival_captain": "+".join(m.name for m in rivals if m.captain_id == pid),
        })
    t = pd.DataFrame(rows)
    t["leverage"] = 1 - t["rivals_owning"] / max(len(rivals), 1)
    return t


def ownership_tag(pid: int, managers: list[Manager]) -> str:
    owners = [m.name for m in managers if not m.is_you and pid in m.squad_ids]
    rivals = sum(1 for m in managers if not m.is_you)
    if not owners:
        return "differential: no rival owns"
    if len(owners) == rivals:
        return "owned by all rivals"
    return "owned by " + ", ".join(owners)
