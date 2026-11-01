"""Gameweek review: grade the decisions, not the predictions.

Player-level accuracy (rank correlation, error) says little about whether a
manager is winning: among players who start, one week's points barely rank
at all, and the league is decided by a handful of choices - who to captain,
who to buy and sell, who sits on the bench. This module scores those.

For each gameweek it splits the change in the gap to each rival into the
players that moved it (each player's points times your multiplier minus
theirs - shared players cancel, captains count double), and across the
season it keeps a scoreboard per manager:
  - transfers: points of each player bought minus the player sold, from the
    gameweek of the move through the latest one, less hits;
  - captaincy: captain's extra points vs the best pick in that XI (hindsight
    ceiling) and vs this model's pick from the prediction log (what following
    the model would have given);
  - bench: points left on the bench after auto-subs.

All of it comes from FPL's public picks/transfers/live endpoints plus the
prediction log, so it can be run retroactively for any played gameweek.
Picks' multipliers already include auto-subs and the vice-captain stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Week:
    gw: int
    points: int  # FPL's total for the week, before hits
    hits: int
    chip: str | None
    mult: dict[int, int]  # element -> multiplier after auto-subs (0 = bench)
    captain: int


def week_from_picks(gw: int, picks: dict) -> Week:
    eh = picks["entry_history"]
    mult = {p["element"]: p["multiplier"] for p in picks["picks"]}
    captain = next(p["element"] for p in picks["picks"] if p["is_captain"])
    return Week(gw, eh["points"], eh["event_transfers_cost"], picks.get("active_chip"), mult, captain)


def week_points(week: Week, points: dict[int, int]) -> int:
    """The week's score before hits, from the picks - FPL's own total reads 0
    until a live gameweek has been processed."""
    return sum(m * points.get(i, 0) for i, m in week.mult.items())


def gap_swing(me: Week, rival: Week, points: dict[int, int]) -> list[tuple[int, int]]:
    """(element, points this player moved the gap by, in my favour), largest
    first. Sums to my week minus theirs, before hits."""
    ids = set(me.mult) | set(rival.mult)
    out = [(i, (me.mult.get(i, 0) - rival.mult.get(i, 0)) * points.get(i, 0)) for i in ids]
    return sorted([x for x in out if x[1]], key=lambda x: -abs(x[1]))


@dataclass
class Scoreboard:
    transfer_net: int = 0
    transfers: int = 0
    hits: int = 0
    captain_pts: int = 0  # captain's extra points (multiplier - 1) x points
    captain_best: int = 0  # same, had the best player in the XI been captain
    captain_model: int = 0  # same, had this model's top-projected starter been captain
    model_weeks: int = 0  # weeks with a model pick in the log
    bench: int = 0
    transfer_rows: list = field(default_factory=list)


def captain_extra(week: Week, element: int, points: dict[int, int]) -> int:
    """Extra points from captaining `element`: the multiplier above 1 (2, or
    3 with Triple Captain) times his points."""
    extra = max(max(week.mult.values()), 2) - 1
    return extra * points.get(element, 0)


def scoreboard(weeks: list[Week], transfers: list[dict], points_by_gw: dict[int, dict[int, int]],
               model_pts_by_gw: dict[int, dict[int, float]] | None = None, through: int | None = None) -> Scoreboard:
    """Season scoreboard. `transfers` is FPL's entry/{id}/transfers/ list;
    moves made in a Wildcard or Free Hit week are left out (a whole-squad
    rebuild isn't a run of single decisions). Transfers are scored from their
    gameweek through `through` (default: the last week given)."""
    model_pts_by_gw = model_pts_by_gw or {}
    through = through or max(w.gw for w in weeks)
    chip_weeks = {w.gw for w in weeks if w.chip in ("wildcard", "freehit")}
    sb = Scoreboard()
    for w in weeks:
        pts = points_by_gw.get(w.gw, {})
        sb.hits += w.hits
        sb.bench += sum(pts.get(i, 0) for i, m in w.mult.items() if m == 0)
        xi = [i for i, m in w.mult.items() if m > 0]
        if not xi:
            continue
        # Whoever actually wore the armband (the vice, if the captain didn't play).
        sb.captain_pts += sum((m - 1) * pts.get(i, 0) for i, m in w.mult.items() if m > 1)
        sb.captain_best += captain_extra(w, max(xi, key=lambda i: pts.get(i, 0)), pts)
        mp = model_pts_by_gw.get(w.gw)
        if mp:
            # The model's pick among the eleven who were picked to start
            # (auto-subs aside), as it stood before the deadline.
            pick = max(xi, key=lambda i: mp.get(i, 0.0))
            sb.captain_model += captain_extra(w, pick, pts)
            sb.model_weeks += 1
    for t in transfers:
        g = t["event"]
        if g in chip_weeks or g > through:
            continue
        # Score the move only while the bought player was kept: up to the
        # week before he was sold on, if he was.
        resold = [u["event"] for u in transfers if u["element_out"] == t["element_in"] and u["event"] > g]
        end = min(resold) - 1 if resold else through
        d = sum(points_by_gw.get(x, {}).get(t["element_in"], 0) - points_by_gw.get(x, {}).get(t["element_out"], 0)
                for x in range(g, end + 1))
        sb.transfer_net += d
        sb.transfers += 1
        sb.transfer_rows.append((g, t["element_out"], t["element_in"], d))
    sb.transfer_net -= sum(w.hits for w in weeks if w.gw not in chip_weeks)
    return sb
