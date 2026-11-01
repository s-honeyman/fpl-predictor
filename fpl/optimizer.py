"""ILP-based squad selection using PuLP (bundled CBC solver).

Picks the 15-man squad, valid starting XI, and captain that maximises
projected points under FPL's real constraints: £100.0m budget, max 3
players per club, and position quotas (2 GKP / 5 DEF / 5 MID / 3 FWD in the
squad; a valid formation - 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD - in the
starting XI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from . import model as m


@dataclass
class SquadResult:
    squad_ids: list[int]
    starting_ids: list[int]
    bench_ids: list[int]
    captain_id: int
    vice_captain_id: int
    total_cost: int  # 0.1m units
    predicted_starting_points: float  # includes captain double, excludes bench


def optimize_squad(
    df: pd.DataFrame,
    budget: int = m.BUDGET_TENTHS,
    old_squad_ids: list[int] | None = None,
    max_transfers: int | None = None,
) -> SquadResult:
    """Solve for the highest-projected-points valid squad.

    If `old_squad_ids` and `max_transfers` are given, constrains the result
    to keep at least (SQUAD_SIZE - max_transfers) players from the old
    squad - used for in-season transfer suggestions.
    """
    # Players already owned stay eligible even if now unavailable: holding is
    # always possible in real FPL, and excluding an injured held player made
    # the max_transfers=0 baseline infeasible and crashed `fpl transfers`.
    # Their predicted_points already scale to ~0 with availability, so the
    # solver benches them naturally.
    owned = set(old_squad_ids or [])
    pool = df[(df["availability"] > 0) | df["id"].isin(owned)].copy()
    ids = pool["id"].tolist()
    if len(ids) < m.SQUAD_SIZE:
        raise RuntimeError("Not enough available players to fill a squad.")

    points = dict(zip(pool.id, pool.predicted_points))
    cost = dict(zip(pool.id, pool.now_cost))
    pos = dict(zip(pool.id, pool.position_id))
    team = dict(zip(pool.id, pool.team))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad_var = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start_var = pulp.LpVariable.dicts("start", ids, cat="Binary")
    cap_var = pulp.LpVariable.dicts("cap", ids, cat="Binary")

    # Objective: starting XI points, with the captain's counted twice.
    prob += pulp.lpSum(points[i] * (start_var[i] + cap_var[i]) for i in ids)

    prob += pulp.lpSum(squad_var[i] for i in ids) == m.SQUAD_SIZE
    prob += pulp.lpSum(cost[i] * squad_var[i] for i in ids) <= budget

    for pos_id, rules in m.FORMATION_RULES.items():
        prob += pulp.lpSum(squad_var[i] for i in ids if pos[i] == pos_id) == rules["squad_select"]

    for t in set(team.values()):
        prob += pulp.lpSum(squad_var[i] for i in ids if team[i] == t) <= m.MAX_PER_TEAM

    for i in ids:
        prob += start_var[i] <= squad_var[i]
        prob += cap_var[i] <= start_var[i]

    prob += pulp.lpSum(start_var[i] for i in ids) == m.STARTING_XI
    prob += pulp.lpSum(cap_var[i] for i in ids) == 1

    for pos_id, rules in m.FORMATION_RULES.items():
        starters_in_pos = pulp.lpSum(start_var[i] for i in ids if pos[i] == pos_id)
        prob += starters_in_pos >= rules["min_play"]
        prob += starters_in_pos <= rules["max_play"]

    if old_squad_ids and max_transfers is not None:
        kept = [i for i in ids if i in old_squad_ids]
        min_keep = max(0, m.SQUAD_SIZE - max_transfers)
        prob += pulp.lpSum(squad_var[i] for i in kept) >= min_keep

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal squad (status={pulp.LpStatus[prob.status]}).")

    squad_ids = [i for i in ids if squad_var[i].value() > 0.5]
    starting_ids = [i for i in ids if start_var[i].value() > 0.5]
    bench_ids = [i for i in squad_ids if i not in starting_ids]
    captain_id = next(i for i in ids if cap_var[i].value() > 0.5)

    bench_sorted = sorted(bench_ids, key=lambda i: points[i], reverse=True)
    starters_sorted_desc = sorted(starting_ids, key=lambda i: points[i], reverse=True)
    vice_captain_id = next(i for i in starters_sorted_desc if i != captain_id)

    predicted_starting_points = sum(points[i] for i in starting_ids) + points[captain_id]

    return SquadResult(
        squad_ids=squad_ids,
        starting_ids=starting_ids,
        bench_ids=bench_sorted,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        total_cost=sum(cost[i] for i in squad_ids),
        predicted_starting_points=round(predicted_starting_points, 2),
    )


def optimize_multiweek(
    proj: dict[int, np.ndarray],
    cost: dict[int, int],
    pos: dict[int, int],
    team: dict[int, int],
    budget: int,
    weeks: list[int],
    bench_boost_week: int | None = None,
    pool_per_position: int = 60,
) -> tuple[list[int], float]:
    """One squad held over several weeks (a Wildcard), with its own best XI
    and captain each week. In `bench_boost_week` every squad member scores,
    so a bench of starters is worth money there and only there.

    Returns (squad ids, projected points over `weeks`). The pool is trimmed to
    each position's top players over the window plus its cheapest, to keep
    the solve fast without losing budget enablers."""
    weeks = list(weeks)
    pool: set[int] = set()
    for p in (1, 2, 3, 4):
        ids = [i for i in proj if pos.get(i) == p and i in cost]
        pool |= set(sorted(ids, key=lambda i: -float(sum(proj[i][w] for w in weeks)))[:pool_per_position])
        pool |= set(sorted(ids, key=lambda i: cost[i])[:15])
    ids = sorted(pool)
    pts = {(i, w): float(proj[i][w]) for i in ids for w in weeks}

    prob = pulp.LpProblem("fpl_multiweek", pulp.LpMaximize)
    sq = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    st = pulp.LpVariable.dicts("start", [(i, w) for i in ids for w in weeks], cat="Binary")
    cp = pulp.LpVariable.dicts("cap", [(i, w) for i in ids for w in weeks], cat="Binary")

    prob += pulp.lpSum(pts[i, w] * ((sq[i] if w == bench_boost_week else st[i, w]) + cp[i, w])
                       for i in ids for w in weeks)
    prob += pulp.lpSum(sq[i] for i in ids) == m.SQUAD_SIZE
    prob += pulp.lpSum(cost[i] * sq[i] for i in ids) <= budget
    for pos_id, rules in m.FORMATION_RULES.items():
        prob += pulp.lpSum(sq[i] for i in ids if pos[i] == pos_id) == rules["squad_select"]
    for t in {team[i] for i in ids}:
        prob += pulp.lpSum(sq[i] for i in ids if team[i] == t) <= m.MAX_PER_TEAM
    for w in weeks:
        prob += pulp.lpSum(cp[i, w] for i in ids) == 1
        if w == bench_boost_week:
            for i in ids:
                prob += cp[i, w] <= sq[i]
            continue
        prob += pulp.lpSum(st[i, w] for i in ids) == m.STARTING_XI
        for i in ids:
            prob += st[i, w] <= sq[i]
            prob += cp[i, w] <= st[i, w]
        for pos_id, rules in m.FORMATION_RULES.items():
            n = pulp.lpSum(st[i, w] for i in ids if pos[i] == pos_id)
            prob += n >= rules["min_play"]
            prob += n <= rules["max_play"]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal squad (status={pulp.LpStatus[prob.status]}).")
    return [i for i in ids if sq[i].value() > 0.5], float(pulp.value(prob.objective))
