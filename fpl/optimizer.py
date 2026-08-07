"""ILP-based squad selection using PuLP (bundled CBC solver).

Picks the 15-man squad, valid starting XI, and captain that maximises
projected points under FPL's real constraints: £100.0m budget, max 3
players per club, and position quotas (2 GKP / 5 DEF / 5 MID / 3 FWD in the
squad; a valid formation - 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD - in the
starting XI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    pool = df[df["availability"] > 0].copy()
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
