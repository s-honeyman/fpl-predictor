"""Multi-week transfer planner.

`fpl transfers` asks a one-shot question: which k transfers made now give the
best total over the next five weeks? Real FPL is a sequence: one free transfer
a week, unused ones roll over (max 5), extra moves cost 4 points each, and a
player worth buying in week 3 shouldn't be bought in week 1. This searches
those sequences week by week (beam search: keep the best few dozen partial
plans at each step) using per-gameweek projections, so it can say things like
"hold this week, use two free transfers next week".

Squad points per week are the best legal XI plus captain from the 15, the
same rules the optimizer uses. Selling price is taken as the current price,
a slight overestimate of buying power (FPL keeps half of any price rise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

MIN_PLAY = {1: 1, 2: 3, 3: 2, 4: 1}
MAX_PLAY = {1: 1, 2: 5, 3: 5, 4: 3}
HIT = 4
MAX_FREE = 5


def best_xi(squad, w: int, proj: dict[int, np.ndarray], pos: dict[int, int], bench_boost: bool = False) -> float:
    """Best legal XI + captain for week index `w`. Taking each position's
    minimum from its best players, then filling the remaining slots with the
    best players left (within each position's maximum), is optimal here."""
    vals = {pid: (proj[pid][w] if pid in proj else 0.0) for pid in squad}
    if bench_boost:
        return sum(vals.values()) + max(vals.values())
    by_pos = {p: sorted((vals[i] for i in squad if pos[i] == p), reverse=True) for p in (1, 2, 3, 4)}
    chosen = []
    rest = []
    for p in (1, 2, 3, 4):
        chosen += by_pos[p][:MIN_PLAY[p]]
        rest += by_pos[p][MIN_PLAY[p]:MAX_PLAY[p]]
    chosen += sorted(rest, reverse=True)[: 11 - len(chosen)]
    return sum(chosen) + max(chosen)


@dataclass
class Plan:
    squad: frozenset
    bank: int
    ft: int
    points: float = 0.0
    hits: int = 0
    moves: list = field(default_factory=list)  # per week: list of (out, in)
    sold: frozenset = frozenset()  # never buy back a player this plan has sold

    @property
    def net(self) -> float:
        return self.points - self.hits


def _clubs_ok(squad, team: dict[int, int]) -> bool:
    counts: dict[int, int] = {}
    for i in squad:
        counts[team[i]] = counts.get(team[i], 0) + 1
        if counts[team[i]] > 3:
            return False
    return True


def _candidates(plan: Plan, w: int, proj, pos, team, price, pool: dict[int, list[int]], k: int):
    singles = []
    for out in plan.squad:
        rem_out = proj.get(out, np.zeros(1))[w:].sum()
        for inn in pool[pos[out]]:
            if inn in plan.squad or inn in plan.sold or price[inn] > plan.bank + price[out]:
                continue
            gain = proj[inn][w:].sum() - rem_out
            if gain > 0:
                singles.append((gain, out, inn))
    singles.sort(reverse=True)
    singles = [s for s in singles if _clubs_ok((plan.squad - {s[1]}) | {s[2]}, team)][:k]
    doubles = []
    for (g1, o1, i1), (g2, o2, i2) in combinations(singles, 2):
        if o1 == o2 or i1 == i2:
            continue
        if price[i1] + price[i2] > plan.bank + price[o1] + price[o2]:
            continue
        sq = (plan.squad - {o1, o2}) | {i1, i2}
        if _clubs_ok(sq, team):
            doubles.append((g1 + g2, ((o1, i1), (o2, i2))))
    doubles.sort(reverse=True)
    return [((o, i),) for _, o, i in singles] + [d for _, d in doubles[:k]]


def plan_transfers(squad, bank: int, free_transfers: int, proj: dict[int, np.ndarray], pos: dict[int, int],
                   team: dict[int, int], price: dict[int, int], n_weeks: int, beam: int = 30, k: int = 8,
                   allow_hits: bool = True, pool_size: int = 40) -> Plan:
    """Best transfer sequence over the next `n_weeks` (projection arrays are
    indexed by week from 0). Returns the winning plan; `.moves[w]` is that
    week's transfers, possibly empty."""
    pool = {}
    for p in (1, 2, 3, 4):
        ids = [i for i in proj if pos.get(i) == p]
        pool[p] = sorted(ids, key=lambda i: -proj[i].sum())[:pool_size]

    def future(sq, w):
        return sum(best_xi(sq, x, proj, pos) for x in range(w, n_weeks))

    plans = [Plan(frozenset(squad), bank, free_transfers)]
    for w in range(n_weeks):
        nxt = {}
        for pl in plans:
            for mv in [()] + _candidates(pl, w, proj, pos, team, price, pool, k):
                n = len(mv)
                hit = HIT * max(0, n - pl.ft)
                if hit and not allow_hits:
                    continue
                sq, bk = pl.squad, pl.bank
                for o, i in mv:
                    sq = (sq - {o}) | {i}
                    bk += price[o] - price[i]
                new = Plan(sq, bk, min(MAX_FREE, max(0, pl.ft - n) + 1),
                           pl.points + best_xi(sq, w, proj, pos), pl.hits + hit, pl.moves + [list(mv)],
                           pl.sold | {o for o, _ in mv})
                key = (sq, new.ft, bk)
                if key not in nxt or nxt[key].net < new.net:
                    nxt[key] = new
        plans = sorted(nxt.values(), key=lambda p: -(p.net + future(p.squad, w + 1)))[:beam]
    return max(plans, key=lambda p: p.net)
