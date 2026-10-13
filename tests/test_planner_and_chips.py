import numpy as np

from fpl import chips, planner


def _squad():
    # 2 GK, 5 DEF, 5 MID, 3 FWD; ids encode position in the tens digit
    ids = [11, 12, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43]
    pos = {i: i // 10 for i in ids}
    return ids, pos


def test_best_xi_respects_formation_and_doubles_captain():
    ids, pos = _squad()
    proj = {i: np.array([1.0]) for i in ids}
    proj[41] = np.array([10.0])  # star forward
    proj[11], proj[12] = np.array([5.0]), np.array([4.0])  # only one keeper may start
    # 1 GK (5) + 10 outfield: forward 10 + nine 1s; captain doubles the 10
    assert planner.best_xi(ids, 0, proj, pos) == 5 + 10 + 9 + 10
    assert planner.best_xi(ids, 0, proj, pos, bench_boost=True) == 5 + 4 + 10 + 12 + 10


def _market(gain_per_week, weeks):
    ids, pos = _squad()
    proj = {i: np.full(weeks, 2.0) for i in ids}
    proj[99] = np.full(weeks, 2.0 + gain_per_week)  # a better midfielder on the market
    pos[99] = 3
    team = {i: i for i in list(ids) + [99]}
    price = {i: 50 for i in list(ids) + [99]}
    return ids, proj, pos, team, price


def test_planner_uses_a_free_transfer_for_a_clear_upgrade():
    ids, proj, pos, team, price = _market(3.0, 3)
    plan = planner.plan_transfers(ids, 0, 1, proj, pos, team, price, 3)
    assert plan.moves[0] and plan.moves[0][0][1] == 99
    assert plan.hits == 0


def test_planner_wont_pay_a_hit_that_doesnt_pay_back():
    ids, proj, pos, team, price = _market(1.0, 2)  # +2 over two weeks < 4-point hit
    plan = planner.plan_transfers(ids, 0, 0, proj, pos, team, price, 2)
    assert plan.hits == 0
    assert all(not m for m in plan.moves[:1])


def test_planner_never_buys_back_a_player_it_sold():
    ids, proj, pos, team, price = _market(3.0, 4)
    plan = planner.plan_transfers(ids, 0, 2, proj, pos, team, price, 4)
    sold = [o for week in plan.moves for o, _ in week]
    bought = [i for week in plan.moves for _, i in week]
    assert not set(sold) & set(bought)


def test_chip_windows_drop_only_the_half_already_used():
    boot = {"chips": [
        {"name": "3xc", "start_event": 1, "stop_event": 19},
        {"name": "3xc", "start_event": 20, "stop_event": 38},
        {"name": "wildcard", "start_event": 2, "stop_event": 19},
    ]}
    left = chips.chip_windows(boot, [{"name": "3xc", "event": 3}])
    assert {(c["name"], c["start"]) for c in left} == {("3xc", 20), ("wildcard", 2)}
