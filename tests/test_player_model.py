import math

import pandas as pd

from fpl import player_model as pm


class FlatRatings:
    """Every fixture identical unless an opponent is marked weak."""

    def __init__(self, weak=()):
        self.weak = set(weak)

    def expected_goals(self, team, opp, home):
        return (2.0, 0.5) if opp in self.weak else (1.4, 1.4)

    def ratios(self, team, opp, home):
        return (2.0 / 1.4 if opp in self.weak else 1.0), 1.0


def _hist(rows):
    return pd.DataFrame(rows, columns=pm.HISTORY_COLS)


def _row(gw, pid, minutes, points, et=3, team=1):
    return (gw, pid, team, et, minutes, points, 0, 0, 0.2 * minutes / 90, 0.1 * minutes / 90, 0, 1, 1)


LOOKUP = {1: {gw: [(3, True, 2)] for gw in range(1, 12)}}


def test_rotation_player_projects_below_regular_with_same_output_per_game():
    # Same points and per-90 output when they play; one plays every week, one every other week.
    rows = [_row(gw, 1, 90, 6) for gw in range(1, 7)]
    rows += [_row(gw, 2, 90 if gw % 2 else 0, 6 if gw % 2 else 0) for gw in range(1, 7)]
    out = pm.project([{"id": 1, "team": 1, "element_type": 3}, {"id": 2, "team": 1, "element_type": 3}],
                     _hist(rows), LOOKUP, [7], FlatRatings(), {}, {})
    assert out[1][0] > out[2][0] * 1.4


def test_weak_opponent_raises_a_defenders_projection_via_clean_sheet():
    rows = [_row(gw, 1, 90, 2, et=2) for gw in range(1, 7)]
    elements = [{"id": 1, "team": 1, "element_type": 2}]
    normal = pm.project(elements, _hist(rows), LOOKUP, [7], FlatRatings(), {}, {})[1][0]
    easy = pm.project(elements, _hist(rows), LOOKUP, [7], FlatRatings(weak={2}), {}, {})[1][0]
    assert easy > normal


def test_conceded_penalty_matches_poisson_expectation():
    lam = 1.4
    exact = sum((g // 2) * math.exp(-lam) * lam ** g / math.factorial(g) for g in range(30))
    assert abs(pm._expected_conceded_penalty(lam) - exact) < 1e-6
    assert pm._expected_conceded_penalty(0.0) == 0.0


def test_injured_now_but_back_later_only_loses_the_first_week():
    rows = [_row(gw, 1, 90, 6) for gw in range(1, 7)]
    el = [{"id": 1, "team": 1, "element_type": 3}]
    healthy = pm.project(el, _hist(rows), LOOKUP, [7, 8], FlatRatings(), {}, {})[1]
    doubt = pm.project(el, _hist(rows), LOOKUP, [7, 8], FlatRatings(), {}, {}, availability_now={1: 0.0})[1]
    assert doubt[1] == 0.0
    assert abs((healthy[0] - healthy[1]) - doubt[0]) < 1e-9
