from fpl import review


def _week(gw, mult, captain, hits=0, chip=None):
    return review.Week(gw, 0, hits, chip, mult, captain)


def test_gap_swing_cancels_shared_players_and_counts_captains_double():
    pts = {1: 10, 2: 6, 3: 4, 4: 2}
    me = _week(1, {1: 2, 2: 1, 3: 0}, captain=1)
    rival = _week(1, {1: 1, 2: 1, 4: 1}, captain=4)
    swing = dict(review.gap_swing(me, rival, pts))
    assert swing == {1: 10, 4: -2}  # shared 2 cancels, my bench player 3 moves nothing
    assert sum(swing.values()) == review.week_points(me, pts) - review.week_points(rival, pts)


def test_transfer_scored_only_while_the_player_was_kept():
    pts = {1: {10: 2, 11: 8}, 2: {10: 2, 11: 8}, 3: {10: 2, 11: 8, 12: 0}}
    weeks = [_week(g, {10: 1, 11: 1, 12: 1}, captain=10) for g in (1, 2, 3)]
    transfers = [{"event": 1, "element_out": 10, "element_in": 11},
                 {"event": 3, "element_out": 11, "element_in": 12}]
    sb = review.scoreboard(weeks, transfers, pts)
    rows = {(o, i): d for _, o, i, d in sb.transfer_rows}
    assert rows[(10, 11)] == 12  # GW1-2 only: +6 a week, not GW3 after he was sold
    assert rows[(11, 12)] == -8


def test_wildcard_week_moves_and_hits_left_out():
    pts = {1: {10: 0, 11: 5}, 2: {10: 0, 11: 5}}
    weeks = [_week(1, {10: 1}, 10, hits=4), _week(2, {11: 1}, 11, chip="wildcard")]
    transfers = [{"event": 1, "element_out": 9, "element_in": 10},
                 {"event": 2, "element_out": 10, "element_in": 11}]
    sb = review.scoreboard(weeks, transfers, pts)
    assert sb.transfers == 1 and sb.transfer_net == 0 - 4


def test_captaincy_against_best_and_model_pick():
    pts = {1: {1: 2, 2: 12, 3: 6}}
    # captain 1 didn't play enough to matter here; armband stayed on him (x2)
    weeks = [_week(1, {1: 2, 2: 1, 3: 1, 4: 0}, captain=1)]
    sb = review.scoreboard(weeks, [], pts, model_pts_by_gw={1: {1: 5.0, 3: 6.0, 4: 9.0}})
    assert sb.captain_pts == 2
    assert sb.captain_best == 12
    assert sb.captain_model == 6  # model's top among the XI (bench player 4 excluded)
    assert sb.model_weeks == 1


def test_vice_captain_stand_in_counts_as_captain_points():
    pts = {1: {1: 0, 2: 7}}
    weeks = [_week(1, {1: 0, 2: 2}, captain=1)]  # captain blanked, vice got the armband
    sb = review.scoreboard(weeks, [], pts)
    assert sb.captain_pts == 7
