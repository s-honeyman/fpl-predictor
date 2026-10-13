import pytest

from fpl import league, live, team_ratings


def _hist(transfers, chips=()):
    return {
        "current": [{"event": gw, "event_transfers": t} for gw, t in enumerate(transfers, start=1)],
        "chips": [{"event": gw, "name": n} for gw, n in chips],
    }


def test_free_transfers_matches_real_season_so_far():
    # GW1-6 transfers 0,0,3,1,0,2 -> one free transfer for GW7 (checked against the app)
    assert live.free_transfers(_hist([0, 0, 3, 1, 0, 2])) == 1


def test_free_transfers_roll_and_cap_at_five():
    assert live.free_transfers(_hist([0, 0, 0])) == 3
    assert live.free_transfers(_hist([0] * 10)) == 5


def test_free_transfers_kept_through_wildcard():
    assert live.free_transfers(_hist([0, 0, 0, 9], chips=[(4, "wildcard")])) == 3


BOOT = {
    "teams": [{"id": 1, "short_name": "HUL"}, {"id": 2, "short_name": "BHA"}, {"id": 3, "short_name": "LEE"}],
    "elements": [
        {"id": 10, "web_name": "Ajayi", "team": 1, "element_type": 2, "now_cost": 42},
        {"id": 20, "web_name": "De Cuyper", "team": 2, "element_type": 2, "now_cost": 51},
        {"id": 30, "web_name": "Wilson", "team": 3, "element_type": 3, "now_cost": 65},
        {"id": 31, "web_name": "Wilson", "team": 1, "element_type": 3, "now_cost": 45},
    ],
}


def _team():
    return live.LiveTeam(entry_id=1, gameweek=6, squad_ids=[10, 30], starting_ids=[30], bench_ids=[10],
                         captain_id=30, vice_captain_id=10, bank_tenths=17, free_transfers=1)


def test_apply_pending_swaps_player_and_updates_bank_and_free_transfers():
    t = live.apply_pending(_team(), BOOT, ["Ajayi>De Cuyper"])
    assert 20 in t.squad_ids and 10 not in t.squad_ids
    assert t.bank_tenths == 17 + 42 - 51
    assert t.free_transfers == 0
    assert t.vice_captain_id == 20


def test_ambiguous_names_are_refused_not_guessed():
    with pytest.raises(ValueError):
        live.resolve_player(BOOT, "Wilson")
    assert live.resolve_player(BOOT, "Wilson:LEE")["id"] == 30


def test_pending_rejects_negative_bank():
    t = _team()
    t.bank_tenths = 0
    with pytest.raises(ValueError):
        live.apply_pending(t, BOOT, ["Ajayi>De Cuyper"])


def _mgr(eid, squad, you=False):
    return league.Manager(entry_id=eid, name=f"m{eid}", team_name="t", total=0, squad_ids=squad,
                          captain_id=squad[0], weekly=[50, 60], is_you=you)


def test_ownership_tags():
    ms = [_mgr(1, [1, 2], you=True), _mgr(2, [3, 4]), _mgr(3, [3, 5])]
    assert league.ownership_tag(3, ms) == "owned by all rivals"
    assert league.ownership_tag(4, ms) == "owned by m2"
    assert league.ownership_tag(1, ms) == "differential: no rival owns"


def test_win_probability_sums_to_one_and_favours_the_leader_when_equal():
    ms = [_mgr(1, [1], you=True), _mgr(2, [2]), _mgr(3, [3])]
    ms[1].total = 40
    wp = league.win_probability(ms, sims=4000)
    assert abs(sum(wp.values()) - 1) < 1e-9
    assert wp[2] == max(wp.values())


def test_team_ratings_recover_obvious_strength_and_home_advantage():
    names = {1: "Strong", 2: "Weak"}
    matches = [team_ratings.Match("Strong", "Weak", 3, 0)] * 10 + [team_ratings.Match("Weak", "Strong", 0, 2)] * 10
    r = team_ratings.fit_ratings(matches, [], names)
    gf, ga = r.expected_goals(1, 2, True)
    assert gf > ga
    assert r.clean_sheet_prob(1, 2, True) > r.clean_sheet_prob(2, 1, True)
