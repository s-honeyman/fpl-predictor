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


def test_selling_price_keeps_half_of_a_rise_and_all_of_a_fall():
    assert live.selling_price(60, 63) == 61  # +0.3 -> +0.1 (rounded down)
    assert live.selling_price(60, 64) == 62
    assert live.selling_price(60, 57) == 57


def test_selling_prices_from_latest_purchase_or_start_price():
    boot = {"elements": [{"id": 10, "now_cost": 46, "cost_change_start": 4},  # held since the start at 4.2
                         {"id": 20, "now_cost": 55, "cost_change_start": 5}]}
    transfers = [{"event": 3, "element_in": 20, "element_in_cost": 50},
                 {"event": 7, "element_in": 20, "element_in_cost": 52}]  # sold and bought back
    assert live.selling_prices([10, 20], transfers, boot) == {10: 44, 20: 53}


def test_apply_pending_sells_at_selling_price():
    t = _team()
    t.sell_price = {10: 40, 30: 65}
    t = live.apply_pending(t, BOOT, ["Ajayi>De Cuyper"])
    assert t.bank_tenths == 17 + 40 - 51
    assert t.sell_price == {30: 65, 20: 51}


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


def test_market_odds_conversion_and_name_matching():
    from fpl import market_odds as mo
    lh, la = mo.implied_goals(2.7, 3.3, 2.7, 1.9, 1.9)
    assert abs(lh - la) < 0.01
    fav_h, fav_a = mo.implied_goals(1.2, 7.0, 13.0, 1.4, 3.0)
    assert fav_h > 2 * fav_a
    names = {1: "Man Utd", 2: "Spurs", 3: "Ipswich Town", 4: "Man City"}
    assert mo.match_team("Man United", names) == 1
    assert mo.match_team("Tottenham", names) == 2
    assert mo.match_team("Ipswich", names) == 3
    assert mo.match_team("Man City", names) == 4


def test_odds_ratings_fall_back_entirely_when_a_fixture_has_no_odds():
    from fpl import market_odds as mo
    names = {1: "A", 2: "B", 3: "C"}
    ms = [team_ratings.Match("A", "B", 2, 1), team_ratings.Match("B", "C", 1, 1), team_ratings.Match("C", "A", 0, 2)] * 3
    base = team_ratings.fit_ratings(ms, [], names)
    r = mo.OddsAdjustedRatings(base, {(1, 2): (2.5, 0.5)}, {(1, 2): 9}, as_of_gw=5)
    assert r.expected_goals(1, 2, True) == (2.5, 0.5)
    assert r.ratios(2, 3, True) == base.ratios(2, 3, True)


def test_bookmaker_player_names_match_strictly_but_tolerate_variants():
    from fpl import odds_api
    c = [
        {"first_name": "David", "second_name": "Raya Martín", "web_name": "Raya"},
        {"first_name": "Martin", "second_name": "Ødegaard", "web_name": "Ødegaard"},
        {"first_name": "Benjamin", "second_name": "White", "web_name": "White"},
        {"first_name": "Josh", "second_name": "Acheampong", "web_name": "Acheampong"},
        {"first_name": "Ferdi", "second_name": "Kadıoğlu", "web_name": "Kadıoğlu"},
        {"first_name": "Pascal", "second_name": "Groß", "web_name": "Groß"},
    ]
    m = lambda n: (odds_api.match_player(n, c) or {}).get("web_name")
    assert m("Martin Odegaard") == "Ødegaard"
    assert m("Martin Odegaard") != "Raya"
    assert m("Ben White") == "White"
    assert m("Joshua Kofi Acheampong") == "Acheampong"
    assert m("Ferdi Kadioglu") == "Kadıoğlu"
    assert m("Pascal Gross") == "Groß"
    assert m("Martin Smith") is None


def test_long_and_compound_bookmaker_names():
    from fpl import odds_api
    c = [
        {"first_name": "Marcelino", "second_name": "Núñez", "web_name": "Núñez"},
        {"first_name": "Promise", "second_name": "David", "web_name": "David"},
        {"first_name": "Ben", "second_name": "Gannon-Doak", "web_name": "Gannon-Doak"},
        {"first_name": "Nilson", "second_name": "Angulo", "web_name": "Angulo"},
        {"first_name": "Juan", "second_name": "Angulo", "web_name": "J.Angulo"},
    ]
    m = lambda n: (odds_api.match_player(n, c) or {}).get("first_name")
    assert m("Marcelino Ignacio Nunez Espinoza") == "Marcelino"
    assert m("Promise Oluwatobi Emmanuel David") == "Promise"
    assert m("Ben Gannondoak") == "Ben"
    assert m("Nilson David Angulo Ramirez") == "Nilson"
    assert m("Juan Riquelme Angulo Caicedo") == "Juan"
