from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import api, chips, league, live, model, odds_api, planner, player_model, prediction_log, review, squad_state, team_ratings
from .optimizer import SquadResult, optimize_squad

POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _warn_if_gameweek_in_progress(bootstrap: dict, fixtures: list[dict]) -> None:
    """FPL resets points_per_game/starts for the whole player pool at the
    start of each gameweek, then fills them in fixture-by-fixture as matches
    are actually played - not all at once. Mid-gameweek, that means players
    whose match already happened show real (possibly inflated-looking)
    stats while everyone else still shows zero, purely because of kickoff
    order, not real form. `_shrink_ppg` has no way to tell "hasn't played
    yet" apart from "played and returned zero", so projections (and any
    transfer/captaincy suggestion built on them) are unreliable until the
    whole gameweek is actually finished. Learned this the hard way - a
    mid-gameweek `fpl transfers` run once dropped Haaland from captain
    purely because Man City hadn't kicked off yet while Arsenal had."""
    current = next((e for e in bootstrap["events"] if not model._gameweek_effectively_finished(e["id"], e, fixtures)), None)
    if current is None:
        return
    gw_fixtures = [f for f in fixtures if f.get("event") == current["id"]]
    any_started = any(f.get("started") for f in gw_fixtures)
    if any_started:
        played = sum(1 for f in gw_fixtures if f.get("started"))
        print(
            f"WARNING: GW{current['id']} is in progress ({played}/{len(gw_fixtures)} fixtures started, "
            f"not finished). Its deadline has passed, so planning starts from the next gameweek, but "
            f"GW{current['id']}'s results aren't in the model yet - re-run once it's finished before acting."
        )


# First-gameweek weight on FPL's own ep_next in the decomposed model: FPL's
# number knows injury/suspension news our history can't (backtest: blending it
# lifted next-week ranking quality); later gameweeks are pure decomposition.
FIRST_WEEK_FPL_WEIGHT = 0.5


def _load_projections(horizon: int, refresh: bool, kind: str = "decomposed") -> tuple[pd.DataFrame, dict]:
    bootstrap = api.get_bootstrap(force_refresh=refresh)
    fixtures = api.get_fixtures(force_refresh=refresh)
    _warn_if_gameweek_in_progress(bootstrap, fixtures)
    df = model.project_players(bootstrap, fixtures, horizon=horizon)
    if kind == "decomposed":
        by_id = {p["id"]: p for p in bootstrap["elements"]}
        now = {i: model._availability_multiplier(p) for i, p in by_id.items()}
        later = {i: model._later_gameweek_availability(p) for i, p in by_id.items()}
        proj = player_model.live_projection(bootstrap, fixtures, df.attrs["gameweeks"], now, later)
        w = FIRST_WEEK_FPL_WEIGHT
        df["model_next_gw"] = df["id"].map(lambda i: round(proj[i][1], 3))
        df["predicted_points"] = [
            round(proj[i][0] - proj[i][1] + (1 - w) * proj[i][1] + w * ep * a, 2)
            for i, ep, a in zip(df["id"], df["ep_next"], df["availability"])
        ]
    df.attrs["model"] = kind
    _sync_prediction_log(bootstrap, fixtures, df)
    return df, bootstrap


def _sync_prediction_log(bootstrap: dict, fixtures: list[dict], df: pd.DataFrame) -> None:
    """The 'get smarter every gameweek' mechanism (see prediction_log.py):
    logs this run's next-gameweek predictions, and fills in real results for
    any previously-logged gameweek that's since finished. Runs automatically
    on every `build`/`transfers` call - no separate step to remember."""
    next_gw = df.attrs.get("next_gw")
    if next_gw is not None:
        # ep_next * availability is the model's actual single-gameweek estimate
        # for the immediate next gameweek - `predicted_points` in `df` is a
        # multi-gameweek horizon SUM, not comparable to one gameweek's actual score.
        log_df = pd.DataFrame(
            {
                "id": df["id"],
                "web_name": df["web_name"],
                "predicted_points": df["ep_next"] * df["availability"],
                "model_points": df["model_next_gw"],
            }
        )
        # next_gw always has its deadline ahead (model.next_gameweeks skips
        # passed deadlines), so refreshing the logged prediction is honest.
        added = prediction_log.record_predictions(log_df, next_gw, update_before_deadline=True)
        if added:
            print(f"(logged {added} predictions for GW{next_gw} - checked against reality once it's played)")

    finished_gws = {e["id"] for e in bootstrap["events"] if model._gameweek_effectively_finished(e["id"], e, fixtures)}
    for gw in prediction_log.gameweeks_needing_actuals(finished_gws):
        live = api.get_gameweek_live(gw)
        actuals = {el["id"]: el["stats"]["total_points"] for el in live["elements"]}
        minutes = {el["id"]: el["stats"]["minutes"] for el in live["elements"]}
        updated = prediction_log.record_actuals(gw, actuals, minutes)
        if updated:
            print(f"(filled in {updated} real results for GW{gw} - run scripts/recalibrate.py to check accuracy)")


def _player_row(df: pd.DataFrame, player_id: int) -> pd.Series:
    return df.loc[df["id"] == player_id].iloc[0]


def _print_squad(df: pd.DataFrame, result: SquadResult) -> None:
    gws = df.attrs.get("gameweeks", [])
    print(f"\nProjected over gameweeks {gws[0]}-{gws[-1]}" if gws else "")
    print(f"Squad cost: £{result.total_cost / 10:.1f}m / £100.0m")
    print(f"Projected starting-XI points (captain doubled): {result.predicted_starting_points:.1f}\n")

    def fmt(pid: int, tag: str = "") -> str:
        row = _player_row(df, pid)
        return (
            f"  {row['position']:<3} {row['web_name']:<18} {row['team_short']:<4} "
            f"£{row['price']:>4.1f}m  xPts {row['predicted_points']:>5.1f}{tag}"
        )

    print("STARTING XI:")
    starters = sorted(result.starting_ids, key=lambda i: POSITION_ORDER[_player_row(df, i)["position"]])
    for pid in starters:
        tag = ""
        if pid == result.captain_id:
            tag = "  (C)"
        elif pid == result.vice_captain_id:
            tag = "  (VC)"
        print(fmt(pid, tag))

    print("\nBENCH:")
    for pid in result.bench_ids:
        print(fmt(pid))
    print()


def cmd_build(args: argparse.Namespace) -> None:
    df, _bootstrap = _load_projections(horizon=args.horizon, refresh=args.refresh, kind=args.model)
    result = optimize_squad(df, budget=int(args.budget * 10))
    _print_squad(df, result)
    if not args.no_save:
        squad_state.save_squad(result, bank_tenths=int(args.budget * 10) - result.total_cost)
        print(f"Saved as your current squad -> run `fpl transfers` in future gameweeks to update it.")


def _live_team(args: argparse.Namespace, bootstrap: dict) -> live.LiveTeam | None:
    """Your real FPL team (as of the last deadline, plus any --pending moves),
    or None if no entry id is configured / --offline was passed."""
    cfg = live.load_config()
    if getattr(args, "offline", False) or not cfg:
        if getattr(args, "pending", None):
            raise SystemExit("--pending needs a linked FPL team - run `fpl sync --entry <id>` first.")
        return None
    team = live.live_team(cfg["entry_id"], refresh=args.refresh, bootstrap=bootstrap)
    return live.apply_pending(team, bootstrap, args.pending or [])


def _managers(args: argparse.Namespace, team: live.LiveTeam | None) -> list[league.Manager] | None:
    cfg = live.load_config()
    if not cfg or not cfg.get("league_id") or team is None:
        return None
    managers = league.load_managers(cfg["league_id"], cfg["entry_id"], refresh=args.refresh)
    for m in managers:  # rivals' public picks lag the same way, but yours we know better
        if m.is_you:
            m.squad_ids = list(team.squad_ids)
    return managers


def _team_source_line(team: live.LiveTeam) -> str:
    extra = f" + {len(team.pending)} pending transfer(s) you passed in" if team.pending else ""
    return (f"Using your live FPL team as of the GW{team.gameweek} deadline{extra}. "
            f"Bank £{team.bank_tenths / 10:.1f}m, {team.free_transfers} free transfer(s) for the next deadline.")


def cmd_transfers(args: argparse.Namespace) -> None:
    df, bootstrap = _load_projections(horizon=args.horizon, refresh=args.refresh, kind=args.model)
    team = _live_team(args, bootstrap)
    if team is not None:
        old_ids, bank = list(team.squad_ids), team.bank_tenths
        free = args.free_transfers if args.free_transfers is not None else team.free_transfers
        print(_team_source_line(team))
    else:
        saved = squad_state.load_squad()
        if saved is None:
            print("No saved squad found. Run `fpl sync --entry <id>` or `fpl build` first.")
            return
        old_ids, bank = saved["squad_ids"], saved.get("bank_tenths", 0)
        free = args.free_transfers if args.free_transfers is not None else 1
        print("Using the locally saved squad (no linked FPL team) - may be out of date.")
    # Budget is what the squad would really fetch (FPL keeps half of any rise)
    # plus the bank, and each owned player is costed at that same selling
    # price, so holding everything is always exactly affordable. Without a
    # linked team, current value stands in: a slight overestimate of buying power.
    sell = team.sell_price if team is not None else {}
    if sell:
        df = df.copy()
        owned = df["id"].isin(sell)
        df.loc[owned, "now_cost"] = df.loc[owned, "id"].map(sell)
    current_squad_value = int(df[df["id"].isin(old_ids)]["now_cost"].sum())
    budget = current_squad_value + bank

    baseline = optimize_squad(df, budget=budget, old_squad_ids=old_ids, max_transfers=0)

    best = baseline
    best_transfers = 0
    best_net = baseline.predicted_starting_points
    for k in range(1, free + 2):  # +1 hit allowed beyond free transfers
        penalty = 4 * max(0, k - free)
        candidate = optimize_squad(df, budget=budget, old_squad_ids=old_ids, max_transfers=k)
        net = candidate.predicted_starting_points - penalty
        if net > best_net:
            best, best_transfers, best_net = candidate, k, net

    if best_transfers == 0:
        print("No transfer improves your projected points enough to be worth it - hold.")
        _print_squad(df, baseline)
        return

    moved_out = [i for i in old_ids if i not in best.squad_ids]
    moved_in = [i for i in best.squad_ids if i not in old_ids]
    managers = _managers(args, team)

    print(f"Recommended: {best_transfers} transfer(s)"
          + (f" (costs {4 * max(0, best_transfers - free)} pts)" if best_transfers > free else " (free)")
          + f" - net {best_net - baseline.predicted_starting_points:+.1f} projected pts vs holding")
    for out_id, in_id in zip(moved_out, moved_in):
        out_row = _player_row(df, out_id)
        in_row = _player_row(df, in_id)
        print(f"  OUT: {out_row['web_name']:<18} xPts {out_row['predicted_points']:>5.1f}"
              f"   ->   IN: {in_row['web_name']:<18} xPts {in_row['predicted_points']:>5.1f}")
        if managers:
            print(f"        out: {league.ownership_tag(out_id, managers)} | in: {league.ownership_tag(in_id, managers)}")

    _print_squad(df, best)

    if not args.no_save:
        squad_state.save_squad(best, bank_tenths=budget - best.total_cost)
        print("Saved as your current squad.")


def cmd_sync(args: argparse.Namespace) -> None:
    cfg = live.load_config() or {}
    entry_id = args.entry or cfg.get("entry_id")
    if not entry_id:
        raise SystemExit("First time: `fpl sync --entry <your FPL team id> --league <mini-league id>`")
    live.save_config(entry_id, args.league or cfg.get("league_id"))
    bootstrap = api.get_bootstrap(force_refresh=args.refresh)
    team = live.apply_pending(live.live_team(entry_id, refresh=True), bootstrap, args.pending or [])

    class _R:  # squad_state.save_squad takes a SquadResult-shaped object
        pass
    r = _R()
    r.squad_ids, r.starting_ids, r.bench_ids = team.squad_ids, team.starting_ids, team.bench_ids
    r.captain_id, r.vice_captain_id = team.captain_id, team.vice_captain_id
    by_id = {p["id"]: p for p in bootstrap["elements"]}
    r.total_cost = sum(by_id[i]["now_cost"] for i in team.squad_ids)
    squad_state.save_squad(r, bank_tenths=team.bank_tenths)
    print(_team_source_line(team))
    print("Chips used: " + (", ".join(team.chips_used) or "none"))
    print("Local squad file now matches your real team.")


def cmd_league(args: argparse.Namespace) -> None:
    cfg = live.load_config()
    if not cfg or not cfg.get("league_id"):
        raise SystemExit("Link your team and league first: `fpl sync --entry <id> --league <id>`")
    df, bootstrap = _load_projections(horizon=args.horizon, refresh=args.refresh, kind=args.model)
    fixtures = api.get_fixtures()
    team = _live_team(args, bootstrap)
    managers = _managers(args, team)
    gws = df.attrs.get("gameweeks", [])
    you = next(m for m in managers if m.is_you)

    print(f"\nLEAGUE (after GW{min(len(m.weekly) for m in managers)})")
    leader = max(m.total for m in managers)
    for m in sorted(managers, key=lambda m: -m.total):
        gap = "" if m.total == leader else f"  ({m.total - leader:+d})"
        print(f"  {m.name:<8} {m.team_name:<20} {m.total:>4}{gap}{'   <- you' if m.is_you else ''}")

    strength = {m.entry_id: league.squad_strength(df, m.squad_ids) for m in managers}
    print(f"\nSQUAD STRENGTH NOW - best XI + captain this model picks from each current 15, GW{gws[0]}-{gws[-1]}:")
    for m in sorted(managers, key=lambda m: -strength[m.entry_id]):
        print(f"  {m.name:<8} {strength[m.entry_id]:6.1f}{'   <- you' if m.is_you else ''}")
    print("  (rivals' squads are as of the last deadline - their pending moves aren't public)")

    wp = league.win_probability(managers)
    mean_s = sum(strength.values()) / len(strength)
    # Projections run ~20% hot in backtests; scale the edge down to match.
    edge = {k: 0.8 * (v - mean_s) / len(gws) for k, v in strength.items()}
    wp_edge = league.win_probability(managers, weekly_edge=edge, edge_weeks=len(gws))
    print("\nROUGH CHANCE OF FINISHING TOP (crude simulation of the remaining gameweeks):")
    print(f"  {'':<8} {'equal squads':>12} {'+ next ' + str(len(gws)) + ' GWs of current squad edge':>32}")
    for m in sorted(managers, key=lambda m: -wp_edge[m.entry_id]):
        print(f"  {m.name:<8} {wp[m.entry_id]:>12.0%} {wp_edge[m.entry_id]:>32.0%}")

    t = league.ownership_table(df, managers)
    rivals = len(managers) - 1
    short = t[(~t.you) & (t.rivals_owning == rivals)].sort_values("xpts", ascending=False).head(6)
    one = t[(~t.you) & (t.rivals_owning > 0) & (t.rivals_owning < rivals)].sort_values("xpts", ascending=False).head(6)
    diff = t[(t.you) & (t.rivals_owning == 0)].sort_values("xpts", ascending=False)
    print("\nYOU'RE SHORT (every rival owns, you don't) - their haul costs you double:")
    for r in short.itertuples():
        print(f"  {r.player:<14}{r.team:<4}{r.pos:<4} xPts {r.xpts:5.1f}{'  [captained by ' + r.rival_captain + ']' if r.rival_captain else ''}")
    print("\nOWNED BY ONE RIVAL, NOT YOU:")
    for r in one.itertuples():
        print(f"  {r.player:<14}{r.team:<4}{r.pos:<4} xPts {r.xpts:5.1f}  ({r.rival_names})")
    print("\nYOUR DIFFERENTIALS (no rival owns) - every point moves the gap:")
    for r in diff.itertuples():
        print(f"  {r.player:<14}{r.team:<4}{r.pos:<4} xPts {r.xpts:5.1f}")

    ratings, _ = team_ratings.live_ratings(bootstrap, fixtures)
    lookup = model.build_team_fixture_lookup(fixtures)
    teams = {tm["id"]: tm["short_name"] for tm in bootstrap["teams"]}
    next_gw = gws[0]
    print(f"\nGW{next_gw} FIXTURE ODDS for your clubs (goals model - information, not yet in the projections):")
    seen = set()
    for pid in team.squad_ids:
        club = _player_row(df, pid)["team"]
        if club in seen:
            continue
        seen.add(club)
        for _, home, opp in lookup.get(club, {}).get(next_gw, []):
            gf, ga = ratings.expected_goals(club, opp, home)
            print(f"  {teams[club]:<4} {'H' if home else 'A'} v {teams[opp]:<4} expected goals {gf:.2f}-{ga:.2f}, "
                  f"clean sheet {ratings.clean_sheet_prob(club, opp, home):.0%}")


def cmd_plan(args: argparse.Namespace) -> None:
    _load_projections(horizon=args.horizon, refresh=args.refresh)  # keeps the prediction log in sync
    bootstrap = api.get_bootstrap()
    fixtures = api.get_fixtures()
    team = _live_team(args, bootstrap)
    if team is None:
        raise SystemExit("`fpl plan` needs your linked FPL team: `fpl sync --entry <id> --league <id>`")
    by_id = {p["id"]: p for p in bootstrap["elements"]}
    names = {i: p["web_name"] for i, p in by_id.items()}
    pos = {i: p["element_type"] for i, p in by_id.items()}
    club = {i: p["team"] for i, p in by_id.items()}
    price = {i: p["now_cost"] for i, p in by_id.items()}
    price.update(team.sell_price)  # your own players are worth what they'd sell for
    upcoming = model.next_gameweeks(bootstrap, fixtures, 38)
    hist = live.history(team.entry_id, args.refresh)
    windows = chips.chip_windows(bootstrap, hist.get("chips", []))
    first_half_end = min((c["stop"] for c in windows if c["start"] <= upcoming[0] <= c["stop"]), default=upcoming[0] + 4)
    gws = [g for g in upcoming if g <= max(first_half_end, upcoming[0] + args.horizon - 1)]

    now = {i: model._availability_multiplier(p) for i, p in by_id.items()}
    later = {i: model._later_gameweek_availability(p) for i, p in by_id.items()}
    h, lookup, ratings, priors, rates = player_model.live_inputs(bootstrap, fixtures)
    params = player_model.fixture_params(bootstrap["elements"], h, lookup, gws, ratings, priors, rates, now, later)
    proj = player_model.by_gameweek(params, gws)
    # Chips compare weeks against each other, so they use this model alone in
    # every week: blending FPL's estimate into week 1 only made week 1 look
    # like the chip week (the Free Hit solver piles into wherever the two disagree).
    proj_model = {i: a.copy() for i, a in proj.items()}
    model_first = {i: a[0] for i, a in proj.items()}  # this model alone, before the FPL blend
    w0 = FIRST_WEEK_FPL_WEIGHT
    for i, a in proj.items():  # same first-week blend with FPL's estimate as `transfers`
        a[0] = (1 - w0) * a[0] + w0 * float(by_id[i]["ep_next"] or 0) * now.get(i, 1.0)
    n = min(args.horizon, len(gws))
    squad = set(team.squad_ids)
    print(_team_source_line(team))

    # 1. Transfer plan
    plan = planner.plan_transfers(squad, team.bank_tenths, team.free_transfers, proj, pos, club, price, n)
    hold = sum(planner.best_xi(squad, w, proj, pos) for w in range(n))
    print(f"\nTRANSFER PLAN, GW{gws[0]}-{gws[n - 1]} (week by week, real free-transfer and -4 rules):")
    for w in range(n):
        mv = plan.moves[w]
        txt = ", ".join(f"{names[o]} -> {names[i]}" for o, i in mv) if mv else "hold"
        print(f"  GW{gws[w]}: {txt}")
    print(f"  Projected: plan {plan.net:.1f} (after -{plan.hits} in hits) vs holding all {n} weeks {hold:.1f}")
    print("  (Backtest: this planner and one-shot `fpl transfers` finished +54 and -15 against each other over two")
    print("   seasons - treat this as a timing view, not a proven upgrade.)")

    # 2. Captaincy - ranked by this model's own average: the rule that beat
    # ceiling-based picks in the weekly backtest. FPL's estimate (mostly recent
    # form) is shown alongside, not mixed in.
    first = params[params["gw"] == gws[0]]
    cands = sorted(squad, key=lambda i: -model_first.get(i, 0))[:5]
    managers = _managers(args, team)
    rival_caps = [names[m.captain_id] for m in managers if not m.is_you] if managers else []
    print(f"\nCAPTAIN, GW{gws[0]} - by this model's average (backtest: beat picking by ceiling in both seasons):")
    print(f"  {'':<14}{'model avg':>10}{'FPL est.':>10}{'chance 10+':>12}{'90th pct':>10}")
    for i in cands:
        sim = player_model.simulate(first[first["player_id"] == i], n=20000, seed=i)
        print(f"  {names[i]:<14}{model_first.get(i, 0):>10.1f}{float(by_id[i]['ep_next'] or 0):>10.1f}"
              f"{(sim >= 10).mean():>12.0%}{np.percentile(sim, 90):>10.0f}{'   <- captain' if i == cands[0] else ''}")
    if rival_caps:
        print(f"  Rivals' captains last deadline: {', '.join(rival_caps)}")

    # 3. Chips
    print("\nCHIPS still available:")
    for c in windows:
        print(f"  {c['name']:<9} GW{c['start']}-{c['stop']}")
    in_window = [w for w, g in enumerate(gws) if g <= first_half_end]
    if in_window:
        bb = chips.bench_boost_values(squad, proj_model, pos, len(gws))
        budget = sum(price[i] for i in squad) + team.bank_tenths
        fh = chips.free_hit_values(squad, budget, proj_model, pos, club, price, len(in_window))
        print(f"\n  Projected chip value by gameweek, current squad, this model only "
              f"(first-half chips expire after GW{first_half_end}):")
        print(f"  {'GW':<5}{'Bench Boost':>12}{'Free Hit':>10}")
        for w in in_window:
            print(f"  {gws[w]:<5}{bb[w]:>12.1f}{fh[w]:>10.1f}")
        playable = {c["name"] for c in windows if c["start"] <= gws[0] <= c["stop"]}
        if "wildcard" in playable:
            wc, normal = chips.wildcard_value(squad, team.bank_tenths, team.free_transfers, proj_model, pos, club, price, n)
            print(f"  Wildcard now: best fresh squad {wc:.1f} vs best normal-transfer plan {normal:.1f} over GW{gws[0]}-{gws[n - 1]} "
                  f"({wc - normal:+.1f})")
        if {"wildcard", "bboost"} <= playable:
            combos = chips.wildcard_then_bench_boost(squad, team.bank_tenths, proj_model, pos, club, price, n)
            base_bb = max(bb[w] for w in range(n))
            print(f"\n  Wildcard now + Bench Boost later, GW{gws[0]}-{gws[n - 1]} (squad bought knowing that week's bench counts),")
            print(f"  vs normal transfers + Bench Boost on your current squad's best week ({normal:.1f} + {base_bb:.1f}):")
            for b, pts, ids in combos:
                bench_pts = pts - sum(planner.best_xi(ids, w, proj_model, pos) for w in range(n))
                print(f"    BB GW{gws[b]}: {pts:.1f} (bench adds {bench_pts:.1f})  {pts - normal - base_bb:+.1f}")
            b, pts, ids = max(combos, key=lambda c: c[1])
            xi = sorted(ids, key=lambda i: (pos[i], -proj_model[i][b]))
            print(f"    Squad for BB GW{gws[b]}: " + ", ".join(f"{names[i]} {proj_model[i][b]:.1f}" for i in xi))
        bench = [i for i in squad if by_id[i]["status"] != "a"]
        if bench:
            print("  Flagged in your 15 (hurts Bench Boost): " + ", ".join(f"{names[i]} ({by_id[i]['status']})" for i in bench))
    if managers:
        for m in managers:
            if not m.is_you:
                used = [f"{c['name']} GW{c['event']}" for c in live.history(m.entry_id, args.refresh).get("chips", [])]
                print(f"  {m.name} has used: {', '.join(used) or 'nothing'}")


def _model_rates(bootstrap: dict, fixtures: list[dict], gw: int) -> dict[str, dict]:
    """This model's own per-player goal/assist rates for `gw`, as known now."""
    by_id = {p["id"]: p for p in bootstrap["elements"]}
    now = {i: model._availability_multiplier(p) for i, p in by_id.items()}
    h, lookup, ratings, priors, rates = player_model.live_inputs(bootstrap, fixtures)
    prm = player_model.fixture_params(bootstrap["elements"], h, lookup, [gw], ratings, priors, rates, now, now)
    out = {}
    for pid, g in prm.groupby("player_id"):
        out[str(pid)] = {"goals90": float(g["lam_goals"].sum()), "assists90": float(g["lam_assists"].sum()),
                         "p_play": float(min(1.0, (g["p60"] + g["psub"]).mean())), "xpts": float(g["xpts"].sum())}
    return out


def _evaluate_odds(bootstrap: dict, fixtures: list[dict]) -> None:
    rows = []
    for e in bootstrap["events"]:
        rec = odds_api.load(e["id"])
        if not rec or "model" not in rec or not model._gameweek_effectively_finished(e["id"], e, fixtures):
            continue
        p_play = {int(k): v["p_play"] for k, v in rec["model"].items()}
        mkt = odds_api.rates_if_playing(rec, p_play)
        live_stats = {x["id"]: x["stats"] for x in api.get_gameweek_live(e["id"])["elements"]}
        for pid, m in mkt.items():
            st, mod = live_stats.get(pid), rec["model"].get(str(pid))
            if not st or not mod or st["minutes"] == 0:
                continue
            rows.append((e["id"], pid, m, mod["goals90"], st["goals_scored"], min(st["minutes"], 90) / 90))
    if not rows:
        print("No finished gameweek with logged odds yet - run this after the gameweek has been played.")
        return
    d = pd.DataFrame(rows, columns=["gw", "pid", "market", "model", "goals", "exposure"])

    def loglik(rate):
        lam = np.clip(rate * d["exposure"], 1e-4, None)
        return float((d["goals"] * np.log(lam) - lam).mean())

    full = d[d["exposure"] >= 60 / 90]
    print(f"Goal-scoring rates vs what happened - {len(d)} player-matches with minutes, GW{', GW'.join(map(str, sorted(d.gw.unique())))}")
    print(f"  {'':<14}{'log-likelihood':>16}{'rank corr (60+ mins)':>22}")
    for col, label in (("model", "our model"), ("market", "bookmakers")):
        sp = full[[col, "goals"]].corr(method="spearman").iloc[0, 1]
        print(f"  {label:<14}{loglik(d[col]):>16.4f}{sp:>22.3f}")
    d["blend"] = np.sqrt(d["model"] * d["market"])
    full = d[d["exposure"] >= 60 / 90]
    sp = full[["blend", "goals"]].corr(method="spearman").iloc[0, 1]
    print(f"  {'50/50 blend':<14}{loglik(d['blend']):>16.4f}{sp:>22.3f}")
    print("  (higher log-likelihood = better. Too little data to act on until several gameweeks are in.)")


def cmd_odds(args: argparse.Namespace) -> None:
    bootstrap = api.get_bootstrap(force_refresh=args.refresh)
    fixtures = api.get_fixtures(force_refresh=args.refresh)
    if args.evaluate:
        _evaluate_odds(bootstrap, fixtures)
        return
    gw = model.next_gameweeks(bootstrap, fixtures, 1)[0]
    rec = odds_api.load(gw)
    if args.fetch or rec is None:
        print(f"Fetching GW{gw} odds from The Odds API (~32 credits)...")
        rec = odds_api.fetch_gameweek(bootstrap, fixtures, gw)
    elif args.reprocess:
        import json as _json
        raw = _json.loads((odds_api.LOG_DIR / f"gw{gw}_raw.json").read_text())
        rec = {**odds_api.process(bootstrap, gw, raw), "fetched_at": rec["fetched_at"],
               "credits_remaining": rec.get("credits_remaining")}
    rec["model"] = _model_rates(bootstrap, fixtures, gw)
    (odds_api.LOG_DIR / f"gw{gw}.json").write_text(__import__("json").dumps(rec, indent=1))

    p_play = {int(k): v["p_play"] for k, v in rec["model"].items()}
    mkt = odds_api.rates_if_playing(rec, p_play)
    names = {p["id"]: p["web_name"] for p in bootstrap["elements"]}
    print(f"GW{gw} odds fetched {rec['fetched_at']} | credits left: {rec.get('credits_remaining')} | "
          f"players matched: {len(rec['players'])}, unmatched names: {len(rec['unmatched_names'])}")
    print("SHADOW MODE: logged for comparison only, not used in projections yet.\n")

    team = _live_team(args, bootstrap)
    if team:
        print("Your squad - expected goals per match IF he plays:")
        print(f"  {'':<14}{'P(plays)':>9}{'our model':>11}{'bookmakers':>12}")
        for pid in sorted(team.squad_ids, key=lambda i: -mkt.get(i, 0)):
            mod = rec["model"].get(str(pid), {})
            if mod.get("goals90", 0) < 0.05 and pid not in mkt:
                continue
            print(f"  {names[pid]:<14}{mod.get('p_play', 0):>9.0%}{mod.get('goals90', 0):>11.2f}"
                  f"{mkt[pid] if pid in mkt else float('nan'):>12.2f}")
    likely = [pid for pid in mkt if rec["model"].get(str(pid), {}).get("p_play", 0) >= 0.6
              and rec["model"][str(pid)]["goals90"] >= 0.05]
    ratio = {pid: mkt[pid] / rec["model"][str(pid)]["goals90"] for pid in likely}
    for title, keys in (("Bookmakers MUCH higher than our model", sorted(ratio, key=lambda p: -ratio[p])[:8]),
                        ("Bookmakers MUCH lower than our model", sorted(ratio, key=lambda p: ratio[p])[:8])):
        print(f"\n{title} (likely starters):")
        for pid in keys:
            print(f"  {names[pid]:<14} model {rec['model'][str(pid)]['goals90']:.2f}  bookmakers {mkt[pid]:.2f}  (x{ratio[pid]:.1f})")


def cmd_review(args: argparse.Namespace) -> None:
    cfg = live.load_config()
    if not cfg or not cfg.get("league_id"):
        raise SystemExit("Link your team and league first: `fpl sync --entry <id> --league <id>`")
    bootstrap = api.get_bootstrap(force_refresh=args.refresh)
    fixtures = api.get_fixtures(force_refresh=args.refresh)
    names = {p["id"]: p["web_name"] for p in bootstrap["elements"]}
    started = [e["id"] for e in bootstrap["events"] if e["finished"] or e["is_current"]]
    gw = args.gw or max(started)
    finished = {e["id"] for e in bootstrap["events"] if e["finished"]}
    # Past gameweeks never change; only the one being reviewed may still be live.
    points = {g: {x["id"]: x["stats"]["total_points"]
                  for x in api.get_gameweek_live(g, force_refresh=args.refresh and g == gw)["elements"]}
              for g in range(1, gw + 1)}
    model_pts: dict[int, dict[int, float]] = {}
    for e in prediction_log._load_raw():
        if e.get("model_points") is not None:
            model_pts.setdefault(e["gw"], {})[e["player_id"]] = e["model_points"]

    standings = live.league_standings(cfg["league_id"], args.refresh)["standings"]["results"]
    managers = []
    for r in standings:
        eid = r["entry"]
        weeks = [review.week_from_picks(g, live.picks(eid, g, refresh=args.refresh and g == gw)) for g in range(1, gw + 1)]
        managers.append((r["player_name"].split()[0], eid, weeks, live.transfers(eid, args.refresh)))
    me = next(m for m in managers if m[1] == cfg["entry_id"])

    played = [f for f in fixtures if f["event"] == gw]
    status = "" if gw in finished else f" - IN PROGRESS, {sum(f['finished'] or f['finished_provisional'] for f in played)} of {len(played)} matches finished"
    print(f"\nGW{gw} REVIEW{status}")
    for name, _, weeks, _ in managers:
        w = weeks[-1]
        chip = f" ({w.chip})" if w.chip else ""
        print(f"  {name:<8} {review.week_points(w, points[gw]) - w.hits:>3}{f' (after -{w.hits})' if w.hits else ''}{chip}"
              f"   captain {names[w.captain]}")

    mine = me[2][-1]
    for name, eid, weeks, _ in managers:
        if eid == me[1]:
            continue
        theirs = weeks[-1]
        swing = review.gap_swing(mine, theirs, points[gw])
        net = sum(s for _, s in swing) - mine.hits + theirs.hits
        print(f"\nGap to {name} moved {net:+d} this week. Biggest movers (+ = for you):")
        for pid, s in swing[:8]:
            a, b = mine.mult.get(pid, 0), theirs.mult.get(pid, 0)
            who = (f"you x{a}, {name} x{b}" if a and b else f"you x{a}" if a else f"{name} x{b}")
            print(f"  {s:+4d}  {names[pid]:<15} {points[gw].get(pid, 0):>3} pts  ({who})")

    through = max(g for g in range(1, gw + 1) if g in finished or g == gw)
    print(f"\nSEASON DECISION SCOREBOARD (GW1-{through}{', current week still live' if gw not in finished else ''})")
    print(f"  {'':<8}{'transfers':>10}{'net vs hold':>12}{'captain':>9}{'best in XI':>11}{'model pick':>11}{'bench':>7}")
    for name, eid, weeks, tr in managers:
        sb = review.scoreboard(weeks, tr, points, model_pts, through)
        mp = f"{sb.captain_model}/{sb.model_weeks}wk" if sb.model_weeks else "-"
        print(f"  {name:<8}{sb.transfers:>10}{sb.transfer_net:>+12d}{sb.captain_pts:>9}{sb.captain_best:>11}{mp:>11}{sb.bench:>7}")
    print("  net vs hold: each bought player's points minus the sold one's, from the move to now, less hits")
    print("  (Wildcard/Free Hit weeks left out). captain: the armband's extra points; best in XI: the hindsight")
    print("  ceiling; model pick: the extra points had you captained this model's top projected starter, over")
    print("  the weeks the log has its numbers for. bench: points left on the bench after auto-subs.")

    sb = review.scoreboard(me[2], me[3], points, model_pts, through)
    if sb.transfer_rows:
        print("\nYour transfers, best and worst:")
        rows = sorted(sb.transfer_rows, key=lambda r: -r[3])
        for g, o, i, d in rows[:3] + (rows[3:][-3:] if len(rows) > 3 else []):
            print(f"  GW{g:<3}{names[o]:>15} -> {names[i]:<15}{d:+4d}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fpl", description="Fantasy Premier League squad predictor/optimizer.")
    parser.add_argument("--horizon", type=int, default=5, help="Number of upcoming gameweeks to project over (default 5).")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh cached FPL API data.")
    parser.add_argument("--no-save", action="store_true", help="Don't persist the result as your current squad.")
    parser.add_argument("--model", choices=["decomposed", "classic"], default="decomposed",
                        help="decomposed (default): minutes x goals/assists/clean sheets per fixture; "
                             "classic: points-per-game x FDR.")

    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build an optimal squad from scratch (e.g. for gameweek 1).")
    build_p.add_argument("--budget", type=float, default=100.0, help="Budget in £m (default 100.0).")
    build_p.set_defaults(func=cmd_build)

    def live_flags(p):
        p.add_argument("--pending", action="append", metavar="OUT>IN",
                       help="A transfer you've made that FPL won't show until the deadline, e.g. 'Ajayi>De Cuyper'. Repeatable.")
        p.add_argument("--offline", action="store_true", help="Ignore your linked FPL team and use the local squad file.")

    transfers_p = sub.add_parser("transfers", help="Suggest transfers for your squad using updated projections.")
    transfers_p.add_argument("--free-transfers", type=int, default=None,
                             help="Free transfers available (default: worked out from your FPL history, else 1).")
    live_flags(transfers_p)
    transfers_p.set_defaults(func=cmd_transfers)

    sync_p = sub.add_parser("sync", help="Link and pull your real FPL team (and mini-league).")
    sync_p.add_argument("--entry", type=int, help="Your FPL team id (from your team's points-page URL).")
    sync_p.add_argument("--league", type=int, help="Your mini-league id.")
    sync_p.add_argument("--pending", action="append", metavar="OUT>IN", help="Transfers made since the last deadline.")
    sync_p.set_defaults(func=cmd_sync)

    league_p = sub.add_parser("league", help="Your mini-league: gaps, squad strength, rival ownership, fixture odds.")
    live_flags(league_p)
    league_p.set_defaults(func=cmd_league)

    plan_p = sub.add_parser("plan", help="Week-by-week transfer plan, captain options and chip outlook.")
    live_flags(plan_p)
    plan_p.set_defaults(func=cmd_plan)

    odds_p = sub.add_parser("odds", help="Goalscorer odds vs the model (shadow mode), and how each did.")
    odds_p.add_argument("--fetch", action="store_true", help="Re-fetch from The Odds API (~32 credits).")
    odds_p.add_argument("--reprocess", action="store_true", help="Re-process saved raw odds (no credits).")
    odds_p.add_argument("--evaluate", action="store_true", help="Score logged odds vs the model on played gameweeks.")
    live_flags(odds_p)
    odds_p.set_defaults(func=cmd_odds)

    review_p = sub.add_parser("review", help="Grade a gameweek's decisions: where the gap to each rival moved, "
                                             "and a season scoreboard of transfers, captaincy and bench.")
    review_p.add_argument("--gw", type=int, default=None, help="Gameweek to review (default: the current one).")
    review_p.set_defaults(func=cmd_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
