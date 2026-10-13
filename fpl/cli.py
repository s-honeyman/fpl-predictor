from __future__ import annotations

import argparse

import pandas as pd

from . import api, league, live, model, prediction_log, squad_state, team_ratings
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
            f"not finished) - projections and any transfer/captaincy suggestion below are unreliable "
            f"right now (players whose match hasn't kicked off yet look artificially worse than players "
            f"who've already played). Wait until GW{current['id']} is fully finished before acting on this."
        )


def _load_projections(horizon: int, refresh: bool) -> tuple[pd.DataFrame, dict]:
    bootstrap = api.get_bootstrap(force_refresh=refresh)
    fixtures = api.get_fixtures(force_refresh=refresh)
    _warn_if_gameweek_in_progress(bootstrap, fixtures)
    df = model.project_players(bootstrap, fixtures, horizon=horizon)
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
        added = prediction_log.record_predictions(log_df, next_gw)
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
    df, _bootstrap = _load_projections(horizon=args.horizon, refresh=args.refresh)
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
    team = live.live_team(cfg["entry_id"], refresh=args.refresh)
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
    df, bootstrap = _load_projections(horizon=args.horizon, refresh=args.refresh)
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
    # Budget must reflect the squad's CURRENT live value, not the stale price
    # it was bought at - prices drift daily (a held player who's risen in
    # price makes "hold everything" look infeasible under the old total,
    # even though holding never costs anything new). Real FPL selling price
    # is current value minus half of any profit, so full current value is a
    # slight overestimate of buying power - never an underestimate, so it
    # can't cause the false-infeasible bug.
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
    df, bootstrap = _load_projections(horizon=args.horizon, refresh=args.refresh)
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="fpl", description="Fantasy Premier League squad predictor/optimizer.")
    parser.add_argument("--horizon", type=int, default=5, help="Number of upcoming gameweeks to project over (default 5).")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh cached FPL API data.")
    parser.add_argument("--no-save", action="store_true", help="Don't persist the result as your current squad.")

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
