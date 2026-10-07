from __future__ import annotations

import argparse

import pandas as pd

from . import api, model, prediction_log, squad_state
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
            {"id": df["id"], "web_name": df["web_name"], "predicted_points": df["ep_next"] * df["availability"]}
        )
        added = prediction_log.record_predictions(log_df, next_gw)
        if added:
            print(f"(logged {added} predictions for GW{next_gw} - checked against reality once it's played)")

    finished_gws = {e["id"] for e in bootstrap["events"] if model._gameweek_effectively_finished(e["id"], e, fixtures)}
    for gw in prediction_log.gameweeks_needing_actuals(finished_gws):
        live = api.get_gameweek_live(gw)
        actuals = {el["id"]: el["stats"]["total_points"] for el in live["elements"]}
        updated = prediction_log.record_actuals(gw, actuals)
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


def cmd_transfers(args: argparse.Namespace) -> None:
    saved = squad_state.load_squad()
    if saved is None:
        print("No saved squad found. Run `fpl build` first to create your starting squad.")
        return

    df, _bootstrap = _load_projections(horizon=args.horizon, refresh=args.refresh)
    old_ids = saved["squad_ids"]
    # Budget must reflect the squad's CURRENT live value, not the stale price
    # it was bought at - prices drift daily (a held player who's risen in
    # price makes "hold everything" look infeasible under the old total,
    # even though holding never costs anything new). Real FPL selling price
    # is actually current value minus half of any profit, which needs a
    # per-entry API call we don't have without a linked FPL team ID - using
    # full current value is a slight overestimate of real buying power, but
    # never an underestimate, so it can't cause the same false-infeasible bug.
    current_squad_value = int(df[df["id"].isin(old_ids)]["now_cost"].sum())
    budget = current_squad_value + saved.get("bank_tenths", 0)

    baseline = optimize_squad(df, budget=budget, old_squad_ids=old_ids, max_transfers=0)

    best = baseline
    best_transfers = 0
    best_net = baseline.predicted_starting_points
    for k in range(1, args.free_transfers + 2):  # +1 hit allowed beyond free transfers
        penalty = 4 * max(0, k - args.free_transfers)
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

    print(f"Recommended: {best_transfers} transfer(s)"
          + (f" (costs {4 * max(0, best_transfers - args.free_transfers)} pts)" if best_transfers > args.free_transfers else " (free)"))
    for out_id, in_id in zip(moved_out, moved_in):
        out_row = _player_row(df, out_id)
        in_row = _player_row(df, in_id)
        print(f"  OUT: {out_row['web_name']:<18} xPts {out_row['predicted_points']:>5.1f}"
              f"   ->   IN: {in_row['web_name']:<18} xPts {in_row['predicted_points']:>5.1f}")

    _print_squad(df, best)

    if not args.no_save:
        squad_state.save_squad(best, bank_tenths=budget - best.total_cost)
        print("Saved as your current squad.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fpl", description="Fantasy Premier League squad predictor/optimizer.")
    parser.add_argument("--horizon", type=int, default=5, help="Number of upcoming gameweeks to project over (default 5).")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh cached FPL API data.")
    parser.add_argument("--no-save", action="store_true", help="Don't persist the result as your current squad.")

    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build an optimal squad from scratch (e.g. for gameweek 1).")
    build_p.add_argument("--budget", type=float, default=100.0, help="Budget in £m (default 100.0).")
    build_p.set_defaults(func=cmd_build)

    transfers_p = sub.add_parser("transfers", help="Suggest transfers to your saved squad using updated projections.")
    transfers_p.add_argument("--free-transfers", type=int, default=1, help="Free transfers available this gameweek (default 1).")
    transfers_p.set_defaults(func=cmd_transfers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
