"""Recalibration: checks the model's real in-season accuracy against the
accumulated prediction log (data/prediction_log.jsonl), and - once enough
real gameweeks have completed - suggests whether SHRINKAGE_K should move.

This is what makes the tool "get smarter every gameweek": each time `fpl
transfers` runs, it logs that gameweek's predictions and fills in actuals for
any gameweek that's since finished (see fpl/prediction_log.py). Run this
script periodically (natural to pair with the 4-gameweek replanning cadence)
to see whether the model's real-world accuracy is holding up, drifting, or
improving - not just trust the historical backtest forever.

Suggestions are printed for manual review, not auto-applied - SHRINKAGE_K
lives in fpl/model.py and should be a deliberate, visible edit, the same way
every other constant in this project has been.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fpl import model, prediction_log


def main() -> None:
    completed = prediction_log.load_completed_log()
    gws = {e["gw"] for e in completed}

    print(f"Prediction log: {len(completed)} logged predictions across {len(gws)} completed gameweek(s).")

    if len(gws) < prediction_log.MIN_GAMEWEEKS_FOR_RECALIBRATION:
        remaining = prediction_log.MIN_GAMEWEEKS_FOR_RECALIBRATION - len(gws)
        print(
            f"Need at least {prediction_log.MIN_GAMEWEEKS_FOR_RECALIBRATION} completed gameweeks before "
            f"recalibrating off real data - {remaining} more to go. This fills in naturally as the season "
            f"progresses and `fpl transfers` keeps running each week - nothing to do manually."
        )
        return

    def report(label: str, key: str, entries: list[dict]) -> float | None:
        if not entries:
            print(f"\n{label}: no logged entries yet.")
            return None
        predicted = np.array([e[key] for e in entries])
        actual = np.array([e["actual_points"] for e in entries])
        bias = float(np.mean(predicted - actual))
        corr = float(np.corrcoef(predicted, actual)[0, 1]) if len(predicted) > 1 else float("nan")
        print(f"\n{label} ({len({e['gw'] for e in entries})} gameweek(s), {len(entries)} predictions):")
        print(f"  Mean absolute error: {float(np.mean(np.abs(predicted - actual))):.2f} pts/player/gameweek")
        print(f"  Bias: {bias:+.2f} ({'over' if bias > 0 else 'under'}-projecting on average)")
        print(f"  Correlation (predicted vs. actual): {corr:.3f}")
        return bias

    # Through GW6 only FPL's own ep_next was logged, so those numbers say
    # nothing about this model - SHRINKAGE_K advice must come from model_points.
    report("FPL's own ep_next (reference only)", "predicted_points", completed)
    model_entries = [e for e in completed if "model_points" in e]
    bias = report("THIS MODEL's own projection", "model_points", model_entries)
    if bias is None or len({e["gw"] for e in model_entries}) < prediction_log.MIN_GAMEWEEKS_FOR_RECALIBRATION:
        print("\nNot enough gameweeks of this model's own projections yet to advise on SHRINKAGE_K.")
        return

    print(f"\nCurrent SHRINKAGE_K = {model.SHRINKAGE_K}")
    if abs(bias) > 1.0:
        direction = "higher (more shrinkage)" if bias > 0 else "lower (less shrinkage)"
        print(
            f"Still meaningfully {'over' if bias > 0 else 'under'}-projecting on real data - consider trying a "
            f"{direction} SHRINKAGE_K. This is a suggestion for manual review: edit fpl/model.py's SHRINKAGE_K "
            f"directly if you act on it, then re-run this script after a few more gameweeks to check the effect - "
            f"don't chase a single week's noise."
        )
    else:
        print("Bias is small on real data so far - no strong signal to change SHRINKAGE_K right now.")


if __name__ == "__main__":
    main()
