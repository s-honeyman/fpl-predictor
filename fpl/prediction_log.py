"""Prediction-log + recalibration: the actual "get smarter every gameweek"
mechanism for this tool.

This is deliberately NOT MiroFish's Zep-based memory approach - that's built
for LLM-agent conversational/episodic memory (tracking entities and their
relationships across a social simulation), which is the wrong tool for a
numeric regression problem. The correct analog here is the standard one from
statistics/ML: a growing log of predicted-vs-actual results, and a
recalibration step that re-fits model constants against real accumulated
error once enough of it exists. Same spirit ("learn from what's happened"),
right tool for this specific job.

Log format: append-only JSON Lines (data/prediction_log.jsonl), one entry per
(gameweek, player). `record_predictions` writes the predicted side as soon as
a gameweek is projected; `record_actuals` fills in the real side once that
gameweek has actually been played.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "prediction_log.jsonl"

# Don't recalibrate off a handful of noisy gameweeks - wait for a real sample.
MIN_GAMEWEEKS_FOR_RECALIBRATION = 5


def _load_raw() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]


def _rewrite(entries: list[dict]) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def record_predictions(df: pd.DataFrame, gw: int) -> int:
    """Logs the predicted_points for the immediate next gameweek (`gw`) - call
    this each time the tool projects, right before that gameweek locks.
    Idempotent: re-running for the same gameweek doesn't duplicate or
    overwrite an already-logged prediction (the point is to capture what was
    predicted BEFORE the result was known, not to let hindsight sneak in).
    Returns how many new entries were written."""
    existing = _load_raw()
    already_logged = {(e["gw"], e["player_id"]) for e in existing}

    new_entries = []
    for _, row in df.iterrows():
        key = (gw, int(row["id"]))
        if key in already_logged:
            continue
        new_entries.append(
            {
                "gw": gw,
                "player_id": int(row["id"]),
                "web_name": row["web_name"],
                "predicted_points": float(row["predicted_points"]),
                "actual_points": None,
            }
        )
    if new_entries:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")
    return len(new_entries)


def record_actuals(gw: int, actuals_by_player_id: dict[int, float]) -> int:
    """Fills in actual_points for a gameweek's already-logged predictions.
    Returns how many entries were updated."""
    entries = _load_raw()
    updated = 0
    for e in entries:
        if e["gw"] == gw and e.get("actual_points") is None and e["player_id"] in actuals_by_player_id:
            e["actual_points"] = actuals_by_player_id[e["player_id"]]
            updated += 1
    if updated:
        _rewrite(entries)
    return updated


def gameweeks_needing_actuals(finished_gws: set[int]) -> set[int]:
    """Which logged, finished gameweeks are still missing actuals - what the
    CLI should fetch on its next run."""
    entries = _load_raw()
    pending = set()
    for e in entries:
        if e["gw"] in finished_gws and e.get("actual_points") is None:
            pending.add(e["gw"])
    return pending


def load_completed_log() -> list[dict]:
    """Only entries with both predicted and actual - what recalibration uses."""
    return [e for e in _load_raw() if e.get("actual_points") is not None]
