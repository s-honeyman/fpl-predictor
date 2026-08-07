"""Persists the user's current squad locally so `transfers` mode can run
in later gameweeks without re-specifying the whole team."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .optimizer import SquadResult

STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "my_squad.json"


def save_squad(result: SquadResult, bank_tenths: int = 0, saved_at: str | None = None) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "squad_ids": result.squad_ids,
        "starting_ids": result.starting_ids,
        "bench_ids": result.bench_ids,
        "captain_id": result.captain_id,
        "vice_captain_id": result.vice_captain_id,
        "total_cost": result.total_cost,
        "bank_tenths": bank_tenths,
        "saved_at": saved_at or datetime.now(timezone.utc).isoformat(),
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2))


def load_squad() -> dict | None:
    if not STATE_PATH.exists():
        return None
    return json.loads(STATE_PATH.read_text())
