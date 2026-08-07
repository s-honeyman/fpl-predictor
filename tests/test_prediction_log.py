import pandas as pd
import pytest

from fpl import prediction_log


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """Every test gets its own empty log file - never touch the real one."""
    monkeypatch.setattr(prediction_log, "LOG_PATH", tmp_path / "prediction_log.jsonl")
    yield


def _df(rows):
    return pd.DataFrame(rows)


def test_record_predictions_writes_new_entries():
    df = _df([{"id": 1, "web_name": "Haaland", "predicted_points": 8.0}, {"id": 2, "web_name": "Salah", "predicted_points": 7.0}])
    added = prediction_log.record_predictions(df, gw=1)
    assert added == 2
    entries = prediction_log._load_raw()
    assert len(entries) == 2
    assert entries[0]["actual_points"] is None


def test_record_predictions_is_idempotent_for_same_gameweek():
    df = _df([{"id": 1, "web_name": "Haaland", "predicted_points": 8.0}])
    prediction_log.record_predictions(df, gw=1)
    added_again = prediction_log.record_predictions(df, gw=1)
    assert added_again == 0
    assert len(prediction_log._load_raw()) == 1


def test_record_predictions_allows_new_gameweek_for_same_player():
    df = _df([{"id": 1, "web_name": "Haaland", "predicted_points": 8.0}])
    prediction_log.record_predictions(df, gw=1)
    added = prediction_log.record_predictions(df, gw=2)
    assert added == 1
    assert len(prediction_log._load_raw()) == 2


def test_record_actuals_fills_in_matching_entries_only():
    df = _df([{"id": 1, "web_name": "Haaland", "predicted_points": 8.0}, {"id": 2, "web_name": "Salah", "predicted_points": 7.0}])
    prediction_log.record_predictions(df, gw=1)

    updated = prediction_log.record_actuals(1, {1: 12.0})  # only player 1's actual known
    assert updated == 1

    entries = {e["player_id"]: e for e in prediction_log._load_raw()}
    assert entries[1]["actual_points"] == 12.0
    assert entries[2]["actual_points"] is None


def test_record_actuals_does_not_overwrite_already_filled_entries():
    df = _df([{"id": 1, "web_name": "Haaland", "predicted_points": 8.0}])
    prediction_log.record_predictions(df, gw=1)
    prediction_log.record_actuals(1, {1: 12.0})

    updated_again = prediction_log.record_actuals(1, {1: 99.0})  # shouldn't clobber real result
    assert updated_again == 0
    entries = prediction_log._load_raw()
    assert entries[0]["actual_points"] == 12.0


def test_gameweeks_needing_actuals_only_flags_finished_and_incomplete():
    df = _df([{"id": 1, "web_name": "Haaland", "predicted_points": 8.0}])
    prediction_log.record_predictions(df, gw=1)
    prediction_log.record_predictions(df, gw=2)
    prediction_log.record_actuals(1, {1: 12.0})  # GW1 now complete

    pending = prediction_log.gameweeks_needing_actuals(finished_gws={1, 2})
    assert pending == {2}  # GW1 already filled, GW3 not finished so not even asked about


def test_load_completed_log_excludes_pending_entries():
    df = _df([{"id": 1, "web_name": "Haaland", "predicted_points": 8.0}, {"id": 2, "web_name": "Salah", "predicted_points": 7.0}])
    prediction_log.record_predictions(df, gw=1)
    prediction_log.record_actuals(1, {1: 12.0})

    completed = prediction_log.load_completed_log()
    assert len(completed) == 1
    assert completed[0]["player_id"] == 1
