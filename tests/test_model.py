from fpl import model


def test_shrink_ppg_full_shrinkage_at_zero_starts():
    # No starts at all -> the raw PPG is pure noise/unknown, trust the positional
    # mean entirely.
    assert model._shrink_ppg(raw_ppg=8.0, starts=0, positional_mean_ppg=3.0) == 3.0


def test_shrink_ppg_half_weight_at_k_starts():
    # At starts == SHRINKAGE_K, raw and positional mean are weighted equally.
    result = model._shrink_ppg(raw_ppg=6.0, starts=int(model.SHRINKAGE_K), positional_mean_ppg=2.0)
    assert result == (2.0 + 6.0) / 2


def test_shrink_ppg_converges_to_raw_with_many_starts():
    # A near-full season of starts should trust the raw figure much more than
    # a handful of starts would - weight = starts / (starts + K), so it never
    # reaches 1.0, but it should clearly outweigh the positional mean by now.
    result = model._shrink_ppg(raw_ppg=8.0, starts=38, positional_mean_ppg=2.0)
    assert result > 6.5  # weight = 38/(38+8) ≈ 0.83 at the current SHRINKAGE_K


def test_shrink_ppg_never_moves_away_from_raw_toward_a_worse_estimate():
    # Shrinkage should always pull the raw value toward the mean, never past it.
    shrunk = model._shrink_ppg(raw_ppg=6.0, starts=4, positional_mean_ppg=2.0)
    assert 2.0 <= shrunk <= 6.0


def test_positional_mean_ppg_is_starts_weighted():
    bootstrap = {
        "elements": [
            {"element_type": 3, "points_per_game": 10.0, "starts": 30},  # heavily weighted
            {"element_type": 3, "points_per_game": 0.0, "starts": 0},  # a bench player, shouldn't drag the mean much
        ]
    }
    means = model._positional_mean_ppg(bootstrap)
    assert means[3] == 10.0  # the 0-start player contributes zero weight


def test_positional_mean_ppg_handles_position_with_no_starts_at_all():
    bootstrap = {"elements": [{"element_type": 1, "points_per_game": 5.0, "starts": 0}]}
    means = model._positional_mean_ppg(bootstrap)
    assert means[1] == 0.0  # no signal to go on - falls back to 0 rather than crashing
