# fpl-predictor

Fantasy Premier League squad/transfer optimizer: projects player points from FPL's own public API, then uses an ILP solver ([PuLP](https://github.com/coin-or/pulp)) to pick the optimal 15-man squad, starting XI, and captain under the real budget/formation/max-3-per-club constraints.

Fully standalone - the FPL API is public, no API key needed.

## How it works

- `fpl/api.py` - fetches and caches `bootstrap-static` (every player's price, form, historical points) and `fixtures` (including FPL's own difficulty ratings) from the public FPL API.
- `fpl/model.py` - projects each player's points over a horizon (default 5 gameweeks): FPL's own `ep_next` for the immediate gameweek, and points-per-game scaled by fixture difficulty for the gameweeks after that.
- `fpl/optimizer.py` - PuLP ILP solver: picks the 15-man squad, valid starting XI, and captain that maximizes projected points under budget/formation/club-limit constraints.
- `fpl/squad_state.py` - persists your current squad locally so `transfers` mode can suggest swaps in later gameweeks without re-specifying the whole team.

## Usage

```bash
uv run fpl build                          # initial squad (e.g. for gameweek 1)
uv run fpl transfers --free-transfers 1   # each gameweek: best transfer(s), or hold
```

## Backtested against two real completed seasons - and the honest result

`scripts/backtest_season.py` reconstructs exactly what data would have been available at the start of a real season (no lookahead - prior season's final stats as the "last season" signal, that season's own fixture difficulty ratings), runs the actual shipped `model.py`/`optimizer.py` code unchanged, and scores the result against what really happened. Run it yourself:

```bash
uv run python scripts/backtest_season.py --prior 2024-25 --current 2025-26
uv run python scripts/backtest_season.py --prior 2023-24 --current 2024-25
```

**Result before the regression-to-mean fix (see next section):**

| | 2025-26 | 2024-25 |
|---|---|---|
| Static GW1-5: model's own projection vs. actual | 309.4 vs **218** | 311.4 vs **230** |
| Static full season, zero transfers (the honest floor) | 2,548.5 vs **1,337** | 2,586.7 vs **1,520** |
| Dynamic - re-optimized every 5 GWs on real results | **1,913** | **2,082** |

For context (from real 2025/26 data): the actual FPL champion won by a 38-point margin at a level clearly above the "top 1k" threshold of ~2,450+; top 10k typically needs ~2,300-2,450; even an *average* manager's season is generally cited around 2,000+. **Before the fix below, this tool's methodology landed around or slightly below an average finish - not competitive for winning, not even clearly above-average.**

## Fixed: regression-to-mean on points-per-game

**Diagnosis:** the optimizer's objective is exactly the thing most vulnerable to noise. It picks players by highest projected points - which means it disproportionately selects players whose *prior-season* points-per-game was inflated by variance (an unusually good run, a soft fixture list, a purple patch), because those are exactly the players a naive optimizer finds most attractive. There was no regression-to-the-mean adjustment anywhere in `model.py` - a player's full-season average was taken at face value and projected forward unchanged. Known failure mode in any system that optimizes against a noisy estimate (sometimes called the optimizer's "winner's curse").

**Fix:** `model._shrink_ppg` now shrinks each player's `points_per_game` toward the starts-weighted positional mean, empirical-Bayes style, weighted by how many starts that PPG figure is actually based on (`SHRINKAGE_K = 8.0` - at 8 starts, raw and positional-mean are weighted equally; below that, the mean dominates; well above it, the raw figure dominates). `ep_next` (FPL's own near-term prediction) is left untouched - the diagnosis specifically implicated the raw-PPG-driven longer-horizon component, not FPL's own tuned single-gameweek figure.

**Re-validated, same two seasons, same method:**

| | 2025-26 (before → after) | 2024-25 (before → after) |
|---|---|---|
| Static GW1-5 over-projection | 91.4 → **48.4** (-47%) | 81.4 → **37.6** (-54%) |
| Static GW1-5 actual points scored | 218 → **242** (+11%) | 230 → **248** (+8%) |
| Dynamic full-season total | 1,913 → **1,933** | 2,082 → **2,103** |

Honest read on this: the fix meaningfully improves day-one squad quality (less over-projection, and the resulting squad genuinely scores more, not just "predicts more honestly") - the winner's-curse diagnosis was correct. The full-season dynamic total moved less (+1% both seasons), because re-optimizing every 5 gameweeks on real results already self-corrects for a lot of the initial bias regardless of the starting squad. Still short of winning-competitive territory - this closes a real gap, it doesn't erase the fundamental one covered in the sections above.

## Not yet fixed (this is the honest state, not a promise)

- **No captaincy-specific strategy** - captain is currently just "highest-projected starter," with no consideration of differential vs. template captaincy, which is a known real skill lever among strong human FPL managers.
- **No chip timing** (Wildcard / Free Hit / Bench Boost / Triple Captain) - flagged as a gap since the very first version of this tool, still not built.
- **Dynamic backtest's transfer budget (5 per 5-gameweek window) is a reasonable but unvalidated proxy** for real free-transfer accrual and point-hit costs, not an exact rules implementation.
- **Backtest tested 2 seasons.** A genuinely rigorous validation would test more (data for several more seasons is available from the same free source - `vaastav/Fantasy-Premier-League` on GitHub) - two was enough to establish the over-projection is a real, repeatable pattern and not a one-season fluke, not enough to precisely quantify it.

## Breakout-player screen (`scripts/breakout_analysis.py`)

The Moneyball question, tested empirically rather than assumed: what actually distinguishes a cheap, low-expectation player who breaks out from one who stays irrelevant? Using 3 real season transitions (2022-23→2023-24, 2023-24→2024-25, 2024-25→2025-26):

- **"Breakout"** = started the season cheap (≤£5.5m) with genuinely low prior FPL output (≤50 points, including anyone new to the league), and finished in the top quartile of points for their position *among that same cheap/unproven pool* - not top quartile overall, which would just find already-known cheap-but-good players (early runs of this without the low-prior-output filter falsely flagged already-established players like Saliba and Declan Rice as "breakouts" simply because they were affordably priced defenders, not because nobody rated them - fixed by requiring low prior output too).
- **Finding**: hidden per-90 efficiency (xG/90, xA/90, ICT/90) is only a **weak** signal (Cohen's d 0.16-0.24) - the "Moneyball" story of spotting efficient-but-overlooked output is real but modest here. The **strong** signal (d=0.82 for prior minutes, d=0.79 for prior starts) is much less glamorous: **a cheap player who'd already earned a real run of minutes despite unspectacular returns is a far better bet than a truly fringe player**, regardless of per-90 rates - the market underprices "already trusted with game time," not hidden efficiency.
- The script then screens the current live player pool with a score weighted roughly by those effect sizes (minutes/starts dominant, per-90 stats a minor tiebreaker) and prints the top candidates.

Run it: `uv run python scripts/breakout_analysis.py`

Honest caveat: ~110 breakout examples per season by this definition is enough to see a real pattern, not enough to treat the exact weights as precisely calibrated - this is a scouting-report tool for transfer/differential ideas, not something folded into the core point-projection model (a different, shorter-horizon prediction task).

## Data sources

- Live squad/fixture data: the public FPL API (`fantasy.premierleague.com/api`), no key needed.
- Historical backtesting data: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) on GitHub - free, no auth, actively maintained season-by-season archives.
