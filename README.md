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
- **Dynamic backtest's transfer budget (5 per 5-gameweek window) is a reasonable but unvalidated proxy** for real free-transfer accrual and point-hit costs, not an exact rules implementation.
- **Backtest tested 2 seasons.** A genuinely rigorous validation would test more (data for several more seasons is available from the same free source - `vaastav/Fantasy-Premier-League` on GitHub) - two was enough to establish the over-projection is a real, repeatable pattern and not a one-season fluke, not enough to precisely quantify it.

## Chip timing (`scripts/chip_timing.py`)

Bench Boost and Triple Captain candidates derived from real fixture-difficulty data for your saved squad (reuses `model.py`'s own shrinkage/fixture-multiplier machinery directly, not `ep_next` - which only means anything for the true next real gameweek). Also checks the published fixture list for double/blank gameweeks. Wildcard timing is stated as general community convention, explicitly labeled as such - it's inherently reactive to how the season actually unfolds, which doesn't exist yet to base a specific gameweek on.

Results are quite flat this early pre-season (candidate gameweeks within ~0.2 pts of each other) - an honest reflection of genuine pre-season uncertainty in the fixture-difficulty ratings themselves, not a bug. Re-run periodically as the season progresses and FDR ratings/doubles get confirmed.

## GW1-5 walk-forward plan (`scripts/plan_gw1_5.py`)

Rolls the 5-gameweek projection horizon forward one week at a time for the first 5 gameweeks, checking whether a fixture-swing-driven transfer clears a real value bar (respecting 1 free transfer/week). **Explicitly fixture-only** - there's no live season data yet to react to; once GW1 actually happens, `fpl transfers` (which does react to live results) supersedes this.

## Investigated: is match-level Understat data worth using? (`scripts/validate_recent_form_signal.py`)

Real question raised: is `points_per_game` (a season-to-date average) actually the best signal available, or is there a sharper "recent form" signal we're leaving on the table? The free historical archive includes match-by-match Understat data (xG, xA, shots, key passes) per player - but **only through 2024-25** (no `understat/` folder exists yet for 2025-26 in the archive - this data source lags a season behind the live game, so it can inform research/backtesting but not the actual current squad).

Tested directly: does a player's recent-form xG+xA (last 5 matches) predict their next 5 gameweeks' points better than season-to-date PPG? On a 74-player real sample (2024-25 season, GW20 checkpoint):

| Signal | Correlation with next-5-GW points |
|---|---|
| Season-to-date points_per_game (what the model already uses) | **r = 0.751** |
| Recent-form xG+xA/90, last 5 matches (Understat) | r = 0.027 |

**Honest result: the signal we already use was clearly stronger, not the one we were considering paying for or engineering harder to get.** Caveat worth stating plainly: this isn't a fully fair comparison - season PPG averages ~19 games vs. recent form's 5, so some of the gap is just larger-sample-size noise reduction, not proof that "recency" itself doesn't matter. A cleaner follow-up would match window sizes (e.g. trailing-19-match xG vs. trailing-19-match points) before concluding Understat data isn't worth integrating at all. Not done here - flagged as the fair next test, not skipped silently.

## Considered: top-ranked managers' actual picks, and paid data sources

Two ideas raised and checked for real feasibility, not just discussed:

- **What do top-20 *managers'* (not just top-scoring players') picks look like historically?** Checked directly against FPL's live API: `leagues-classic/314/standings/` (the "Overall" global league) only reflects the *current* season's live standings - there's no free way to retroactively pull who ranked highly in a past season or what they picked. This is buildable **going forward**: once a season is live, the top-N entries and their gameweek-by-gameweek picks are freely fetchable in real time. Not retroactive.
- **Paid data APIs** - considered specifically (not just "get some football API"): player prop betting odds (e.g. [The Odds API](https://the-odds-api.com)) are the actual highest-value recommendation, not generic stats - reasoning by analogy to `pl-club-forecast`'s bookmaker-odds comparison, where market odds sat right at the practical prediction ceiling. Not pursued yet - the free Understat check above was prioritized first, and turned out to be the more useful thing to test before spending anything.

## Breakout-player screen (`scripts/breakout_analysis.py`)

The Moneyball question, tested empirically rather than assumed: what actually distinguishes a cheap, low-expectation player who breaks out from one who stays irrelevant? Using 3 real season transitions (2022-23→2023-24, 2023-24→2024-25, 2024-25→2025-26):

- **"Breakout"** = started the season cheap (≤£5.5m) with genuinely low prior FPL output (≤50 points, including anyone new to the league), and finished in the top quartile of points for their position *among that same cheap/unproven pool* - not top quartile overall, which would just find already-known cheap-but-good players (early runs of this without the low-prior-output filter falsely flagged already-established players like Saliba and Declan Rice as "breakouts" simply because they were affordably priced defenders, not because nobody rated them - fixed by requiring low prior output too).
- **Finding**: hidden per-90 efficiency (xG/90, xA/90, ICT/90) is only a **weak** signal (Cohen's d 0.16-0.24) - the "Moneyball" story of spotting efficient-but-overlooked output is real but modest here. The **strong** signal (d=0.82 for prior minutes, d=0.79 for prior starts) is much less glamorous: **a cheap player who'd already earned a real run of minutes despite unspectacular returns is a far better bet than a truly fringe player**, regardless of per-90 rates - the market underprices "already trusted with game time," not hidden efficiency.
- The script then screens the current live player pool with a score weighted roughly by those effect sizes (minutes/starts dominant, per-90 stats a minor tiebreaker) and prints the top candidates.

Run it: `uv run python scripts/breakout_analysis.py`

Honest caveat: ~110 breakout examples per season by this definition is enough to see a real pattern, not enough to treat the exact weights as precisely calibrated - this is a scouting-report tool for transfer/differential ideas, not something folded into the core point-projection model (a different, shorter-horizon prediction task).

## Data sources

- Live squad/fixture data: the public FPL API (`fantasy.premierleague.com/api`), no key needed.
- Historical backtesting data: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) on GitHub - free, no auth, actively maintained season-by-season archives. Includes per-player match-level Understat data (xG/xA/shots) through 2024-25 - see "Investigated" section above for what that is and isn't useful for.
