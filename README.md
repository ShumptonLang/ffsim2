# ffsim

A Monte Carlo engine for Sleeper fantasy football leagues. It ranks a manager's
decisions (start/sit, waivers, FAAB, trades) by how much each one changes their
**championship probability**, not by projected points. See [docs/PLAN.md](docs/PLAN.md)
for the full design.

**Status:** all six milestones in the plan are done.

1. Sleeper client and cache, league-specific scoring (including superflex and IDP),
   and the league model.
2. Skewed per-player outcome distributions fitted from 2023–2025 Sleeper projections
   vs. actual results, projection-based lineups, and a full simulation of the
   remaining regular season and playoff bracket (`ffsim odds`).
3. Bayesian updating (preseason prior + games played, blended with Sleeper's weekly
   projections), a per-player talent factor shared by all remaining weeks, this
   week's injury designations, new injuries later in the season, and bench
   substitutes for starters who are out.
4. Within-game correlation through a factor-model copula. Factors cover each
   game's environment, each team's offense and pass/run split, and teammates'
   usage shares; loadings by role are fitted to 2023–2025 residual correlations
   (QB–WR1 +0.27, RB1–RB2 −0.07, DEF vs. opposing QB −0.40).

5. Decisions ranked by change in championship probability. Every option is
   scored against the same simulated seasons (common random numbers), so each
   result is a paired difference with its standard error: start/sit, waiver
   pickups (with a FAAB bid heuristic in FAAB leagues), trade evaluation for both
   sides, a trade finder, and a one-page weekly plan.

6. Calibration and backtesting. `ffsim calibrate` checks every model on held-out
   seasons; `ffsim backtest` replays past NFL seasons and scores the predicted
   playoff and title odds against what happened.

**Dynamic planning** (`ffsim plan --dynamic`) runs alongside the static plan. The
static engine scores each option as one fixed roster for the rest of the season,
so the "do nothing" baseline never fills a hole: a kicker on bye scores zero, so
"drop your RB2 for a kicker now" looks good even when waiting until the bye week
is better, and a streaming strategy can't be expressed at all. The dynamic
engine plays every simulated season week by week. In each one, every manager
claims and drops players using only what they could know then: beliefs updated
from the simulated scores, projection news revealed as the games approach, and
announced injuries. Claims resolve under the league's FAAB, rolling or
reverse-standings waivers. Your options this week are ranked as "this move,
then the same policy" against "the policy from now". Waiting, streaming and a
shrinking free-agent pool come out of the simulation with no position-specific
rules. See [docs/PLAN.md](docs/PLAN.md#9-dynamic-planning-ffsim-plan---dynamic).

## How well it works

Held-out checks on 2023–2025 (each season scored by models fitted on the other two):

- **Outcome distributions:** for startable players, actual scores land below the
  model's 10th / 50th / 90th percentiles 8–11% / 47–51% / 89–91% of the time with
  the bootstrap family. It beats the parametric family on CRPS at every position,
  so `ffsim calibrate --apply` and the shipped defaults use it.
- **Within-game correlation:** observed vs. model on held-out seasons: QB–WR1
  +0.27 / +0.27, QB–TE1 +0.18 / +0.17, RB1–RB2 −0.07 / −0.05, DEF vs. opposing
  QB −0.41 / −0.39.
- **Bayesian blend:** for offense, Sleeper's weekly projections are hard to improve
  (held-out change −0.6% to +0.2% in next-week squared error). The blend matters
  for IDP, where Sleeper's projections are weak.
- **Season odds (backtest):** 45 mock leagues in a 10-team, 6-team-playoff PPR
  format, drafted from preseason projections and played out with real 2023–2025
  scores. P(playoffs) beats a records-only, coin-flip baseline at every cutoff
  week: **+6% Brier skill overall** (90% interval +3% to +9%, resampling whole
  leagues), and +8% at week 12. Calibration is good. P(champion) is about even with
  the baseline (+0.4%, interval −2% to +3%). Once you're in a 3-round, single-week
  bracket, the title is mostly luck; the model's edge is in who gets in and who
  gets a bye.

## Limitations

- In the static engine, rosters only change through the move being evaluated.
  Nobody picks up free agents to cover injuries, so late-season scoring is
  slightly understated. `plan --dynamic` removes this limit, at a large cost in
  run time.
- In the dynamic engine, every manager follows the same greedy policy. It makes
  one add/drop at a time against a value that assumes no later moves, defers a
  move until the week it helps unless a rival is claiming the player, and bids
  FAAB so each budget is spent over the season. There are no in-season trades
  between simulated managers. How much projection news is still to come comes
  from game-week projection history, because Sleeper doesn't publish older
  snapshots.
- Managers are assumed to set projection-optimal lineups.
- Projections come from Sleeper's undocumented app API.
- Two inputs are assumptions rather than fits: the chance a Questionable player
  plays (75%) and the static plan's FAAB bid rule. The dynamic plan fits its bids.
- Trade acceptability is a proxy (perceived value and projected lineup points),
  not a model of the other manager.
- The backtest uses mock leagues because no completed Sleeper league was
  available. `ffsim backtest --replay <league_id>` replays a real one.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[fast]"   # optional: numba, about 3x faster `plan --dynamic`
```

## Usage

```bash
ffsim league -l <LEAGUE_ID> -m <manager>          # format, playoff structure, standings, schedule
ffsim roster -l <LEAGUE_ID> -m <manager> -w 9     # roster with Sleeper projections scored by your league's rules
ffsim odds   -l <LEAGUE_ID> -m <manager>          # playoff / bye / title odds for every team
ffsim fit -l <LEAGUE_ID>                          # fit all models to your league's scoring

ffsim plan        -l <LEAGUE_ID> -m <manager>     # this week: odds, lineup, pickups, trade ideas
ffsim plan        -l <LEAGUE_ID> -m <manager> --dynamic [--fast] [--paths 3000] [--iterations 2]
                                                  # ...plus moves ranked with every manager acting all season
ffsim lineup      -l <LEAGUE_ID> -m <manager>     # start/sit options by Δ title odds
ffsim waivers     -l <LEAGUE_ID> -m <manager>     # add/drop pairs by Δ title odds (+ FAAB bids)
ffsim trade       -l <LEAGUE_ID> -m <manager> --give "James Cook" --get "Christian Watson"
ffsim find-trades -l <LEAGUE_ID> -m <manager> [--max-size 2] [--with <team>]
```

Model checks:

```bash
ffsim calibrate -l <LEAGUE_ID> [--apply]          # held-out checks; --apply saves the best family per position
ffsim backtest  -l <LEAGUE_ID> [--leagues 10] [--cutoffs 3,6,9,12] [--replay <old_league_id> ...]
```

Δ values are percentage points of probability. They are dimmed when within two
standard errors of zero. `find-trades` only proposes offers that don't cost the
other manager perceived value (rest-of-season points above replacement) or
projected starting-lineup points.

`plan --dynamic` prints the static plan first, unchanged, then a table with each
move's Δ title odds under both engines, how often the claim goes through, and the
FAAB bid (including the bid that beats every rival's for this week). It then shows
what the policy goes on to do for you, for example "adds this kicker in week 12
in 85% of seasons". A 12-team league takes about 40 minutes at the default
settings on a 4-core machine with numba, and a few minutes with `--fast`.

`ffsim odds` uses models fitted for your league's scoring once you have run
`ffsim fit`. Until then it uses the shipped defaults, which were fitted with
Sleeper's default PPR (+IDP) scoring.

`--manager` accepts a display name, team name, user ID or roster ID. API responses
are cached in `~/.cache/ffsim` (override with `FFSIM_CACHE_DIR` or `--cache-dir`);
pass `--refresh` to bypass the cache.

Projections come from `api.sleeper.com`, the undocumented endpoint behind the
Sleeper app. It may change without notice.

## Tests

```bash
pytest                                  # offline, against tests/fixtures
FFSIM_LIVE_LEAGUE=<id> pytest -m live   # checks scoring against Sleeper's own per-player points
```

The offline fixture (`tests/fixtures/sleeper_superflex_idp.json`) is a synthetic
4-team superflex/IDP league written to match Sleeper's documented response format.
It is not a recording of a real league.
