# ffsim

A Monte Carlo engine for Sleeper fantasy football leagues. It ranks a manager's
decisions (start/sit, waivers, FAAB, trades) by how much each one changes their
**championship probability**, not by projected points. See [docs/PLAN.md](docs/PLAN.md)
for the full design.

**Status:** milestones 1–5 of 6 are done.

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

Next: calibration and backtesting against past seasons (milestone 6).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
ffsim league -l <LEAGUE_ID> -m <manager>          # format, playoff structure, standings, schedule
ffsim roster -l <LEAGUE_ID> -m <manager> -w 9     # roster with Sleeper projections scored by your league's rules
ffsim odds   -l <LEAGUE_ID> -m <manager>          # playoff / bye / title odds for every team
ffsim fit -l <LEAGUE_ID>                          # fit all models to your league's scoring

ffsim plan        -l <LEAGUE_ID> -m <manager>     # this week: odds, lineup, pickups, trade ideas
ffsim lineup      -l <LEAGUE_ID> -m <manager>     # start/sit options by Δ title odds
ffsim waivers     -l <LEAGUE_ID> -m <manager>     # add/drop pairs by Δ title odds (+ FAAB bids)
ffsim trade       -l <LEAGUE_ID> -m <manager> --give "James Cook" --get "Christian Watson"
ffsim find-trades -l <LEAGUE_ID> -m <manager> [--max-size 2] [--with <team>]
```

Δ values are percentage points of probability. They are dimmed when within two
standard errors of zero. `find-trades` only proposes offers that don't cost the
other manager perceived value (rest-of-season points above replacement) or
projected starting-lineup points.

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
