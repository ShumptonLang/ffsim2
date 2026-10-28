# ffsim

A Monte Carlo engine for Sleeper fantasy football leagues. It ranks a manager's
decisions (start/sit, waivers, FAAB, trades) by how much each one changes their
**championship probability**, not by projected points. See [docs/PLAN.md](docs/PLAN.md)
for the full design.

**Status:** milestones 1–3 of 6 are done.

1. Sleeper client and cache, league-specific scoring (including superflex and IDP),
   and the league model.
2. Skewed per-player outcome distributions fitted from 2023–2025 Sleeper projections
   vs. actual results, projection-based lineups, and a full simulation of the
   remaining regular season and playoff bracket (`ffsim odds`).
3. Bayesian updating (preseason prior + games played, blended with Sleeper's weekly
   projections), a per-player talent factor shared by all remaining weeks, this
   week's injury designations, new injuries later in the season, and bench
   substitutes for starters who are out.

Players on different NFL teams, and teammates, are still simulated independently
within a week. Correlation (QB–WR stacks, shootouts) arrives in milestone 4.

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
```

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
