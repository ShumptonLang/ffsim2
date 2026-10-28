# ffsim

A Monte Carlo engine for Sleeper fantasy football leagues. It ranks a manager's
decisions (start/sit, waivers, FAAB, trades) by how much each one changes their
**championship probability**, not by projected points. See [docs/PLAN.md](docs/PLAN.md)
for the full design.

**Status:** milestone 1 of 6 is done: the Sleeper client and cache, league-specific
scoring (including superflex and IDP), and the league model. The simulator itself
comes in later milestones.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
ffsim league -l <LEAGUE_ID> -m <manager>          # format, playoff structure, standings, schedule
ffsim roster -l <LEAGUE_ID> -m <manager> -w 9     # roster with Sleeper projections scored by your league's rules
```

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
