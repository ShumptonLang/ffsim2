# ffsim2: Sleeper championship-probability Monte Carlo engine (plan)

## Context
The repo `ShumptonLang/ffsim2` is empty, so this starts from nothing. The goal is a tool that takes a Sleeper league plus one manager (the "focal" team) and ranks every available decision (start/sit, waiver adds/drops, FAAB bids, trades) by **how much it changes that manager's championship probability**. It does not rank by projected points. The four requirements behind the design:
1. **Probabilistic, right-skewed player distributions** with Bayesian shrinkage toward preseason priors.
2. **Correlated draws**: QB↔own pass catchers, both sides of a shootout, RB↔own DEF, and offense↔opposing DEF (negative).
3. **A full league-season simulation**: every remaining matchup, standings and tiebreakers, playoff seeding, byes, and bracket rounds, run thousands of times.
4. **The objective is Δ P(championship)**, computed with common random numbers so small differences between decisions can be measured.

Decisions made with the user: **Python**, **Sleeper's unofficial projections API** as the source of projection centers (behind a pluggable interface), **CLI + reports**, and v1 support for **redraft H2H, including Superflex and IDP**.

## Stack
Python 3.11+, `numpy`, `scipy` (stats, special), `pandas`, `httpx` (Sleeper API), `typer` + `rich` (CLI and tables), `pydantic` (settings and models), `pyarrow`/parquet (cache), `pytest`. Optional: `numba` for the bracket and lineup inner loops if profiling shows a need. Tooling: `ruff` and `mypy`. Packaging uses `pyproject.toml` with a `ffsim` console script.

## Package layout
```
src/ffsim/
  data/
    sleeper.py        # HTTP client + disk cache (league, rosters, users, matchups, players, transactions, brackets, state)
    projections.py    # ProjectionSource protocol; SleeperProjectionSource (api.sleeper.com/projections|stats)
    nfl_schedule.py   # NFL schedule, byes, Vegas spread/total (nflverse schedules release CSV)
    history.py        # pulls past weeks' projections + actuals for error-model fitting/backtests
  scoring.py          # stat-line -> fantasy points using league.scoring_settings (handles IDP, bonuses, negatives)
  league.py           # League model: roster slots (incl. SUPER_FLEX, IDP_FLEX, DL/LB/DB), playoff settings, schedule
  model/
    errors.py         # projection-error model: fit + quantile functions per position/proj bucket
    bayes.py          # preseason prior + in-season shrinkage -> per-player ROS talent posterior
    availability.py   # injury status / bye / depth -> P(plays) per week
    correlation.py    # factor-model Gaussian/t copula: game, team-offense, pass, script factors
  sim/
    draws.py          # (n_sims, n_weeks, n_players) correlated point tensor
    lineups.py        # projection-based lineup setter per team/week (greedy + flex/superflex solve)
    season.py         # vectorized remaining regular season -> W/L/PF per team per sim
    playoffs.py       # seeding, tiebreakers, byes, reseeding, 1- or 2-week rounds, bracket
    engine.py         # run(league_state, overrides) -> SimResult; common-random-number support
  decisions/
    lineup.py         # start/sit: candidate lineups ranked by ΔP(champ) (+ P(win week))
    waivers.py        # add/drop pairs ranked by ΔP; FAAB bid suggestion
    trades.py         # evaluate a given trade; trade finder w/ acceptability filter
    search.py         # successive-halving screen (few sims -> many sims on survivors)
  report.py           # rich tables / markdown export
  cli.py              # ffsim odds | lineup | waivers | trade | find-trades | calibrate | backtest
tests/ (fixtures = recorded Sleeper JSON)
```

## 1. Data layer
- **Sleeper v1 API** (documented): `/state/nfl`, `/league/{id}` (scoring_settings, roster_positions, settings.playoff_week_start, playoff_teams, playoff_round_type, playoff_seed_type, waiver_budget), `/rosters`, `/users`, `/matchups/{week}` (Sleeper pre-generates future `matchup_id`s, which gives the remaining schedule), `/transactions/{week}`, `/winners_bracket`, `/players/nfl` (about 5MB, cache once a day). The focal manager can be identified by username or roster_id.
- **Projections and stats** (unofficial): `api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular&position[]=…` and `/stats/nfl/{season}/{week}`. **Use raw stat lines and score them with the league's own `scoring_settings`** in `scoring.py`. This makes custom scoring, superflex and IDP (tackles, sacks, etc.) work automatically. Store the endpoint behind the `ProjectionSource` protocol so another source (for example a CSV) can be added later. Integration tests run against recorded fixtures, so changes to the endpoint show up quickly.
- **NFL context**: the nflverse `schedules` release CSV (game_id, teams, week, `spread_line`, `total_line`) supplies bye weeks, game pairing (needed for the correlation factors) and implied team totals.
- **Cache**: `~/.cache/ffsim/`, parquet or JSON keyed by endpoint and date, with a `--refresh` flag.

## 2. Player outcome distributions (`model/errors.py`)
- **Fit from history.** Pull 2–4 past seasons of (Sleeper weekly projection, actual) pairs scored with the league's settings. Model the outcome as a **zero/low-floor mass plus a skewed continuous component**:
  - With p₀ = P(dud/early exit | played), the rest is a **lognormal on the ratio actual/projection**, fitted per position × projection bucket (e.g. QB, RB1-tier vs RB-flex-tier). Parameters (μ, σ) are smoothed across buckets: σ shrinks as projection grows.
  - K and DST, which can be zero or negative, use a **shifted gamma**. They can fall back to an **empirical bootstrap** of residual ratios. The method is selectable in config (`lognormal|gamma|bootstrap`) so the methods can be compared during calibration.
- Each player-week exposes `quantile(u) -> points` (an inverse CDF). This is what the copula feeds. A test checks that the mean of the distribution matches the projection center after the skew is applied.

## 3. Bayesian projection updating (`model/bayes.py`)
- **Prior**: the preseason per-game projection (Sleeper week-0/season projections), with prior strength k_pos in "games-equivalent". k_pos is position-specific (stability differs) and fitted in calibration by maximizing out-of-sample log-likelihood.
- **Posterior ROS talent**: a normal-normal update on the per-game fantasy rate (or on the log rate). The posterior mean is `(k·prior + n·obs_mean)/(k+n)`, with posterior variance. Early-season results are therefore shrunk, not taken at face value.
- **Weekly center**:
  - For weeks where Sleeper has a weekly projection, blend it with the posterior. Sleeper already reacts to news, so its weight is configurable.
  - For further-out weeks, use `posterior rate × matchup/implied-team-total factor`.
- **Talent uncertainty persists across weeks.** In each simulation, draw one season-level talent multiplier per player from the posterior and apply it to every remaining week, then add weekly noise on top. Without this, team strength would be independent from week to week and the spread of title odds would come out too narrow.
- **Availability** (`availability.py`): injury_status maps to P(plays) (e.g. Q≈0.8, D≈0.3, O/IR=0, with a return-week prior for IR), and bye weeks give 0. A player who doesn't play scores 0 unless the manager swaps him; see lineups.

## 4. Correlation (`model/correlation.py`)
- **Factor-model Gaussian copula.** For each NFL game g and team t in week w:
  - `G_g`: game-environment/shootout factor, shared by both teams and all players in the game.
  - `T_t`: team offensive success.
  - `P_t`: pass-vs-run split, so QB/WR/TE load positively and RB negatively.
  - `S_t`: game script (leading), so own RB and own DEF load positively and opposing QB/WR load positively (catch-up throws). Opposing DEF loads negatively on `T_opp`.
  - Player latent: `z = Σ λ_role,f · F_f + sqrt(1−Σλ²)·ε`, then `u = Φ(z)`, then `points = quantile(u)`. Optionally use a **t-copula** (a shared chi-square scaling per game) to fatten joint tails.
- **Loadings λ by role** (QB, WR1/2/3, TE, RB1/RB2, K, DST, IDP-by-position) are **fitted from historical residual correlations** (the same history pull as above) by least squares between the implied and observed role-pair correlations. Sensible defaults ship in config: for example QB–WR1 ≈ 0.35–0.45, QB–oppWR ≈ 0.15–0.2, RB–own DST ≈ 0.1–0.15, QB–opp DST ≈ −0.3.
- A factor model avoids a dense 400×400 Cholesky each week, is O(players) to sample, and is guaranteed positive-definite.
- Roles come from the depth chart and projection rank within the team (Sleeper `players` gives depth_chart_order and team).

## 5. League season simulation (`sim/`)
- **Draw tensor**: points with shape `(n_sims, n_weeks_remaining, n_players)` for all rostered players plus the top-N free agents per position, which are needed for waiver evaluation. 10k sims × 12 weeks × 600 players ≈ 72M float32, about 290MB, so generate it in week chunks.
- **Lineups**: managers set lineups from *projections*, not outcomes. Each team-week gets a mean-maximizing lineup. The solver handles FLEX, SUPER_FLEX and IDP_FLEX eligibility and fills the most restrictive slots first with an exact small assignment. Byes are skipped and players who are out are replaced. Inactive draws that were not known beforehand score 0, which is realistic. Lineups do not depend on the simulation except through known availability, so they are computed once per week.
- **Season**: score each team-week, apply the pre-generated matchups, and accumulate W/L/T and PF, starting from the current real standings.
- **Playoffs** (`playoffs.py`):
  - Sleeper settings: playoff_teams (4/6/8), byes for top seeds, `playoff_seed_type` (reseed or fixed), `playoff_round_type` (one week, or two weeks for the championship or for every round), and tiebreakers (record, then PF, as Sleeper does). Divisions are supported if present.
  - When the league is already in the playoffs, initialize from `/winners_bracket`.
  - Fully vectorized over simulations.
- **Output (`SimResult`)**: per team, P(playoffs), P(bye), the seed distribution, P(finals) and **P(champion)**; per focal week, P(win). Standard errors are reported for each.

## 6. Decision engine (objective = Δ P(champ))
- **Common random numbers**: the baseline and every alternative reuse the *same* uniform draws, talent draws and availability draws (one cached seed per player-week). Only roster or lineup assignments change, so `ΔP` is a **paired difference** with a small standard error, which is what makes 0.3% edges detectable. Every recommendation reports ΔP ± SE.
- **Start/sit** (`decisions/lineup.py`): enumerate the top-K lineups by mean (plus swaps of high-variance players), then rank by ΔP(champ). Also show P(win this week), the mean and the 90th-percentile ceiling. This exposes cases where an underdog should pick a higher-variance or stacked lineup.
- **Waivers/FAAB** (`decisions/waivers.py`): evaluate every (FA add, roster drop) pair, after pruning to plausible drops, by ΔP. The FAAB suggestion scales the bid with ΔP relative to the remaining budget and the remaining weeks. It also estimates competition from the other teams' ΔP for the same player, since that player helps them too. This is stated in reports as a heuristic.
- **Trades** (`decisions/trades.py`):
  - `evaluate`: apply the swap to *both* rosters. This matters because a trade can strengthen a rival or a playoff opponent. Report my ΔP(champ) and the counterparty's ΔP, ΔP(playoffs) and ΔROS points.
  - `find-trades`: search 1-for-1, 2-for-1 and 2-for-2 deals against every other roster. Filter for **acceptability**: the counterparty gains on ROS-projected points or on its own P(champ), since real managers judge by perceived value. Rank by my ΔP.
  - **Search** (`search.py`): screen with successive halving (500 sims, then 2k on the top 20%, then 10k on the top 10). This is where results like "lower expected points, higher title odds" appear: playoff-week upside, bye coverage and positional scarcity for superflex.

## 7. Calibration and backtesting (`ffsim calibrate`, `ffsim backtest`)
- **Distributions**: PIT histogram and 10/50/90 coverage by position on held-out seasons. Pick lognormal, gamma or bootstrap per position based on log score.
- **Bayes**: fit k_pos by out-of-sample likelihood of rest-of-season actuals from week-n data.
- **Correlation**: compare the simulated against the observed role-pair correlation tables.
- **League**: replay past Sleeper seasons of the user's league (`previous_league_id` chain) from week n, and compute the Brier and log loss of P(playoffs) and P(champ) against the actual outcomes.

## 8. CLI
```
ffsim odds        --league ID --manager NAME [--sims 10000]   # title/playoff/seed table for all teams
ffsim lineup      --league ID --manager NAME --week W
ffsim waivers     --league ID --manager NAME [--faab]
ffsim trade       --league ID --manager NAME --give "A,B" --get "C" --with TEAM
ffsim find-trades --league ID --manager NAME [--max-size 2]
ffsim plan        --league ID --manager NAME   # weekly "path to title": best lineup + top waiver + top trades by ΔP
ffsim calibrate / backtest --seasons 2023,2024,2025
```
Config lives in `ffsim.toml`: sims, seed, distribution family, prior weights, injury probabilities and correlation defaults.

## 9. Dynamic planning (`ffsim plan --dynamic`)
Added after the six milestones. The static decision engine scores every option as **one roster per team, fixed for the rest of the season**. Lineups are re-chosen weekly but rosters never change, which causes three problems:
- The "do nothing" baseline never fills holes. A kicker on bye scores zero, so "drop an RB, add a kicker now" wins by default.
- Timing can't be expressed. "Add the kicker the week of the bye" isn't a scenario.
- Streaming can't be expressed, because it's a policy, not a player. The best all-season DEF always wins.

`ffsim.dynamic` instead simulates the season as a sequence of decisions, with no position-specific rules.

**Paths** (`dynamic/paths.py`). Outcomes come from the static engine's `draw_batch`, so a path where nobody moves reproduces the static engine exactly (tested). Each path also carries what managers can observe:
- **Latent values.** The latent value implied by each score. A score in the flat dud stretch only reveals the stretch.
- **Absences.** The absence state after each week: healthy, a season-ending injury (announced), or the games missed so far. The drawn length of an absence stays hidden.
- **Projection news.** For week w, an increment with variance s²(k) − s²(k−1) is revealed when the lead drops below k. The week's points are drawn on the fully revealed projection, so what managers learn is consistent with outcomes.

**Projection news model** (`model/reveal.py`, part of `ModelSet`, refit by `ffsim fit`). The variance of log-projection news still to come, by position and lead k. History only stores game-week projections, so it is measured from the drift between game-week projections k weeks apart (a variogram), after removing the part explained by performance in between:
- Weeks Sleeper hasn't projected yet are carried forward by the simulator, so s²_unpub(k) = V(k).
- Published weeks are adjusted for the change of opponent first, and week t's own one-off component is removed: s²(k) = V(k) − V(0)/2.

Fitted on 2023–2025 at default scoring, DEF news is about ±26% one week out when a week isn't projected yet and ±12% when it is. QB and K news is about ±5–8%.

**Beliefs** (`dynamic/beliefs.py`). These use the engine's own generative model, z = √ρ·η + √(1−ρ)·ε. After n observed games: E[z] = ρ·n·z̄/(1+(n−1)ρ) and Var[z] = 1−ρ + ρ/(1+nρ/(1−ρ)). `ErrorModel.expected_points` turns that into expected points through precomputed tables. Before any simulated game, beliefs equal the starting projection exactly.

**Policy** (`dynamic/policy.py`, `dynamic/value.py`). Every manager in every path, every week:
- **Roster value.** The manager values a roster as Σ_w L[team, w] · E[lineup points in week w]. The expectation runs over their beliefs and over imagined injury scenarios drawn from the availability model, conditioned on known injuries. That is what gives depth, handcuffs and bye cover their value.
- **Leverage.** L is the title-odds leverage per point, estimated from the simulation by finite differences with common random numbers (`dynamic/leverage.py`). Each team's week profile is shrunk toward the league's.
- **Search.** Screen free agents by points over the weakest eligible starter, screen drops by leave-one-out loss, then fully evaluate every (add, drop) pair. Take the best, and repeat up to `max_moves`.
- **Timing.** A move that doesn't raise this week's value is deferred, since the next decision re-checks it. A deferred move is claimed now only if a rival is claiming the same player this week.
- **FAAB bids.** Bids are price[team] × value gained. Each team's price is fitted during policy iteration so that it spends its budget over the season.

**Claims** (`dynamic/claims.py`). FAAB goes to the highest bid, with ties broken by waiver priority. Rolling priority sends a successful claimant to the back of the line; reverse standings uses each path's own record. Dropped players sit on waivers until the next decision step.

**Ranking** (`dynamic/decide.py`):
1. **Leverage.** Fit leverage on seasons where nobody moves.
2. **Policy iteration.** Simulate with the policy, then refit leverage and FAAB prices. The first round, before any price exists, uses rolling priority.
3. **First moves.** First-step decisions are identical in every path, so they are made once and shared. A forced first move is therefore compared with the policy's own on common random numbers. Candidates are: no move, the policy's deferred move, its runner-up moves, the static plan's pickups and trades, and in FAAB leagues each claim at the exact bid that beats every rival's this week.
4. **Held-out scoring.** Screen candidates on one set of paths, then score the finalists on a fresh set, which avoids the winner's curse in the static engine's successive halving.
5. **Explanation.** Report what the policy goes on to do for you in the baseline paths.

**Speed.** Lineups are filled greedily, most restrictive slot first. This is exact for nested or disjoint slot sets with single-position players, tested against the Hungarian solver; the actual weekly lineups use the exact solver otherwise. Roster valuation uses an optional numba kernel (`pip install ffsim[fast]`).

**Simplifications.**
- The value function assumes no later moves; only the deferral rule and the top-level ranking see timing.
- Leverage doesn't depend on a path's standings.
- All managers follow the same policy (a symmetric approximation, not an equilibrium), with no in-path trades.
- Waivers are processed once per week.

## Milestones
1. Scaffolding, the Sleeper client with cache and fixtures, `scoring.py` (with IDP), and the league model with roster slots.
2. Independent distributions, lineups, the season and playoff sim, and `ffsim odds`. This gives a working end-to-end baseline.
3. Bayesian updating, the talent persistence draw, and availability.
4. The correlation factor copula and its history-based fitting.
5. Common random numbers and the decision modules: lineup, then waivers, then trades and the trade finder; plus `ffsim plan`.
6. Calibration and backtest commands, then performance tuning (numba if needed).

## Verification
- **Unit tests (pytest)**:
  - Scoring reproduces Sleeper's own `points` in the matchup fixtures for standard, superflex and IDP leagues.
  - Each quantile function is monotone and its mean matches the center.
  - Sampled factor-copula correlations recover the target role correlations within tolerance.
  - The lineup solver returns legal, optimal lineups on brute-forceable cases.
  - Standings tiebreakers and the bracket (byes, reseeding, two-week finals) are checked against hand-built scenarios.
  - Under common random numbers, a no-op change gives ΔP = 0 exactly, and the SE of a paired change is far below the unpaired SE.
- **Sanity sims**:
  - A league of identical teams gives P(champ) ≈ 1/N.
  - A clinched team gives P(playoffs) = 1.
  - When the focal team is a big underdog, adding correlation measurably raises the value of a stacked lineup.
- **End-to-end**: run `ffsim odds` and `ffsim plan` against a real public Sleeper league ID. Confirm that the playoff odds total to playoff_teams and that the championship odds total 1. Inspect the trade and waiver rankings.
- **Backtest**: `ffsim backtest` on prior seasons. The Brier score for P(playoffs) should beat a records-only baseline.
