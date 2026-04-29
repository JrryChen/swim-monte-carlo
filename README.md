# swim-monte-carlo

A Monte Carlo simulation that models finishing-position probabilities for competitive swimming finals using historical LCM (long course metre) results.

All 28 individual swimming events from the **2024 Paris Olympics** are supported. Hyperparameters are tuned via Bayesian optimization (Optuna) against actual Paris 2024 results, and benchmarked against a crowdsourced pick-em baseline of 1,037 respondents.

---

## Repository Structure

```
run.py                  # Main entry point — run the simulation
tune_hyperparams.py     # Bayesian hyperparameter tuning (Optuna)
validate.py             # Quick Brier score check against Paris 2024 results
audit_times.py          # Inspect raw athlete times for short-course contamination
fetch_actual_results.py # Fetch actual Paris 2024 results
run_headless.py         # Headless simulation (saves charts, no GUI)
src/
  config.py             # All tunable hyperparameters
  events.py             # Event catalogues with world records; set EVENTS = target Olympics
  fetcher.py            # Swimming results data client (local only, not committed)
  models.py             # Data classes (Athlete, RaceModel, SimResult)
  simulation.py         # Model fitting and Monte Carlo simulation
  output.py             # Printing, charting, CSV/JSON export
# Local-only (not committed to git)
fetch_actual_results.py # Fetches actual Paris 2024 results from results data source
src/fetcher.py          # Data client with API endpoints

validation/
  actual_results.csv    # Paris 2024 top-4 finishers (ground truth)
  athlete_cache/        # Cached athlete histories (JSON, one file per event)
  optuna-main.db        # Optuna study database (SQLite, branch-specific)
```

---

## Setup

### 1. Create the conda environment

```bash
conda create -n swim-monte-carlo python=3.11
conda activate swim-monte-carlo
pip install -r requirements.txt
```

### 2. Run the simulation

```bash
# Default event (Men's 50m Freestyle)
python run.py

# Specify any 2024 Paris Olympics event
python run.py --event men_200_breast
python run.py --event women_100_fly
python run.py --event men_400_im

# Headless mode — saves charts to results/ without opening windows
python run_headless.py --event men_200_breast
```

Available event slugs follow the pattern `{men|women}_{distance}_{stroke}`, e.g. `men_50_free`, `women_200_back`, `men_400_im`. Run `python run.py --help` to list all 28 options.

Results are written to `results/`:

| File | Contents |
|---|---|
| `probabilities.csv` | Finishing-position probabilities for each swimmer |
| `probabilities.json` | Same data in JSON format |
| `distributions.png` | Ex-Gaussian PDF chart per swimmer |
| `win_probabilities.png` | Win probability bar chart |

### 3. Run the tests

```bash
pytest tests/
```

---

## Hyperparameter Tuning

Hyperparameters in `src/config.py` are tuned via Bayesian optimization using [Optuna](https://optuna.org/), minimising Brier score against actual Paris 2024 top-4 results across all 28 events.

### Workflow

```bash
# Build (or refresh) the athlete history cache for all events
python tune_hyperparams.py --cache-only

# Score the current config.py values against actual results
python tune_hyperparams.py --score-current

# Run 200 Optuna trials (resumable — Ctrl+C and rerun to continue)
python tune_hyperparams.py --trials 200

# Run more trials on top of existing ones
python tune_hyperparams.py --trials 1000

# Print suggested config.py edits for the best parameters found
python tune_hyperparams.py --apply-best

# Quick validation without tuning
python validate.py
```

Optuna stores trials in `validation/optuna-{branch}.db` — each git branch gets its own database so trial history doesn't bleed across model variants.

### Tuned Parameters (Paris 2024)

| Parameter | Tuned Value |
|---|---|
| `SEASON_DECAY` | 0.4730 |
| `MAX_SEASONS` | 3 |
| `BEST_TIME_DECAY` | 3.4295 |
| `DECAY_DISTANCE_EXP` | 1.0585 |
| `SIGMA_DISTANCE_EXP` | 0.3104 |
| `DEFAULT_SIGMA` | 0.4004 |
| `DEFAULT_TAU` | 0.5210 |

| Model | Brier Score |
|---|---|
| Simulator (tuned) | **0.1626** |
| Crowd baseline (1,037 respondents) | 0.1885 |

The simulator beats the crowd pick-em by a margin of **0.026 Brier**.

---

## Auditing for Short-Course Contamination

Short-course (25m pool) times are typically 1.5–3% faster than LCM and will corrupt the model if included. Use `audit_times.py` to inspect raw times for any event and flag suspicious results.

```bash
# Show all swimmers' times with weights and suspicion flags
python audit_times.py --event men_100_back

# Filter to one swimmer
python audit_times.py --event men_100_back --swimmer CECCON

# Only show swimmers with at least one flagged time
python audit_times.py --event men_100_back --fast-only
```

If a competition looks suspicious, add a substring of its name to `EXCLUDED_COMPETITIONS` in `src/config.py`, delete the event's cache file (`validation/athlete_cache/{event}.json`), and rebuild with `python tune_hyperparams.py --cache-only`.

---

## Configuration (`src/config.py`)

| Parameter | Tuned Value | What it controls |
|---|---|---|
| `N_SIMULATIONS` | `10_000` | Number of races simulated. Higher = more stable probabilities, slower. |
| `DEFAULT_SIGMA` | `0.4004` | Fallback Gaussian spread (seconds per 50m) used when a swimmer has only one recorded time. Higher = more assumed uncertainty. In swimming terms: how much race-to-race variation to assign when you barely know the swimmer. |
| `DEFAULT_TAU` | `0.5210` | Fallback exponential tail (seconds per 50m) used when fewer than 3 results are available. Higher = fatter right tail, more blown races assumed. In swimming terms: how often to expect a badly off day from a data-scarce swimmer. |
| `SEASON_DECAY` | `0.4730` | Fraction of weight retained per older season. `0.5` = each season back is half as influential; lower values push the model toward recent form. In swimming terms: how much last year’s times should matter compared to this year’s. |
| `MAX_SEASONS` | `3` | How many seasons of history to include. Results older than this are dropped entirely. |
| `BEST_TIME_DECAY` | `3.4295` | Steepness of the proximity weighting curve toward the world record. `weight = exp(-effective_decay × (time − WR))`. Higher = the model trusts a swimmer’s peak over their average more aggressively. In swimming terms: how much a single fast swim should dominate the model versus the swimmer’s typical results. |
| `DECAY_DISTANCE_EXP` | `1.0585` | Softens `BEST_TIME_DECAY` for longer events: `effective_decay = BEST_TIME_DECAY / (distance / 50) ^ exp`. At `0.0` all events use the same decay; at `1.0` a 200m gets half the decay of a 50m. In swimming terms: distance swimmers are more consistent — the gap between a good and bad 1500m is narrower than in the 50m, so peak-chasing matters less. |
| `SIGMA_DISTANCE_EXP` | `0.3104` | Scales fallback σ and τ by event distance: `effective = default × (distance / 50) ^ exp`. At `0.0` all distances use the same fallback; at `1.0` scaling is linear with distance. In swimming terms: longer races have more room for variation in absolute seconds, so sparse-data uncertainty should grow with distance. |
| `EXCLUDED_COMPETITIONS` | (list) | Competitions excluded by name substring — used to filter short-course and non-standard meets. |
| `DEFAULT_EVENT` | `"men_50_free"` | Event run when no `--event` flag is passed. |

---

## The Model

### 1. Data collection & filtering

Historical LCM results are fetched for each finalist. Only times recorded **before the event date** are kept. Short-course and non-standard meets are excluded via `EXCLUDED_COMPETITIONS`.

### 2. Seasonal weighting

Each result is assigned a season weight of `SEASON_DECAY ^ seasons_ago`. The most recent season receives full weight (1.0); each older season is discounted. Results older than `MAX_SEASONS` seasons are dropped.

### 3. Proximity weighting

When estimating spread (σ), faster times receive more weight via `exp(-effective_decay × (time − WR))`, where `effective_decay = BEST_TIME_DECAY / (distance / 50) ^ DECAY_DISTANCE_EXP`. The distance scaling ensures proximity weighting is proportionally equivalent across sprint and distance events.

### 4. Season-drop (taper) adjustment

Swimmers peak at championship meets through tapering. For each season, the model computes a relative drop: `(season_avg − season_best) / season_avg`. These are averaged across seasons (weighted by recency) to estimate each swimmer's typical taper improvement.

The projected mean is computed from the **season-weighted average** (no proximity bias), then adjusted downward by the taper estimate:

```
mu_season = season-weighted average of times
mu = mu_season × (1 − season_drop)
```

This separates the taper signal from proximity weighting. A swimmer who consistently drops 2% at major meets projects meaningfully faster than one who swims to their season average.

### 5. Ex-Gaussian distribution

Each swimmer's race time is modelled as an **ex-Gaussian** random variable:

```
X = Normal(μ − τ, σ_n) + Exponential(τ)
```

- The **normal component** anchors peak performance near the projected mean.
- The **exponential component** (τ) generates the right-skewed tail for off-days.
- τ is estimated per swimmer from the weighted third central moment. Falls back to `DEFAULT_TAU` when fewer than 3 results are available.

**Further reading:**
- [Wikipedia: Exponentially modified Gaussian distribution](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
- Palmer et al. (2011). What are the shapes of response time distributions in visual search? *Journal of Experimental Psychology*, 37(1), 58–71. [DOI](https://doi.org/10.1037/a0020747)

---

## Validation — Paris 2024

The simulator was validated against all 28 individual Paris 2024 Olympic finals using Brier score (mean squared error between predicted top-4 probability and actual 0/1 outcome). Lower is better.

| Model | Brier Score |
|---|---|
| Simulator (tuned) | **0.1626** |
| Crowd pick-em (1,037 respondents) | 0.1885 |
| Improvement | +0.026 |

Hyperparameters were optimised over 1,000 Optuna trials.
---

## Targeting a Different Olympics

To run the simulator for a different Games, add a new event catalogue to  and update the  alias:

```python
# src/events.py
EVENTS_2028_LA = {
    "men_100_free": EventConfig(...),
    ...
}

EVENTS = EVENTS_2028_LA  # ← swap this line
```

All scripts (, , , ) pick up the new catalogue automatically.

---

## Sample Output — Men's 50m Freestyle

Run with
```bash
python run.py --event men_50_free
```

### Swimmer models

| Swimmer | PB | Proj. Mean (μ) | Std Dev (σ) | Tau (τ) | Season Drop |
|---|---|---|---|---|---|
| MCEVOY Cameron | 21.06s | 21.256s | 0.289s | 0.196s | 2.11% |
| PROUD Benjamin | 21.25s | 21.393s | 0.205s | 0.151s | 1.14% |
| MANAUDOU Florent | 21.54s | 21.616s | 0.200s | 0.101s | 1.36% |
| LIENDO Josh | 21.48s | 21.602s | 0.245s | 0.218s | 1.95% |
| GKOLOMEEV Kristian | 21.72s | 21.770s | 0.173s | 0.146s | 1.15% |
| DRESSEL Caeleb | 21.29s | 21.624s | 0.275s | 0.237s | 1.85% |
| DEPLANO Leonardo | 21.60s | 21.789s | 0.197s | 0.177s | 1.77% |
| CROOKS Jordan | 21.51s | 21.659s | 0.207s | 0.185s | 1.35% |

### Finishing-position probabilities

| Swimmer | P(1) | P(2) | P(3) | P(4) | P(5) | P(6) | P(7) | P(8) |
|---|---|---|---|---|---|---|---|---|
| MCEVOY Cameron | 59.4% | 17.0% | 8.0% | 4.9% | 3.4% | 2.7% | 2.2% | 2.4% |
| PROUD Benjamin | 25.2% | 36.0% | 16.2% | 9.0% | 5.5% | 3.6% | 2.7% | 2.0% |
| DRESSEL Caeleb | 5.4% | 14.0% | 17.4% | 15.4% | 12.8% | 10.9% | 10.4% | 13.7% |
| LIENDO Josh | 4.7% | 14.2% | 20.6% | 17.4% | 13.1% | 10.0% | 9.3% | 10.7% |
| MANAUDOU Florent | 3.8% | 10.5% | 15.9% | 17.0% | 16.1% | 14.7% | 12.8% | 9.2% |
| CROOKS Jordan | 1.3% | 6.7% | 14.8% | 19.2% | 18.6% | 15.0% | 12.4% | 12.1% |
| GKOLOMEEV Kristian | 0.1% | 0.8% | 3.6% | 8.8% | 15.6% | 22.1% | 25.6% | 23.3% |
| DEPLANO Leonardo | 0.1% 