# swim-monte-carlo

A Monte Carlo simulation that models finishing-position probabilities for competitive swimming finals using historical LCM (long course metre) results.

The project includes the 28 individual swimming events from the **2024 Paris Olympics** and can also load World Aquatics competition datasets such as `competition_2943` (Paris 2024) and `competition_4725` (2025 World Championships). Competition datasets can include additional individual 50m stroke events, so some competitions have 34 supported individual events.

Hyperparameters are tuned via Bayesian optimization (Optuna) against actual top-4 results and can be benchmarked against a crowdsourced pick-em baseline.

---

## Repository Structure

```
run.py                  # Main entry point — run the simulation
tune_hyperparams.py     # Bayesian hyperparameter tuning (Optuna)
validate.py             # Quick Brier score check against Paris 2024 results
audit_times.py          # Inspect raw athlete times for short-course contamination
fetch_actual_results.py # Fetch actual results and competition manifests
run_headless.py         # Headless simulation (saves charts, no GUI)
configs/
  default.json          # JSON hyperparameter preset example
src/
  config.py             # All tunable hyperparameters
  config_presets.py     # Load/write JSON config presets
  events.py             # Default event catalogue and event metadata
  fetcher.py            # Swimming results data client (local only, not committed)
  models.py             # Data classes (Athlete, RaceModel, SimResult)
  simulation.py         # Model fitting and Monte Carlo simulation
  output.py             # Printing, charting, CSV/JSON export
# Local-only (not committed to git)
src/fetcher.py          # Data client with API endpoints

validation/
  actual_results.csv    # Paris 2024 top-4 finishers (ground truth)
  competition_<id>/     # Competition-specific results, manifest, metadata, cache
    actual_results.csv
    events_manifest.csv
    competition_metadata.json
    athlete_cache/      # Cached athlete histories (JSON, one file per event)
  optuna-*.db           # Local Optuna study databases (ignored by git)
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

# Run a competition dataset from cache
python run.py --competition-id 4725 --event men_50_back --from-cache

# Use a saved JSON hyperparameter preset
python run.py --competition-id 4725 --event men_50_back --from-cache --config configs/default.json

# Quick CLI run without plots
python run.py --competition-id 4725 --event men_50_back --from-cache --n-sims 1000 --no-plots

# Headless mode — saves charts to results/ without opening windows
python run_headless.py --event men_200_breast
```

Available event slugs follow the pattern `{men|women}_{distance}_{stroke}`, e.g. `men_50_free`, `women_200_back`, `men_400_im`. Competition datasets may also include events such as `men_50_back`, `women_50_breast`, and `men_50_fly`. Run `python run.py --help` for all flags.

#### Running without the data fetcher

`src/fetcher.py` is not included in this repository. If you have cloned the repo and do not have access to the fetcher, you can still run the simulator using the pre-built athlete cache committed under `validation/competition_<id>/athlete_cache/`:

```bash
python run.py --competition-id 2943 --from-cache --event men_50_free
python run.py --competition-id 4725 --from-cache --event men_50_back
```

This loads athlete data directly from the cached JSON files instead of calling the API. The `--from-cache` flag is also useful for reproducibility because it guarantees the simulation runs against the exact same historical data used for tuning and validation.

Results are written by competition and event:

```text
results/competition_4725/men_50_back/
results/competition_2943/men_100_free/
results/default/men_50_free/
```

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

Hyperparameters in `src/config.py` or a JSON preset under `configs/` are tuned via Bayesian optimization using [Optuna](https://optuna.org/), minimising Brier score against actual top-4 results.

The standard competition workflow uses repeated holdout cross-validation: tune on most events, score on held-out events, and repeat across multiple folds. A final all-event Optuna fit is still useful after validation. For `competition_2943`, this usually means 28 events. For `competition_4725`, this means 34 individual events, including the extra 50m stroke finals.

### Workflow

```bash
# Build (or refresh) the athlete history cache for all events
python tune_hyperparams.py --competition-id 4725 --cache-only
python tune_hyperparams.py --competition-id 4725 --cache-only --refresh-cache

# Score the current config.py values or a JSON preset against actual results
python tune_hyperparams.py --competition-id 4725 --score-current
python tune_hyperparams.py --competition-id 4725 --score-current --config configs/default.json

# Quick validation without tuning
python validate.py

# Preferred out-of-sample validation: tune on train events, score 3 held-out events.
# Omitting --cv-trials chooses a compute-equivalent budget to 1000 all-event trials
# based on the number of cached competition events.
python tune_hyperparams.py --competition-id 4725 --cv-tune --cv-folds 20 --cv-test-size 3

# Stricter but slower out-of-sample validation: tune on N-1 events, score one held-out event
python tune_hyperparams.py --competition-id 4725 --loo-tune --loo-trials 100

# Final fit after validation: tune on all competition events
python tune_hyperparams.py --competition-id 4725 --trials 1000 --apply-best

# Save recommended/best hyperparameters to a JSON preset instead of editing src/config.py
python tune_hyperparams.py --competition-id 4725 --cv-tune --cv-folds 20 --cv-test-size 3 --save-config configs/competition_4725_cv.json
python tune_hyperparams.py --competition-id 4725 --trials 1000 --save-config configs/competition_4725_best.json
```

Optuna stores trials in `validation/optuna-{branch}-{dataset}.db` — each git branch and competition dataset gets its own database so trial history doesn't bleed across model variants.
Repeated holdout CV stores one Optuna study per fold in that same database, so interrupted CV runs can resume without losing completed fold trials.

### Tuned Parameters (Paris 2024)

| Parameter | Tuned Value |
|---|---|
| `SEASON_DECAY` | 0.5458 |
| `MAX_SEASONS` | 3 |
| `BEST_TIME_DECAY` | 1.5751 |
| `DECAY_DISTANCE_EXP` | 1.0613 |
| `SIGMA_DISTANCE_EXP` | 0.6010 |
| `DEFAULT_SIGMA` | 0.3486 |
| `DEFAULT_TAU` | 0.2596 |

| Model | Brier Score |
|---|---|
| Repeated 25/3 CV test | **0.1709** |
| Simulator (tuned) | **0.1634** |
| Crowd baseline (937 respondents) | 0.1885 |

The simulator beats the crowd pick-em by a margin of **0.0251 Brier**.

---

## Config Presets

You do not need to edit `src/config.py` for every tuned setup. JSON presets under `configs/` can override any tuned hyperparameter:

```json
{
  "hyperparams": {
    "season_decay": 0.5486,
    "max_seasons": 4,
    "best_time_decay": 0.3296,
    "decay_distance_exp": 1.3976,
    "sigma_distance_exp": 1.0711,
    "default_sigma": 0.2023,
    "default_tau": 0.0793
  }
}
```

Use a preset when running:

```bash
python run.py --competition-id 4725 --event men_50_back --from-cache --config configs/competition_4725_cv.json
```

Use a preset when scoring:

```bash
python tune_hyperparams.py --competition-id 4725 --score-current --config configs/competition_4725_cv.json
```

Write a preset from tuning:

```bash
python tune_hyperparams.py --competition-id 4725 --trials 1000 --save-config configs/competition_4725_best.json
```

`src/config.py` remains the default fallback. A preset only overrides the hyperparameters it contains.

---

## Auditing for Short-Course Contamination

Short-course (25m pool) times are typically 1.5–3% faster than LCM and will corrupt the model if included. Use `audit_times.py` to inspect raw times for any event and flag suspicious results.

```bash
# Show all swimmers' times with weights and suspicion flags
python audit_times.py --event men_100_back
python audit_times.py --competition-id 4725 --event men_50_back

# Filter to one swimmer
python audit_times.py --competition-id 4725 --event men_100_free --swimmer POPOVICI

# Only show swimmers with at least one flagged time
python audit_times.py --competition-id 4725 --event men_100_back --fast-only
```

If a competition looks suspicious, add a substring of its name to `EXCLUDED_COMPETITIONS` in `src/config.py`, delete the event's cache file (`validation/competition_<id>/athlete_cache/{event}.json`), and rebuild with `python tune_hyperparams.py --competition-id <id> --cache-only --refresh-cache`.

`--fast-only` needs a stored world record for the event because it flags times faster than the WR-based suspicion threshold. It still works for the default Olympic events. For extra competition events without a stored WR, run without `--fast-only` to inspect all cached times.

---

## Configuration (`src/config.py`)

| Parameter | Tuned Value | What it controls |
|---|---|---|
| `N_SIMULATIONS` | `100_000` | Number of races simulated. Higher = more stable probabilities, slower. |
| `DEFAULT_SIGMA` | `0.2023` | Fallback Gaussian spread (seconds per 50m) used when a swimmer has only one recorded time. Higher = more assumed uncertainty. In swimming terms: how much race-to-race variation to assign when you barely know the swimmer. |
| `DEFAULT_TAU` | `0.0793` | Fallback exponential tail (seconds per 50m) used when fewer than 3 results are available. Higher = fatter right tail, more blown races assumed. In swimming terms: how often to expect a badly off day from a data-scarce swimmer. |
| `SEASON_DECAY` | `0.5486` | Fraction of weight retained per older season. `0.5` = each season back is half as influential; lower values push the model toward recent form. In swimming terms: how much last year's times should matter compared to this year's. |
| `MAX_SEASONS` | `4` | How many seasons of history to include. Results older than this are dropped entirely. |
| `BEST_TIME_DECAY` | `0.3296` | Steepness of the proximity weighting curve toward each swimmer's PB. `weight = exp(-effective_decay x (time - PB))`. Higher = the model trusts a swimmer's peak over their average more aggressively. In swimming terms: how much a single fast swim should dominate the model versus the swimmer's typical results. |
| `DECAY_DISTANCE_EXP` | `1.3976` | Softens `BEST_TIME_DECAY` for longer events: `effective_decay = BEST_TIME_DECAY / (distance / 50) ^ exp`. At `0.0` all events use the same decay; at `1.0` a 200m gets half the decay of a 50m. In swimming terms: distance swimmers are more consistent - the gap between a good and bad 1500m is narrower than in the 50m, so peak-chasing matters less. |
| `SIGMA_DISTANCE_EXP` | `1.0711` | Scales fallback sigma and tau by event distance: `effective = default x (distance / 50) ^ exp`. At `0.0` all distances use the same fallback; at `1.0` scaling is linear with distance. In swimming terms: longer races have more room for variation in absolute seconds, so sparse-data uncertainty should grow with distance. |
| `EXCLUDED_COMPETITIONS` | (list) | Competitions excluded by name substring — used to filter short-course and non-standard meets. |
| `DEFAULT_EVENT` | `"men_50_free"` | Event run when no `--event` flag is passed. |

---

## The Model

### 1. Data collection & filtering

Historical LCM results are fetched for each finalist. Only times recorded **before the event date** are kept. Short-course and non-standard meets are excluded via `EXCLUDED_COMPETITIONS`.

### 2. Seasonal weighting

Each result is assigned a season weight of `SEASON_DECAY ^ seasons_ago`. The most recent season receives full weight (1.0); each older season is discounted. Results older than `MAX_SEASONS` seasons are dropped.

### 3. Proximity weighting

Combined weights `season_weight x proximity_weight` are applied when estimating spread (sigma), tail skew (tau), and an intermediate mean `mu_raw` used as the variance/moment anchor. The proximity term is `exp(-effective_decay x (time - PB))`, where `PB` is the swimmer's personal best within the model window and `effective_decay = BEST_TIME_DECAY / (distance / 50) ^ DECAY_DISTANCE_EXP`. Faster PB-adjacent times receive more weight, so sigma and tau reflect the swimmer's high-end variability rather than their average variability. The distance scaling ensures proximity weighting is proportionally equivalent across sprint and distance events. Note: the final projected mean mu is computed separately from season-only weights (see step 4), so proximity does not bias the race-time center.

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
- τ is estimated per swimmer from the weighted third central moment. Falls back to `DEFAULT_TAU` when fewer than 3 results are available. τ is capped at `0.9 × σ` to keep the normal component's variance positive in the sampler.

**Further reading:**
- [Wikipedia: Exponentially modified Gaussian distribution](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)
- Palmer et al. (2011). What are the shapes of response time distributions in visual search? *Journal of Experimental Psychology*, 37(1), 58–71. [DOI](https://doi.org/10.1037/a0020747)

---

## Validation — Paris 2024

The simulator is scored against all 28 individual Paris 2024 Olympic finals using Brier score (mean squared error between predicted top-4 probability and actual 0/1 outcome). Lower is better.

| Model | Brier Score |
|---|---|
| Repeated 25/3 CV test | **0.1709** |
| Simulator (tuned) | **0.1634** |
| Crowd pick-em (937 respondents) | 0.1885 |
| Improvement | +0.0252 |

The table above uses the final all-events fit, so it should be interpreted as an optimistic in-sample score. Repeated 25/3 holdout CV was added to estimate out-of-sample performance: each fold tunes hyperparameters on 25 events, scores 3 unseen events, and reports both fold-level Brier scores and a median fold-winner config recommendation.

---

## Targeting a Different Competition

For World Aquatics competition IDs, fetch the results and event manifest first:

```bash
python fetch_actual_results.py --competition-id 4725
```

This creates:

```text
validation/competition_4725/actual_results.csv
validation/competition_4725/events_manifest.csv
validation/competition_4725/competition_metadata.json
```

Then build the athlete cache:

```bash
python tune_hyperparams.py --competition-id 4725 --cache-only
```

Run, audit, score, and tune with the same `--competition-id`:

```bash
python run.py --competition-id 4725 --event men_50_back --from-cache
python audit_times.py --competition-id 4725 --event men_50_back
python tune_hyperparams.py --competition-id 4725 --score-current
```

The competition manifest supplies the event-specific World Aquatics discipline IDs. The default `src/events.py` catalogue remains useful for base Olympic event metadata and default runs without `--competition-id`.

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
| MCEVOY Cameron | 59.5% | 17.2% | 8.0% | 4.9% | 3.4% | 2.5% | 2.3% | 2.3% |
| PROUD Benjamin | 25.2% | 35.7% | 16.0% | 9.1% | 5.4% | 3.7% | 2.7% | 2.1% |
| DRESSEL Caeleb | 5.5% | 13.8% | 17.9% | 15.5% | 12.6% | 10.6% | 10.3% | 13.7% |
| LIENDO Josh | 4.7% | 14.4% | 20.3% | 17.4% | 13.2% | 10.2% | 9.1% | 10.7% |
| MANAUDOU Florent | 3.6% | 10.7% | 15.8% | 16.7% | 16.4% | 14.7% | 12.6% | 9.4% |
| CROOKS Jordan | 1.3% | 6.5% | 14.9% | 19.1% | 18.5% | 15.2% | 12.5% | 12.0% |
| GKOLOMEEV Kristian | 0.1% | 0.9% | 3.7% | 8.9% | 15.5% | 22.3% | 25.4% | 23.2% |
| DEPLANO Leonardo | 0.1% | 0.8% | 3.4% | 8.4% | 14.9% | 20.8% | 25.0% | 26.6% |

### Sportsbook Odds

| Swimmer | To Win | Top 3 |
|---|---|---|
| MCEVOY Cameron | -147 | -551 |
| PROUD Benjamin | +296 | -335 |
| DRESSEL Caeleb | +1711 | +168 |
| LIENDO Josh | +2036 | +154 |
| MANAUDOU Florent | +2667 | +232 |
| CROOKS Jordan | +7719 | +342 |
| DEPLANO Leonardo | +100910 | +2235 |
| GKOLOMEEV Kristian | +104067 | +2040 |

### Winning Time O/U Lines

Projected winning time: **21.156s** (median 21.175s)

| Line | Under | Over |
|---|---|---|
| 21.10s | +191 | -191 |
| 21.15s | +125 | -125 |
| 21.20s | -125 | +125 |
| 21.25s | -205 | +205 |
| 21.30s | -355 | +355 |

### Charts

![Swimmer Time Distributions](sample_results/distributions.png)

![Win Probabilities](sample_results/win_probabilities.png)
