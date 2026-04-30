#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
"""
Hyperparameter tuner for the swim Monte Carlo simulator.

Optimizes config.py parameters against Paris 2024 Olympic actual results,
benchmarking against the crowdsourced pick-em survey as a human baseline.

WORKFLOW
--------
1. Pre-fetch and cache athlete data (one-time, requires internet, ~5 min):
       python tune_hyperparams.py --cache-only

2. Score current config vs crowd baseline (no tuning):
       python tune_hyperparams.py --score-current

3. Run Bayesian optimization:
       python tune_hyperparams.py --trials 200

4. Apply the best found parameters to config.py:
       python tune_hyperparams.py --apply-best

HYPERPARAMETERS TUNED
---------------------
  season_decay      — weight given to each prior season (config: SEASON_DECAY)
  max_seasons       — how many seasons of history to include (config: MAX_SEASONS)
  best_time_decay   — proximity-to-WR weighting steepness (config: BEST_TIME_DECAY)
  default_sigma     — fallback σ for athletes with 1 result (config: DEFAULT_SIGMA)
  default_tau       — fallback τ (right-tail skew) (config: DEFAULT_TAU)

SCORING
-------
  Brier score on top-4 probability across all events:
    For each swimmer in each event:
      score += (P(top-4) - actual_top4)²   where actual_top4 ∈ {0, 1}
  Lower Brier score = better calibration.
  The crowd pick-em survey serves as a human baseline to beat.
"""

import argparse
import json
import re
import sys
import unicodedata
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
VALIDATION_DIR = ROOT / "validation"
CACHE_DIR = VALIDATION_DIR / "athlete_cache"
XLSX_PATH = ROOT / "Swimming_PickEm_Frequency_Analysis.xlsx"

# Mapping from event slug → XLSX column prefix (Sheet1 header base name)
SLUG_TO_XLSX = {
    "men_50_free":      "Men Freestyle 50m",
    "men_100_free":     "Men Freestyle 100m",
    "men_200_free":     "Men Freestyle 200m",
    "men_400_free":     "Men Freestyle 400m",
    "men_800_free":     "Men Freestyle 800m",
    "men_1500_free":    "Men Freestyle 1500m",
    "men_100_back":     "Men Backstroke 100m",
    "men_200_back":     "Men Backstroke 200m",
    "men_100_breast":   "Men Breaststroke 100m",
    "men_200_breast":   "Men Breaststroke 200m",
    "men_100_fly":      "Men Butterfly 100m",
    "men_200_fly":      "Men Butterfly 200m",
    "men_200_im":       "Men Medley 200m",
    "men_400_im":       "Men Medley 400m",
    "women_50_free":    "Women Freestyle 50m",
    "women_100_free":   "Women Freestyle 100m",
    "women_200_free":   "Women Freestyle 200m",
    "women_400_free":   "Women Freestyle 400m",
    "women_800_free":   "Women Freestyle 800m",
    "women_1500_free":  "Women Freestyle 1500m",
    "women_100_back":   "Women Backstroke 100m",
    "women_200_back":   "Women Backstroke 200m",
    "women_100_breast": "Women Breaststroke 100m",
    "women_200_breast": "Women Breaststroke 200m",
    "women_100_fly":    "Women Butterfly 100m",
    "women_200_fly":    "Women Butterfly 200m",
    "women_200_im":     "Women Medley 200m",
    "women_400_im":     "Women Medley 400m",
}


# ─── Name normalisation ────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_str.lower().split())


def names_match(api_name: str, csv_name: str) -> bool:
    """Return True if two swimmer names refer to the same athlete.

    Handles case differences, diacritics, and common variants such as
    "O'CALLAGHAN" vs "OCALLAGHAN". Falls back to last-name + first-initial
    matching (never last name alone, to avoid false positives on Smith, etc.).
    """
    a = _normalize(api_name)
    b = _normalize(csv_name)
    # Remove apostrophes/hyphens for comparison
    a_clean = re.sub(r"['\-]", "", a)
    b_clean = re.sub(r"['\-]", "", b)
    if a_clean == b_clean:
        return True
    # Last name + first-name initial match (requires at least 2 tokens each)
    a_parts = a_clean.split()
    b_parts = b_clean.split()
    if len(a_parts) >= 2 and len(b_parts) >= 2:
        if a_parts[0] == b_parts[0] and a_parts[1][0] == b_parts[1][0]:
            return True
    return False


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_actual_results() -> dict[str, list[str]]:
    """Load Paris 2024 actual top-4 from validation/actual_results.csv.

    Returns {event_slug: [1st_name, 2nd_name, 3rd_name, 4th_name]}.
    Lines starting with '#' are ignored.
    """
    from events import EVENTS
    path = VALIDATION_DIR / "actual_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"actual_results.csv not found at {path}\n"
            "Create it or copy the template from validation/actual_results.csv"
        )
    results: dict[str, list[str]] = {}
    with open(path) as f:
        headers = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if headers is None:
                headers = [h.strip() for h in line.split(",")]
                continue
            parts = [p.strip() for p in line.split(",")]
            row = dict(zip(headers, parts))
            slug = row.get("event_slug", "")
            if slug in EVENTS:
                results[slug] = [row.get(f"place_{i}", "") for i in range(1, 5)]
    return results


def load_crowd_top4_probs(
    xlsx_path: Path = XLSX_PATH,
    event_slugs: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Parse the pick-em XLSX (Sheet1) and compute crowd top-4 probabilities.

    For each event, counts how many respondents picked each athlete in any of
    the four positions, then divides by total respondents.

    Returns {event_slug: {athlete_last_name_first: crowd_top4_prob}}.
    Athlete keys are 'LASTNAME Firstname' as extracted from the survey cells.
    """
    try:
        import openpyxl
    except ImportError:
        print("  openpyxl not installed — skipping crowd baseline.")
        return {}

    if not xlsx_path.exists():
        print(f"  XLSX not found at {xlsx_path} — skipping crowd baseline.")
        return {}

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    ws = wb["Sheet1"]

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {}

    headers = [str(h) if h is not None else "" for h in rows[0]]
    n_respondents = len(rows) - 1

    slugs = event_slugs or list(SLUG_TO_XLSX.keys())
    crowd_probs: dict[str, dict[str, float]] = {}

    for slug in slugs:
        base = SLUG_TO_XLSX.get(slug)
        if base is None:
            continue

        # Find column indices for each of the 4 places
        col_indices: list[int | None] = []
        for suffix in ["1st", "2nd", "3rd", "4th"]:
            col_name = f"{base}, {suffix}"
            try:
                col_indices.append(headers.index(col_name))
            except ValueError:
                col_indices.append(None)

        if all(c is None for c in col_indices):
            continue

        name_counts: dict[str, int] = {}
        for row in rows[1:]:
            picked_this_row: set[str] = set()
            for ci in col_indices:
                if ci is None:
                    continue
                cell = row[ci]
                if cell and isinstance(cell, str):
                    # Cell format: "LASTNAME Firstname, NAT, time"
                    athlete_name = cell.split(",")[0].strip()
                    picked_this_row.add(athlete_name)
            for name in picked_this_row:
                name_counts[name] = name_counts.get(name, 0) + 1

        crowd_probs[slug] = {
            name: count / n_respondents
            for name, count in name_counts.items()
        }

    wb.close()
    return crowd_probs


# ─── Athlete data caching ──────────────────────────────────────────────────────

def _athlete_to_dict(athlete) -> dict:
    return {
        "id": athlete.id,
        "name": athlete.name,
        "results": [
            {"competition": r.competition, "time_seconds": r.time_seconds, "date": r.date}
            for r in athlete.results
        ],
    }


def _athlete_from_dict(d: dict):
    from models import Athlete, SwimResult
    a = Athlete(id=d["id"], name=d["name"])
    a.results = [
        SwimResult(
            competition=r["competition"],
            time_seconds=r["time_seconds"],
            date=r["date"],
        )
        for r in d["results"]
    ]
    return a


def get_or_cache_athletes(event_slug: str, event, force: bool = False) -> tuple[list, str]:
    """Return (athletes, event_date), fetching from API or local JSON cache.

    Cache lives at validation/athlete_cache/{event_slug}.json.
    Pass force=True to re-fetch even if cache exists.
    """
    from fetcher import get_finalists, get_athlete_times

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{event_slug}.json"

    if cache_file.exists() and not force:
        data = json.loads(cache_file.read_text())
        athletes = [_athlete_from_dict(d) for d in data["athletes"]]
        return athletes, data["event_date"]

    athletes, event_date = get_finalists(event)
    for athlete in athletes:
        get_athlete_times(athlete, before_date=event_date, discipline_name=event.discipline_name)

    cache_file.write_text(
        json.dumps(
            {"event_date": event_date, "athletes": [_athlete_to_dict(a) for a in athletes]},
            indent=2,
        )
    )
    return athletes, event_date


# ─── Scoring ───────────────────────────────────────────────────────────────────

def brier_score(
    sim_results,
    actual_top4: list[str],
    crowd_probs: dict[str, float] | None = None,
) -> tuple[float, float | None]:
    """Compute Brier score for top-4 prediction across all swimmers in one event.

    sim_results   — list[SimResult] from run_fast()
    actual_top4   — list of 4 athlete names who actually finished in the top 4
    crowd_probs   — optional {athlete_name: crowd_top4_prob}

    Returns (sim_brier, crowd_brier). crowd_brier is None if crowd_probs is None.
    """
    sim_scores, crowd_scores = [], []

    for result in sim_results:
        p_top4 = sum(result.place_probs.get(p, 0.0) for p in range(1, 5))
        actual = 1.0 if any(names_match(result.name, a) for a in actual_top4) else 0.0
        sim_scores.append((p_top4 - actual) ** 2)

        if crowd_probs is not None:
            # Find best-matching crowd prob for this swimmer
            cp = 0.0
            for crowd_name, prob in crowd_probs.items():
                if names_match(crowd_name, result.name):
                    cp = prob
                    break
            crowd_scores.append((cp - actual) ** 2)

    sim_b = float(np.mean(sim_scores)) if sim_scores else float("nan")
    crowd_b = float(np.mean(crowd_scores)) if crowd_scores else None
    return sim_b, crowd_b


def score_all_events(
    validation_athletes: dict[str, list],
    actual_results: dict[str, list[str]],
    crowd_probs: dict[str, dict[str, float]] | None = None,
    n_sims: int = 2_000,
    hyperparams: dict | None = None,
) -> tuple[float, float | None]:
    """Run simulations for every event in validation_athletes and return aggregate Brier scores.

    hyperparams — optional dict of kwargs for build_model (season_decay, etc.)
    Returns (mean_sim_brier, mean_crowd_brier).
    """
    from events import EVENTS
    from simulation import build_model, run_fast

    hp = hyperparams or {}
    rng = np.random.default_rng(42)

    sim_briers, crowd_briers = [], []
    for slug, athletes in validation_athletes.items():
        if slug not in actual_results:
            continue
        event = EVENTS[slug]
        try:
            models = [build_model(a, event, **hp) for a in athletes]
        except Exception:
            continue

        results = run_fast(models, n=n_sims, rng=rng)
        cp = crowd_probs.get(slug) if crowd_probs else None
        s_b, c_b = brier_score(results, actual_results[slug], cp)

        if not np.isnan(s_b):
            sim_briers.append(s_b)
        if c_b is not None:
            crowd_briers.append(c_b)


    mean_sim = float(np.mean(sim_briers)) if sim_briers else float("nan")
    mean_crowd = float(np.mean(crowd_briers)) if crowd_briers else None
    return mean_sim, mean_crowd


def run_loo_score(
    validation_athletes: dict[str, list],
    actual_results: dict[str, list[str]],
    n_sims: int = 2_000,
    hyperparams: dict | None = None,
) -> tuple[float, list[tuple[str, float]]]:
    """Score the model event-by-event using current hyperparams.

    Each event is treated as held-out: the score reported for it is computed
    independently, not as part of the training objective. With global hyperparams
    this is equivalent to per-event Brier scoring — the value is in seeing which
    events the model handles well and which it doesn't, and getting an unbiased
    estimate of mean performance (since hyperparams were fixed before this call).

    Returns (mean_brier, [(slug, brier), ...]) sorted worst → best.
    """
    from events import EVENTS
    from simulation import build_model, run_fast

    hp = hyperparams or {}
    rng = np.random.default_rng(42)
    rows: list[tuple[str, float]] = []

    for slug, athletes in validation_athletes.items():
        if slug not in actual_results:
            continue
        event = EVENTS[slug]
        try:
            models = [build_model(a, event, **hp) for a in athletes]
        except Exception:
            continue
        results = run_fast(models, n=n_sims, rng=rng)
        s_b, _ = brier_score(results, actual_results[slug])
        if not np.isnan(s_b):
            rows.append((slug, s_b))

    rows.sort(key=lambda x: -x[1])
    mean = float(np.mean([s for _, s in rows])) if rows else float("nan")
    return mean, rows


def _loo_tune_objective(
    trial,
    validation_athletes: dict[str, list],
    actual_results: dict[str, list[str]],
    held_out: str,
    n_sims: int,
) -> float:
    """Optuna objective for one LOO fold: optimise on all events except held_out."""
    params = {
        "season_decay":       trial.suggest_float("season_decay",       0.05, 0.90),
        "max_seasons":        trial.suggest_int(  "max_seasons",        2,    6),
        "best_time_decay":    trial.suggest_float("best_time_decay",    0.3,  6.0),
        "decay_distance_exp": trial.suggest_float("decay_distance_exp", 0.0,  1.5),
        "sigma_distance_exp": trial.suggest_float("sigma_distance_exp", 0.0,  1.5),
        "default_sigma":      trial.suggest_float("default_sigma",      0.05, 1.5,  log=True),
        "default_tau":        trial.suggest_float("default_tau",        0.02, 0.60, log=True),
    }
    subset = {k: v for k, v in validation_athletes.items() if k != held_out}
    sim_brier, _ = score_all_events(subset, actual_results, hyperparams=params, n_sims=n_sims)
    return sim_brier


def run_loo_tuning(
    validation_athletes: dict[str, list],
    actual_results: dict[str, list[str]],
    n_trials: int = 100,
    n_sims: int = 2_000,
) -> tuple[float, list[tuple[str, float, dict]]]:
    """Leave-one-event-out hyperparameter tuning.

    For each of the N events:
      1. Run n_trials Optuna trials optimising Brier on the other N-1 events.
      2. Score the best params on the held-out event.

    Returns (mean_loo_brier, [(slug, held_out_brier, best_params), ...]).
    This gives a honest out-of-sample estimate of model performance.
    """
    import optuna
    from functools import partial
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows: list[tuple[str, float, dict]] = []
    slugs = list(validation_athletes.keys())
    n = len(slugs)

    for i, held_out in enumerate(slugs):
        print(f"  LOO fold {i+1}/{n}  held out: {held_out}")
        study = optuna.create_study(direction="minimize")
        obj = partial(
            _loo_tune_objective,
            validation_athletes=validation_athletes,
            actual_results=actual_results,
            held_out=held_out,
            n_sims=n_sims,
        )
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

        # Score held-out event with best params found on the other 27
        from events import EVENTS
        from simulation import build_model, run_fast
        best_hp = study.best_params
        event = EVENTS[held_out]
        athletes = validation_athletes[held_out]
        try:
            models = [build_model(a, event, **best_hp) for a in athletes]
            results = run_fast(models, n=n_sims)
            s_b, _ = brier_score(results, actual_results[held_out])
        except Exception as e:
            print(f"    error scoring {held_out}: {e}")
            s_b = float("nan")

        rows.append((held_out, s_b, best_hp))
        print(f"    held-out Brier: {s_b:.4f}")

    valid = [s for _, s, _ in rows if not np.isnan(s)]
    mean_loo = float(np.mean(valid)) if valid else float("nan")
    return mean_loo, rows


# ─── Optuna objective ──────────────────────────────────────────────────────────

def _objective(
    trial,
    validation_athletes: dict[str, list],
    actual_results: dict[str, list[str]],
    n_sims: int,
) -> float:
    params = {
        "season_decay":    trial.suggest_float("season_decay",    0.05, 0.90),
        "max_seasons":     trial.suggest_int(  "max_seasons",     2,    6),
        "best_time_decay": trial.suggest_float("best_time_decay", 0.3,  6.0),
        "decay_distance_exp": trial.suggest_float("decay_distance_exp", 0.0, 1.5),
        "sigma_distance_exp": trial.suggest_float("sigma_distance_exp", 0.0, 1.5),
        "default_sigma":   trial.suggest_float("default_sigma",   0.05, 1.5,  log=True),
        "default_tau":     trial.suggest_float("default_tau",     0.02, 0.60, log=True),
    }
    sim_brier, _ = score_all_events(
        validation_athletes, actual_results, hyperparams=params, n_sims=n_sims
    )
    return sim_brier


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _build_validation_athletes(
    actual_results: dict[str, list[str]], force_refresh: bool = False
) -> dict[str, list]:
    from events import EVENTS

    validation_athletes: dict[str, list] = {}
    for slug in actual_results:
        if slug not in EVENTS:
            continue
        event = EVENTS[slug]
        try:
            athletes, _ = get_or_cache_athletes(slug, event, force=force_refresh)
            validation_athletes[slug] = athletes
            print(f"  ✓ {slug:30s}  {len(athletes)} athletes")
        except Exception as e:
            print(f"  ✗ {slug:30s}  {e}")
    return validation_athletes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune swim Monte Carlo hyperparameters via Bayesian optimisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-only",    action="store_true", help="Pre-fetch & cache athlete data, then exit")
    parser.add_argument("--refresh-cache", action="store_true", help="Re-fetch athlete data even if cached")
    parser.add_argument("--score-current", action="store_true", help="Score current config vs crowd baseline, then exit")
    parser.add_argument("--trials",  type=int, default=200, help="Number of Optuna trials (default: 200)")
    parser.add_argument("--n-sims",  type=int, default=2_000, help="Simulations per event per trial (default: 2000)")
    parser.add_argument("--apply-best", action="store_true", help="Print recommended config.py edits for best params found")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel Optuna jobs (default: 1)")
    parser.add_argument("--loo-score",  action="store_true", help="Score current config event-by-event (LOO-style breakdown)")
    parser.add_argument("--loo-tune",   action="store_true", help="Full LOO tuning: tune on N-1 events, score on held-out (slow)")
    parser.add_argument("--loo-trials", type=int, default=100, help="Optuna trials per LOO fold (default: 100)")
    args = parser.parse_args()

    # ── Load ground truth ──────────────────────────────────────────────────────
    print("Loading actual results...")
    actual_results = load_actual_results()
    print(f"  {len(actual_results)} events loaded from actual_results.csv\n")

    # ── Build/load athlete cache ───────────────────────────────────────────────
    print("Loading athlete data (from cache or API)...")
    validation_athletes = _build_validation_athletes(
        actual_results, force_refresh=args.refresh_cache
    )
    print(f"\n  {len(validation_athletes)} events ready for simulation\n")

    if args.cache_only:
        print("Cache complete. Run again without --cache-only to tune.")
        return

    # ── LOO score (fast) ───────────────────────────────────────────────────────
    if args.loo_score:
        import config
        hp = {
            "season_decay": config.SEASON_DECAY, "max_seasons": config.MAX_SEASONS,
            "best_time_decay": config.BEST_TIME_DECAY, "decay_distance_exp": config.DECAY_DISTANCE_EXP,
            "sigma_distance_exp": config.SIGMA_DISTANCE_EXP, "default_sigma": config.DEFAULT_SIGMA,
            "default_tau": config.DEFAULT_TAU,
        }
        print("LOO scoring current config (each event treated as held-out)...")
        mean_loo, rows = run_loo_score(validation_athletes, actual_results, n_sims=args.n_sims, hyperparams=hp)
        print(f"  {'Event':<30} {'Brier':>8}")
        print(f"  {'─'*28} {'─'*8}")
        for slug, s_b in rows:
            print(f"  {slug:<30} {s_b:>8.4f}")
        print("\n  Mean LOO Brier:", f"{mean_loo:.4f}")
        return

    # ── LOO tune (slow) ────────────────────────────────────────────────────────
    if args.loo_tune:
        print(f"LOO tuning: {len(validation_athletes)} folds x {args.loo_trials} trials each")
        est = len(validation_athletes) * args.loo_trials * args.n_sims // 500_000 + 1
        print(f"  Estimated time: ~{est} min")
        mean_loo, rows = run_loo_tuning(
            validation_athletes, actual_results,
            n_trials=args.loo_trials, n_sims=args.n_sims,
        )
        print("\n" + "="*55)
        print(f"  LOO TUNING RESULTS")
        print(f"{'='*55}")
        print(f"  {'Event':<30} {'Held-out Brier':>14}")
        print(f"  {'─'*28} {'─'*14}")
        for slug, s_b, _ in sorted(rows, key=lambda x: -x[1]):
            print(f"  {slug:<30} {s_b:>14.4f}")
        print("\n  Mean LOO Brier (out-of-sample):", f"{mean_loo:.4f}")
        return

    # ── Load crowd baseline ────────────────────────────────────────────────────
    print("Parsing crowd pick-em baseline...")
    crowd_probs = load_crowd_top4_probs(event_slugs=list(validation_athletes.keys()))
    has_crowd = bool(crowd_probs)
    print(f"  {'Crowd data loaded' if has_crowd else 'Crowd data unavailable (no XLSX)'}\n")

    # ── Score current config ───────────────────────────────────────────────────
    print("Scoring current config.py hyperparameters...")
    current_sim_brier, current_crowd_brier = score_all_events(
        validation_athletes, actual_results,
        crowd_probs=crowd_probs if has_crowd else None,
        n_sims=args.n_sims,
    )
    print(f"  Current config Brier score : {current_sim_brier:.4f}")
    if current_crowd_brier is not None:
        print(f"  Crowd baseline Brier score : {current_crowd_brier:.4f}")
        gap = current_crowd_brier - current_sim_brier
        verdict = f"{'BETTER' if gap > 0 else 'WORSE'} than crowd by {abs(gap):.4f}"
        print(f"  Simulator vs crowd         : {verdict}")

    if args.score_current:
        return

    # ── Run Optuna optimisation ────────────────────────────────────────────────
    try:
        import optuna
    except ImportError:
        print("\noptuna not installed. Run:  pip install optuna --break-system-packages")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    import subprocess
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip().replace("/", "-")
    except Exception:
        branch = "main"
    db_path = VALIDATION_DIR / f"optuna-{branch}.db"
    study_name = f"swim-hyperparams-{branch}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    n_done = len(study.trials)
    n_remaining = max(0, args.trials - n_done)
    print(f"\nRunning Optuna ({n_remaining} new trials, {n_done} already in DB)...")
    print(f"  Sims per trial: {args.n_sims:,} × {len(validation_athletes)} events\n")

    objective = partial(
        _objective,
        validation_athletes=validation_athletes,
        actual_results=actual_results,
        n_sims=args.n_sims,
    )
    study.optimize(objective, n_trials=n_remaining, n_jobs=args.jobs, show_progress_bar=True)

    # ── Report results ─────────────────────────────────────────────────────────
    best = study.best_params
    best_brier = study.best_value

    print(f"\n{'='*60}")
    print(f"  BEST HYPERPARAMETERS  (Brier: {best_brier:.4f})")
    print(f"{'='*60}")
    for k, v in best.items():
        current_val = _get_current_config(k)
        flag = "  ← changed" if current_val is not None and abs(float(current_val) - float(v)) > 1e-4 else ""
        fmt = ".0f" if k == "max_seasons" else ".4f"
        print(f"  {k:<22} {v:{fmt}}{flag}")

    improvement = current_sim_brier - best_brier
    print(f"\n  Improvement over current config : {improvement:+.4f}")
    if current_crowd_brier is not None:
        crowd_gap = current_crowd_brier - best_brier
        print(f"  Best config vs crowd            : {'+' if crowd_gap > 0 else ''}{crowd_gap:.4f}")

    if args.apply_best:
        _print_config_patch(best)


def _get_current_config(param_name: str):
    """Return the current value of a config param, or None if not found."""
    import config
    mapping = {
        "season_decay":    "SEASON_DECAY",
        "max_seasons":     "MAX_SEASONS",
        "best_time_decay": "BEST_TIME_DECAY",
        "decay_distance_exp": "DECAY_DISTANCE_EXP",
        "sigma_distance_exp": "SIGMA_DISTANCE_EXP",
        "default_sigma":   "DEFAULT_SIGMA",
        "default_tau":     "DEFAULT_TAU",
    }
    attr = mapping.get(param_name)
    return getattr(config, attr, None) if attr else None


def _print_config_patch(best_params: dict) -> None:
    """Print the config.py lines to update with the best found parameters."""
    mapping = {
        "season_decay":    "SEASON_DECAY",
        "max_seasons":     "MAX_SEASONS",
        "best_time_decay": "BEST_TIME_DECAY",
        "decay_distance_exp": "DECAY_DISTANCE_EXP",
        "sigma_distance_exp": "SIGMA_DISTANCE_EXP",
        "default_sigma":   "DEFAULT_SIGMA",
        "default_tau":     "DEFAULT_TAU",
    }
    print("\n" + "=" * 60)
    print("  SUGGESTED config.py EDITS  (replace existing lines)")
    print("=" * 60)
    for k, v in best_params.items():
        cfg_name = mapping.get(k, k.upper())
        if k == "max_seasons":
            print(f"  {cfg_name} = {int(v)}")
        else:
            print(f"  {cfg_name} = {v:.4f}")


if __name__ == "__main__":
    main()
