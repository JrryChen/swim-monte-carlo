#!/usr/bin/env python3
"""
Quick scorer — evaluate current config.py hyperparameters against Paris 2024
actual results and the crowdsourced pick-em survey, without running any
hyperparameter optimisation.

Usage:
    python validate.py               # score all cached events
    python validate.py --event men_100_free   # score one event in detail
    python validate.py --verbose     # show per-event breakdown

Requires athlete cache to be populated first:
    python tune_hyperparams.py --cache-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from tune_hyperparams import (
    load_actual_results,
    load_crowd_top4_probs,
    get_or_cache_athletes,
    brier_score,
    names_match,
    SLUG_TO_XLSX,
)
from events import EVENTS_2024_PARIS
from simulation import build_model, run_fast
import config


def score_event_detail(slug: str, n_sims: int = 5_000) -> None:
    """Print a detailed breakdown for a single event."""
    actual_results = load_actual_results()
    if slug not in actual_results:
        print(f"No actual results for '{slug}' in actual_results.csv")
        return
    if slug not in EVENTS_2024_PARIS:
        print(f"Unknown event slug: {slug}")
        return

    event = EVENTS_2024_PARIS[slug]
    cache_path = ROOT / "validation" / "athlete_cache" / f"{slug}.json"
    if not cache_path.exists():
        print(f"No cached data for {slug}. Run:  python tune_hyperparams.py --cache-only")
        return

    athletes, _ = get_or_cache_athletes(slug, event)
    models = [build_model(a, event) for a in athletes]
    results = run_fast(models, n=n_sims)

    actual_top4 = actual_results[slug]
    crowd_probs = load_crowd_top4_probs(event_slugs=[slug]).get(slug, {})

    print(f"\n{'─'*70}")
    print(f"  {event.name}  ({n_sims:,} simulations)")
    print(f"{'─'*70}")
    print(f"  {'Swimmer':<30} {'Sim P(top4)':>11} {'Crowd P(top4)':>13} {'Actual':>8}")
    print(f"  {'─'*28} {'─'*11} {'─'*13} {'─'*8}")

    for res in sorted(results, key=lambda r: -sum(r.place_probs.get(p, 0) for p in range(1, 5))):
        p_top4 = sum(res.place_probs.get(p, 0) for p in range(1, 5))
        cp = next((prob for name, prob in crowd_probs.items() if names_match(name, res.name)), 0.0)
        actual = "✓ top-4" if any(names_match(res.name, a) for a in actual_top4) else ""
        print(f"  {res.name:<30} {p_top4:>10.1%} {cp:>12.1%} {actual:>8}")

    sim_b, crowd_b = brier_score(results, actual_top4, crowd_probs or None)
    print(f"\n  Sim Brier: {sim_b:.4f}   Crowd Brier: {crowd_b:.4f if crowd_b else 'N/A'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score current config vs crowd baseline.")
    parser.add_argument("--event",   default=None, help="Score a single event in detail")
    parser.add_argument("--verbose", action="store_true", help="Show per-event Brier scores")
    parser.add_argument("--n-sims",  type=int, default=5_000, help="Simulations per event (default: 5000)")
    args = parser.parse_args()

    if args.event:
        score_event_detail(args.event, n_sims=args.n_sims)
        return

    # ── Full validation run ────────────────────────────────────────────────────
    actual_results = load_actual_results()
    crowd_probs = load_crowd_top4_probs(event_slugs=list(actual_results.keys()))
    has_crowd = bool(crowd_probs)

    print(f"\nValidating {len(actual_results)} events with {args.n_sims:,} sims each...\n")
    print(f"  Current hyperparameters:")
    print(f"    SEASON_DECAY    = {config.SEASON_DECAY}")
    print(f"    MAX_SEASONS     = {config.MAX_SEASONS}")
    print(f"    BEST_TIME_DECAY = {config.BEST_TIME_DECAY}")
    print(f"    DEFAULT_SIGMA   = {config.DEFAULT_SIGMA}")
    print(f"    DEFAULT_TAU     = {config.DEFAULT_TAU}\n")

    sim_briers, crowd_briers = [], []
    rng = np.random.default_rng(42)

    rows = []
    for slug, top4 in actual_results.items():
        if slug not in EVENTS_2024_PARIS:
            continue
        cache_path = ROOT / "validation" / "athlete_cache" / f"{slug}.json"
        if not cache_path.exists():
            if args.verbose:
                print(f"  SKIP  {slug}  (not cached — run --cache-only first)")
            continue

        event = EVENTS_2024_PARIS[slug]
        athletes, _ = get_or_cache_athletes(slug, event)
        try:
            models = [build_model(a, event) for a in athletes]
        except Exception as e:
            if args.verbose:
                print(f"  ERROR {slug}  {e}")
            continue

        results = run_fast(models, n=args.n_sims, rng=rng)
        cp = crowd_probs.get(slug) if has_crowd else None
        s_b, c_b = brier_score(results, top4, cp)

        sim_briers.append(s_b)
        if c_b is not None:
            crowd_briers.append(c_b)
        rows.append((slug, s_b, c_b))

    if args.verbose and rows:
        print(f"  {'Event':<30} {'Sim Brier':>10} {'Crowd Brier':>12}")
        print(f"  {'─'*28} {'─'*10} {'─'*12}")
        for slug, s_b, c_b in sorted(rows, key=lambda x: x[1], reverse=True):
            crowd_str = f"{c_b:.4f}" if c_b is not None else "  N/A"
            print(f"  {slug:<30} {s_b:>10.4f} {crowd_str:>12}")
        print()

    print(f"{'='*50}")
    if sim_briers:
        mean_sim = np.mean(sim_briers)
        print(f"  Sim Brier score   (mean): {mean_sim:.4f}")
    if crowd_briers:
        mean_crowd = np.mean(crowd_briers)
        print(f"  Crowd Brier score (mean): {mean_crowd:.4f}")
        gap = mean_crowd - mean_sim
        verdict = f"{'BETTER ↑' if gap > 0 else 'WORSE  ↓'} than crowd by {abs(gap):.4f}"
        print(f"  Simulator vs crowd      : {verdict}")
    else:
        print(f"  (No crowd baseline available)")

    n_cached = len(rows)
    n_total = len(actual_results)
    if n_cached < n_total:
        print(f"\n  ⚠  {n_total - n_cached} events skipped (not cached).")
        print(f"     Run: python tune_hyperparams.py --cache-only")
    print()


if __name__ == "__main__":
    main()
