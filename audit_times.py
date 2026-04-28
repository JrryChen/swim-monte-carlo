#!/usr/bin/env python3
"""
Inspect raw athlete times for a given event to identify short-course or
otherwise suspicious results contaminating the model.

Usage:
    python audit_times.py --event men_100_free
    python audit_times.py --event men_100_free --swimmer CECCON
    python audit_times.py --event men_100_free --fast-only   # only show flagged times

For each swimmer, prints every result used to build their model, showing:
  - Competition name and date
  - Time in seconds and formatted MM:SS.hh
  - Season weight and proximity weight
  - ⚠ flag if the time looks suspiciously fast (possible short course)

Any competition that appears suspicious can be added to EXCLUDED_COMPETITIONS
in config.py to remove it from future model builds. Then delete the event's
cache file (validation/athlete_cache/{event}.json) and re-run --cache-only.
"""

import argparse
from pathlib import Path

import numpy as np
from tabulate import tabulate

ROOT = Path(__file__).parent


def fmt_time(seconds: float) -> str:
    """Format seconds as M:SS.hh or SS.hh."""
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m}:{s:05.2f}"
    return f"{seconds:.2f}"


def _get_season_year(date: str, season_start_month: int = 9) -> int:
    year, month = int(date[:4]), int(date[5:7])
    return year if month >= season_start_month else year - 1


def inspect_event(event_slug: str, swimmer_filter: str | None, fast_only: bool) -> None:
    from events import EVENTS_2024_PARIS
    from config import SEASON_DECAY, MAX_SEASONS, BEST_TIME_DECAY, DECAY_DISTANCE_EXP
    from simulation import build_model
    from tune_hyperparams import get_or_cache_athletes

    if event_slug not in EVENTS_2024_PARIS:
        print(f"Unknown event: {event_slug}")
        print(f"Available: {', '.join(EVENTS_2024_PARIS)}")
        return

    event = EVENTS_2024_PARIS[event_slug]
    cache_path = ROOT / "validation" / "athlete_cache" / f"{event_slug}.json"

    if not cache_path.exists():
        print(f"No cache for {event_slug}. Run:  python tune_hyperparams.py --cache-only")
        return

    athletes, event_date = get_or_cache_athletes(event_slug, event)

    if swimmer_filter:
        athletes = [a for a in athletes if swimmer_filter.upper() in a.name.upper()]
        if not athletes:
            print(f"No swimmers matching '{swimmer_filter}'")
            return

    print(f"\n{'═'*80}")
    print(f"  {event.name}  —  Finals date: {event_date}")
    print(f"  WR: {fmt_time(event.world_record)}  |  "
          f"SEASON_DECAY={SEASON_DECAY}  MAX_SEASONS={MAX_SEASONS}  "
          f"BEST_TIME_DECAY={BEST_TIME_DECAY}  DECAY_DISTANCE_EXP={DECAY_DISTANCE_EXP}")
    print(f"{'═'*80}")

    for athlete in athletes:
        dated = [r for r in athlete.results if r.date]
        if not dated:
            print(f"\n  {athlete.name}: no dated results")
            continue

        # Replicate build_model weighting logic
        effective_decay = BEST_TIME_DECAY / (event.distance / 50) ** DECAY_DISTANCE_EXP

        most_recent = max(_get_season_year(r.date) for r in dated)
        cutoff = most_recent - MAX_SEASONS
        in_window = [r for r in dated if _get_season_year(r.date) > cutoff]
        excluded_old = [r for r in dated if _get_season_year(r.date) <= cutoff]

        times = np.array([r.time_seconds for r in in_window])
        seasons = np.array([_get_season_year(r.date) for r in in_window])
        season_weights = np.array([SEASON_DECAY ** (most_recent - s) for s in seasons])
        proximity_weights = np.exp(-effective_decay * (times - event.world_record))
        total_weights = season_weights * proximity_weights
        if total_weights.sum() > 0:
            norm_weights = total_weights / total_weights.sum()
        else:
            norm_weights = np.ones(len(total_weights)) / len(total_weights)

        # Fit the model for display
        try:
            model = build_model(athlete, event)
            model_summary = (f"μ={model.mu:.3f}s  σ={model.sigma:.3f}s  "
                             f"τ={model.tau:.3f}s  PB={fmt_time(model.pb)}  "
                             f"drop={model.season_drop:.1%}")
        except Exception as e:
            model_summary = f"(model error: {e})"

        # Suspicious threshold: faster than WR * 1.005
        # Short course times are typically 1.5–3% faster than long course
        suspicion_threshold = event.world_record * 1.005

        rows = []
        any_flagged = False
        for i, r in enumerate(sorted(in_window, key=lambda x: x.time_seconds)):
            idx = in_window.index(r)
            s_wt = season_weights[idx]
            p_wt = proximity_weights[idx]
            n_wt = norm_weights[idx]
            season = _get_season_year(r.date)
            flagged = r.time_seconds < suspicion_threshold
            if flagged:
                any_flagged = True
            flag = "  ⚠ FAST" if flagged else ""
            rows.append([
                f"{flag}",
                fmt_time(r.time_seconds),
                f"{r.time_seconds:.2f}s",
                r.date,
                f"S{season}",
                f"{s_wt:.3f}",
                f"{p_wt:.3f}",
                f"{n_wt:.1%}",
                r.competition,
            ])

        if fast_only and not any_flagged:
            continue

        print(f"\n  ┌─ {athlete.name}")
        print(f"  │  {model_summary}")
        if excluded_old:
            print(f"  │  {len(excluded_old)} result(s) outside {MAX_SEASONS}-season window (not shown)")

        # Also show any results excluded by competition name filter
        all_raw = athlete.results  # these are already post-exclusion-filter from fetcher
        print(f"  │  {len(in_window)} result(s) in model window")

        headers = ["", "Time", "Seconds", "Date", "Season", "S.Wt", "P.Wt", "Norm.Wt", "Competition"]
        print(tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
            colalign=("left", "right", "right", "left", "left", "right", "right", "right", "left"),
        ))

        if any_flagged:
            print(f"  │  ⚠  Times below {fmt_time(suspicion_threshold)} "
                  f"({suspicion_threshold:.2f}s) may be short-course. "
                  f"Check competition and add to EXCLUDED_COMPETITIONS in config.py if so.")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raw athlete times to spot short-course contamination.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--event",    required=True, help="Event slug, e.g. men_100_free")
    parser.add_argument("--swimmer",  default=None,  help="Filter to one swimmer (partial name match)")
    parser.add_argument("--fast-only", action="store_true",
                        help="Only show swimmers with at least one flagged time")
    args = parser.parse_args()

    inspect_event(args.event, args.swimmer, args.fast_only)


if __name__ == "__main__":
    main()
