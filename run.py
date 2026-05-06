#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import json
from simulation import build_model, run
from output import print_models, print_table, print_odds, show_distributions, show_chart, save_csv, save_json
from events import EVENTS
from config import N_SIMULATIONS, DEFAULT_EVENT
from config_presets import load_preset

ROOT = Path(__file__).parent
VALIDATION_DIR = ROOT / "validation"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _competition_context(competition_id: int | None):
    if competition_id is None:
        return EVENTS, ROOT / "validation" / "athlete_cache", None

    from tune_hyperparams import load_competition_events

    events_map, cutoff_date = load_competition_events(
        competition_id,
        include_unmodeled=True,
    )
    cache_dir = VALIDATION_DIR / f"competition_{competition_id}" / "athlete_cache"
    return events_map, cache_dir, cutoff_date


def _load_from_cache(event_slug: str, cache_dir: Path, competition_id: int | None):
    """Load athlete data from the selected validation athlete cache."""
    from models import Athlete, SwimResult

    cache_path = cache_dir / f"{event_slug}.json"
    if not cache_path.exists():
        cache_command = (
            f"python tune_hyperparams.py --competition-id {competition_id} --cache-only"
            if competition_id is not None
            else "python tune_hyperparams.py --cache-only"
        )
        raise FileNotFoundError(
            f"No cache found for '{event_slug}' at {cache_path}\n"
            f"Run: {cache_command}"
        )
    data = json.loads(cache_path.read_text())
    athletes = []
    for d in data["athletes"]:
        a = Athlete(id=d["id"], name=d["name"])
        a.results = [
            SwimResult(competition=r["competition"], time_seconds=r["time_seconds"], date=r["date"])
            for r in d["results"]
        ]
        athletes.append(a)
    return athletes, data["event_date"]


def _results_dir(competition_id: int | None, event_slug: str) -> Path:
    competition_part = f"competition_{competition_id}" if competition_id is not None else "default"
    return ROOT / "results" / competition_part / event_slug


def main() -> None:
    parser = argparse.ArgumentParser(description="Swim race Monte Carlo simulator")
    parser.add_argument(
        "--event",
        default=DEFAULT_EVENT,
        help=f"Event slug (default: {DEFAULT_EVENT}).",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        default=None,
        help="Use validation/competition_<id>/ manifest and athlete cache.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Load athlete data from validation cache instead of fetching from the API. "
             "Useful if you do not have access to src/fetcher.py.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=N_SIMULATIONS,
        help=f"Number of simulations to run (default: {N_SIMULATIONS}).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib charts; still prints tables and saves CSV/JSON.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON config preset with hyperparameter overrides.",
    )
    args = parser.parse_args()

    events_map, cache_dir, cutoff_date = _competition_context(args.competition_id)
    event_slug = args.event
    if event_slug not in events_map:
        parser.error(f"Unknown event: {event_slug}. Available: {', '.join(events_map)}")

    event = events_map[event_slug]
    output_dir = _results_dir(args.competition_id, event_slug)
    hyperparams = load_preset(args.config)

    print(f"Event: {event.name}")
    if args.competition_id is not None:
        print(f"Competition: {args.competition_id}")
    if args.config:
        print(f"Config preset: {args.config}")
    print(f"Results directory: {output_dir.relative_to(ROOT)}")

    if args.from_cache:
        print(f"Loading athlete data from cache...")
        athletes, event_date = _load_from_cache(event_slug, cache_dir, args.competition_id)
        print(f"Event date: {event_date}")
        print(f"Found {len(athletes)} finalists: {[a.name for a in athletes]}\n")
    else:
        from fetcher import get_finalists, get_athlete_times
        print("Fetching finalists...")
        athletes, event_date = get_finalists(event)
        print(f"Event date: {event_date}")
        print(f"Found {len(athletes)} finalists: {[a.name for a in athletes]}\n")
        before_date = cutoff_date or event_date
        print(f"Fetching historical times (before {before_date})...")
        for athlete in athletes:
            get_athlete_times(athlete, before_date=before_date, discipline_name=event.discipline_name)
            print(f"  {athlete.name}: {len(athlete.results)} results")

    models = [build_model(a, event, **hyperparams) for a in athletes]
    print_models(models)

    print(f"\nRunning {args.n_sims:,} simulations...")
    results, winning_times = run(models, n=args.n_sims)

    print_table(results)
    print_odds(results, winning_times)
    save_csv(results, output_dir=str(output_dir))
    save_json(results, output_dir=str(output_dir))
    if not args.no_plots:
        show_distributions(models, event_name=event.name, output_dir=str(output_dir))
        show_chart(results, event_name=event.name, output_dir=str(output_dir))


if __name__ == "__main__":
    main()
