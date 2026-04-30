#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
from simulation import build_model, run
from output import print_models, print_table, print_odds, show_distributions, show_chart, save_csv, save_json
from events import EVENTS
from config import N_SIMULATIONS, DEFAULT_EVENT

ROOT = Path(__file__).parent


def _load_from_cache(event_slug: str, event):
    """Load athlete data from validation/athlete_cache/{event_slug}.json."""
    import json
    from models import Athlete, SwimResult

    cache_path = ROOT / "validation" / "athlete_cache" / f"{event_slug}.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cache found for '{event_slug}' at {cache_path}\n"
            f"Run: python tune_hyperparams.py --cache-only"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Swim race Monte Carlo simulator")
    parser.add_argument(
        "--event",
        default=DEFAULT_EVENT,
        choices=EVENTS.keys(),
        help=f"Event slug (default: {DEFAULT_EVENT}). Available: {', '.join(EVENTS)}",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Load athlete data from validation/athlete_cache/ instead of fetching from the API. "
             "Useful if you do not have access to src/fetcher.py.",
    )
    args = parser.parse_args()
    event_slug = args.event
    event = EVENTS[event_slug]

    print(f"Event: {event.name}")

    if args.from_cache:
        print(f"Loading athlete data from cache...")
        athletes, event_date = _load_from_cache(event_slug, event)
        print(f"Event date: {event_date}")
        print(f"Found {len(athletes)} finalists: {[a.name for a in athletes]}\n")
    else:
        from fetcher import get_finalists, get_athlete_times
        print("Fetching finalists...")
        athletes, event_date = get_finalists(event)
        print(f"Event date: {event_date}")
        print(f"Found {len(athletes)} finalists: {[a.name for a in athletes]}\n")
        print(f"Fetching historical times (before {event_date})...")
        for athlete in athletes:
            get_athlete_times(athlete, before_date=event_date, discipline_name=event.discipline_name)
            print(f"  {athlete.name}: {len(athlete.results)} results")

    models = [build_model(a, event) for a in athletes]
    print_models(models)

    print(f"\nRunning {N_SIMULATIONS:,} simulations...")
    results, winning_times = run(models, n=N_SIMULATIONS)

    print_table(results)
    print_odds(results, winning_times)
    save_csv(results)
    save_json(results)
    show_distributions(models, event_name=event.name)
    show_chart(results, event_name=event.name)


if __name__ == "__main__":
    main()
