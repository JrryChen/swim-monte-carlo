"""Headless run — skips the matplotlib GUI for CI/terminal verification."""
import argparse
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no window

from fetcher import get_finalists, get_athlete_times
from simulation import build_model, run
from output import print_models, print_table, print_odds, show_distributions, show_chart, save_csv, save_json
from events import EVENTS_2024_PARIS
from config import N_SIMULATIONS, DEFAULT_EVENT

parser = argparse.ArgumentParser()
parser.add_argument("--event", default=DEFAULT_EVENT, choices=EVENTS_2024_PARIS.keys())
args = parser.parse_args()
event = EVENTS_2024_PARIS[args.event]

print(f"Event: {event.name}")
print("Fetching finalists...")
athletes, event_date = get_finalists(event)
print(f"Event date: {event_date}")
print(f"Found {len(athletes)} finalists: {[a.name for a in athletes]}\n")

print(f"Fetching historical times (before {event_date})...")
for athlete in athletes:
    get_athlete_times(athlete, before_date=event_date, discipline_name=event.discipline_name)
    print(f"  {athlete.name}: {len(athlete.times)} results")

models = [build_model(a, event) for a in athletes]
print_models(models)

print(f"\nRunning {N_SIMULATIONS:,} simulations...")
results, winning_times = run(models, n=N_SIMULATIONS)

print_table(results)
print_odds(results, winning_times)
show_distributions(models, event_name=event.name)
show_chart(results, event_name=event.name)
save_csv(results)
save_json(results)
print("\nDone.")
