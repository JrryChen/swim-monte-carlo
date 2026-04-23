"""Headless run — skips the matplotlib GUI for CI/terminal verification."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no window

from fetcher import get_finalists, get_athlete_times
from simulation import build_model, run
from output import print_models, print_table, print_odds, save_csv, save_json
from config import N_SIMULATIONS

print("Fetching finalists...")
athletes, event_date = get_finalists()
print(f"Event date: {event_date}")
print(f"Found {len(athletes)} finalists: {[a.name for a in athletes]}\n")

print(f"Fetching historical times (before {event_date})...")
for athlete in athletes:
    get_athlete_times(athlete, before_date=event_date)
    print(f"  {athlete.name}: {len(athlete.times)} LCM 50m free results")

models = [build_model(a) for a in athletes]
print_models(models)

print(f"\nRunning {N_SIMULATIONS:,} simulations...")
results, winning_times = run(models, n=N_SIMULATIONS)

print_table(results)
print_odds(results, winning_times)
save_csv(results)
save_json(results)
print("\nDone.")
