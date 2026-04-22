from fetcher import get_finalists, get_athlete_times
from simulation import build_model, run
from output import print_models, print_table, show_chart, save_csv, save_json
from config import N_SIMULATIONS


def main() -> None:
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
    results = run(models, n=N_SIMULATIONS)

    print_table(results)
    save_csv(results)
    save_json(results)
    show_chart(results)


if __name__ == "__main__":
    main()
