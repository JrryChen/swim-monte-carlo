import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from models import RaceModel, SimResult

RESULTS_DIR = "results"


def print_models(models: list[RaceModel]) -> None:
    """Print a summary table of each swimmer's fitted distribution."""
    rows = [
        [m.name, f"{m.pb:.3f}s", f"{m.mu:.3f}s", f"{m.sigma:.3f}s", f"{m.season_drop:.2%}"]
        for m in models
    ]
    print("\n=== Swimmer Performance Models (pre-event LCM 50m free) ===")
    print(tabulate(
        rows,
        headers=["Swimmer", "PB", "Proj. Mean (μ)", "Std Dev (σ)", "Season Drop"],
        tablefmt="rounded_outline",
    ))


def _sorted_results(results: list[SimResult]) -> list[SimResult]:
    return sorted(results, key=lambda r: r.place_probs[1], reverse=True)


def print_table(results: list[SimResult]) -> None:
    """Print a probability table sorted by win probability."""
    sorted_results = _sorted_results(results)
    num_places = len(results)

    headers = ["Swimmer"] + [f"P({i})" for i in range(1, num_places + 1)]
    rows = []
    for r in sorted_results:
        row = [r.name] + [f"{r.place_probs[i]:.1%}" for i in range(1, num_places + 1)]
        rows.append(row)

    print("\n=== Monte Carlo Simulation Results ===")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


def show_chart(results: list[SimResult]) -> None:
    """Display a horizontal bar chart of win probabilities."""
    sorted_results = _sorted_results(results)
    names = [r.name for r in sorted_results]
    win_probs = [r.place_probs[1] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, [p * 100 for p in win_probs], color="steelblue")
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("Win Probability (%)")
    ax.set_title("2024 Paris Olympics — Men's 50m Freestyle Final\nWin Probability (Monte Carlo)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def save_csv(results: list[SimResult]) -> None:
    """Save full probability table to results/probabilities.csv."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    num_places = len(results)
    rows = []
    for r in _sorted_results(results):
        row = {"Swimmer": r.name}
        for i in range(1, num_places + 1):
            row[f"P(Place {i})"] = round(r.place_probs[i], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "probabilities.csv")
    df.to_csv(path, index=False)
    print(f"CSV saved to {path}")


def save_json(results: list[SimResult]) -> None:
    """Save full probability table to results/probabilities.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = []
    for r in _sorted_results(results):
        output.append({
            "swimmer": r.name,
            "place_probabilities": {str(k): round(v, 4) for k, v in r.place_probs.items()},
        })

    path = os.path.join(RESULTS_DIR, "probabilities.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved to {path}")
