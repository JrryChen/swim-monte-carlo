import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
from tabulate import tabulate
from models import RaceModel, SimResult

RESULTS_DIR = "results"


def print_models(models: list[RaceModel]) -> None:
    """Print a summary table of each swimmer's fitted distribution."""
    rows = [
        [m.name, f"{m.pb:.2f}s", f"{m.mu:.3f}s", f"{m.sigma:.3f}s", f"{m.tau:.3f}s", f"{m.season_drop:.2%}"]
        for m in models
    ]
    print("\n=== Swimmer Performance Models ===")
    print(tabulate(
        rows,
        headers=["Swimmer", "PB", "Proj. Mean (μ)", "Std Dev (σ)", "Tau (τ)", "Season Drop"],
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


def _american_odds(p: float) -> str:
    if p <= 0 or p >= 1:
        return "N/A"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    return f"+{round((1 - p) / p * 100)}"


def print_odds(results: list[SimResult], winning_times: np.ndarray) -> None:
    """Print sportsbook-style odds: win, top-3, and winning-time O/U lines."""
    sorted_results = _sorted_results(results)

    # --- Win / Top 3 table ---
    rows = []
    for r in sorted_results:
        win_p = r.place_probs[1]
        top3_p = sum(r.place_probs.get(i, 0) for i in range(1, 4))
        rows.append([r.name, _american_odds(win_p), _american_odds(top3_p)])

    print("\n=== Sportsbook Odds ===")
    print(tabulate(rows, headers=["Swimmer", "To Win", "Top 3"], tablefmt="rounded_outline"))

    # --- Winning time O/U lines ---
    median = float(np.median(winning_times))
    # Round to nearest 0.05 for clean lines
    base = round(round(median / 0.05) * 0.05, 2)
    lines = [base - 0.10, base - 0.05, base, base + 0.05, base + 0.10]

    ou_rows = []
    for line in lines:
        under_p = float(np.mean(winning_times < line))
        ou_rows.append([
            f"{line:.2f}s",
            _american_odds(under_p),
            _american_odds(1 - under_p),
        ])

    print("\n=== Winning Time O/U Lines ===")
    print(tabulate(
        ou_rows,
        headers=["Line", "Under", "Over"],
        tablefmt="rounded_outline",
    ))
    print(f"  Projected winning time: {np.mean(winning_times):.3f}s  (median {np.median(winning_times):.3f}s)")


def show_distributions(models: list[RaceModel], event_name: str = "Men's 50m Freestyle") -> None:
    """Plot each swimmer's ex-Gaussian time distribution with a mean line."""
    # Sort by projected mean so the legend reads fastest → slowest
    sorted_models = sorted(models, key=lambda m: m.mu)

    x_min = min(m.pb for m in sorted_models) - 0.3
    x_max = max(m.mu for m in sorted_models) + 1.2
    x = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10.colors

    pdfs = []
    for i, m in enumerate(sorted_models):
        sigma_n = float(np.sqrt(max(m.sigma**2 - m.tau**2, 1e-9)))
        K = m.tau / sigma_n if m.tau > 0 else 1e-6
        pdfs.append(exponnorm.pdf(x, K=K, loc=m.mu - m.tau, scale=sigma_n))

    y_max = max(pdf.max() for pdf in pdfs)

    for i, (m, pdf) in enumerate(zip(sorted_models, pdfs)):
        color = colors[i % len(colors)]
        ax.plot(x, pdf, color=color, linewidth=1.8, label=f"{m.name}  (μ={m.mu:.3f}s)")
        ax.axvline(m.mu, color=color, linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_xlabel("Race Time (s)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"2024 Paris Olympics — {event_name} Final\nSwimmer Time Distributions (Ex-Gaussian)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "distributions.png")
    fig.savefig(path, dpi=150)
    print(f"Distributions chart saved to {path}")
    plt.show()


def show_chart(results: list[SimResult], event_name: str = "Men's 50m Freestyle") -> None:
    """Display a horizontal bar chart of win probabilities."""
    sorted_results = _sorted_results(results)
    names = [r.name for r in sorted_results]
    win_probs = [r.place_probs[1] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(names, [p * 100 for p in win_probs], color="steelblue")
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("Win Probability (%)")
    ax.set_title(f"2024 Paris Olympics — {event_name} Final\nWin Probability (Monte Carlo)")
    ax.invert_yaxis()
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "win_probabilities.png")
    fig.savefig(path, dpi=150)
    print(f"Win probability chart saved to {path}")
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
