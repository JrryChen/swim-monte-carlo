import numpy as np
from models import Athlete, RaceModel, SimResult
from config import DEFAULT_SIGMA, N_SIMULATIONS


def build_model(athlete: Athlete) -> RaceModel:
    """Fit a normal distribution to the athlete's historical times."""
    times = athlete.times
    if len(times) == 0:
        raise ValueError(f"No LCM 50m freestyle times found for {athlete.name}.")

    mu = float(np.mean(times))
    sigma = float(np.std(times)) if len(times) > 1 else DEFAULT_SIGMA

    return RaceModel(name=athlete.name, mu=mu, sigma=sigma)


def run(models: list[RaceModel], n: int = N_SIMULATIONS) -> list[SimResult]:
    """
    Simulate n races and return finishing-position probabilities per swimmer.
    Lower time = better finish.
    """
    num_swimmers = len(models)
    position_counts: dict[str, list[int]] = {m.name: [0] * num_swimmers for m in models}

    rng = np.random.default_rng()

    for _ in range(n):
        sampled_times = np.array([
            rng.normal(m.mu, m.sigma) for m in models
        ])
        # argsort ascending: index 0 = fastest swimmer
        ranks = np.argsort(sampled_times)
        for place, swimmer_idx in enumerate(ranks):
            position_counts[models[swimmer_idx].name][place] += 1

    results = []
    for model in models:
        counts = position_counts[model.name]
        place_probs = {place + 1: count / n for place, count in enumerate(counts)}
        results.append(SimResult(name=model.name, place_probs=place_probs))

    return results
