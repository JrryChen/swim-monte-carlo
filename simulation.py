import numpy as np
from models import Athlete, RaceModel, SimResult
from config import DEFAULT_SIGMA, N_SIMULATIONS, SEASON_DECAY, SEASON_START_MONTH


def _get_season_year(date: str) -> int:
    """Return the season start year for a YYYY-MM-DD date.
    Seasons start in September: Sep–Dec of year Y is season Y,
    Jan–Aug of year Y is season Y-1.
    """
    year, month = int(date[:4]), int(date[5:7])
    return year if month >= SEASON_START_MONTH else year - 1


def build_model(athlete: Athlete) -> RaceModel:
    """Fit a seasonally-weighted normal distribution to the athlete's times.

    Results from the most recent season carry weight 1.0; each prior season
    is multiplied by SEASON_DECAY, so older data has diminishing influence.
    """
    dated = [r for r in athlete.results if r.date]
    if not dated:
        raise ValueError(f"No LCM 50m freestyle times found for {athlete.name}.")

    times = np.array([r.time_seconds for r in dated])
    seasons = np.array([_get_season_year(r.date) for r in dated])

    most_recent = int(seasons.max())
    weights = np.array([SEASON_DECAY ** (most_recent - s) for s in seasons])

    mu = float(np.average(times, weights=weights))

    if len(times) == 1:
        sigma = DEFAULT_SIGMA
    else:
        variance = float(np.average((times - mu) ** 2, weights=weights))
        sigma = float(np.sqrt(variance))

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
