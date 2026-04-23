import numpy as np
from models import Athlete, RaceModel, SimResult
from config import DEFAULT_SIGMA, N_SIMULATIONS, SEASON_DECAY, SEASON_START_MONTH, MAX_SEASONS


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

    most_recent = max(_get_season_year(r.date) for r in dated)
    cutoff = most_recent - MAX_SEASONS  # seasons strictly older than this are dropped
    dated = [r for r in dated if _get_season_year(r.date) > cutoff]

    times = np.array([r.time_seconds for r in dated])
    seasons = np.array([_get_season_year(r.date) for r in dated])

    weights = np.array([SEASON_DECAY ** (most_recent - s) for s in seasons])

    mu_raw = float(np.average(times, weights=weights))

    if len(times) == 1:
        sigma = DEFAULT_SIGMA
    else:
        variance = float(np.average((times - mu_raw) ** 2, weights=weights))
        sigma = float(np.sqrt(variance))

    # Season-best adjustment: compute each season's avg→best drop, then
    # weight-average those drops (same decay) and subtract from mu.
    unique_seasons = sorted(set(int(s) for s in seasons))
    drops, drop_weights = [], []
    for s in unique_seasons:
        s_times = times[seasons == s]
        drops.append(float(np.mean(s_times) - np.min(s_times)))
        drop_weights.append(SEASON_DECAY ** (most_recent - s))

    season_drop = float(np.average(drops, weights=drop_weights))
    mu = mu_raw - season_drop

    return RaceModel(name=athlete.name, mu=mu, sigma=sigma, season_drop=season_drop)


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
