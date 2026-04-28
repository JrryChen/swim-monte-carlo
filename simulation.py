import numpy as np
from models import Athlete, RaceModel, SimResult
from config import DEFAULT_SIGMA, DEFAULT_TAU, N_SIMULATIONS, SEASON_DECAY, SEASON_START_MONTH, MAX_SEASONS, BEST_TIME_DECAY, DECAY_DISTANCE_EXP
from events import EventConfig


def _get_season_year(date: str) -> int:
    """Return the season start year for a YYYY-MM-DD date.
    Seasons start in September: Sep–Dec of year Y is season Y,
    Jan–Aug of year Y is season Y-1.
    """
    year, month = int(date[:4]), int(date[5:7])
    return year if month >= SEASON_START_MONTH else year - 1


def build_model(
    athlete: Athlete,
    event: EventConfig,
    *,
    season_decay: float = SEASON_DECAY,
    max_seasons: int = MAX_SEASONS,
    best_time_decay: float = BEST_TIME_DECAY,
    decay_distance_exp: float = DECAY_DISTANCE_EXP,
    default_sigma: float = DEFAULT_SIGMA,
    default_tau: float = DEFAULT_TAU,
) -> RaceModel:
    """Fit a seasonally-weighted ex-Gaussian to the athlete's times.

    Results from the most recent season carry weight 1.0; each prior season
    is multiplied by season_decay, so older data has diminishing influence.

    default_sigma, default_tau, and best_time_decay are tuned for 50m sprints.
    They are scaled by distance/50 (or 50/distance for decay) so that fallback
    values and proximity weighting are proportionally equivalent across events.

    All numeric hyperparameters default to the values in config.py; pass explicit
    values to override (e.g. during hyperparameter tuning with tune_hyperparams.py).
    """
    dated = [r for r in athlete.results if r.date]
    if not dated:
        raise ValueError(f"No LCM times found for {athlete.name}.")

    # scale = event.distance / 50
    scale = 1
    effective_sigma = default_sigma * scale
    effective_tau = default_tau * scale
    effective_decay = best_time_decay / (event.distance / 50) ** decay_distance_exp

    most_recent = max(_get_season_year(r.date) for r in dated)
    cutoff = most_recent - max_seasons
    dated = [r for r in dated if _get_season_year(r.date) > cutoff]

    times = np.array([r.time_seconds for r in dated])
    seasons = np.array([_get_season_year(r.date) for r in dated])

    season_weights = np.array([season_decay ** (most_recent - s) for s in seasons])

    proximity_weights = np.exp(-effective_decay * (times - event.world_record))
    weights = season_weights * proximity_weights

    mu_raw = float(np.average(times, weights=weights))

    if len(times) == 1:
        sigma = effective_sigma
    else:
        variance = float(np.average((times - mu_raw) ** 2, weights=weights))
        sigma = float(np.sqrt(variance))

    unique_seasons = sorted(set(int(s) for s in seasons))
    rel_drops, drop_weights = [], []
    for s in unique_seasons:
        s_times = times[seasons == s]
        s_avg = float(np.mean(s_times))
        rel_drops.append((s_avg - float(np.min(s_times))) / s_avg)
        drop_weights.append(season_decay ** (most_recent - s))

    season_drop = float(np.average(rel_drops, weights=drop_weights))
    mu = mu_raw

    pb = float(np.min(times))

    if len(times) >= 3:
        m3 = float(np.average((times - mu_raw) ** 3, weights=weights))
        tau = float((m3 / 2) ** (1 / 3)) if m3 > 0 else effective_tau
    else:
        tau = effective_tau
    tau = min(tau, sigma * 0.9)

    return RaceModel(name=athlete.name, mu=mu, sigma=sigma, tau=tau, season_drop=season_drop, pb=pb)


def run(models: list[RaceModel], n: int = N_SIMULATIONS) -> tuple[list[SimResult], np.ndarray]:
    """
    Simulate n races and return (place-probability results, winning times).
    Lower time = better finish.
    """
    num_swimmers = len(models)
    position_counts: dict[str, list[int]] = {m.name: [0] * num_swimmers for m in models}
    winning_times: list[float] = []

    rng = np.random.default_rng()

    for _ in range(n):
        # Ex-Gaussian: Normal(mu - tau, sigma_n) + Exponential(tau)
        # Expected value = (mu - tau) + tau = mu  (preserves projected mean)
        sampled_times = np.array([
            rng.normal(m.mu - m.tau, max(np.sqrt(m.sigma**2 - m.tau**2), 1e-6))
            + rng.exponential(m.tau) if m.tau > 0
            else rng.normal(m.mu, m.sigma)
            for m in models
        ])
        winning_times.append(float(np.min(sampled_times)))
        # argsort ascending: index 0 = fastest swimmer
        ranks = np.argsort(sampled_times)
        for place, swimmer_idx in enumerate(ranks):
            position_counts[models[swimmer_idx].name][place] += 1

    results = []
    for model in models:
        counts = position_counts[model.name]
        place_probs = {place + 1: count / n for place, count in enumerate(counts)}
        results.append(SimResult(name=model.name, place_probs=place_probs))

    return results, np.array(winning_times)


def run_fast(
    models: list[RaceModel],
    n: int = N_SIMULATIONS,
    rng: np.random.Generator | None = None,
) -> list[SimResult]:
    """Fully vectorized simulation — ~10–20x faster than run(), designed for hyperparameter tuning.

    Samples all n races in one NumPy call per swimmer, then counts placements with bincount.
    Returns place-probability results only (no winning times array).
    """
    if rng is None:
        rng = np.random.default_rng()
    num = len(models)

    # Shape (n_races, n_swimmers) — sample all races at once
    times = np.zeros((n, num))
    for i, m in enumerate(models):
        if m.tau > 0:
            sigma_n = float(np.sqrt(max(m.sigma ** 2 - m.tau ** 2, 0.0))) or 1e-6
            times[:, i] = rng.normal(m.mu - m.tau, sigma_n, n) + rng.exponential(m.tau, n)
        else:
            times[:, i] = rng.normal(m.mu, m.sigma, n)

    # ranks[race_i, place_j] = swimmer_idx who finished in place j in race i (0-indexed)
    ranks = np.argsort(times, axis=1)

    # place_matrix[swimmer_idx, place_idx] = count of times that swimmer finished in that place
    place_matrix = np.zeros((num, num), dtype=np.int64)
    for place_idx in range(num):
        swimmers_at_place = ranks[:, place_idx]  # shape (n,)
        place_matrix[:, place_idx] = np.bincount(swimmers_at_place, minlength=num)

    return [
        SimResult(
            name=m.name,
            place_probs={p + 1: int(place_matrix[i, p]) / n for p in range(num)},
        )
        for i, m in enumerate(models)
    ]
