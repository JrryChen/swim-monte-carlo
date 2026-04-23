import numpy as np
from models import Athlete, RaceModel, SimResult
from config import DEFAULT_SIGMA, DEFAULT_TAU, N_SIMULATIONS, SEASON_DECAY, SEASON_START_MONTH, MAX_SEASONS, BEST_TIME_DECAY


def _get_season_year(date: str) -> int:
    """Return the season start year for a YYYY-MM-DD date.
    Seasons start in September: Sep–Dec of year Y is season Y,
    Jan–Aug of year Y is season Y-1.
    """
    year, month = int(date[:4]), int(date[5:7])
    return year if month >= SEASON_START_MONTH else year - 1


def build_model(athlete: Athlete, world_record: float = 20.91) -> RaceModel:
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

    season_weights = np.array([SEASON_DECAY ** (most_recent - s) for s in seasons])

    # Proximity weighting: times closer to the world record get exponentially more
    # weight, so elite performances pull the mean down more than off-days.
    proximity_weights = np.exp(-BEST_TIME_DECAY * (times - world_record))
    weights = season_weights * proximity_weights

    mu_raw = float(np.average(times, weights=weights))

    if len(times) == 1:
        sigma = DEFAULT_SIGMA
    else:
        variance = float(np.average((times - mu_raw) ** 2, weights=weights))
        sigma = float(np.sqrt(variance))

    # Season-best adjustment: express each season's drop as a fraction of
    # that season's average so that dropping 0.3s from 21.5 is treated as
    # a harder achievement than dropping 0.3s from 21.9.
    unique_seasons = sorted(set(int(s) for s in seasons))
    rel_drops, drop_weights = [], []
    for s in unique_seasons:
        s_times = times[seasons == s]
        s_avg = float(np.mean(s_times))
        rel_drops.append((s_avg - float(np.min(s_times))) / s_avg)
        drop_weights.append(SEASON_DECAY ** (most_recent - s))

    season_drop = float(np.average(rel_drops, weights=drop_weights))
    mu = mu_raw * (1 - season_drop)

    # Hard cap: never project faster than the swimmer's actual best in the window.
    pb = float(np.min(times))
    mu = max(mu, pb)

    # Estimate tau (exponential component) from weighted third central moment.
    # For ex-Gaussian: third central moment = 2*tau^3, so tau = (m3/2)^(1/3).
    # Fall back to DEFAULT_TAU if data is too sparse or skew is non-positive.
    if len(times) >= 3:
        m3 = float(np.average((times - mu_raw) ** 3, weights=weights))
        tau = float((m3 / 2) ** (1 / 3)) if m3 > 0 else DEFAULT_TAU
    else:
        tau = DEFAULT_TAU
    # tau can't exceed sigma (would leave no room for the normal component)
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
