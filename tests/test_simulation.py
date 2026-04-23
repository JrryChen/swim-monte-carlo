import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from models import Athlete, SwimResult
from simulation import build_model, run, _get_season_year


def make_athlete(name: str, times: list[float], date: str = "2024-01-01") -> Athlete:
    athlete = Athlete(id="0", name=name)
    athlete.results = [SwimResult(competition="Test", time_seconds=t, date=date) for t in times]
    return athlete


def test_build_model_computes_mean_and_std():
    # Times [21.0, 21.5, 22.0] in one season: raw_mean=21.5, best=21.0
    # relative_drop = 0.5/21.5; adjusted mu = 21.5 * (1 - 0.5/21.5) = 21.0
    athlete = make_athlete("Alice", [21.0, 21.5, 22.0])
    model = build_model(athlete)

    assert model.name == "Alice"
    assert abs(model.mu - 21.0) < 1e-9
    assert abs(model.season_drop - 0.5 / 21.5) < 1e-9
    assert model.sigma > 0


def test_build_model_uses_default_sigma_for_single_time():
    from config import DEFAULT_SIGMA
    athlete = make_athlete("Bob", [21.0])
    model = build_model(athlete)

    assert model.sigma == DEFAULT_SIGMA


def test_build_model_raises_for_no_times():
    athlete = Athlete(id="0", name="Empty")
    with pytest.raises(ValueError):
        build_model(athlete)


def test_run_returns_one_result_per_swimmer():
    athletes = [make_athlete(f"Swimmer{i}", [21.0 + i * 0.1]) for i in range(8)]
    models = [build_model(a) for a in athletes]
    results = run(models, n=100)

    assert len(results) == 8


def test_run_probabilities_sum_to_one():
    athletes = [make_athlete(f"Swimmer{i}", [21.0 + i * 0.05, 21.1 + i * 0.05]) for i in range(8)]
    models = [build_model(a) for a in athletes]
    results = run(models, n=1000)

    # Each swimmer's probs across all places should sum to 1
    for r in results:
        total = sum(r.place_probs.values())
        assert abs(total - 1.0) < 1e-9

    # Each place's probs across all swimmers should sum to 1
    for place in range(1, 9):
        total = sum(r.place_probs[place] for r in results)
        assert abs(total - 1.0) < 0.01  # small tolerance for Monte Carlo noise


def test_run_faster_swimmer_wins_more_often():
    fast = make_athlete("Fast", [21.0, 21.0])
    slow = make_athlete("Slow", [23.0, 23.0])
    models = [build_model(fast), build_model(slow)]
    results = run(models, n=1000)

    fast_result = next(r for r in results if r.name == "Fast")
    slow_result = next(r for r in results if r.name == "Slow")

    assert fast_result.place_probs[1] > slow_result.place_probs[1]


def test_get_season_year_september_starts_new_season():
    assert _get_season_year("2024-09-01") == 2024  # Sep = new season starts
    assert _get_season_year("2024-08-31") == 2023  # Aug = still old season
    assert _get_season_year("2024-01-15") == 2023  # Jan = still old season
    assert _get_season_year("2023-12-01") == 2023  # Dec = mid-season


def test_build_model_ignores_results_beyond_four_seasons():
    """Results more than 4 seasons old must not influence the model."""
    athlete = Athlete(id="0", name="Test")
    athlete.results = [
        SwimResult("VeryOld", 25.0, "2018-01-01"),  # season 2017 — 6 seasons ago, excluded
        SwimResult("Recent",  21.0, "2024-01-01"),  # season 2023 — included
        SwimResult("Recent",  21.0, "2024-06-01"),  # season 2023 — included
    ]
    model = build_model(athlete)
    # If the 25.0 were included, mu would be pulled well above 21.0
    assert model.mu < 22.0


def test_season_drop_lowers_mu_for_variable_swimmer():
    """A swimmer with a big relative gap between avg and best gets a lower projected μ."""
    consistent = Athlete(id="0", name="Consistent")
    consistent.results = [SwimResult("T", 21.5, "2024-01-01")] * 5  # no variance

    variable = Athlete(id="1", name="Variable")
    variable.results = [
        SwimResult("T", 21.7, "2024-01-01"),
        SwimResult("T", 21.7, "2024-02-01"),
        SwimResult("T", 21.0, "2024-03-01"),  # big championship drop
    ]

    m_consistent = build_model(consistent)
    m_variable = build_model(variable)

    assert m_consistent.season_drop == 0.0
    assert m_variable.season_drop > 0.01         # significant relative drop (>1%)
    assert m_variable.mu < m_consistent.mu       # variable swimmer projects faster


def test_season_drop_zero_for_single_result():
    """A single result per season has no drop (avg == best)."""
    athlete = make_athlete("Solo", [21.5])
    model = build_model(athlete)
    assert model.season_drop == 0.0


def test_build_model_weights_recent_seasons_more():
    """The weighted mean should be pulled toward the most recent season's times."""
    athlete = Athlete(id="0", name="Test")
    athlete.results = [
        # Old season (2019-2020): slow times
        SwimResult("Old", 23.0, "2020-01-01"),
        SwimResult("Old", 23.0, "2020-06-01"),
        # Recent season (2023-2024): fast times
        SwimResult("Recent", 21.0, "2024-01-01"),
        SwimResult("Recent", 21.0, "2024-06-01"),
    ]
    model = build_model(athlete)
    unweighted_mean = np.mean([23.0, 23.0, 21.0, 21.0])  # = 22.0

    # Weighted mean must be pulled below 22.0 toward the recent fast times
    assert model.mu < unweighted_mean
