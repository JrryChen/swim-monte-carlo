import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import Athlete, SwimResult
from simulation import build_model, run


def make_athlete(name: str, times: list[float]) -> Athlete:
    athlete = Athlete(id="0", name=name)
    athlete.results = [SwimResult(competition="Test", time_seconds=t, date="2024-01-01") for t in times]
    return athlete


def test_build_model_computes_mean_and_std():
    athlete = make_athlete("Alice", [21.0, 21.5, 22.0])
    model = build_model(athlete)

    assert model.name == "Alice"
    assert abs(model.mu - 21.5) < 1e-9
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
