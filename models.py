from dataclasses import dataclass, field


@dataclass
class SwimResult:
    competition: str
    time_seconds: float
    date: str


@dataclass
class Athlete:
    id: str
    name: str
    results: list[SwimResult] = field(default_factory=list)

    @property
    def times(self) -> list[float]:
        return [r.time_seconds for r in self.results]


@dataclass
class RaceModel:
    name: str
    mu: float         # championship-adjusted mean time in seconds
    sigma: float      # std dev in seconds
    season_drop: float = 0.0  # relative drop fraction: (season_avg - season_best) / season_avg
    pb: float = 0.0           # personal best in the 4-season window


@dataclass
class SimResult:
    name: str
    place_probs: dict[int, float]  # {1: 0.31, 2: 0.24, ...}
