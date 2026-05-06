import json
from pathlib import Path
from typing import Any

import config


HYPERPARAM_TO_CONFIG = {
    "season_decay": "SEASON_DECAY",
    "max_seasons": "MAX_SEASONS",
    "best_time_decay": "BEST_TIME_DECAY",
    "decay_distance_exp": "DECAY_DISTANCE_EXP",
    "sigma_distance_exp": "SIGMA_DISTANCE_EXP",
    "default_sigma": "DEFAULT_SIGMA",
    "default_tau": "DEFAULT_TAU",
}


def current_hyperparams() -> dict[str, float | int]:
    """Return build_model hyperparameters from src/config.py."""
    values: dict[str, float | int] = {}
    for key, attr in HYPERPARAM_TO_CONFIG.items():
        values[key] = getattr(config, attr)
    return values


def load_preset(path: str | Path | None) -> dict[str, float | int]:
    """Load a JSON hyperparameter preset, returning only provided overrides."""
    if path is None:
        return {}

    preset_path = Path(path)
    data = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config preset must be a JSON object: {preset_path}")

    params = data.get("hyperparams", data)
    if not isinstance(params, dict):
        raise ValueError(f"Config preset hyperparams must be a JSON object: {preset_path}")

    unknown = sorted(set(params) - set(HYPERPARAM_TO_CONFIG))
    if unknown:
        raise ValueError(f"Unknown config preset keys in {preset_path}: {', '.join(unknown)}")

    loaded: dict[str, float | int] = {}
    for key, value in params.items():
        loaded[key] = int(value) if key == "max_seasons" else float(value)
    return loaded


def merged_hyperparams(path: str | Path | None = None) -> dict[str, float | int]:
    """Return src/config.py hyperparameters overlaid with a JSON preset."""
    params = current_hyperparams()
    params.update(load_preset(path))
    return params


def write_preset(
    path: str | Path,
    params: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON config preset for later use by run.py/tune_hyperparams.py."""
    preset_path = Path(path)
    unknown = sorted(set(params) - set(HYPERPARAM_TO_CONFIG))
    if unknown:
        raise ValueError(f"Unknown config preset keys for {preset_path}: {', '.join(unknown)}")

    cleaned: dict[str, float | int] = {}
    for key in HYPERPARAM_TO_CONFIG:
        if key not in params:
            continue
        value = params[key]
        cleaned[key] = int(value) if key == "max_seasons" else round(float(value), 6)

    output: dict[str, Any] = {"hyperparams": cleaned}
    if metadata:
        output["metadata"] = metadata

    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return preset_path
