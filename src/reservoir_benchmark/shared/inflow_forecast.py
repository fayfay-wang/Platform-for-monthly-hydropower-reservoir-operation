"""Shared one-month-ahead inflow scenarios for the forecast sensitivity experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reservoir_benchmark.shared.io_utils import load_algorithm_settings
from reservoir_benchmark.shared.project_paths import get_data_dir


@dataclass(frozen=True)
class ForecastScenario:
    mode: str
    cv: float = 0.0
    training_seed: int = 2026
    evaluation_seed: int = 2027

    @property
    def name(self) -> str:
        return f"noisy_cv{self.cv}" if self.mode == "noisy" else self.mode

    def observe(self, actual_next, seed: int) -> np.ndarray:
        """Apply a mean-one lognormal error with the configured relative CV."""
        actual_next = np.asarray(actual_next, dtype=float)
        if self.mode != "noisy":
            return actual_next
        variance = np.log1p(self.cv ** 2)
        errors = np.random.default_rng(seed).standard_normal(actual_next.shape)
        return actual_next * np.exp(np.sqrt(variance) * errors - variance / 2)

    def next_month(self, inflows, seed: int) -> np.ndarray | None:
        if self.mode == "none":
            return None
        inflows = np.asarray(inflows, dtype=float)
        flat = inflows.ravel()
        # The final step has no next-month record and is omitted from reporting.
        actual_next = np.concatenate((flat[1:], flat[-1:]))
        return self.observe(actual_next, seed).reshape(inflows.shape)


def _load_settings() -> dict:
    return load_algorithm_settings(
        get_data_dir() / "experiment_settings.yaml", "forecast_sensitivity"
    )


def load_forecast_scenarios() -> list[ForecastScenario]:
    settings = _load_settings()
    training_seed = int(settings["training_seed"])
    return [
        ForecastScenario("perfect", training_seed=training_seed),
        ForecastScenario("none", training_seed=training_seed),
        *[
            ForecastScenario("noisy", cv=float(cv), training_seed=training_seed)
            for cv in settings["error_cvs"]
        ],
    ]


def load_forecast_evaluation_seeds() -> list[int]:
    return [int(seed) for seed in _load_settings()["evaluation_seeds"]]


def load_forecast_penalty_candidates(algorithm: str) -> list[float]:
    return [float(value) for value in _load_settings()["penalty_candidates"][algorithm]]


def forecast_evaluation_dir(
    base_dir: Path,
    forecast: ForecastScenario | None,
    policy_forecast: ForecastScenario | None = None,
) -> Path:
    if forecast is None:
        return base_dir
    output_dir = base_dir / forecast.name
    if policy_forecast is not None and policy_forecast.name != forecast.name:
        output_dir /= f"policy_{policy_forecast.name}"
    if forecast.mode == "noisy":
        output_dir /= f"forecast_seed_{forecast.evaluation_seed}"
    return output_dir
