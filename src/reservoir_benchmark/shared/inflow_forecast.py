"""Shared month-start forecasts of the inflow over the operating month."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reservoir_benchmark.shared.io_utils import (
    load_algorithm_settings,
    load_monthly_inflows_from_excel,
)
from reservoir_benchmark.shared.project_paths import get_data_dir


@dataclass(frozen=True)
class ForecastScenario:
    mode: str
    cv: float = 0.0
    training_seed: int = 2026
    evaluation_seed: int = 2027
    target_nse: float | None = None

    @property
    def name(self) -> str:
        if self.mode != "noisy":
            return self.mode
        if self.target_nse is None:
            return f"noisy_gaussian_cv{self.cv:g}"
        return f"noisy_nse{self.target_nse:g}"

    def observe(self, actual, seed: int) -> np.ndarray:
        """Apply Gaussian relative error and floor forecasts at zero."""
        actual = np.asarray(actual, dtype=float)
        if self.mode != "noisy":
            return actual
        errors = np.random.default_rng(seed).standard_normal(actual.shape)
        return np.maximum(actual * (1.0 + self.cv * errors), 0.0)

    def current_month(self, inflows, seed: int) -> np.ndarray | None:
        if self.mode == "none":
            return None
        return self.observe(inflows, seed)


def _load_settings() -> dict:
    return load_algorithm_settings(
        get_data_dir() / "experiment_settings.yaml", "forecast_sensitivity"
    )


def load_forecast_scenarios() -> list[ForecastScenario]:
    settings = _load_settings()
    training_seed = int(settings["training_seed"])
    training_inflow = load_monthly_inflows_from_excel(
        get_data_dir() / "SynMonthlyInflow_train.xlsx"
    ).to_numpy(dtype=float)
    # Yang et al. (2020), Eq. (16), with one training-only scale per target NSE.
    relative_variance = np.sum((training_inflow - training_inflow.mean()) ** 2) / np.sum(
        training_inflow ** 2
    )
    return [
        ForecastScenario("perfect", training_seed=training_seed),
        ForecastScenario("none", training_seed=training_seed),
        *[
            ForecastScenario(
                "noisy",
                cv=float(np.sqrt((1.0 - nse) * relative_variance)),
                training_seed=training_seed,
                target_nse=float(nse),
            )
            for nse in settings["target_nse"]
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
    output_dir = base_dir / "current_month" / forecast.name
    if policy_forecast is not None and policy_forecast.name != forecast.name:
        output_dir /= f"policy_{policy_forecast.name}"
    if forecast.mode == "noisy":
        output_dir /= f"forecast_seed_{forecast.evaluation_seed}"
    return output_dir
