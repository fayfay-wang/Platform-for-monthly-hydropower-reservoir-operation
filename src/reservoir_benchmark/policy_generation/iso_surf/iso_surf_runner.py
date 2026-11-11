"""
ISO-SURF policy fitting and persistence module for the unified reservoir
operation framework.

This module fits monthly ISO-SURF rule surfaces from upstream DP training
trajectories and persists the fitted policy as a PKL artifact, including:
1) preparing month-wise training pairs with optional next-month forecasts,
2) fitting monthly spline surfaces with health checks,
3) falling back to linear planes when spline fitting is unstable,
4) predicting next-period water levels from the fitted rule surface, and
5) saving and loading the fitted ISO-SURF policy artifact.

The purpose of this module is to keep ISO-SURF policy generation independent
from evaluation and to standardize policy persistence under the unified
repository structure.
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline

from reservoir_benchmark.policy_generation.iso_surf.iso_preprocessor import ISOPreprocessor
from reservoir_benchmark.shared.inflow_forecast import (
    ForecastScenario,
    load_forecast_penalty_candidates,
)
from reservoir_benchmark.shared.io_utils import (
    load_algorithm_settings,
    load_penalty_candidates,
    require_keys,
)
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_policy_generation_dir,
)


class ISOSurfPolicyFitter:
    """
    Fit and persist monthly ISO-SURF policy surfaces.
    """

    def __init__(
        self,
        smoothing_factor: float = 1.0,
        spline_order: tuple[int, int] = (3, 3),
        grid_size: int = 25,
        max_factor: float = 10000.0,
        min_samples_spline: int = 20,
    ):
        self.smoothing_factor = float(smoothing_factor)
        self.spline_order = tuple(int(x) for x in spline_order)
        self.grid_size = int(grid_size)
        self.max_factor = float(max_factor)
        self.min_samples_spline = int(min_samples_spline)

        if len(self.spline_order) != 2:
            raise ValueError("iso_surf.spline_order must contain exactly two integers.")

        self.month_models: dict[int, dict] = {}
        self.output_dir = get_policy_generation_dir("iso_surf") / "policy"

    def _prepare_dataframe(
        self, dp_df: pd.DataFrame, forecast: ForecastScenario | None = None
    ) -> pd.DataFrame:
        """
        Prepare DP trajectory data for month-wise ISO-SURF fitting.
        """
        # The upstream table supplies month, end-of-month water level, and inflow.
        required_cols = ["Month", "Water_Level_m", "Inflow_m3s"]
        for col in required_cols:
            if col not in dp_df.columns:
                raise ValueError(f"DP training DataFrame is missing required column: {col}")

        df = dp_df.copy()

        if forecast is None:
            # Preserve the original training-pair convention for legacy policies.
            df["Next_Level_m"] = df["Water_Level_m"].shift(-1)
            df = df.iloc[:-1, :].copy()
        else:
            # The DP table stores end-of-month levels. Match each decision to
            # its start level, current inflow, and optional next-month forecast.
            df["Next_Level_m"] = df["Water_Level_m"]
            df["Water_Level_m"] = df["Water_Level_m"].shift(1)
            next_inflows = forecast.next_month(df["Inflow_m3s"].to_numpy(), forecast.training_seed)
            if next_inflows is not None:
                df["Next_Inflow_Forecast_m3s"] = next_inflows
            # Give all sensitivity fits the same rows with known boundary states and inflows.
            df = df.iloc[1:-1, :].copy()

        # Convert the continuous month index in the DP trajectory into a month-of-year
        # label in the range 1-12, so each month can be fitted independently.
        df["MonthOfYear"] = ((df["Month"] - 1) % 12) + 1

        # Remove any incomplete rows before fitting.
        df = df.dropna(subset=["Water_Level_m", "Inflow_m3s", "Next_Level_m"])
        return df

    def fit_from_dp_dataframe(
        self, dp_df: pd.DataFrame, forecast: ForecastScenario | None = None
    ) -> None:
        """
        Fit month-wise ISO-SURF rule surfaces from a DP monthly trajectory DataFrame.
        """
        df = self._prepare_dataframe(dp_df, forecast)
        if "Next_Inflow_Forecast_m3s" in df and not df.empty:
            # Fit the original bivariate surface within each monthly forecast band.
            forecast_by_month = df.groupby("MonthOfYear")["Next_Inflow_Forecast_m3s"]
            lower = forecast_by_month.transform("quantile", q=1 / 3)
            upper = forecast_by_month.transform("quantile", q=2 / 3)
            observed = df["Next_Inflow_Forecast_m3s"]
            forecast_bands = observed.ge(lower).astype(int) + observed.ge(upper).astype(int)

            self.month_models = {}
            for _, band in df.groupby(forecast_bands):
                surfaces = self._fit_month_models(band)
                medians = band.groupby("MonthOfYear")["Next_Inflow_Forecast_m3s"].median()
                for month, median in medians.items():
                    model = self.month_models.setdefault(
                        month, {"mode": "forecast", "centers": [], "surfaces": []}
                    )
                    model["centers"].append(float(median))
                    model["surfaces"].append(surfaces[month])
        else:
            self.month_models = self._fit_month_models(df)

        print("ISO-SURF month models fitted for months:", sorted(self.month_models))

    def _fit_month_models(self, df: pd.DataFrame) -> dict[int, dict]:

        # Use a positive smoothing factor. If the user provides a non-positive
        # value, fall back to a default value of 1.0.
        base_sf = self.smoothing_factor if self.smoothing_factor > 0 else 1.0

        def _check_spline_health(S_norm, I_norm, Z_data, spline, grid_size=25):
            """
            Check whether a fitted spline behaves reasonably on a regular grid.

            The spline is rejected if:
            1) it produces non-finite values,
            2) its predicted range is far outside the observed training range, or
            3) its overall variation is excessively large compared with the data.
            """
            Sg = np.linspace(S_norm.min(), S_norm.max(), grid_size)
            Ig = np.linspace(I_norm.min(), I_norm.max(), grid_size)
            Zg = spline(Sg, Ig)

            if not np.all(np.isfinite(Zg)):
                return False, np.nan, np.nan, np.nan, np.nan

            z_min = float(Zg.min())
            z_max = float(Zg.max())

            data_min = float(Z_data.min())
            data_max = float(Z_data.max())
            data_range = max(data_max - data_min, 1e-6)
            grid_range = z_max - z_min

            # Allow a moderate extrapolation margin, but reject strongly abnormal
            # predictions that go far beyond the observed target range.
            margin = max(5.0 * data_range, 50.0)
            if z_min < data_min - margin or z_max > data_max + margin:
                return False, z_min, z_max, data_min, data_max

            # Reject splines that oscillate too strongly over the fitted domain.
            max_allowed_range = max(10.0 * data_range, 200.0)
            if grid_range > max_allowed_range:
                return False, z_min, z_max, data_min, data_max

            return True, z_min, z_max, data_min, data_max

        month_models = {}

        # Fit one rule surface per calendar month.
        for month in range(1, 13):
            df_m = df[df["MonthOfYear"] == month].copy()
            n_m = len(df_m)

            # If the available monthly samples are too few, skip spline fitting
            # and directly use the linear fallback model.
            if n_m < self.min_samples_spline:
                month_models[month] = self._fit_linear_only(df_m)
                continue

            S_t = df_m["Water_Level_m"].values
            I_t = df_m["Inflow_m3s"].values
            Z = df_m["Next_Level_m"].values

            # Normalize the two input dimensions before spline fitting so that
            # water level and inflow have comparable numerical scales.
            S_mean, S_std = S_t.mean(), S_t.std()
            I_mean, I_std = I_t.mean(), I_t.std()
            S_std = S_std if S_std > 0 else 1.0
            I_std = I_std if I_std > 0 else 1.0

            S_norm = (S_t - S_mean) / S_std
            I_norm = (I_t - I_mean) / I_std

            # Start from the base smoothing level and gradually increase it if
            # the spline is unstable or Fitpack reports warnings.
            factor = base_sf
            max_factor = base_sf * self.max_factor
            best_spline = None

            while factor <= max_factor:
                s_val = factor * n_m

                # Capture warnings raised during spline fitting. Some warnings
                # indicate that the smoothing is too small or the fit is
                # numerically unstable.
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    try:
                        spline = SmoothBivariateSpline(
                            S_norm,
                            I_norm,
                            Z,
                            kx=self.spline_order[0],
                            ky=self.spline_order[1],
                            s=s_val,
                        )
                    except Exception:
                        # If the fit fails completely, increase smoothing and retry.
                        factor *= 5.0
                        continue

                has_fitpack_warning = False
                for w in wlist:
                    msg = str(w.message)
                    if (
                        "storage space" in msg
                        or "s too small" in msg
                        or "The required storage space" in msg
                    ):
                        has_fitpack_warning = True
                        break

                # If Fitpack indicates a problematic fit, use a larger smoothing value.
                if has_fitpack_warning:
                    factor *= 5.0
                    continue

                # Even if fitting formally succeeds, reject the spline if it is
                # numerically unhealthy on a regular evaluation grid.
                ok, _, _, _, _ = _check_spline_health(
                    S_norm,
                    I_norm,
                    Z,
                    spline,
                    grid_size=self.grid_size,
                )
                if not ok:
                    factor *= 5.0
                    continue

                # Accept the first spline that passes all health checks.
                best_spline = spline
                break

            # If no acceptable spline can be obtained, fall back to a linear surface.
            if best_spline is None:
                month_models[month] = self._fit_linear_only(df_m)
                continue

            # Store the accepted month model together with the normalization
            # statistics required later for prediction.
            month_models[month] = {
                "mode": "spline",
                "spline": best_spline,
                "S_mean": S_mean,
                "S_std": S_std,
                "I_mean": I_mean,
                "I_std": I_std,
                "S_range": (float(S_t.min()), float(S_t.max())),
                "I_range": (float(I_t.min()), float(I_t.max())),
                "n_samples": n_m,
            }

        return month_models

    def _fit_linear_only(self, df_m: pd.DataFrame) -> dict:
        """
        Fit a linear fallback plane:
            S_{t+1} = a + b S_t + c I_t
        """
        if df_m.empty:
            return {
                "mode": "identity",
                "n_samples": 0,
            }

        S_t = df_m["Water_Level_m"].values
        I_t = df_m["Inflow_m3s"].values
        Z = df_m["Next_Level_m"].values

        # Fit a simple least-squares linear plane when spline fitting is not
        # possible or not reliable.
        X = np.column_stack([np.ones_like(S_t), S_t, I_t])
        coef, *_ = np.linalg.lstsq(X, Z, rcond=None)

        return {
            "mode": "linear",
            "coef": coef,
            "S_range": (float(S_t.min()), float(S_t.max())),
            "I_range": (float(I_t.min()), float(I_t.max())),
            "n_samples": len(df_m),
        }

    def predict_next_level(
        self, month: int, S_t: float, I_t: float, next_inflow_forecast: float | None = None
    ) -> float:
        """
        Predict next-period water level S_{t+1} for a given month and state.
        """
        info = self.month_models.get(month, {"mode": "identity"})
        if info.get("mode") == "forecast":
            if next_inflow_forecast is None:
                raise ValueError("This ISO-SURF policy requires a next-month inflow forecast.")
            levels = [self._predict_surface(surface, S_t, I_t) for surface in info["surfaces"]]
            return float(np.interp(next_inflow_forecast, info["centers"], levels))

        return self._predict_surface(info, S_t, I_t)

    @staticmethod
    def _predict_surface(info: dict, S_t: float, I_t: float) -> float:
        mode = info.get("mode", "spline")

        if mode == "identity":
            return float(S_t)

        if mode == "spline":
            spline = info["spline"]
            S_mean, S_std = info["S_mean"], info["S_std"]
            I_mean, I_std = info["I_mean"], info["I_std"]
            S_std = S_std if S_std > 0 else 1.0
            I_std = I_std if I_std > 0 else 1.0

            # Apply the same normalization used during fitting before calling
            # the spline predictor.
            S_norm = (S_t - S_mean) / S_std
            I_norm = (I_t - I_mean) / I_std
            z_pred = spline(S_norm, I_norm)[0][0]
            return float(z_pred)

        if mode == "linear":
            a, b, c = info["coef"]
            z_pred = a + b * S_t + c * I_t
            return float(z_pred)

        return float(S_t)

    def save_policy(self, lam: float, forecast: ForecastScenario | None = None) -> Path:
        """
        Save the fitted ISO-SURF policy as a PKL file.
        """
        lam_str = format_lambda_for_filename(lam)
        output_dir = self.output_dir
        if forecast is not None:
            output_dir /= forecast.name
        output_dir.mkdir(parents=True, exist_ok=True)
        policy_path = output_dir / f"iso_surf_train_lambda_{lam_str}.pkl"

        # Save only the information needed later for evaluation and prediction.
        payload = {
            "smoothing_factor": self.smoothing_factor,
            "spline_order": self.spline_order,
            "grid_size": self.grid_size,
            "max_factor": self.max_factor,
            "min_samples_spline": self.min_samples_spline,
            "month_models": self.month_models,
        }

        with policy_path.open("wb") as f:
            pickle.dump(payload, f)

        print(f"ISO-SURF policy saved to: {policy_path}")
        return policy_path

    @classmethod
    def load_policy(cls, policy_path: str | Path) -> "ISOSurfPolicyFitter":
        """
        Load a fitted ISO-SURF policy from a PKL file.
        """
        policy_path = Path(policy_path)
        with policy_path.open("rb") as f:
            payload = pickle.load(f)

        obj = cls(
            smoothing_factor=payload.get("smoothing_factor", 1.0),
            spline_order=tuple(payload.get("spline_order", (3, 3))),
            grid_size=payload.get("grid_size", 25),
            max_factor=payload.get("max_factor", 10000.0),
            min_samples_spline=payload.get("min_samples_spline", 20),
        )
        obj.month_models = payload["month_models"]
        return obj


def main_iso_surf(forecast: ForecastScenario | Sequence[ForecastScenario] | None = None) -> None:
    """
    Generate upstream DP training data first, then fit and save ISO-SURF policies.
    """

    data_dir = get_data_dir()
    experiment_settings_file = data_dir / "experiment_settings.yaml"
    iso_settings = load_algorithm_settings(experiment_settings_file, "iso_surf")
    require_keys(
        iso_settings,
        [
            "smoothing_factor",
            "spline_order",
            "grid_size",
            "max_factor",
            "min_samples_spline",
        ],
        "iso_surf",
    )

    spline_order = tuple(int(x) for x in iso_settings["spline_order"])
    if len(spline_order) != 2:
        raise ValueError("iso_surf.spline_order must contain exactly two integers.")

    iso_root_dir = get_policy_generation_dir("iso_surf")
    preprocessed_dir = iso_root_dir / "preprocessed"
    policy_dir = iso_root_dir / "policy"
    penalty_settings_file = data_dir / "penalty_settings.yaml"
    preprocessor = ISOPreprocessor()
    if forecast is None:
        scenarios = [None]
        penalty_lambdas = load_penalty_candidates(penalty_settings_file, "iso_surf")
        preprocessor.run_all()
    else:
        scenarios = [forecast] if isinstance(forecast, ForecastScenario) else forecast
        penalty_lambdas = load_forecast_penalty_candidates("iso_surf")

    print("\n============================================================")
    print("ISO-SURF policy fitting started")
    print("============================================================")
    print(f"Penalty settings file: {penalty_settings_file}")
    print(f"Preprocessed directory: {preprocessed_dir}")
    print(f"Policy directory: {policy_dir}")
    print(f"Penalty lambdas: {penalty_lambdas}")
    print("============================================================")

    for lam in penalty_lambdas:
        lam_str = format_lambda_for_filename(lam)
        dp_csv_path = preprocessed_dir / f"iso_upstream_dp_train_lambda_{lam_str}.csv"
        if forecast is not None:
            # One full-horizon training DP supplies the targets for all scenarios.
            dp_csv_path = preprocessor.run_one(float(lam))

        if not dp_csv_path.exists():
            raise FileNotFoundError(
                f"Required upstream DP training file not found: {dp_csv_path}"
            )

        print("\n------------------------------------------------------------")
        print(f"Fitting ISO-SURF policy for lambda = {lam}")
        print("------------------------------------------------------------")

        # Each lambda corresponds to one upstream DP trajectory file. That file
        # is loaded and used to fit a separate ISO-SURF monthly rule set.
        dp_df = pd.read_csv(dp_csv_path)

        fitter = ISOSurfPolicyFitter(
            smoothing_factor=float(iso_settings["smoothing_factor"]),
            spline_order=spline_order,
            grid_size=int(iso_settings["grid_size"]),
            max_factor=float(iso_settings["max_factor"]),
            min_samples_spline=int(iso_settings["min_samples_spline"]),
        )
        for scenario in scenarios:
            fitter.fit_from_dp_dataframe(dp_df, forecast=scenario)
            fitter.save_policy(float(lam), forecast=scenario)


if __name__ == "__main__":
    main_iso_surf()
