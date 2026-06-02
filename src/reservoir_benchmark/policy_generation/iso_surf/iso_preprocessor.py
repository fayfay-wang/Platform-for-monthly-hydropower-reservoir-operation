"""
ISO-SURF preprocessing module for the unified reservoir operation framework.

This module prepares the upstream deterministic dynamic programming (DP)
training trajectories required for ISO-SURF policy fitting, including:
1) loading the synthetic training inflow series,
2) loading ISO-SURF penalty coefficient candidates,
3) reusing the shared DP solver under the unified reservoir definition, and
4) exporting standardized monthly DP trajectory tables for subsequent
   ISO-SURF fitting.

The purpose of this module is to provide an ISO-specific wrapper around the
existing DP solver without duplicating the DP core algorithm.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservoir_benchmark.policy_generation.ddp.ddp_solver import ReservoirOperation as DPSolver
from reservoir_benchmark.shared.io_utils import (
    load_monthly_inflows_from_excel,
    load_penalty_candidates,
)
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_policy_generation_dir,
)
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


def save_iso_upstream_dp_results(
    result_dir: Path,
    lam: float,
    optimal_levels,
    optimal_storages,
    optimal_powers,
    optimal_electricity,
    optimal_discharges,
    optimal_turbine_flows,
    optimal_spill_flows,
    inflows,
) -> Path:
    """
    Save upstream DP training results for ISO-SURF fitting.

    File name pattern:
        iso_upstream_dp_train_lambda_{lam}.csv
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    lam_str = format_lambda_for_filename(lam)
    csv_path = result_dir / f"iso_upstream_dp_train_lambda_{lam_str}.csv"

    monthly_results = pd.DataFrame(
        {
            "Month": range(1, len(optimal_levels) + 1),
            "Water_Level_m": optimal_levels,
            "Storage_1e9_m3": optimal_storages,
            "Power_kW": optimal_powers,
            "Electricity_GWh": optimal_electricity,
            "Total_Discharge_m3s": optimal_discharges,
            "Turbine_Discharge_m3s": optimal_turbine_flows,
            "Spill_Discharge_m3s": optimal_spill_flows,
            "Inflow_m3s": inflows[: len(optimal_levels)],
        }
    )

    monthly_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved ISO upstream DP results: {csv_path.resolve()}")
    return csv_path


class ISOPreprocessor:
    """
    Prepare and export upstream DP training trajectories for ISO-SURF fitting.
    """

    def __init__(self):
        self.data_dir = get_data_dir()
        self.output_dir = get_policy_generation_dir("iso_surf") / "preprocessed"

        self.train_inflow_file = self.data_dir / "SynMonthlyInflow_train.xlsx"
        self.penalty_settings_file = self.data_dir / "penalty_settings.yaml"

        self.inflows = load_monthly_inflows_from_excel(self.train_inflow_file).values
        self.penalty_lambdas = load_penalty_candidates(self.penalty_settings_file, "iso_surf")

        self.reservoir = SharedReservoirModel(data_dir=self.data_dir)

    def run_one(self, lam: float) -> Path:
        """
        Run the shared DP solver for one ISO-SURF lambda and save the upstream
        DP monthly trajectory table.
        """
        dp_solver = DPSolver(reservoir=self.reservoir, inflows=self.inflows)

        (
            value_function,
            decisions,
            optimal_levels,
            optimal_storages,
            optimal_powers,
            optimal_electricity,
            optimal_discharges,
            optimal_turbine_flows,
            optimal_spill_flows,
            inflows_flat,
        ) = dp_solver.optimize(firm_power_penalty_lambda=float(lam))

        return save_iso_upstream_dp_results(
            result_dir=self.output_dir,
            lam=float(lam),
            optimal_levels=optimal_levels,
            optimal_storages=optimal_storages,
            optimal_powers=optimal_powers,
            optimal_electricity=optimal_electricity,
            optimal_discharges=optimal_discharges,
            optimal_turbine_flows=optimal_turbine_flows,
            optimal_spill_flows=optimal_spill_flows,
            inflows=inflows_flat,
        )

    def run_all(self) -> None:
        """
        Generate and save upstream DP training trajectories for all ISO-SURF lambdas.
        """
        print("\n============================================================")
        print("ISO-SURF preprocessing started")
        print("============================================================")
        print(f"Training inflow file: {self.train_inflow_file}")
        print(f"Penalty settings file: {self.penalty_settings_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"Penalty lambdas: {self.penalty_lambdas}")
        print("============================================================")

        for lam in self.penalty_lambdas:
            print("\n------------------------------------------------------------")
            print(f"Generating ISO upstream DP data for lambda = {lam}")
            print("------------------------------------------------------------")
            self.run_one(float(lam))


def main_iso_preprocessor() -> None:
    preprocessor = ISOPreprocessor()
    preprocessor.run_all()


if __name__ == "__main__":
    main_iso_preprocessor()
