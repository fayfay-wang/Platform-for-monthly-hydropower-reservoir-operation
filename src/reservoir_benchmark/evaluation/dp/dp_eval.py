"""
DP execution and result-export module for the unified reservoir operation
framework.

This module connects the deterministic dynamic programming (DP) solver to the
shared project infrastructure, including:
1) loading training inflow data,
2) loading penalty coefficient candidates for DP runs,
3) constructing the shared reservoir object,
4) executing the DP solver for each specified penalty coefficient, and
5) exporting monthly optimization results to the standardized output directory.

The purpose of this module is to separate experiment orchestration from the DP
solver itself, thereby keeping the optimization logic independent from path
management, data loading, parameter looping, and result saving.
"""

from pathlib import Path

import pandas as pd

from reservoir_benchmark.policy_generation.dp.dp_solver import ReservoirOperation
from reservoir_benchmark.shared.io_utils import (
    load_monthly_inflows_from_excel,
    load_penalty_candidates,
)
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_outputs_dir,
)
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


def save_dp_results(
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
) -> None:
    """
    Save monthly DP results as a single CSV file.

    File name pattern:
        dp_eval_lambda_{lam}.csv
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    lam_str = format_lambda_for_filename(lam)
    csv_filename = f"dp_eval_lambda_{lam_str}.csv"
    csv_path = result_dir / csv_filename

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
    print(f"Saved DP results: {csv_path.resolve()}")


def main_dp_eval() -> None:
    """
    Run DP for all lambda values defined in data/penalty_settings.yaml.
    """
    data_dir = get_data_dir()
    result_dir = get_outputs_dir() / "evaluation" / "dp"

    inflow_path = data_dir / "SynMonthlyInflow_evaluation.xlsx"
    penalty_settings_path = data_dir / "penalty_settings.yaml"

    inflows_df = load_monthly_inflows_from_excel(inflow_path)
    inflows = inflows_df.values

    penalty_lambdas = load_penalty_candidates(penalty_settings_path, "dp")

    reservoir = SharedReservoirModel(data_dir=data_dir)
    dp_solver = ReservoirOperation(reservoir=reservoir, inflows=inflows)

    print("\n====================================")
    print("DP evaluation started")
    print(f"Inflow file: {inflow_path}")
    print(f"Penalty settings file: {penalty_settings_path}")
    print(f"Output directory: {result_dir}")
    print(f"Penalty lambdas: {penalty_lambdas}")
    print("====================================")

    for lam in penalty_lambdas:
        print("\n------------------------------------")
        print(f"Running firm_power_penalty_lambda = {lam}")
        print("------------------------------------")

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
        ) = dp_solver.optimize(firm_power_penalty_lambda=lam)

        save_dp_results(
            result_dir=result_dir,
            lam=lam,
            optimal_levels=optimal_levels,
            optimal_storages=optimal_storages,
            optimal_powers=optimal_powers,
            optimal_electricity=optimal_electricity,
            optimal_discharges=optimal_discharges,
            optimal_turbine_flows=optimal_turbine_flows,
            optimal_spill_flows=optimal_spill_flows,
            inflows=inflows_flat,
        )
