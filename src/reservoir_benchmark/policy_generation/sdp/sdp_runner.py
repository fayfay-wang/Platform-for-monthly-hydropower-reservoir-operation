
"""
SDP execution and rule-export module for the unified reservoir operation
framework.

This module connects the stochastic dynamic programming (SDP) solver to the
shared project infrastructure, including:
1) generating discretized inflow intervals from the synthetic training inflow,
2) loading penalty coefficient candidates for SDP runs,
3) constructing the shared reservoir object,
4) executing the SDP solver for each specified penalty coefficient, and
5) exporting trained SDP rule tables to the standardized output directory.

The purpose of this module is to separate experiment orchestration from the SDP
solver itself, thereby keeping the optimization logic independent from path
management, shared input loading, parameter looping, and result export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_benchmark.policy_generation.sdp.sdp_preprocessor import (
    SDPInflowDiscretizer,
    estimate_inflow_observations,
)
from reservoir_benchmark.policy_generation.sdp.sdp_solver import ReservoirOperation
from reservoir_benchmark.shared.inflow_forecast import (
    ForecastScenario,
    load_forecast_penalty_candidates,
)
from reservoir_benchmark.shared.io_utils import (
    load_algorithm_settings,
    load_monthly_inflows_from_excel,
    load_penalty_candidates,
    load_sdp_inflow_intervals,
    require_keys,
)
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_policy_generation_dir,
)
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


def export_operation_rules(
    decisions,
    levels,
    flow_intervals,
    n_flows: int,
    output_dir: Path,
    lam: float,
) -> Path:
    """
    Export the SDP operation rule table to Excel.

    The Excel file is saved under:
        outputs/policy_generation/sdp/policy/

    File name pattern:
        sdp_train_lambda_{lambda}.xlsx
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lam_str = format_lambda_for_filename(lam)
    filepath = output_dir / f"sdp_train_lambda_{lam_str}.xlsx"

    with pd.ExcelWriter(filepath) as writer:
        for month in range(12):
            columns = []
            for flow_idx in range(n_flows):
                lower, upper = flow_intervals.loc[month + 1, flow_idx]
                columns.append(f"{lower:.1f}")
                if flow_idx < n_flows - 1:
                    columns.append(f"{upper:.1f}")
            _, last_upper = flow_intervals.loc[month + 1, n_flows - 1]
            columns.append(f"{last_upper:.1f}")

            data = []
            for level_idx in range(len(levels)):
                row = []
                for flow_idx in range(n_flows):
                    decision = decisions[month, level_idx, flow_idx]
                    row.append(decision)
                    if flow_idx < n_flows - 1:
                        row.append(decision)
                row.append(decisions[month, level_idx, n_flows - 1])
                data.append(row)

            rule_df = pd.DataFrame(
                data,
                index=[f"{h:.0f}m" for h in levels],
                columns=columns,
            )
            rule_df.to_excel(writer, sheet_name=f"{month + 1}月")

    print(f"Operation rule table saved to: {filepath}")
    return filepath


def _export_forecast_rules(
    decisions: np.ndarray,
    levels: np.ndarray,
    forecast_bounds: np.ndarray | None,
    output_dir: Path,
    lam: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"sdp_train_lambda_{format_lambda_for_filename(lam)}.xlsx"

    with pd.ExcelWriter(filepath) as writer:
        for month in range(12):
            if forecast_bounds is None:
                columns = ["Target water level (m)"]
            else:
                columns = forecast_bounds[month]
            index = pd.Index(levels, name="Water level (m)")
            rule_df = pd.DataFrame(decisions[month], index=index, columns=columns)
            rule_df.to_excel(writer, sheet_name=f"{month + 1}月")

    print(f"Operation rule table saved to: {filepath}")
    return filepath


def main_sdp(forecast: ForecastScenario | None = None) -> None:
    """
    Main entry point for SDP training.
    """
    data_dir = get_data_dir()
    output_dir = get_policy_generation_dir("sdp") / "policy"

    train_inflow_file = data_dir / "SynMonthlyInflow_train.xlsx"
    penalty_settings_file = data_dir / "penalty_settings.yaml"
    experiment_settings_file = data_dir / "experiment_settings.yaml"

    sdp_settings = load_algorithm_settings(experiment_settings_file, "sdp")
    require_keys(
        sdp_settings,
        ["n_inflow_classes", "threshold"],
        "sdp",
    )

    n_inflow_discrete = int(sdp_settings["n_inflow_classes"])
    threshold = float(sdp_settings["threshold"])

    if forecast is not None:
        train_inflows = load_monthly_inflows_from_excel(train_inflow_file).to_numpy(dtype=float)
        inflows, forecast_bounds, inflow_probs, forecast_probs = estimate_inflow_observations(
            train_inflows,
            n_inflow_discrete,
            forecast,
            n_forecast_classes=int(sdp_settings.get("n_forecast_classes", 5)),
        )
        reservoir_model = SharedReservoirModel(data_dir=data_dir)
        for lam in load_forecast_penalty_candidates("sdp"):
            print(f"\nTraining SDP for {forecast.name}, lambda = {lam}")
            solver = ReservoirOperation(reservoir=reservoir_model, inflows=inflows)
            solver.config["threshold"] = threshold
            solver.config["firm_power_penalty_lambda"] = lam
            decisions = solver.optimize_with_forecast(inflow_probs, forecast_probs)
            _export_forecast_rules(
                decisions,
                solver.levels,
                forecast_bounds,
                output_dir / "current_month" / forecast.name,
                lam,
            )
        return

    # Step 1: generate SDP inflow discretization file automatically
    discretizer = SDPInflowDiscretizer(
        n_inflow_classes=n_inflow_discrete,
    )
    inflow_intervals_file = discretizer.generate()

    # Step 2: load inputs for SDP training
    penalty_lambdas = load_penalty_candidates(penalty_settings_file, "sdp")
    inflows, flow_probs, flow_intervals = load_sdp_inflow_intervals(
        inflow_intervals_file,
        n_inflow_discrete=n_inflow_discrete,
    )
    reservoir_model = SharedReservoirModel(data_dir=data_dir)

    print("\n============================================================")
    print("SDP training started")
    print("============================================================")
    print(f"Training inflow file: {train_inflow_file}")
    print(f"Inflow interval file: {inflow_intervals_file}")
    print(f"Penalty settings file: {penalty_settings_file}")
    print(f"Output directory: {output_dir}")
    print(f"Penalty lambdas: {penalty_lambdas}")
    print("============================================================")

    for lam in penalty_lambdas:
        print("\n============================================================")
        print(f"Starting SDP training with firm_power_penalty_lambda = {lam}")
        print("============================================================")

        solver = ReservoirOperation(
            reservoir=reservoir_model,
            inflows=inflows,
            flow_probs=flow_probs,
            flow_intervals=flow_intervals,
        )
        solver.config["threshold"] = threshold
        solver.config["firm_power_penalty_lambda"] = float(lam)

        expected_total_energy, optimal_decisions, annual_energy, history = solver.optimize()

        export_operation_rules(
            decisions=optimal_decisions,
            levels=solver.levels,
            flow_intervals=solver.flow_intervals,
            n_flows=solver.n_flows,
            output_dir=output_dir,
            lam=lam,
        )

        print("\n============================================================")
        print("SDP training completed.")
        if annual_energy:
            print(f"Final expected annual electricity: {annual_energy[-1] / 1e6:.2f} GWh")
            print(f"Number of iterations: {len(annual_energy)}")
        print(f"Current lambda: {lam}")
        print(f"Training inflow file: {train_inflow_file}")
        print(f"Inflow interval file: {inflow_intervals_file}")
        print("============================================================")


if __name__ == "__main__":
    main_sdp()
