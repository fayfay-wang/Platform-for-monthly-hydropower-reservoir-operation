"""
SDP evaluation module for the unified reservoir operation framework.

This module evaluates previously generated SDP rule tables under the shared
reservoir definition, including:
1) loading the synthetic evaluation inflow series,
2) loading SDP rule tables from Excel files,
3) running month-by-month closed-loop simulation using the rule table,
4) computing reliability statistics, and
5) exporting standardized SDP evaluation results and diagnostic figures.

The purpose of this module is to keep SDP evaluation independent from SDP
policy generation, so that evaluation only consumes saved rule tables.
"""

from __future__ import annotations

import time
from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd

from reservoir_benchmark.shared.inflow_forecast import (
    ForecastScenario,
    forecast_evaluation_dir,
    load_forecast_penalty_candidates,
)
from reservoir_benchmark.shared.io_utils import (
    load_monthly_inflows_from_excel,
    load_penalty_candidates,
)
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_outputs_dir,
    get_policy_generation_dir,
)
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


class SDPReservoirSimulator:
    """
    Closed-loop simulator using previously generated SDP rule tables.
    """

    class DecisionLookupTable:
        """
        Efficient month-level / water-level / inflow lookup table for SDP rules.
        """

        def __init__(self, excel_path: str | Path, forecast: ForecastScenario | None = None):
            self.forecast = forecast
            self.tables = {}
            self.water_levels = None
            self.inflow_intervals = {}

            with pd.ExcelFile(excel_path) as workbook:
                for month in range(1, 13):
                    df = workbook.parse(sheet_name=f"{month}月", header=None)
                    forecast_bounds = None
                    if forecast is None:
                        inflow_bounds = df.iloc[0, 1:].to_numpy(dtype=float)
                        water_levels = np.array([float(str(level).replace("m", "")) for level in df.iloc[1:, 0]])
                        decision_data = df.iloc[1:, 1:].to_numpy(dtype=float)
                    else:
                        water_levels = df.iloc[1:, 0].drop_duplicates().to_numpy(dtype=float)
                        n_inflows = (len(df) - 1) // len(water_levels)
                        inflow_bounds = df.iloc[1:1 + n_inflows, 1].to_numpy(dtype=float)
                        decision_data = df.iloc[1:, 2:].to_numpy(dtype=float)
                        if forecast.mode == "none":
                            decision_data = decision_data.reshape(len(water_levels), n_inflows)
                        else:
                            forecast_bounds = df.iloc[0, 2:].to_numpy(dtype=float)
                            decision_data = decision_data.reshape(len(water_levels), n_inflows, len(forecast_bounds))

                    self.water_levels = water_levels
                    self.inflow_intervals[month] = inflow_bounds
                    self.tables[month] = {
                        "inflow_bounds": inflow_bounds,
                        "water_levels": water_levels,
                        "forecast_bounds": forecast_bounds,
                        "decisions": decision_data,
                    }

        def find_decision_level(
            self,
            month: int,
            current_level: float,
            current_inflow: float,
            next_month_forecast: float | None = None,
        ) -> float:
            """
            Find the target water level for a given month, current level, and inflow.
            """
            table = self.tables[month]

            inflow_bounds = table["inflow_bounds"]

            # Find inflow interval index
            if self.forecast is None and current_inflow <= inflow_bounds[0]:
                inflow_idx = 0
            else:
                inflow_idx = bisect_right(inflow_bounds, current_inflow)
                if inflow_idx >= len(inflow_bounds):
                    inflow_idx = len(inflow_bounds) - 1

            # Find water-level row index
            water_levels = table["water_levels"]
            level_idx = bisect_right(water_levels, current_level)
            if level_idx > 0:
                level_idx -= 1

            level_idx = max(0, min(level_idx, len(water_levels) - 1))

            target_level = table["decisions"][level_idx, inflow_idx]
            forecast_bounds = table["forecast_bounds"]
            if forecast_bounds is not None:
                if next_month_forecast is None:
                    raise ValueError("This SDP rule requires a one-month-ahead inflow forecast.")
                forecast_idx = min(bisect_right(forecast_bounds, next_month_forecast), len(forecast_bounds) - 1)
                target_level = target_level[forecast_idx]
            return float(target_level)

        def get_info(self) -> dict:
            """
            Return summary information about the SDP decision table.
            """
            info = {
                "months": len(self.tables),
                "water_levels_range": (self.water_levels[0], self.water_levels[-1]),
                "num_water_levels": len(self.water_levels),
            }

            for month in range(1, 13):
                inflow_bounds = self.tables[month]["inflow_bounds"]
                info[f"month_{month}_inflow_range"] = (inflow_bounds[0], inflow_bounds[-1])

            return info

    def __init__(
        self,
        lam_value: float,
        forecast: ForecastScenario | None = None,
        policy_forecast: ForecastScenario | None = None,
    ):
        self.data_dir = get_data_dir()
        self.forecast = forecast
        self.policy_forecast = policy_forecast or forecast
        self.policy_dir = get_policy_generation_dir("sdp") / "policy"
        if self.policy_forecast is not None:
            self.policy_dir /= self.policy_forecast.name
        self.output_dir = forecast_evaluation_dir(
            get_outputs_dir() / "evaluation" / "sdp", forecast, self.policy_forecast
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_inflow_file = self.data_dir / "SynMonthlyInflow_evaluation.xlsx"
        self.penalty_settings_file = self.data_dir / "penalty_settings.yaml"

        self.reservoir_model = SharedReservoirModel(data_dir=self.data_dir)

        # Shared parameters from SharedReservoirModel
        self.dead_level = self.reservoir_model.dead_level
        self.normal_level = self.reservoir_model.normal_level
        self.initial_level = self.reservoir_model.initial_level
        self.max_turbine_discharge = self.reservoir_model.max_turbine_discharge
        self.max_power = self.reservoir_model.max_power
        self.firm_power = self.reservoir_model.firm_power
        self.power_coefficient = self.reservoir_model.power_coefficient
        self.delta_t_days = self.reservoir_model.delta_t_days

        self.lam_value = float(lam_value)
        self.lam_str = format_lambda_for_filename(self.lam_value)

        self.decision_table_path = self.policy_dir / f"sdp_train_lambda_{self.lam_str}.xlsx"
        self.csv_result_filename = f"sdp_eval_lambda_{self.lam_str}.csv"
        self.output_tag = f"lambda_{self.lam_str}"

        self.lookup_table = None

    def simulate_operation(
        self,
        inflows: np.ndarray,
        initial_level: float | None = None,
        forecasts: np.ndarray | None = None,
    ):
        """
        Simulate one year of reservoir operation using the SDP decision table.
        The initial level should be passed from the previous year's ending level
        during multi-year evaluation.
        """

        current_level = self.initial_level if initial_level is None else float(initial_level)

        monthly_levels = [current_level]
        monthly_energy = []
        monthly_power = []
        monthly_spill = []
        monthly_turbine_discharge = []
        monthly_total_discharge = []

        for month, monthly_inflow in enumerate(inflows[:12]):
            inflow = float(monthly_inflow)
            target_level = self.lookup_table.find_decision_level(
                month + 1, current_level, inflow, None if forecasts is None else forecasts[month]
            )

            current_storage = self.reservoir_model.calculate_storage(current_level)
            target_storage = self.reservoir_model.calculate_storage(target_level)

            total_discharge = (
                    (current_storage - target_storage) * 1e9
                    / (self.delta_t_days * 24.0 * 60.0 * 60.0)
                    + inflow
            )

            if total_discharge < 0:
                adjusted_target_storage = (
                        current_storage
                        + inflow * self.delta_t_days * 24.0 * 60.0 * 60.0 / 1e9
                )
                target_level = self.reservoir_model.calculate_level(adjusted_target_storage)
                turbine_discharge = 0.0
                spill_discharge = 0.0
                power_kw = 0.0
                total_discharge = 0.0
            else:
                turbine_discharge = total_discharge
                spill_discharge = 0.0

                power_kw = self.reservoir_model.calculate_power(
                    turbine_discharge,
                    total_discharge,
                    current_level,
                    target_level,
                )

                if power_kw > self.max_power:
                    average_upstream_level = (target_level + current_level) / 2.0
                    tailwater = self.reservoir_model.calculate_tailwater(total_discharge)
                    average_head = max(average_upstream_level - tailwater, 0.01)

                    turbine_discharge = self.max_power / (
                            self.power_coefficient * average_head
                    )
                    spill_discharge = max(total_discharge - turbine_discharge, 0.0)

                    power_kw = self.reservoir_model.calculate_power(
                        turbine_discharge,
                        total_discharge,
                        current_level,
                        target_level,
                    )

                if turbine_discharge > self.max_turbine_discharge:
                    turbine_discharge = self.max_turbine_discharge
                    spill_discharge = max(total_discharge - turbine_discharge, 0.0)

                    power_kw = self.reservoir_model.calculate_power(
                        turbine_discharge,
                        total_discharge,
                        current_level,
                        target_level,
                    )

            energy_gwh = power_kw * self.delta_t_days * 24.0 / 1e6

            monthly_power.append(power_kw)
            monthly_energy.append(energy_gwh)
            monthly_turbine_discharge.append(turbine_discharge)
            monthly_spill.append(spill_discharge)
            monthly_total_discharge.append(total_discharge)

            monthly_levels.append(target_level)
            current_level = target_level

        return (
            monthly_levels,
            monthly_energy,
            monthly_power,
            monthly_spill,
            monthly_turbine_discharge,
            monthly_total_discharge,
            current_level,
        )

    def compute_reliability_from_csv(self, csv_path: str | Path):
        """
        Compute time-based firm-power reliability from a saved CSV result.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"[Reliability] Result file not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        if "Power_kW" not in df.columns:
            print("[Reliability] Missing 'Power_kW' column in result CSV.")
            return

        power_series = df["Power_kW"].values
        total_months = len(power_series)
        satisfied_months = int(np.sum(power_series >= self.firm_power))
        time_reliability = np.mean(power_series >= self.firm_power)

        print("\n===== SDP Firm-Power Reliability Statistics =====")
        print(f"Firm power threshold: {self.firm_power:.0f} kW")
        print(f"Months meeting firm power / total months: {satisfied_months}/{total_months}")
        print(f"Time-based reliability: {time_reliability * 100:.2f}%")

    def run_simulation(self):
        """
        Main SDP evaluation workflow.
        """
        start_t = time.perf_counter()

        if not self.decision_table_path.exists():
            raise FileNotFoundError(f"SDP rule table not found: {self.decision_table_path}")

        print("\n============================================================")
        print("Current SDP evaluation configuration")
        print("============================================================")
        print(f"Evaluation inflow file: {self.evaluation_inflow_file}")
        print(f"Rule table file: {self.decision_table_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"CSV result file: {self.csv_result_filename}")
        print("============================================================\n")

        print("Loading evaluation inflow data...")
        inflows = load_monthly_inflows_from_excel(self.evaluation_inflow_file).values
        forecasts = None
        if self.forecast is not None:
            forecasts = self.forecast.next_month(inflows, self.forecast.evaluation_seed)

        print("Initializing SDP decision lookup table...")
        self.lookup_table = self.DecisionLookupTable(self.decision_table_path, self.policy_forecast)

        info = self.lookup_table.get_info()
        print("\nDecision table summary:")
        print(f"  Number of months: {info['months']}")
        print(f"  Water-level range: {info['water_levels_range'][0]:.0f} m - {info['water_levels_range'][1]:.0f} m")
        print(f"  Number of water-level points: {info['num_water_levels']}")
        print(
            f"  January inflow range: "
            f"{info['month_1_inflow_range'][0]:.1f} - {info['month_1_inflow_range'][1]:.1f} m³/s"
        )

        print("\nStarting evaluation simulation...\n")
        all_monthly_levels = [self.initial_level]
        all_monthly_energy = []
        all_monthly_power = []
        all_monthly_spill = []
        all_monthly_turbine_discharge = []
        all_monthly_total_discharge = []

        years = len(inflows)
        current_level = self.initial_level

        for year in range(years):
            print(f"Simulating year {year + 1}...")
            year_inflows = inflows[year]
            if self.forecast is not None and year == years - 1:
                year_inflows = year_inflows[:-1]

            (
                monthly_levels,
                monthly_energy,
                monthly_power,
                monthly_spill,
                monthly_turbine_discharge,
                monthly_total_discharge,
                current_level,
            ) = self.simulate_operation(
                year_inflows,
                initial_level=current_level,
                forecasts=None if forecasts is None else forecasts[year],
            )

            all_monthly_levels.extend(monthly_levels[1:])
            all_monthly_energy.extend(monthly_energy)
            all_monthly_power.extend(monthly_power)
            all_monthly_spill.extend(monthly_spill)
            all_monthly_turbine_discharge.extend(monthly_turbine_discharge)
            all_monthly_total_discharge.extend(monthly_total_discharge)

            year_energy = sum(monthly_energy)
            avg_power = sum(monthly_power) / len(monthly_power)
            year_spill = sum(monthly_spill)
            period = "Annual" if len(monthly_power) == 12 else "Partial-year"
            print(
                f"  {period} electricity: {year_energy:.2f} GWh, "
                f"mean power: {avg_power / 1000:.2f} MW, "
                f"spilled water: {year_spill:.2f} m³/s"
            )

        print("\nGenerating evaluation CSV...")

        inflow_flat = inflows.reshape(-1)
        n = len(all_monthly_power)

        sdp_results = pd.DataFrame(
            {
                "Month": range(1, n + 1),
                "Water_Level_m": all_monthly_levels[1: n + 1],
                "Power_kW": all_monthly_power,
                "Electricity_GWh": all_monthly_energy,
                "Total_Discharge_m3s": all_monthly_total_discharge,
                "Turbine_Discharge_m3s": all_monthly_turbine_discharge,
                "Spill_Discharge_m3s": all_monthly_spill,
                "Inflow_m3s": inflow_flat[:n],
            }
        )

        csv_path = self.output_dir / self.csv_result_filename
        sdp_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved SDP evaluation CSV: {csv_path}")

        power_vec = np.asarray(all_monthly_power, dtype=float)
        total_months = len(power_vec)
        satisfied_months = int(np.sum(power_vec >= self.firm_power))
        guarantee_rate = (satisfied_months / total_months) if total_months > 0 else 0.0

        print("\n===== Current simulation statistics =====")
        print(f"Months meeting firm power / total months: {satisfied_months}/{total_months}")
        print(f"Time-based reliability: {guarantee_rate * 100:.2f}%")

        self.compute_reliability_from_csv(csv_path)

        end_t = time.perf_counter()
        print("\n===== Runtime statistics =====")
        print(f"Elapsed time: {end_t - start_t:.3f} seconds")


def main_sdp_eval(
    forecast: ForecastScenario | None = None,
    policy_forecast: ForecastScenario | None = None,
) -> None:
    """
    Main entry point for SDP evaluation.
    """
    data_dir = get_data_dir()
    penalty_settings_file = data_dir / "penalty_settings.yaml"
    if forecast is None:
        penalty_lambdas = load_penalty_candidates(penalty_settings_file, "sdp")
    else:
        penalty_lambdas = load_forecast_penalty_candidates("sdp")

    print("\n============================================================")
    print("SDP evaluation started")
    print("============================================================")
    print(f"Evaluation inflow file: {data_dir / 'SynMonthlyInflow_evaluation.xlsx'}")
    print(f"Penalty settings file: {penalty_settings_file}")
    print(f"Penalty lambdas: {penalty_lambdas}")
    print("============================================================")

    for lam in penalty_lambdas:
        print("\n------------------------------------------------------------")
        print(f"Evaluating SDP rule table for lambda = {lam}")
        print("------------------------------------------------------------")

        simulator = SDPReservoirSimulator(
            lam_value=float(lam), forecast=forecast, policy_forecast=policy_forecast
        )
        simulator.run_simulation()


if __name__ == "__main__":
    main_sdp_eval()
