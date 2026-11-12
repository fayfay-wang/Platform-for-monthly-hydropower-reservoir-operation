"""
ISO-SURF evaluation module for the unified reservoir operation framework.

This module evaluates previously saved ISO-SURF policies under the unified
reservoir definition, including:
1) loading fitted ISO-SURF policy PKL artifacts,
2) running closed-loop simulation on the synthetic evaluation inflow series,
3) computing reliability and average annual electricity, and
4) exporting standardized ISO-SURF evaluation results.

The purpose of this module is to keep ISO-SURF evaluation independent from
upstream DP preprocessing and policy fitting, so that evaluation only consumes
saved policy artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservoir_benchmark.policy_generation.iso_surf.iso_surf_runner import ISOSurfPolicyFitter
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


class ISOReservoirSimulator:
    """
    Closed-loop simulator using fitted ISO-SURF policy surfaces.
    """

    def __init__(
        self, forecast: ForecastScenario | None = None, policy_forecast: ForecastScenario | None = None
    ):
        self.data_dir = get_data_dir()
        self.forecast = forecast
        self.policy_dir = get_policy_generation_dir("iso_surf") / "policy"
        policy_forecast = policy_forecast or forecast
        if policy_forecast is not None:
            self.policy_dir = self.policy_dir / "current_month" / policy_forecast.name
        self.output_dir = forecast_evaluation_dir(
            get_outputs_dir() / "evaluation" / "iso_surf", forecast, policy_forecast
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_inflow_file = self.data_dir / "SynMonthlyInflow_evaluation.xlsx"
        self.penalty_settings_file = self.data_dir / "penalty_settings.yaml"
        if forecast is None:
            self.penalty_lambdas = load_penalty_candidates(self.penalty_settings_file, "iso_surf")
        else:
            self.penalty_lambdas = load_forecast_penalty_candidates("iso_surf")

        self.reservoir = SharedReservoirModel(data_dir=self.data_dir)

        self.dead_level = self.reservoir.dead_level
        self.normal_level = self.reservoir.normal_level
        self.initial_level = self.reservoir.initial_level
        self.max_turbine_discharge = self.reservoir.max_turbine_discharge
        self.max_power = self.reservoir.max_power
        self.firm_power = self.reservoir.firm_power
        self.power_coefficient = self.reservoir.power_coefficient
        self.delta_t_days = self.reservoir.delta_t_days

    def simulate_one_year(
        self, year_inflows, initial_level: float, policy: ISOSurfPolicyFitter, year_forecasts=None
    ):
        """
        Simulate one year of operation using the fitted ISO-SURF policy.
        """
        monthly_levels = [initial_level]
        monthly_energy = []
        monthly_power = []
        monthly_spill = []
        monthly_turbine_discharge = []
        monthly_total_discharge = []

        current_level = initial_level

        for month, month_inflow in enumerate(year_inflows):
            inflow = float(month_inflow)
            month_of_year = month + 1
            policy_inflow = inflow if self.forecast is None else None
            if year_forecasts is not None:
                policy_inflow = float(year_forecasts[month])

            target_level = policy.predict_next_level(month_of_year, current_level, policy_inflow)
            target_level = max(self.dead_level, min(self.normal_level, target_level))

            current_storage = self.reservoir.calculate_storage(current_level)
            target_storage = self.reservoir.calculate_storage(target_level)

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
                target_level = self.reservoir.calculate_level(adjusted_target_storage)
                turbine_discharge = 0.0
                spill_discharge = 0.0
                power_kw = 0.0
                total_discharge = 0.0
            else:
                turbine_discharge = total_discharge
                spill_discharge = 0.0

                power_kw = self.reservoir.calculate_power(
                    turbine_discharge,
                    total_discharge,
                    current_level,
                    target_level,
                )

                if power_kw > self.max_power:
                    average_upstream_level = (target_level + current_level) / 2.0
                    tailwater = self.reservoir.calculate_tailwater(total_discharge)
                    average_head = max(average_upstream_level - tailwater, 0.01)

                    turbine_discharge = self.max_power / (self.power_coefficient * average_head)
                    spill_discharge = max(total_discharge - turbine_discharge, 0.0)

                    power_kw = self.reservoir.calculate_power(
                        turbine_discharge,
                        total_discharge,
                        current_level,
                        target_level,
                    )

                if turbine_discharge > self.max_turbine_discharge:
                    turbine_discharge = self.max_turbine_discharge
                    spill_discharge = max(total_discharge - turbine_discharge, 0.0)

                    power_kw = self.reservoir.calculate_power(
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
        )

    def run_simulation(self, policy: ISOSurfPolicyFitter):
        """
        Run ISO-SURF evaluation using the synthetic evaluation inflow series.
        """
        inflows_data = load_monthly_inflows_from_excel(self.evaluation_inflow_file)
        inflows = inflows_data.values
        years = len(inflows)
        forecasts = None
        if self.forecast is not None:
            forecasts = self.forecast.current_month(inflows, self.forecast.evaluation_seed)

        all_monthly_levels = [self.initial_level]
        all_monthly_energy = []
        all_monthly_power = []
        all_monthly_spill = []
        all_monthly_turbine_discharge = []
        all_monthly_total_discharge = []

        annual_energy_list = []

        for year in range(years):
            year_inflows = inflows[year]
            year_forecasts = None if forecasts is None else forecasts[year]

            if year == 0:
                initial_level = self.initial_level
            else:
                initial_level = all_monthly_levels[-1]

            (
                monthly_levels,
                monthly_energy,
                monthly_power,
                monthly_spill,
                monthly_turbine_discharge,
                monthly_total_discharge,
            ) = self.simulate_one_year(year_inflows, initial_level, policy, year_forecasts)

            all_monthly_levels.extend(monthly_levels[1:])
            all_monthly_energy.extend(monthly_energy)
            all_monthly_power.extend(monthly_power)
            all_monthly_spill.extend(monthly_spill)
            all_monthly_turbine_discharge.extend(monthly_turbine_discharge)
            all_monthly_total_discharge.extend(monthly_total_discharge)

            annual_energy_list.append(float(np.sum(monthly_energy)))

        inflow_flat = inflows.reshape(-1)
        n = len(all_monthly_power)

        df_iso = pd.DataFrame(
            {
                "Month": range(1, n + 1),
                "Water_Level_m": all_monthly_levels[1 : n + 1],
                "Power_kW": all_monthly_power,
                "Electricity_GWh": all_monthly_energy,
                "Total_Discharge_m3s": all_monthly_total_discharge,
                "Turbine_Discharge_m3s": all_monthly_turbine_discharge,
                "Spill_Discharge_m3s": all_monthly_spill,
                "Inflow_m3s": inflow_flat[:n],
            }
        )

        power_series = np.asarray(all_monthly_power, dtype=float)
        reliability = float(np.mean(power_series >= self.firm_power)) if len(power_series) > 0 else None
        avg_annual_energy = float(np.mean(annual_energy_list)) if len(annual_energy_list) > 0 else 0.0
        return df_iso, avg_annual_energy, reliability

    def save_results(self, df_iso: pd.DataFrame, lam: float) -> Path:
        """
        Save ISO-SURF evaluation results to CSV.
        """
        lam_str = format_lambda_for_filename(lam)
        csv_path = self.output_dir / f"iso_surf_eval_lambda_{lam_str}.csv"
        df_iso.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"ISO-SURF evaluation results saved to: {csv_path}")
        return csv_path

    def run_all(self) -> None:
        """
        Load existing ISO-SURF policies and evaluate them on the evaluation inflow series.
        """
        print("\n============================================================")
        print("ISO-SURF evaluation started")
        print("============================================================")
        print(f"Evaluation inflow file: {self.evaluation_inflow_file}")
        print(f"Penalty settings file: {self.penalty_settings_file}")
        print(f"Policy directory: {self.policy_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Penalty lambdas: {self.penalty_lambdas}")
        print("============================================================")

        for lam in self.penalty_lambdas:
            lam_str = format_lambda_for_filename(lam)
            policy_path = self.policy_dir / f"iso_surf_train_lambda_{lam_str}.pkl"

            if not policy_path.exists():
                raise FileNotFoundError(
                    f"Required ISO-SURF policy file not found: {policy_path}"
                )

            print("\n------------------------------------------------------------")
            print(f"Evaluating ISO-SURF policy for lambda = {lam}")
            print("------------------------------------------------------------")

            policy = ISOSurfPolicyFitter.load_policy(policy_path)
            df_iso, avg_annual_energy, reliability = self.run_simulation(policy)
            self.save_results(df_iso, lam)

            if reliability is None:
                print(
                    f"lam={lam_str}  reliability=NA  "
                    f"avg annual electricity={avg_annual_energy:.2f} GWh/year"
                )
            else:
                print(
                    f"lam={lam_str}  reliability={reliability * 100:.2f}%  "
                    f"avg annual electricity={avg_annual_energy:.2f} GWh/year"
                )


def main_iso_surf_eval(
    forecast: ForecastScenario | None = None, policy_forecast: ForecastScenario | None = None
) -> None:
    simulator = ISOReservoirSimulator(forecast=forecast, policy_forecast=policy_forecast)
    simulator.run_all()


if __name__ == "__main__":
    main_iso_surf_eval()
