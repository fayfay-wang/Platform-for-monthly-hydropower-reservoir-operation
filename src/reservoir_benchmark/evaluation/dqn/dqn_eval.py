"""
DQN evaluation module for the unified reservoir operation framework.

This module evaluates previously trained DQN policies, including:
1) loading lambda candidates from penalty_settings.yaml,
2) loading the corresponding trained model artifact,
3) running closed-loop evaluation on the synthetic evaluation inflow series, and
4) exporting standardized evaluation CSV files.

The purpose of this module is to keep DQN evaluation independent from policy
generation and aligned with the evaluation structure used by DP, SDP, and
ISO-SURF.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservoir_benchmark.policy_generation.dqn.dqn_reservoir import ReservoirEnv
from reservoir_benchmark.policy_generation.dqn.dqn_agent import (
    Config,
    BasicInfo,
    DQNAgent,
    set_global_seed,
)
from reservoir_benchmark.shared.io_utils import load_penalty_candidates
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_policy_generation_dir,
    get_outputs_dir,
)


class ReservoirEvaluator:
    """
    DQN evaluation helper.
    """

    def __init__(self, env, agent: DQNAgent, output_dir: Path):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detailed_records = []

    def run_evaluation(
        self,
        model_path,
        num_episodes,
        start_year=0,
        end_year=None,
        output_csv_name: str = "dqn_eval_lambda.csv",
    ):
        """
        Run DQN evaluation and export a single CSV result file.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return False

        if not self.agent.load_model(model_path):
            return False

        print("\nStarting evaluation...")

        data = self.env.train_data if self.env.is_training else self.env.test_data
        max_years = len(data)
        end_year = end_year if end_year is not None else max_years - 1
        start_year = max(0, min(start_year, max_years - 1))
        end_year = max(start_year, min(end_year, max_years - 1))
        num_years = end_year - start_year + 1

        print(
            f"Evaluation year range: "
            f"{start_year + 1} - {end_year + 1} (total {num_years} years)"
        )

        self.detailed_records = []

        self.env.max_steps = num_years * 12

        for episode in range(num_episodes):
            print(f"Evaluation episode {episode + 1}/{num_episodes}")
            self.env.current_year_index = start_year
            self._evaluate_single_episode(end_year)

            total_electricity_gwh = sum(
                [
                    record["Electricity_GWh"]
                    for record in self.detailed_records
                    if record["Calendar_Month"] > 0
                ]
            )
            print(
                f"Episode {episode + 1}: "
                f"total electricity = {total_electricity_gwh:.2f} GWh"
            )

        self.save_aligned_csv_to(self.output_dir, output_csv_name)
        self._calculate_firm_power_reliability()

        print("\nEvaluation completed.")
        return True

    def _evaluate_single_episode(self, end_year):
        observation, _ = self.env.reset()
        initial_storage = float(observation[0])
        initial_level = float(self.env.utils.storage_to_level(initial_storage))

        initial_record = {
            "Year": self.env.current_year_index + 1,
            "Calendar_Month": 0,
            "Start_Water_Level_m": initial_level,
            "Start_Storage_1e9_m3": initial_storage,
            "Inflow_m3s": float(observation[1]),
            "Action": None,
            "Target_Storage_1e9_m3": None,
            "Total_Discharge_m3s": 0.0,
            "Turbine_Discharge_m3s": 0.0,
            "Spill_Discharge_m3s": 0.0,
            "Water_Level_m": initial_level,
            "End_Storage_1e9_m3": initial_storage,
            "Tailwater_Level_m": 0.0,
            "Head_m": 0.0,
            "Power_kW": 0.0,
            "Electricity_GWh": 0.0,
            "Max_Power_kW": self.env.basic_info.max_power,
        }
        self.detailed_records.append(initial_record)

        while True:
            action = self.agent.choose_action(observation, self.env, evaluate=True)
            next_observation, reward, terminated, truncated, info = self.env.step(action)

            detailed_record = {
                "Year": info["year"],
                "Calendar_Month": info["month"],
                "Start_Water_Level_m": info.get("start_water_level", 0.0),
                "Start_Storage_1e9_m3": info.get("start_storage", 0.0),
                "Inflow_m3s": info.get("inflow", 0.0),
                "Action": action,
                "Target_Storage_1e9_m3": info.get("target_storage", 0.0),
                "Total_Discharge_m3s": info.get("total_discharge", 0.0),
                "Turbine_Discharge_m3s": info.get("turbine_discharge", 0.0),
                "Spill_Discharge_m3s": info.get("spill_discharge", 0.0),
                "Water_Level_m": info.get("water_level", 0.0),
                "End_Storage_1e9_m3": info.get("end_storage", 0.0),
                "Tailwater_Level_m": info.get("tailwater_level", 0.0),
                "Head_m": info.get("head", 0.0),
                "Power_kW": info.get("power_output", 0.0),
                "Electricity_GWh": info.get("electricity_kwh", 0.0) / 1e6,
                "Max_Power_kW": self.env.basic_info.max_power,
            }
            self.detailed_records.append(detailed_record)

            if self.env.current_year_index > end_year:
                terminated = True

            if terminated or truncated:
                break

            observation = next_observation

    def _calculate_firm_power_reliability(self):
        """
        Compute month-based firm-power reliability.
        """
        if not self.detailed_records:
            print("Detailed records are empty. Cannot calculate reliability.")
            return

        firm_power = float(self.env.basic_info.firm_power)
        if firm_power <= 0:
            print("firm_power <= 0. Skip firm-power reliability calculation.")
            return

        df = pd.DataFrame(self.detailed_records)
        df = df[df["Calendar_Month"] > 0].copy()
        if df.empty:
            print("No valid monthly records found. Cannot calculate reliability.")
            return

        power_series = pd.to_numeric(df["Power_kW"], errors="coerce").fillna(0.0).values

        total_months = len(power_series)
        satisfied_months = int((power_series >= firm_power).sum())
        reliability = satisfied_months / total_months if total_months > 0 else 0.0

        print("\n===== Firm-Power Reliability Statistics =====")
        print(f"Total months: {total_months}")
        print(f"Months with power >= {firm_power:.0f} kW: {satisfied_months}")
        print(f"Month-based reliability: {reliability * 100:.2f}%")

    def save_aligned_csv_to(self, out_dir, filename: str):
        """
        Export aligned DQN evaluation results to CSV.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.detailed_records:
            raise ValueError("No detailed records found. Please run evaluation first.")

        df = pd.DataFrame(self.detailed_records)
        df = df[df["Calendar_Month"] > 0].copy()

        aligned = pd.DataFrame(
            {
                "Month": range(1, len(df) + 1),
                "Water_Level_m": df["Water_Level_m"].values,
                "Power_kW": df["Power_kW"].values,
                "Electricity_GWh": df["Electricity_GWh"].values,
                "Total_Discharge_m3s": df["Total_Discharge_m3s"].values,
                "Turbine_Discharge_m3s": df["Turbine_Discharge_m3s"].values,
                "Spill_Discharge_m3s": df["Spill_Discharge_m3s"].values,
                "Inflow_m3s": df["Inflow_m3s"].values,
            }
        )

        out_path = out_dir / filename
        aligned.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Aligned CSV exported to: {out_path}")


def run_evaluation():
    """
    Evaluate all trained DQN models defined by dqn.candidates.
    """
    data_dir = get_data_dir()
    penalty_settings_path = data_dir / "penalty_settings.yaml"
    lam_list = load_penalty_candidates(penalty_settings_path, "dqn")

    model_dir = get_policy_generation_dir("dqn") / "policy"
    eval_output_dir = get_outputs_dir() / "evaluation" / "dqn"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    for lam in lam_list:
        lam_str = format_lambda_for_filename(float(lam))
        model_path = model_dir / f"dqn_train_lambda_{lam_str}.pth"

        if not model_path.exists():
            print(f"Skip lambda={lam}: model file not found -> {model_path}")
            continue

        print("\n" + "=" * 80)
        print(f"Start evaluating lambda = {lam}")
        print("=" * 80)

        config = Config()
        config.agent_config["firm_penalty_lambda"] = float(lam)

        set_global_seed(config.get_seed())

        basic_info = BasicInfo(config)
        env = ReservoirEnv(basic_info)
        env.is_training = False

        agent = DQNAgent(
            n_actions=env.action_space.n,
            n_features=env.observation_space.shape[0],
            config=config,
            basic_info=basic_info,
        )

        evaluator = ReservoirEvaluator(env, agent, eval_output_dir)

        evaluator.run_evaluation(
            model_path=model_path,
            num_episodes=1,
            start_year=0,
            end_year=len(basic_info.test_inflow) - 1,
            output_csv_name=f"dqn_eval_lambda_{lam_str}.csv",
        )


def main_dqn_test() -> None:
    run_evaluation()


if __name__ == "__main__":
    main_dqn_test()