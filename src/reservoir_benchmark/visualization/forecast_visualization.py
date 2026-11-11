"""Report the forecast sensitivity experiment from the evaluation CSV files."""

from dataclasses import replace

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from reservoir_benchmark.shared.inflow_forecast import (
    forecast_evaluation_dir,
    load_forecast_evaluation_seeds,
    load_forecast_penalty_candidates,
    load_forecast_scenarios,
)
from reservoir_benchmark.shared.io_utils import load_algorithm_settings
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_outputs_dir,
)
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel
from reservoir_benchmark.visualization.overall_visualization import COLORS


def main() -> None:
    scenarios = load_forecast_scenarios()
    perfect = scenarios[0]
    evaluation_seeds = load_forecast_evaluation_seeds()
    data_dir = get_data_dir()
    dqn_seeds = load_algorithm_settings(
        data_dir / "experiment_settings.yaml", "dqn"
    )["experiment"]["seed_list"]
    firm_power = SharedReservoirModel(data_dir=data_dir).firm_power
    evaluation_dir = get_outputs_dir() / "evaluation"
    output_dir = get_outputs_dir() / "visualization" / "forecast_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = {"sdp": "SDP", "iso_surf": "ISO-SURF", "dqn": "DQN"}
    records = []
    for algorithm, method in methods.items():
        training_seeds = dqn_seeds if algorithm == "dqn" else [None]
        for lam in load_forecast_penalty_candidates(algorithm):
            filename = f"{algorithm}_eval_lambda_{format_lambda_for_filename(lam)}"
            for scenario in scenarios:
                noisy = scenario.mode == "noisy"
                for seed in evaluation_seeds if noisy else [None]:
                    observed = replace(scenario, evaluation_seed=seed) if noisy else scenario
                    for policy in (scenario, perfect) if noisy else (scenario,):
                        folder = forecast_evaluation_dir(
                            evaluation_dir / algorithm, observed, policy
                        )
                        for training_seed in training_seeds:
                            csv_path = (
                                folder / "by_seed" / f"{filename}_seed_{training_seed}.csv"
                                if training_seed is not None else folder / f"{filename}.csv"
                            )
                            data = pd.read_csv(csv_path)
                            records.append({
                                "Method": method,
                                "Lambda": lam,
                                "Forecast": scenario.name,
                                "Policy": policy.name,
                                "Error_CV": scenario.cv if scenario.mode != "none" else None,
                                "Evaluation_Seed": seed,
                                "DQN_Seed": training_seed,
                                "Months": len(data),
                                "Electricity_GWh_per_year": 12 * data["Electricity_GWh"].mean(),
                                "Firm_Power_Reliability_pct": 100 * (data["Power_kW"] >= firm_power).mean(),
                            })

    detail = pd.DataFrame(records)
    metrics = ["Electricity_GWh_per_year", "Firm_Power_Reliability_pct"]
    mean = detail.groupby(
        ["Method", "Lambda", "Forecast", "Policy"], as_index=False
    )[metrics].mean()
    detail.to_csv(output_dir / "forecast_sensitivity_by_seed.csv", index=False)
    mean.to_csv(output_dir / "forecast_sensitivity_summary.csv", index=False)

    noisy_scenarios = sorted(
        (scenario for scenario in scenarios if scenario.mode == "noisy"),
        key=lambda scenario: scenario.cv,
    )
    forecast_names = [perfect.name, *(scenario.name for scenario in noisy_scenarios)]
    x = [0.0, *(scenario.cv for scenario in noisy_scenarios)]
    no_forecast_x = x[-1] + max(0.05, x[-1] * 0.3)
    labels = ["0%", *(f"{scenario.cv:.0%}" for scenario in noisy_scenarios), "No forecast"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    y_labels = ["Annualized electricity (GWh)", "Firm-power reliability (%)"]
    multiple_penalties = mean.groupby("Method")["Lambda"].nunique().gt(1)
    for ax, metric, ylabel in zip(axes, metrics, y_labels):
        for (method, lam), values in mean.groupby(["Method", "Lambda"], sort=False):
            matched = values[values["Forecast"] == values["Policy"]].set_index("Forecast")
            frozen = values[values["Policy"] == perfect.name].set_index("Forecast")
            label = f"{method} (λ={lam:g})" if multiple_penalties[method] else method
            color = COLORS[method]
            ax.plot(x, matched.loc[forecast_names, metric], "o-", label=label, color=color)
            ax.plot(x, frozen.loc[forecast_names, metric], "--", color=color)
            ax.scatter(no_forecast_x, matched.loc["none", metric], color=color, marker="s")
        ax.set_xticks([*x, no_forecast_x], labels)
        ax.set_xlabel("Relative forecast error (CV)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncol=3)
    axes[1].legend(
        handles=[
            Line2D([], [], color="black", marker="o", label="Scenario-trained"),
            Line2D([], [], color="black", linestyle="--", label="Frozen perfect-forecast"),
            Line2D([], [], color="black", linestyle="none", marker="s", label="No forecast"),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncol=3,
        fontsize=9,
    )
    fig.savefig(output_dir / "forecast_sensitivity.pdf")
    fig.savefig(output_dir / "forecast_sensitivity.png", dpi=600)
    plt.close(fig)
