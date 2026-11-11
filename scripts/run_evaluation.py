"""
Script entry point for evaluation in the unified reservoir operation framework.

This script runs:
1) DP evaluation,
2) SDP evaluation,
3) ISO-SURF evaluation, and
4) DQN evaluation.

The purpose of this script is to keep executable evaluation workflows separate
from reusable library modules.
"""

import argparse
from dataclasses import replace

from reservoir_benchmark.evaluation.ddp.ddp_eval import main_ddp_eval
from reservoir_benchmark.evaluation.sdp.sdp_eval import main_sdp_eval
from reservoir_benchmark.evaluation.iso_surf.iso_surf_eval import main_iso_surf_eval
from reservoir_benchmark.evaluation.dqn.dqn_eval import main_dqn_eval
from reservoir_benchmark.shared.inflow_forecast import (
    load_forecast_evaluation_seeds,
    load_forecast_scenarios,
)


def main(forecast_sensitivity: bool = False) -> None:
    if forecast_sensitivity:
        scenarios = load_forecast_scenarios()
        perfect = scenarios[0]
        evaluation_seeds = load_forecast_evaluation_seeds()
        for forecast in scenarios:
            for evaluate in (main_sdp_eval, main_iso_surf_eval, main_dqn_eval):
                if forecast.mode == "noisy":
                    for seed in evaluation_seeds:
                        observed = replace(forecast, evaluation_seed=seed)
                        evaluate(forecast=observed)
                        evaluate(forecast=observed, policy_forecast=perfect)
                else:
                    evaluate(forecast=forecast)
        return

    try:
        # print("=" * 80)
        # print("Running DDP evaluation...")
        # print("=" * 80)
        # main_ddp_eval()
        #
        # print("\n" + "=" * 80)
        # print("Running SDP evaluation...")
        # print("=" * 80)
        # main_sdp_eval()
        #
        # print("\n" + "=" * 80)
        # print("Running ISO-SURF evaluation...")
        # print("=" * 80)
        # main_iso_surf_eval()

        print("\n" + "=" * 80)
        print("Running DQN evaluation...")
        print("=" * 80)
        main_dqn_eval()

        print("\n" + "=" * 80)
        print("All evaluation tasks completed.")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reservoir operation policies.")
    parser.add_argument("--forecast-sensitivity", action="store_true")
    main(forecast_sensitivity=parser.parse_args().forecast_sensitivity)
