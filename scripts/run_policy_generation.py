"""
Script entry point for policy generation in the unified reservoir operation framework.

This script runs:
1) SDP policy generation,
2) ISO-SURF policy generation, and
3) DQN policy generation.

The purpose of this script is to keep executable policy-generation workflows
separate from reusable library modules.
"""

import argparse

from reservoir_benchmark.policy_generation.sdp.sdp_runner import main_sdp
from reservoir_benchmark.policy_generation.iso_surf.iso_surf_runner import main_iso_surf
from reservoir_benchmark.policy_generation.dqn.dqn_runner import main_dqn
from reservoir_benchmark.shared.inflow_forecast import load_forecast_scenarios


def main(forecast_sensitivity: bool = False) -> None:
    if forecast_sensitivity:
        scenarios = load_forecast_scenarios()
        main_iso_surf(forecast=scenarios)
        for forecast in scenarios:
            print(f"\nGenerating SDP policies for {forecast.name}")
            main_sdp(forecast=forecast)
        for forecast in scenarios:
            print(f"\nGenerating DQN policies for {forecast.name}")
            main_dqn(forecast=forecast)
        return

    try:
        # print("=" * 80)
        print("Running SDP policy generation...")
        print("=" * 80)
        main_sdp()

        print("\n" + "=" * 80)
        print("Running ISO-SURF policy generation...")
        print("=" * 80)
        main_iso_surf()

        print("\n" + "=" * 80)
        print("Running DQN policy generation...")
        print("=" * 80)
        main_dqn()

        print("\n" + "=" * 80)
        print("All policy-generation tasks completed.")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(f"Policy generation failed: {type(exc).__name__}: {exc}")
        print("=" * 80)
        raise
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reservoir operation policies.")
    parser.add_argument("--forecast-sensitivity", action="store_true")
    main(forecast_sensitivity=parser.parse_args().forecast_sensitivity)
