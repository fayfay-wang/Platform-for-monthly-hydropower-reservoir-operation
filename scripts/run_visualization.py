"""
Script entry point for reporting in the unified reservoir operation framework.

This script runs:
1) scenario-level reporting and visualization, and
2) overall trade-off reporting and visualization.

The purpose of this script is to keep executable reporting workflows separate
from reusable reporting modules.
"""

import argparse

from reservoir_benchmark.visualization.forecast_visualization import main as main_forecast_visualization
from reservoir_benchmark.visualization.scenario_visualization import main as main_scenario_visualization
from reservoir_benchmark.visualization.overall_visualization import main as main_overall_visualization


def main(forecast_sensitivity: bool = False) -> None:
    if forecast_sensitivity:
        main_forecast_visualization()
        return

    try:
        print("=" * 80)
        print("Running scenario-level reporting...")
        print("=" * 80)
        main_scenario_visualization()

        print("\n" + "=" * 80)
        print("Running overall reporting...")
        print("=" * 80)
        main_overall_visualization()

        print("\n" + "=" * 80)
        print("All reporting tasks completed.")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(f"Reporting failed: {type(exc).__name__}: {exc}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize reservoir operation results.")
    parser.add_argument("--forecast-sensitivity", action="store_true")
    main(forecast_sensitivity=parser.parse_args().forecast_sensitivity)
