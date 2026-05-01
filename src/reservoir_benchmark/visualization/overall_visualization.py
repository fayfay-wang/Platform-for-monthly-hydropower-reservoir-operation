from __future__ import annotations

"""
Overall trade-off visualization for the unified reservoir operation
framework.

This script scans all standardized evaluation CSV files under outputs/evaluation,
computes the annual electricity generation versus firm-power reliability points,
and saves:
1) the raw point table,
2) interpolated curve data, and
3) the overall generation-reliability figure.

The visualization outputs are saved under outputs/visualization.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoir_benchmark.shared.project_paths import get_data_dir, get_outputs_dir
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel

ALGO_ORDER = ["DP", "SDP", "ISO-SURF", "DQN"]
ALGO_DIR_TO_LABEL = {
    "dp": "DP",
    "sdp": "SDP",
    "iso_surf": "ISO-SURF",
    "dqn": "DQN",
}

LABEL_FT = 14
TICK_FT = 12
LEGEND_FT = 12
COLORS = {
    "DP": "#1f77b4",
    "SDP": "#2ca02c",
    "ISO-SURF": "#d62728",
    "DQN": "#9467bd",
}

X_MIN = 90.0
X_MAX = 99.0


def compute_point_from_csv(csv_path: Path, firm_power: float) -> Tuple[float, float]:
    """
    Compute one generation-reliability point from a standardized evaluation CSV.
    """
    required_columns = ["Power_kW", "Electricity_GWh"]
    df = pd.read_csv(csv_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"{csv_path.name} is missing required columns: {missing}")

    power = pd.to_numeric(df["Power_kW"], errors="coerce")
    electricity = pd.to_numeric(df["Electricity_GWh"], errors="coerce")

    valid = power.notna() & electricity.notna()
    dfv = df.loc[valid].copy()
    if len(dfv) == 0:
        raise ValueError("No valid rows after numeric coercion.")

    power_values = pd.to_numeric(dfv["Power_kW"], errors="coerce").to_numpy(dtype=float)
    electricity_values = pd.to_numeric(dfv["Electricity_GWh"], errors="coerce").to_numpy(dtype=float)

    n_months = len(power_values)
    years = n_months / 12.0
    if years <= 0:
        raise ValueError("Invalid month/year length.")

    reliability_pct = float((power_values >= firm_power).sum() / n_months * 100.0)
    generation_gwh = float(np.nansum(electricity_values) / years)
    return reliability_pct, generation_gwh


def load_all_points(eval_root: Path, firm_power: float) -> Dict[str, pd.DataFrame]:
    """
    Load all standardized evaluation CSV files under outputs/evaluation
    using the fixed repository structure.
    """
    rows = {algo: [] for algo in ALGO_ORDER}
    total_csv = 0
    skipped = 0

    for algo_dir_name, algo_label in ALGO_DIR_TO_LABEL.items():
        algo_dir = eval_root / algo_dir_name
        pattern = f"{algo_dir_name}_eval_lambda_*.csv"
        csv_files = sorted(algo_dir.glob(pattern))

        total_csv += len(csv_files)

        for csv_path in csv_files:
            try:
                reliability_pct, generation_gwh = compute_point_from_csv(
                    csv_path,
                    firm_power=firm_power,
                )
                rows[algo_label].append(
                    {
                        "algorithm": algo_label,
                        "reliability_pct": reliability_pct,
                        "generation_gwh": generation_gwh,
                        "source_csv": str(csv_path.as_posix()),
                    }
                )
            except Exception as exc:
                skipped += 1
                print(f"[SKIP] {csv_path} -> {type(exc).__name__}: {exc}")

    if total_csv == 0:
        raise FileNotFoundError(f"No evaluation CSV files found under: {eval_root}")

    dfs = {
        algo: pd.DataFrame(
            rows[algo],
            columns=["algorithm", "reliability_pct", "generation_gwh", "source_csv"],
        )
        for algo in ALGO_ORDER
    }

    print("=" * 60)
    print(f"Scan dir: {eval_root}")
    print(f"Total CSV found: {total_csv} | Skipped: {skipped}")
    for algo in ALGO_ORDER:
        print(f"{algo}: {len(dfs[algo])} points")
    print("=" * 60)

    return dfs

def prep_curve(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return sorted unique generation-reliability points with duplicates averaged.
    """
    grouped = (
        df.dropna(subset=["reliability_pct", "generation_gwh"])
        .groupby("reliability_pct", as_index=False)["generation_gwh"]
        .mean()
        .sort_values("reliability_pct")
    )
    x = grouped["reliability_pct"].to_numpy()
    y = grouped["generation_gwh"].to_numpy()
    return x, y


def interp_within_range(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """
    Perform piecewise-linear interpolation only within the observed x range.
    """
    if len(x) < 2:
        return np.full_like(x_grid, np.nan, dtype=float)

    y_grid = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
    mask = (x_grid >= np.nanmin(x)) & (x_grid <= np.nanmax(x))
    return np.where(mask, y_grid, np.nan)


def save_raw_data_to_csv(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Save all raw trade-off points to a single CSV file.
    """
    all_data = pd.concat([dfs[algo] for algo in ALGO_ORDER if not dfs[algo].empty], ignore_index=True)
    all_data = all_data.sort_values(["algorithm", "reliability_pct"])
    all_data.to_csv(out_path, index=False)
    print(f"Raw data saved to: {out_path.resolve()}")


def save_interpolated_curves_to_csv(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], x_grid: np.ndarray, out_path: Path) -> None:
    """
    Save interpolated trade-off curves to a CSV file.
    """
    data = {"reliability_pct": x_grid}
    for algo in ALGO_ORDER:
        if algo in curves:
            x, y = curves[algo]
            data[f"{algo}_generation_gwh"] = interp_within_range(x, y, x_grid)

    df_interp = pd.DataFrame(data)
    df_interp.to_csv(out_path, index=False)
    print(f"Interpolated curves saved to: {out_path.resolve()}")


def main() -> None:
    outputs_dir = get_outputs_dir()
    visualization_dir = outputs_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    reservoir = SharedReservoirModel(data_dir=get_data_dir())
    eval_root = outputs_dir / "evaluation"
    x_grid = np.linspace(X_MIN, X_MAX, 200)

    dfs = load_all_points(eval_root, firm_power=reservoir.firm_power)
    curves = {algo: prep_curve(dfs[algo]) for algo in ALGO_ORDER}

    raw_csv_path = visualization_dir / "overall_generation_reliability_raw_data.csv"
    curve_csv_path = visualization_dir / "overall_generation_reliability_interpolated_curves.csv"
    pdf_path = visualization_dir / "overall_generation_reliability.pdf"
    png_path = visualization_dir / "overall_generation_reliability.png"

    save_raw_data_to_csv(dfs, raw_csv_path)
    save_interpolated_curves_to_csv(curves, x_grid, curve_csv_path)

    fig = plt.figure(figsize=(8, 4), dpi=600)
    ax = plt.gca()

    for algo in ALGO_ORDER:
        x, y = curves[algo]
        color = COLORS[algo]

        if len(x) == 0:
            print(f"[WARN] No data points for {algo}, skipping plot.")
            continue

        selected = (x >= X_MIN) & (x <= X_MAX)
        ax.scatter(x[selected], y[selected], s=16, color=color, alpha=0.9)

        y_grid = interp_within_range(x, y, x_grid)
        ax.plot(x_grid, y_grid, color=color, linewidth=1.8, label=algo)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("Reliability (%)", fontsize=LABEL_FT)
    ax.set_ylabel("Electricity Generation (GWh/yr)", fontsize=LABEL_FT)
    ax.tick_params(axis="both", labelsize=TICK_FT)
    ax.grid(True, alpha=0.25)

    legend = ax.legend(
        fontsize=LEGEND_FT,
        ncol=1,
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
    )
    legend.get_frame().set_alpha(0.85)

    plt.tight_layout()
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight", format="pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\nGenerated files:")
    print(f"  1. {raw_csv_path.name}")
    print(f"  2. {curve_csv_path.name}")
    print(f"  3. {pdf_path.name}")
    print(f"  4. {png_path.name}")


if __name__ == "__main__":
    main()
