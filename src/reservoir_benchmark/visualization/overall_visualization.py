from __future__ import annotations

"""
Overall trade-off visualization for the unified reservoir operation
framework.

This script scans all standardized evaluation CSV files under outputs/evaluation,
computes the annual electricity generation versus firm-power reliability points,
and saves the overall generation-reliability figure.

The x-axis range is determined by the minimum and maximum reliability values
computed from all available evaluation CSV files.

The visualization outputs are saved under outputs/visualization.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoir_benchmark.shared.project_paths import get_data_dir, get_outputs_dir
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel

import matplotlib as mpl

FONT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.sans-serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
mpl.rcParams.update(FONT_RC)

# Keep directory names unchanged, but use publication-friendly labels.
ALGO_ORDER = ["DDP", "SDP", "ISO-SURF", "DQN"]
ALGO_DIR_TO_LABEL = {
    "ddp": "DDP",
    "sdp": "SDP",
    "iso_surf": "ISO-SURF",
    "dqn": "DQN",
}

LABEL_FT = 14
TICK_FT = 12
LEGEND_FT = 12
COLORS = {
    "DDP": "#1f77b4",
    "SDP": "#2ca02c",
    "ISO-SURF": "#d62728",
    "DQN": "#9467bd",
}


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
    x = grouped["reliability_pct"].to_numpy(dtype=float)
    y = grouped["generation_gwh"].to_numpy(dtype=float)
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


def get_all_reliability_values(dfs: Dict[str, pd.DataFrame]) -> np.ndarray:
    """
    Collect all finite reliability values from the algorithm data frames.
    """
    values = []
    for algo in ALGO_ORDER:
        if dfs[algo].empty:
            continue
        vals = pd.to_numeric(dfs[algo]["reliability_pct"], errors="coerce").to_numpy(dtype=float)
        values.append(vals[np.isfinite(vals)])

    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def get_reliability_axis_range(dfs: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    """
    Return a padded reliability range from all valid evaluation points.
    """
    values = get_all_reliability_values(dfs)
    if len(values) == 0:
        raise ValueError("No finite reliability values available for x-axis range.")

    raw_x_min = float(np.nanmin(values))
    raw_x_max = float(np.nanmax(values))

    x_min = max(0.0, raw_x_min - 3)
    x_max = min(100.0, raw_x_max + 3)

    print("=" * 60)
    print("Reliability-axis range")
    print(f"  Raw reliability range: [{raw_x_min:.2f}, {raw_x_max:.2f}]")
    print(f"  Plot x-axis: [{x_min:.2f}, {x_max:.2f}]")
    print("=" * 60)

    return x_min, x_max


def main() -> None:
    outputs_dir = get_outputs_dir()
    visualization_dir = outputs_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    reservoir = SharedReservoirModel(data_dir=get_data_dir())
    eval_root = outputs_dir / "evaluation"

    dfs = load_all_points(eval_root, firm_power=reservoir.firm_power)
    curves = {algo: prep_curve(dfs[algo]) for algo in ALGO_ORDER}

    x_min, x_max = get_reliability_axis_range(dfs)
    # x_min = 80
    # x_max = 93

    x_grid = np.linspace(x_min, x_max, 200)

    pdf_path = visualization_dir / "overall_generation_reliability.pdf"
    png_path = visualization_dir / "overall_generation_reliability.png"

    fig = plt.figure(figsize=(8.8, 4), dpi=600)
    ax = plt.gca()

    for algo in ALGO_ORDER:
        x, y_gwh = curves[algo]
        y = y_gwh / 1000.0  # GWh/yr -> TWh/yr
        color = COLORS[algo]

        if len(x) == 0:
            print(f"[WARN] No data points for {algo}, skipping plot.")
            continue

        ax.scatter(x, y, s=16, color=color, alpha=0.9)

        y_grid = interp_within_range(x, y, x_grid)
        ax.plot(x_grid, y_grid, color=color, linewidth=1.8, label=algo)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Firm-power reliability (%)", fontsize=LABEL_FT)
    ax.set_ylabel("Electricity generation (TWh/yr)", fontsize=LABEL_FT)
    ax.tick_params(axis="both", labelsize=TICK_FT)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        fontsize=LEGEND_FT,
    )

    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.925])
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight", format="pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\nGenerated files:")
    print(f"  1. {pdf_path.name}")
    print(f"  2. {png_path.name}")


if __name__ == "__main__":
    main()
