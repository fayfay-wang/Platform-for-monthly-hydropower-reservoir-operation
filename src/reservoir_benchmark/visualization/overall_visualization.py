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

Compared with a fixed reliability-axis range, this version automatically selects
an x-axis focus window from the empirical distribution of reliability points.
The focus window is the shortest interval that contains a target fraction of all
available reliability points, with a small padding added for readability. This
keeps the figure focused on the dense part of the trade-off points while still
preserving all raw points in the exported CSV file.

The visualization outputs are saved under outputs/visualization.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoir_benchmark.shared.project_paths import get_data_dir, get_outputs_dir
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel

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

# -----------------------------------------------------------------------------
# Reliability-axis selection
# -----------------------------------------------------------------------------
# Use a data-driven focus window instead of a fixed reliability interval.
# Example: 0.90 means the x-axis shows the shortest interval covering 90% of all
# reliability points. Sparse tails are not plotted, but all raw points are still
# saved in overall_generation_reliability_raw_data.csv.
AUTO_X_RANGE = True
FOCUS_FRACTION = 0.90
X_PADDING_RATIO = 0.08
MIN_X_WIDTH = 5.0
X_LOWER_BOUND = 0.0
X_UPPER_BOUND = 100.0

# Fallback range if AUTO_X_RANGE is disabled.
MANUAL_X_MIN = 80.0
MANUAL_X_MAX = 94.0


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


def shortest_interval(values: np.ndarray, fraction: float) -> Tuple[float, float, int, int]:
    """
    Return the shortest interval that covers the requested fraction of values.

    Parameters
    ----------
    values : np.ndarray
        Reliability values in percent.
    fraction : float
        Fraction of points to retain, e.g. 0.90 for 90%.

    Returns
    -------
    xmin, xmax, n_selected, n_total
    """
    clean = np.sort(values[np.isfinite(values)])
    n_total = len(clean)
    if n_total == 0:
        raise ValueError("No finite reliability values available for x-axis selection.")

    fraction = float(np.clip(fraction, 0.01, 1.0))
    n_selected = int(np.ceil(n_total * fraction))
    n_selected = max(1, min(n_selected, n_total))

    if n_selected == n_total:
        return float(clean[0]), float(clean[-1]), n_selected, n_total

    widths = clean[n_selected - 1 :] - clean[: n_total - n_selected + 1]
    min_width = np.nanmin(widths)
    candidate_idx = np.flatnonzero(np.isclose(widths, min_width))

    # Tie-breaker: choose the candidate interval whose center is closest to the
    # global median. This prevents arbitrary choices in symmetric cases.
    median_value = np.nanmedian(clean)
    centers = (clean[candidate_idx] + clean[candidate_idx + n_selected - 1]) / 2.0
    best_local = int(np.argmin(np.abs(centers - median_value)))
    best_idx = int(candidate_idx[best_local])

    xmin = float(clean[best_idx])
    xmax = float(clean[best_idx + n_selected - 1])
    return xmin, xmax, n_selected, n_total


def choose_reliability_axis_range(dfs: Dict[str, pd.DataFrame]) -> Tuple[float, float, pd.DataFrame]:
    """
    Choose an automatic reliability-axis range based on point density.

    The raw data are not deleted. This function only determines what is displayed
    in the figure and what x-grid is used for the interpolated curve CSV.
    """
    if not AUTO_X_RANGE:
        info = pd.DataFrame(
            [
                {
                    "mode": "manual",
                    "focus_fraction": np.nan,
                    "n_points_total": np.nan,
                    "n_points_in_focus_before_padding": np.nan,
                    "x_min": MANUAL_X_MIN,
                    "x_max": MANUAL_X_MAX,
                }
            ]
        )
        return MANUAL_X_MIN, MANUAL_X_MAX, info

    values = get_all_reliability_values(dfs)
    base_min, base_max, n_focus, n_total = shortest_interval(values, FOCUS_FRACTION)

    width = base_max - base_min
    if width <= 0:
        width = MIN_X_WIDTH

    padding = max(width * X_PADDING_RATIO, 0.5)
    x_min = base_min - padding
    x_max = base_max + padding

    # Avoid excessive zooming when the dense interval is very narrow.
    if (x_max - x_min) < MIN_X_WIDTH:
        center = (x_min + x_max) / 2.0
        half_width = MIN_X_WIDTH / 2.0
        x_min = center - half_width
        x_max = center + half_width

    x_min = max(X_LOWER_BOUND, x_min)
    x_max = min(X_UPPER_BOUND, x_max)

    # If clipping has reduced the width too much, expand inward where possible.
    if (x_max - x_min) < MIN_X_WIDTH:
        if x_min <= X_LOWER_BOUND:
            x_max = min(X_UPPER_BOUND, x_min + MIN_X_WIDTH)
        elif x_max >= X_UPPER_BOUND:
            x_min = max(X_LOWER_BOUND, x_max - MIN_X_WIDTH)

    info = pd.DataFrame(
        [
            {
                "mode": "auto_shortest_density_interval",
                "focus_fraction": FOCUS_FRACTION,
                "n_points_total": n_total,
                "n_points_in_focus_before_padding": n_focus,
                "raw_focus_min": base_min,
                "raw_focus_max": base_max,
                "x_min_after_padding": x_min,
                "x_max_after_padding": x_max,
                "min_x_width": MIN_X_WIDTH,
                "padding_ratio": X_PADDING_RATIO,
            }
        ]
    )

    print("=" * 60)
    print("Automatic reliability-axis range")
    print(f"  Focus fraction: {FOCUS_FRACTION:.2f}")
    print(f"  Dense interval before padding: [{base_min:.2f}, {base_max:.2f}]")
    print(f"  Plot x-axis after padding:      [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Points covered before padding: {n_focus}/{n_total}")
    print("=" * 60)

    return float(x_min), float(x_max), info


def save_raw_data_to_csv(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Save all raw trade-off points to a single CSV file.
    """
    all_data = pd.concat([dfs[algo] for algo in ALGO_ORDER if not dfs[algo].empty], ignore_index=True)
    all_data = all_data.sort_values(["algorithm", "reliability_pct"])
    all_data.to_csv(out_path, index=False)
    print(f"Raw data saved to: {out_path.resolve()}")


def save_focused_raw_data_to_csv(
    dfs: Dict[str, pd.DataFrame],
    x_min: float,
    x_max: float,
    out_path: Path,
) -> None:
    """
    Save the points that are displayed inside the selected reliability window.
    """
    all_data = pd.concat([dfs[algo] for algo in ALGO_ORDER if not dfs[algo].empty], ignore_index=True)
    selected = all_data[
        (all_data["reliability_pct"] >= x_min)
        & (all_data["reliability_pct"] <= x_max)
    ].copy()
    selected = selected.sort_values(["algorithm", "reliability_pct"])
    selected.to_csv(out_path, index=False)
    print(f"Focused raw data saved to: {out_path.resolve()}")


def save_interpolated_curves_to_csv(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_grid: np.ndarray,
    out_path: Path,
) -> None:
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

    dfs = load_all_points(eval_root, firm_power=reservoir.firm_power)
    curves = {algo: prep_curve(dfs[algo]) for algo in ALGO_ORDER}

    x_min, x_max, focus_info = choose_reliability_axis_range(dfs)
    x_grid = np.linspace(x_min, x_max, 200)

    raw_csv_path = visualization_dir / "overall_generation_reliability_raw_data.csv"
    focused_raw_csv_path = visualization_dir / "overall_generation_reliability_focused_raw_data.csv"
    curve_csv_path = visualization_dir / "overall_generation_reliability_interpolated_curves.csv"
    focus_info_path = visualization_dir / "overall_generation_reliability_focus_range.csv"
    pdf_path = visualization_dir / "overall_generation_reliability.pdf"
    png_path = visualization_dir / "overall_generation_reliability.png"

    save_raw_data_to_csv(dfs, raw_csv_path)
    save_focused_raw_data_to_csv(dfs, x_min, x_max, focused_raw_csv_path)
    save_interpolated_curves_to_csv(curves, x_grid, curve_csv_path)
    focus_info.to_csv(focus_info_path, index=False)
    print(f"Focus range info saved to: {focus_info_path.resolve()}")

    fig = plt.figure(figsize=(8, 4), dpi=600)
    ax = plt.gca()

    for algo in ALGO_ORDER:
        x, y = curves[algo]
        color = COLORS[algo]

        if len(x) == 0:
            print(f"[WARN] No data points for {algo}, skipping plot.")
            continue

        selected = (x >= x_min) & (x <= x_max)
        if selected.sum() == 0:
            print(
                f"[WARN] No {algo} points inside selected reliability window "
                f"[{x_min:.2f}, {x_max:.2f}], skipping scatter."
            )
        else:
            ax.scatter(x[selected], y[selected], s=16, color=color, alpha=0.9)

        y_grid = interp_within_range(x, y, x_grid)
        ax.plot(x_grid, y_grid, color=color, linewidth=1.8, label=algo)

    ax.set_xlim(x_min, x_max)
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
    print(f"  2. {focused_raw_csv_path.name}")
    print(f"  3. {curve_csv_path.name}")
    print(f"  4. {focus_info_path.name}")
    print(f"  5. {pdf_path.name}")
    print(f"  6. {png_path.name}")


if __name__ == "__main__":
    main()
