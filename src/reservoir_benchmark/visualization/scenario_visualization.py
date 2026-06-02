from __future__ import annotations
"""
Scenario-level visualization for the unified reservoir operation framework.

This script builds fixed scenario comparison outputs from the standardized
algorithm evaluation CSV files, including:
1) an energy-inflow-level bubble plot,
2) representative-year time-series plots,
3) a 2×2 monthly distribution figure, and
4) a LaTeX summary table corresponding to the monthly comparison statistics.

Design choices
--------------
- The script reads only the standardized evaluation results under
  outputs/evaluation/{algorithm}.
- Because interactive file selection has been removed, the script automatically
  selects one evaluation CSV for each algorithm by optimizing the consistency of
  month-based reliability across algorithms.
- The preferred combination is the one in which all algorithms satisfy the
  reliability floor (default: 90%) and the variance of the four reliability
  values is minimized.
- If multiple combinations are tied, the script further prefers the one with
  the smallest reliability range and then the smallest sum of lambda values.
- If no combination satisfies the reliability floor for all algorithms, the
  script falls back to all available combinations and still chooses the one with
  the smallest reliability variance.
- The visualization outputs are saved under outputs/visualization.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from reservoir_benchmark.shared.io_utils import load_monthly_inflows_from_excel
from reservoir_benchmark.shared.project_paths import get_data_dir, get_outputs_dir
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel

ALGO_DIR_MAP: Dict[str, str] = {
    "DDP": "dp",
    "SDP": "sdp",
    "ISO-SURF": "iso_surf",
    "DQN": "dqn",
}
ALGO_ORDER: List[str] = ["DDP", "SDP", "ISO-SURF", "DQN"]

TITLE_FT = 16
LABEL_FT = 14
TICK_FT = 12
LEGEND_FT = 12

COLORS = {
    "DDP": "#1f77b4",
    "SDP": "#2ca02c",
    "ISO-SURF": "#d62728",
    "DQN": "#9467bd",
}
INFLOW_COLOR = "#7f7f7f"
PANEL_FACE_COLORS = [
    (0.965, 0.965, 0.965, 1.0),
    (0.985, 0.985, 0.985, 1.0),
    (0.985, 0.985, 0.985, 1.0),
    (0.965, 0.965, 0.965, 1.0),
]

RELIABILITY_DRY_QUANTILE = 0.2
RELIABILITY_WET_QUANTILE = 0.8
RELIABILITY_FLOOR_PCT = 80.0


def infer_year_month_from_month_index(month_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Infer year and calendar month from the standardized Month column.
    """
    month_values = pd.to_numeric(month_series, errors="coerce")
    if month_values.isna().all():
        raise ValueError("Month column cannot be parsed as numeric values.")

    max_month = int(month_values.max())
    if max_month > 12:
        year = ((month_values - 1) // 12 + 1).astype(int)
        month = ((month_values - 1) % 12 + 1).astype(int)
    else:
        n_rows = len(month_values)
        year = pd.Series((np.arange(n_rows) // 12 + 1).astype(int), index=month_values.index)
        month = month_values.astype(int)

    return year, month


def extract_lambda_from_path(csv_path: Path) -> float:
    """
    Extract lambda value from a standardized evaluation filename.
    """
    match = re.search(r"_lambda_([\d\.]+)", csv_path.stem)
    if not match:
        return float("inf")
    try:
        return float(match.group(1))
    except ValueError:
        return float("inf")


def compute_reliability_pct(csv_path: Path, firm_power: float) -> float:
    """
    Compute month-based reliability (%) from one standardized evaluation CSV.
    """
    df = pd.read_csv(csv_path)
    required = {"Power_kW"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    power = pd.to_numeric(df["Power_kW"], errors="coerce").dropna()
    if len(power) == 0:
        raise ValueError(f"{csv_path.name} does not contain valid Power_kW values.")

    return float((power >= firm_power).mean() * 100.0)


def collect_eval_candidates(algo_label: str, eval_dir: Path, firm_power: float) -> List[Tuple[Path, float, float]]:
    """
    Collect standardized evaluation CSV candidates for one algorithm.

    Returns a list of tuples:
        (csv_path, lambda_value, reliability_pct)
    sorted by lambda value and filename.
    """
    pattern = f"{ALGO_DIR_MAP[algo_label]}_eval_lambda_*.csv"
    candidates = sorted(eval_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No evaluation CSV found for {algo_label} in {eval_dir}")

    records: List[Tuple[Path, float, float]] = []
    for csv_path in candidates:
        lambda_value = extract_lambda_from_path(csv_path)
        reliability_pct = compute_reliability_pct(csv_path, firm_power=firm_power)
        records.append((csv_path, lambda_value, reliability_pct))

    records.sort(key=lambda item: (item[1], item[0].name.lower()))
    return records


def select_best_eval_file_combination(outputs_dir: Path, firm_power: float, reliability_floor_pct: float = RELIABILITY_FLOOR_PCT) -> Dict[str, Path]:
    """
    Select one evaluation CSV per algorithm using a strict reliability floor.

    Selection rule:
    1) discard every candidate whose reliability is below reliability_floor_pct;
    2) among the remaining cross-algorithm combinations, choose the one with the
       smallest reliability variance;
    3) break ties using the smallest reliability range, then the smallest sum of
       lambda values.

    Unlike the previous version, this function does not fall back to
    lower-reliability scenarios. If any algorithm has no candidate with
    reliability >= reliability_floor_pct, it raises a ValueError.
    """
    candidate_map: Dict[str, List[Tuple[Path, float, float]]] = {}

    for algo_label, algo_dir_name in ALGO_DIR_MAP.items():
        eval_dir = outputs_dir / "evaluation" / algo_dir_name
        raw_candidates = collect_eval_candidates(algo_label, eval_dir, firm_power=firm_power)
        valid_candidates = [
            item for item in raw_candidates
            if item[2] >= reliability_floor_pct
        ]

        if not valid_candidates:
            best_candidate = max(raw_candidates, key=lambda item: item[2])
            raise ValueError(
                f"No evaluation CSV for {algo_label} satisfies the reliability floor "
                f">= {reliability_floor_pct:.2f}%. "
                f"Best available file is {best_candidate[0].name} "
                f"with reliability = {best_candidate[2]:.2f}%."
            )

        candidate_map[algo_label] = valid_candidates

    algo_labels = list(ALGO_ORDER)
    all_products = product(*(candidate_map[algo] for algo in algo_labels))

    feasible: List[Tuple[float, float, float, Dict[str, Path]]] = []

    for combo in all_products:
        reliabilities = np.array([item[2] for item in combo], dtype=float)
        lambdas = np.array([item[1] for item in combo], dtype=float)
        selected = {algo_labels[i]: combo[i][0] for i in range(len(algo_labels))}

        variance = float(np.var(reliabilities))
        reliability_range = float(np.max(reliabilities) - np.min(reliabilities))
        lambda_sum = float(np.sum(lambdas))

        feasible.append((variance, reliability_range, lambda_sum, selected))

    best = min(feasible, key=lambda item: (item[0], item[1], item[2]))
    return best[3]

def load_single_result(algo_label: str, csv_path: Path) -> pd.DataFrame:
    """
    Load one standardized evaluation CSV and return a normalized DataFrame.
    """
    required_columns = [
        "Month",
        "Water_Level_m",
        "Power_kW",
        "Electricity_GWh",
        "Total_Discharge_m3s",
        "Turbine_Discharge_m3s",
        "Spill_Discharge_m3s",
        "Inflow_m3s",
    ]

    df = pd.read_csv(csv_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"{csv_path.name} is missing required columns: {missing}")

    year, month = infer_year_month_from_month_index(df["Month"])

    out = pd.DataFrame(
        {
            "algo": algo_label,
            "year": year.astype(int),
            "month": month.astype(int),
            "level": pd.to_numeric(df["Water_Level_m"], errors="coerce"),
            "power": pd.to_numeric(df["Power_kW"], errors="coerce"),
            "energy": pd.to_numeric(df["Electricity_GWh"], errors="coerce"),
            "discharge": pd.to_numeric(df["Total_Discharge_m3s"], errors="coerce"),
            "turbine": pd.to_numeric(df["Turbine_Discharge_m3s"], errors="coerce"),
            "spill": pd.to_numeric(df["Spill_Discharge_m3s"], errors="coerce"),
            "inflow": pd.to_numeric(df["Inflow_m3s"], errors="coerce"),
            "source_file": csv_path.name,
        }
    )

    return out.sort_values(["year", "month"]).reset_index(drop=True)


def load_reference_inflow_table() -> pd.DataFrame:
    """
    Load the unified evaluation inflow table from data/SynMonthlyInflow_evaluation.xlsx.
    """
    data_dir = get_data_dir()
    inflow_path = data_dir / "SynMonthlyInflow_evaluation.xlsx"
    inflow_df = load_monthly_inflows_from_excel(inflow_path).copy()
    inflow_df["year"] = np.arange(1, len(inflow_df) + 1, dtype=int)
    return inflow_df


def get_algos_present(df: pd.DataFrame) -> List[str]:
    present = set(pd.unique(df["algo"]))
    return [algo for algo in ALGO_ORDER if algo in present]


def algo_color(algo: str) -> str:
    return COLORS[algo]


def compute_firm_power_rates(df: pd.DataFrame, firm_power: float, algos: Iterable[str]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for algo in algos:
        subset = df[df["algo"] == algo]
        power = pd.to_numeric(subset["power"], errors="coerce").dropna()
        if len(power) == 0:
            continue
        rates[algo] = float((power >= firm_power).mean())
    return rates


def compute_long_term_means(df: pd.DataFrame, value_col: str, algos: Iterable[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for algo in algos:
        subset = df[df["algo"] == algo]
        if subset.empty:
            continue
        mean_monthly = pd.to_numeric(subset[value_col], errors="coerce").dropna().mean()
        if value_col == "energy":
            stats[algo] = float(mean_monthly * 12.0)
        else:
            stats[algo] = float(mean_monthly)
    return stats


def build_summary_table_df(all_df: pd.DataFrame, firm_power: float, algos: Iterable[str]) -> pd.DataFrame:
    """
    Build the summary table corresponding to the monthly comparison figure.
    """
    firm_rates = compute_firm_power_rates(all_df, firm_power=firm_power, algos=algos)
    mean_level = compute_long_term_means(all_df, "level", algos=algos)
    mean_energy = compute_long_term_means(all_df, "energy", algos=algos)
    mean_spill = compute_long_term_means(all_df, "spill", algos=algos)

    rows = [
        {
            "Metric": "Firm power rate (\\%)",
            **{algo: f"{firm_rates.get(algo, np.nan) * 100:.2f}" if algo in firm_rates else "" for algo in algos},
        },
        {
            "Metric": "Mean level (m)",
            **{algo: f"{mean_level.get(algo, np.nan):.2f}" if algo in mean_level else "" for algo in algos},
        },
        {
            "Metric": "Mean annual generation (GWh/yr)",
            **{algo: f"{mean_energy.get(algo, np.nan):.1f}" if algo in mean_energy else "" for algo in algos},
        },
        {
            "Metric": r"Mean monthly spill (m$^3$/s)",
            **{algo: f"{mean_spill.get(algo, np.nan):.1f}" if algo in mean_spill else "" for algo in algos},
        },
    ]
    return pd.DataFrame(rows)


def export_summary_table_to_latex(all_df: pd.DataFrame, firm_power: float, tex_path: Path, algos: Iterable[str]) -> None:
    """
    Export the summary statistics table as LaTeX.
    """
    df_tab = build_summary_table_df(all_df, firm_power=firm_power, algos=algos)

    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{Summary statistics corresponding to the scenario comparison annotations.}",
        r"\label{tab:scenario_summary_stats}",
        r"\begin{tabular}{l" + "c" * len(list(algos)) + r"}",
        r"\toprule",
        "Metric & " + " & ".join(algos) + r" \\",
        r"\midrule",
    ]

    for _, row in df_tab.iterrows():
        values = [str(row["Metric"])] + [str(row[algo]) for algo in algos]
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path.resolve()}")


def pick_representative_years(reference_inflow_df: pd.DataFrame) -> Dict[str, int]:
    """
    Select representative dry and wet years from the unified evaluation inflow series.
    """
    annual = (
        reference_inflow_df.melt(id_vars="year", var_name="month_label", value_name="inflow")
        .groupby("year", as_index=False)["inflow"]
        .mean()
        .sort_values("year")
        .reset_index(drop=True)
    )

    dry_target = annual["inflow"].quantile(RELIABILITY_DRY_QUANTILE)
    wet_target = annual["inflow"].quantile(RELIABILITY_WET_QUANTILE)

    dry_idx = (annual["inflow"] - dry_target).abs().idxmin()
    wet_idx = (annual["inflow"] - wet_target).abs().idxmin()

    dry_year = int(annual.loc[dry_idx, "year"])
    wet_year = int(annual.loc[wet_idx, "year"])

    if dry_year == wet_year and len(annual) >= 2:
        wet_idx_2 = (annual["inflow"] - wet_target).abs().nsmallest(2).index[-1]
        wet_year = int(annual.loc[wet_idx_2, "year"])

    return {"dry": dry_year, "wet": wet_year}


def month_boxdata(df: pd.DataFrame, algo: str, value_col: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    subset = df[df["algo"] == algo]
    for month in range(1, 13):
        values = pd.to_numeric(subset.loc[subset["month"] == month, value_col], errors="coerce").dropna().values
        out.append(values)
    return out


def style_boxplot(bp, face: str, edge: str = "#000000", median: str = "#000000") -> None:
    for box in bp["boxes"]:
        box.set_facecolor(face)
        box.set_alpha(0.65)
        box.set_edgecolor(edge)
        box.set_linewidth(1)
    for whisker in bp["whiskers"]:
        whisker.set_color(edge)
        whisker.set_linewidth(1)
    for cap in bp["caps"]:
        cap.set_color(edge)
        cap.set_linewidth(1)
    for med in bp["medians"]:
        med.set_color(median)
        med.set_linewidth(1.4)


def plot_energy_inflow_level(all_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save the scenario energy-inflow-level bubble plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "scenario_energy_inflow_level_bubble.pdf"

    annual = (
        all_df.groupby(["algo", "year"], as_index=False)
        .agg(
            inflow_mean=("inflow", "mean"),
            energy_sum=("energy", "sum"),
        )
    )

    ri_list = []
    for (algo, year), group in all_df.groupby(["algo", "year"]):
        group = group.sort_values("month")
        levels = pd.to_numeric(group["level"], errors="coerce").to_numpy()
        ramping_index = float(np.nanmean(np.abs(np.diff(levels)))) if len(levels) >= 2 else 0.0
        ri_list.append({"algo": algo, "year": year, "ramping_index": ramping_index})

    annual = annual.merge(pd.DataFrame(ri_list), on=["algo", "year"], how="left")

    size_min, size_max = 30, 220
    values = annual["ramping_index"].to_numpy()
    v_min = np.nanmin(values) if np.isfinite(values).any() else 0.0
    v_max = np.nanmax(values) if np.isfinite(values).any() else 1.0
    annual["_size"] = np.interp(values, [v_min, v_max], [size_min, size_max]) if v_max > v_min else size_min

    with plt.rc_context({"font.sans-serif": ["DejaVu Sans", "Arial"], "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=600)

        for algo in get_algos_present(annual):
            subset = annual[annual["algo"] == algo]
            ax.scatter(
                subset["inflow_mean"],
                subset["energy_sum"],
                s=subset["_size"],
                color=algo_color(algo),
                alpha=0.68,
                label=algo,
                edgecolors="black",
                linewidths=0.6,
            )

        ax.set_xlabel("Mean Annual Inflow (m³/s)", fontsize=LABEL_FT)
        ax.set_ylabel("Total Annual Electricity (GWh)", fontsize=LABEL_FT)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis="both", labelsize=TICK_FT)

        algo_legend = ax.legend(
            fontsize=LEGEND_FT,
            title="Algorithm",
            title_fontsize=LEGEND_FT,
            frameon=False,
            loc="best",
        )
        ax.add_artist(algo_legend)

        if np.isfinite(values).any():
            ri_ticks = np.array([v_min, np.nanmedian(values), v_max], dtype=float)
            size_ticks = np.interp(ri_ticks, [v_min, v_max], [size_min, size_max]) if v_max > v_min else np.full(3, size_min)
            size_handles = [
                ax.scatter([], [], s=size, edgecolors="black", facecolors="none", linewidths=0.9)
                for size in size_ticks
            ]
            size_labels = [f"{ri:.2f} m/month" for ri in ri_ticks]
            size_legend = ax.legend(
                size_handles,
                size_labels,
                title=r"Bubble size = mean $|\Delta h_t|$",
                fontsize=LEGEND_FT,
                title_fontsize=LEGEND_FT,
                frameon=False,
                loc="lower right",
            )
            ax.add_artist(size_legend)

        fig.tight_layout()
        fig.savefig(pdf_path, format="pdf", dpi=600)
        plt.close(fig)

    print(f"Saved bubble plot: {pdf_path.resolve()}")


def add_panel_textbox(ax, text: str | None, loc: str = "lower left", fontsize: int = 9) -> None:
    if not text:
        return

    loc_map = {
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
    }
    x, y, ha, va = loc_map.get(loc, loc_map["lower left"])

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.8,
            alpha=0.88,
        ),
    )


def draw_grouped_box_panel(
    ax,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    panel_label: str,
    panel_facecolor,
    stat_text: str | None = None,
    stats_loc: str = "lower left",
    hlines: List[Tuple[float, str, str, str]] | None = None,
    hlines_legend_loc: str = "upper right",
    show_xlabel: bool = True,
    ylim_expand_top: float = 0.0,
) -> None:
    """
    Draw one panel of the monthly comparison boxplot suite.
    """
    months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax.set_facecolor(panel_facecolor)
    ax.set_axisbelow(True)

    base_pos = np.arange(1, 13, dtype=float)
    algos = get_algos_present(df)
    n = len(algos)

    step = 0.18
    offsets = (np.arange(n) - (n - 1) / 2.0) * step
    box_w = min(0.15, step * 0.82)

    for i, algo in enumerate(algos):
        data = month_boxdata(df, algo, value_col)
        positions = base_pos + offsets[i]
        color = algo_color(algo)

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            widths=box_w,
            showfliers=True,
            flierprops=dict(
                marker="o",
                markersize=2,
                alpha=0.55,
                markerfacecolor=color,
                markeredgecolor=color,
            ),
        )
        style_boxplot(bp, face=color, edge="#000000", median="#000000")

    if hlines:
        line_handles = []
        line_labels = []
        for y, label, color, linestyle in hlines:
            line = ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.95)
            line_handles.append(line)
            line_labels.append(label)

        ax.legend(
            line_handles,
            line_labels,
            fontsize=LEGEND_FT - 1,
            frameon=True,
            fancybox=False,
            framealpha=0.9,
            edgecolor="0.55",
            loc=hlines_legend_loc,
        )

    ax.set_xlim(0.45, 12.55)
    ax.set_xticks(base_pos)
    ax.set_xticklabels(months_abbr, fontsize=TICK_FT)
    ax.tick_params(axis="y", labelsize=TICK_FT)
    ax.tick_params(axis="x", labelsize=TICK_FT)

    ax.set_xlabel("Month" if show_xlabel else "", fontsize=LABEL_FT)
    ax.set_ylabel(ylabel, fontsize=LABEL_FT)
    ax.set_title(title, fontsize=LABEL_FT, pad=6)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")

    ax.text(0.015, 0.985, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=13, fontweight="bold")

    if ylim_expand_top > 0:
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        ax.set_ylim(y0, y1 + yr * ylim_expand_top)

    add_panel_textbox(ax, stat_text, loc=stats_loc, fontsize=LEGEND_FT - 1)


def plot_monthly_boxplots_suite(all_df: pd.DataFrame, output_dir: Path, reservoir: SharedReservoirModel) -> None:
    """
    Save the 2×2 monthly comparison figure as PDF and PNG.
    """
    pdf_path = output_dir / "scenario_monthly_comparison_boxplots.pdf"
    png_path = output_dir / "scenario_monthly_comparison_boxplots.png"

    algos = get_algos_present(all_df)
    if not algos:
        raise ValueError("No algorithm data available for plotting.")

    with plt.rc_context({"font.sans-serif": ["DejaVu Sans", "Arial"], "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), dpi=600)
        axes = axes.flatten()

        power_hlines = [
            (reservoir.max_power, "Installed capacity", "#d62728", "--"),
            (reservoir.firm_power, "Firm power", "#2ca02c", "--"),
        ]
        draw_grouped_box_panel(
            ax=axes[0],
            df=all_df,
            value_col="power",
            ylabel="Power (kW)",
            title="Power output",
            panel_label="(a)",
            panel_facecolor=PANEL_FACE_COLORS[0],
            stat_text=None,
            stats_loc="lower left",
            hlines=power_hlines,
            hlines_legend_loc="upper right",
            show_xlabel=False,
            ylim_expand_top=0.22,
        )

        level_hlines = [
            (reservoir.normal_level, "Normal water level", "#d62728", "--"),
            (reservoir.dead_level, "Dead water level", "#ff7f0e", "--"),
        ]
        draw_grouped_box_panel(
            ax=axes[1],
            df=all_df,
            value_col="level",
            ylabel="Water level (m)",
            title="Water level",
            panel_label="(b)",
            panel_facecolor=PANEL_FACE_COLORS[1],
            stat_text=None,
            stats_loc="lower left",
            hlines=level_hlines,
            hlines_legend_loc="upper right",
            show_xlabel=False,
            ylim_expand_top=0.22,
        )

        draw_grouped_box_panel(
            ax=axes[2],
            df=all_df,
            value_col="energy",
            ylabel="Electricity (GWh)",
            title="Electricity generation",
            panel_label="(c)",
            panel_facecolor=PANEL_FACE_COLORS[2],
            stat_text=None,
            stats_loc="lower left",
            hlines=None,
            show_xlabel=True,
        )

        draw_grouped_box_panel(
            ax=axes[3],
            df=all_df,
            value_col="spill",
            ylabel=r"Spill discharge (m$^3$/s)",
            title="Spill discharge",
            panel_label="(d)",
            panel_facecolor=PANEL_FACE_COLORS[3],
            stat_text=None,
            stats_loc="lower left",
            hlines=None,
            show_xlabel=True,
        )

        algo_handles = [
            Patch(facecolor=algo_color(algo), edgecolor="black", linewidth=0.8, alpha=0.65, label=algo)
            for algo in algos
        ]
        fig.legend(
            handles=algo_handles,
            labels=algos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(algos),
            frameon=False,
            title="Algorithm",
            title_fontsize=LEGEND_FT,
            fontsize=LEGEND_FT,
        )

        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    print(f"Saved boxplot suite PDF: {pdf_path.resolve()}")
    print(f"Saved boxplot suite PNG: {png_path.resolve()}")


def plot_representative_years_timeseries(
    all_df: pd.DataFrame,
    reference_inflow_df: pd.DataFrame,
    output_dir: Path,
    reservoir: SharedReservoirModel,
) -> None:
    """
    Save representative-year time-series plots for the dry and wet years.
    """
    pdf_path = output_dir / "scenario_representative_year_timeseries.pdf"
    png_path = output_dir / "scenario_representative_year_timeseries.png"

    algos = get_algos_present(all_df)
    if not algos:
        raise ValueError("No algorithm data available for plotting.")

    rep_years = pick_representative_years(reference_inflow_df)
    year_items = [("dry", rep_years["dry"]), ("wet", rep_years["wet"])]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(1, 13)

    inflow_long = reference_inflow_df.melt(id_vars="year", var_name="month_label", value_name="inflow")
    inflow_long["month"] = np.tile(np.arange(1, 13), len(reference_inflow_df))

    with plt.rc_context({"font.sans-serif": ["DejaVu Sans", "Arial"], "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), dpi=600, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, (tag, year) in enumerate(year_items):
            ax = axes[i]
            sub = all_df[all_df["year"] == year].copy()
            if sub.empty:
                continue

            inflow_ref = inflow_long[inflow_long["year"] == year].sort_values("month")

            ax2 = ax.twinx()
            ax2.bar(
                inflow_ref["month"],
                inflow_ref["inflow"],
                width=0.62,
                color=INFLOW_COLOR,
                alpha=0.25,
                edgecolor="none",
                zorder=1,
            )

            ax.set_zorder(2)
            ax.patch.set_alpha(0.0)

            for algo in algos:
                series = sub[sub["algo"] == algo].sort_values("month")
                if series.empty:
                    continue
                ax.plot(
                    series["month"],
                    series["level"],
                    color=algo_color(algo),
                    linewidth=1.8,
                    marker="o",
                    markersize=3.5,
                    label=algo,
                    zorder=3,
                )

            ax.axhline(reservoir.normal_level, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.9)
            ax.axhline(reservoir.dead_level, color="#ff7f0e", linestyle="--", linewidth=1.1, alpha=0.9)

            panel = "(a)" if i == 0 else "(b)"
            year_type = "Representative dry year" if tag == "dry" else "Representative wet year"
            ax.set_title(f"{panel} {year_type} (Year {year})", fontsize=LABEL_FT, loc="left", pad=6)

            ax.set_ylabel("Water level (m)", fontsize=LABEL_FT)
            ax2.set_ylabel(r"Inflow (m$^3$/s)", fontsize=LABEL_FT)

            ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5)
            ax.tick_params(axis="both", labelsize=TICK_FT)
            ax2.tick_params(axis="y", labelsize=TICK_FT)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.35")
            for spine in ax2.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.35")

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(month_labels, fontsize=TICK_FT)
        axes[-1].set_xlabel("Month", fontsize=LABEL_FT)

        algo_handles = [
            Line2D([0], [0], color=algo_color(algo), lw=2, marker="o", markersize=4, label=algo)
            for algo in algos
        ]
        extra_handles = [Patch(facecolor=INFLOW_COLOR, alpha=0.25, edgecolor="none", label="Inflow")]
        extra_handles.append(Line2D([0], [0], color="#d62728", lw=1.2, linestyle="--", label="Normal water level"))
        extra_handles.append(Line2D([0], [0], color="#ff7f0e", lw=1.2, linestyle="--", label="Dead water level"))

        fig.legend(
            handles=algo_handles + extra_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(7, len(algo_handles) + len(extra_handles)),
            frameon=False,
            fontsize=LEGEND_FT,
        )

        fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    print(f"Saved representative-year PDF: {pdf_path.resolve()}")
    print(f"Saved representative-year PNG: {png_path.resolve()}")
    print(f"Representative years: dry = {rep_years['dry']}, wet = {rep_years['wet']}")


def main() -> None:
    outputs_dir = get_outputs_dir()
    visualization_dir = outputs_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    reservoir = SharedReservoirModel(data_dir=get_data_dir())

    selected_files = select_best_eval_file_combination(
        outputs_dir=outputs_dir,
        firm_power=reservoir.firm_power,
        reliability_floor_pct=RELIABILITY_FLOOR_PCT,
    )

    frames: List[pd.DataFrame] = []
    for algo_label in ALGO_ORDER:
        selected_file = selected_files[algo_label]
        reliability_pct = compute_reliability_pct(selected_file, firm_power=reservoir.firm_power)
        frames.append(load_single_result(algo_label, selected_file))
        print(
            f"Selected {algo_label} file: {selected_file.name} "
            f"| reliability = {reliability_pct:.2f}%"
        )

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["algo", "year", "month"]).reset_index(drop=True)
    algos = get_algos_present(all_df)

    reference_inflow_df = load_reference_inflow_table()

    export_summary_table_to_latex(
        all_df=all_df,
        firm_power=reservoir.firm_power,
        tex_path=visualization_dir / "scenario_monthly_summary_table.tex",
        algos=algos,
    )
    plot_energy_inflow_level(all_df, visualization_dir)
    plot_representative_years_timeseries(all_df, reference_inflow_df, visualization_dir, reservoir)
    plot_monthly_boxplots_suite(all_df, visualization_dir, reservoir)

    print("\nScenario visualization completed.")
    print(f"visualization directory: {visualization_dir.resolve()}")


if __name__ == "__main__":
    main()
