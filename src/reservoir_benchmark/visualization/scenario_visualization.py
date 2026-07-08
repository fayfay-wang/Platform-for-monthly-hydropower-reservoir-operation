from __future__ import annotations
"""
Scenario-level visualization for the unified reservoir operation framework.

This script builds fixed scenario comparison outputs from the standardized
algorithm evaluation CSV files, including:
1) an energy-inflow-level bubble plot,
2) an energy-inflow envelope comparison plot,
3) representative-year time-series plots,
4) a 2×2 monthly distribution figure, and
5) a LaTeX summary table corresponding to the monthly comparison statistics.

Design choices
--------------
- The script reads standardized evaluation results under
  outputs/evaluation/{algorithm}.
- One evaluation CSV is selected for each algorithm through a reliability
  common-window search across the method-specific penalty sweeps.
- High-reliability candidates are prioritized whenever every algorithm has
  at least one candidate in the high-reliability interval.
- Candidate combinations are ranked primarily by cross-algorithm reliability
  consistency, then by mean reliability and lambda parsimony.
- The visualization outputs are saved under outputs/visualization.
"""

import re
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # Keep the plotting script usable even when SciPy is unavailable.
    PchipInterpolator = None

from reservoir_benchmark.shared.io_utils import load_monthly_inflows_from_excel
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

ALGO_DIR_MAP: Dict[str, str] = {
    "DDP": "ddp",
    "SDP": "sdp",
    "ISO-SURF": "iso_surf",
    "DQN": "dqn",
}
ALGO_ORDER: List[str] = ["DDP", "SDP", "ISO-SURF", "DQN"]

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

# Candidate-pool diagnostics: (reliability floor %, historical minimum candidate ratio).
RELIABILITY_FLOOR_RULES: List[Tuple[float, float]] = [
    (80.0, 0.30),
    (60.0, 0.50),
]


MONTH_COLUMNS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
MONTH_TO_INDEX = {name: i + 1 for i, name in enumerate(MONTH_COLUMNS)}

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


def choose_adaptive_reliability_floor(
    raw_candidate_map: Dict[str, List[Tuple[Path, float, float]]],
    floor_rules: List[Tuple[float, float]],
) -> Tuple[float | None, Dict[float, Dict[str, float]]]:
    """
    Choose a reliability floor according to candidate availability.

    For each reliability floor, the share of candidates satisfying the floor is
    computed separately for each algorithm. A floor is activated only when every
    algorithm meets the required candidate ratio and has at least one valid file.
    """
    ratio_record: Dict[float, Dict[str, float]] = {}

    for floor_pct, min_ratio in floor_rules:
        ratios: Dict[str, float] = {}
        floor_is_available = True

        for algo_label in ALGO_ORDER:
            candidates = raw_candidate_map[algo_label]
            n_total = len(candidates)
            n_valid = sum(1 for item in candidates if item[2] >= floor_pct)
            ratio = n_valid / n_total if n_total > 0 else 0.0
            ratios[algo_label] = ratio

            if n_valid == 0 or ratio < min_ratio:
                floor_is_available = False

        ratio_record[floor_pct] = ratios

        if floor_is_available:
            return floor_pct, ratio_record

    return None, ratio_record


def _candidate_in_interval(reliability_pct: float, lower: float, upper: float) -> bool:
    if upper >= 100.0:
        return lower <= reliability_pct <= upper
    return lower <= reliability_pct < upper


def _local_reference_tolerance(reference_values: np.ndarray, idx: int, max_tolerance: float) -> float:
    """
    Compute a local matching tolerance around a reference reliability value.
    """
    if len(reference_values) <= 1:
        return float(max_tolerance)

    gaps = []
    if idx > 0:
        gaps.append(reference_values[idx] - reference_values[idx - 1])
    if idx < len(reference_values) - 1:
        gaps.append(reference_values[idx + 1] - reference_values[idx])

    local_gap = float(np.nanmean(gaps)) if gaps else float(max_tolerance)
    local_tol = max(1.0, local_gap * 1.25)
    return float(min(max_tolerance, local_tol))


def _find_nearest_match_for_reference(
    ref_reliability: float,
    ref_idx: int,
    reference_values: np.ndarray,
    interval_candidate_map: Dict[str, List[Tuple[Path, float, float]]],
    max_tolerance: float,
) -> Dict[str, object]:
    """
    Match each algorithm to the nearest candidate around one reference reliability.
    """
    adaptive_tol = _local_reference_tolerance(reference_values, ref_idx, max_tolerance)
    selected: Dict[str, Path] = {}
    reliabilities: Dict[str, float | None] = {}
    lambdas: Dict[str, float | None] = {}
    diffs: Dict[str, float | None] = {}

    for algo_label in ALGO_ORDER:
        candidates = interval_candidate_map.get(algo_label, [])
        if not candidates:
            reliabilities[algo_label] = None
            lambdas[algo_label] = None
            diffs[algo_label] = None
            continue

        nearest = min(candidates, key=lambda item: (abs(item[2] - ref_reliability), item[1], item[0].name.lower()))
        diff = abs(nearest[2] - ref_reliability)

        if diff <= adaptive_tol:
            selected[algo_label] = nearest[0]
            reliabilities[algo_label] = float(nearest[2])
            lambdas[algo_label] = float(nearest[1])
            diffs[algo_label] = float(diff)
        else:
            reliabilities[algo_label] = None
            lambdas[algo_label] = None
            diffs[algo_label] = None

    finite_diffs = [value for value in diffs.values() if value is not None]
    finite_lambdas = [value for value in lambdas.values() if value is not None and np.isfinite(value)]
    finite_reliabilities = [value for value in reliabilities.values() if value is not None]

    return {
        "reference": float(ref_reliability),
        "adaptive_tolerance": float(adaptive_tol),
        "selected": selected,
        "reliabilities": reliabilities,
        "lambdas": lambdas,
        "diffs": diffs,
        "n_matched": len(selected),
        "max_diff": float(max(finite_diffs)) if finite_diffs else float("inf"),
        "mean_diff": float(np.mean(finite_diffs)) if finite_diffs else float("inf"),
        "lambda_sum": float(np.sum(finite_lambdas)) if finite_lambdas else float("inf"),
        "mean_reliability": float(np.mean(finite_reliabilities)) if finite_reliabilities else float("nan"),
    }


def _build_match_record(
    combo: Tuple[Tuple[Path, float, float], ...],
    reference: float | None = None,
) -> Dict[str, object]:
    """
    Build a standard match record from one candidate combination.

    Each candidate tuple is (csv_path, lambda_value, reliability_pct), ordered
    consistently with ALGO_ORDER.
    """
    selected: Dict[str, Path] = {}
    reliabilities: Dict[str, float] = {}
    lambdas: Dict[str, float] = {}

    for algo_label, item in zip(ALGO_ORDER, combo):
        selected[algo_label] = item[0]
        lambdas[algo_label] = float(item[1])
        reliabilities[algo_label] = float(item[2])

    reliability_values = np.array([reliabilities[algo] for algo in ALGO_ORDER], dtype=float)
    lambda_values = np.array(
        [lambdas[algo] for algo in ALGO_ORDER if np.isfinite(lambdas[algo])],
        dtype=float,
    )

    if reference is None:
        reference_value = float(np.mean(reliability_values))
    else:
        reference_value = float(reference)

    diffs = {
        algo_label: float(abs(reliabilities[algo_label] - reference_value))
        for algo_label in ALGO_ORDER
    }

    min_reliability = float(np.min(reliability_values))
    max_reliability = float(np.max(reliability_values))
    reliability_range = float(max_reliability - min_reliability)
    mean_reliability = float(np.mean(reliability_values))
    max_diff = float(max(diffs.values()))
    mean_diff = float(np.mean(list(diffs.values())))
    lambda_sum = float(np.sum(lambda_values)) if len(lambda_values) else float("inf")

    return {
        "reference": reference_value,
        "selected": selected,
        "reliabilities": reliabilities,
        "lambdas": lambdas,
        "diffs": diffs,
        "n_matched": len(selected),
        "min_reliability": min_reliability,
        "max_reliability": max_reliability,
        "reliability_range": reliability_range,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "lambda_sum": lambda_sum,
        "mean_reliability": mean_reliability,
    }


def _match_sort_key(
    match: Dict[str, object],
    prefer_higher_reliability: bool = False,
) -> Tuple[float, float, float, float, float, float]:
    """
    Ranking rule for scenario selection.

    When a tolerance-constrained high-reliability window exists, the first goal
    is to keep all algorithms as reliable as possible, so the ranking maximizes
    the minimum selected reliability and then the mean selected reliability.
    Within that high-reliability feasible set, the reliability range remains the
    next key comparability criterion.

    For lower-reliability fallback or no-tolerance cases, the ranking first
    minimizes the reliability range.
    """
    if prefer_higher_reliability:
        return (
            -float(match["min_reliability"]),
            -float(match["mean_reliability"]),
            float(match["reliability_range"]),
            float(match["max_diff"]),
            float(match["mean_diff"]),
            float(match["lambda_sum"]),
        )

    return (
        float(match["reliability_range"]),
        float(match["max_diff"]),
        float(match["mean_diff"]),
        -float(match["mean_reliability"]),
        float(match["lambda_sum"]),
        -float(match["min_reliability"]),
    )


def _select_best_interval_common_window(
    raw_candidate_map: Dict[str, List[Tuple[Path, float, float]]],
    lower: float,
    upper: float,
    max_reliability_range: float | None = None,
    force_interval: bool = False,
    exact_combo_limit: int = 10_000_000,
    prefer_higher_reliability_in_limit: bool = False,
) -> Dict[str, object] | None:
    """
    Select one candidate per algorithm within a reliability interval.

    This function replaces the previous reference-point nearest-neighbor gate.
    It searches for the best common reliability window, where the main window
    width is max(selected reliability) - min(selected reliability). This avoids
    missing valid comparable points when candidate reliabilities are dense and
    the local adaptive tolerance becomes too restrictive.

    If max_reliability_range is provided, the function first tries to return a
    combination whose reliability range is within that limit. If
    prefer_higher_reliability_in_limit is True, feasible combinations are ranked
    by the highest minimum reliability before secondary comparability criteria.
    When force_interval is True, it still returns the best interval-internal
    combination even if the
    range exceeds the requested limit; this is used only for the high-reliability
    interval so that the final selection does not fall back to lower reliability
    while high-reliability candidates exist for all algorithms.
    """
    interval_candidate_map: Dict[str, List[Tuple[Path, float, float]]] = {}

    for algo_label in ALGO_ORDER:
        interval_candidate_map[algo_label] = [
            item for item in raw_candidate_map[algo_label]
            if _candidate_in_interval(item[2], lower, upper)
        ]

    print(f"Common-window matching in [{lower:.0f}, {upper:.0f}]%:")
    for algo_label in ALGO_ORDER:
        candidates = interval_candidate_map[algo_label]
        if candidates:
            rel_values = [item[2] for item in candidates]
            print(
                f"  {algo_label}: {len(candidates)} candidates | "
                f"reliability range = {min(rel_values):.2f}--{max(rel_values):.2f}%"
            )
        else:
            print(f"  {algo_label}: 0 candidates")

    missing_algos = [
        algo_label
        for algo_label in ALGO_ORDER
        if len(interval_candidate_map[algo_label]) == 0
    ]
    if missing_algos:
        print(
            f"  interval [{lower:.0f}, {upper:.0f}]% is unavailable because "
            f"these algorithms have no candidates: {missing_algos}"
        )
        return None

    candidate_lists = [interval_candidate_map[algo_label] for algo_label in ALGO_ORDER]
    total_combinations = int(np.prod([len(items) for items in candidate_lists]))
    print(f"  candidate combinations to check: {total_combinations}")

    best_any: Dict[str, object] | None = None
    best_within_limit: Dict[str, object] | None = None

    if total_combinations <= exact_combo_limit:
        for combo in product(*candidate_lists):
            match = _build_match_record(combo)
            key_any = _match_sort_key(match, prefer_higher_reliability=False)
            key_within = _match_sort_key(
                match,
                prefer_higher_reliability=prefer_higher_reliability_in_limit,
            )

            if best_any is None or key_any < _match_sort_key(best_any, prefer_higher_reliability=False):
                best_any = match

            if max_reliability_range is None or match["reliability_range"] <= max_reliability_range:
                if (
                    best_within_limit is None
                    or key_within < _match_sort_key(
                        best_within_limit,
                        prefer_higher_reliability=prefer_higher_reliability_in_limit,
                    )
                ):
                    best_within_limit = match
    else:
        print(
            f"  exact enumeration skipped because {total_combinations} combinations "
            f"exceed exact_combo_limit={exact_combo_limit}; using reference scan."
        )
        reference_values = sorted(
            set(
                round(float(item[2]), 6)
                for candidates in candidate_lists
                for item in candidates
            )
        )

        for ref_reliability in reference_values:
            combo = tuple(
                min(
                    candidates,
                    key=lambda item: (
                        abs(item[2] - ref_reliability),
                        item[1],
                        item[0].name.lower(),
                    ),
                )
                for candidates in candidate_lists
            )
            match = _build_match_record(combo, reference=ref_reliability)
            key_any = _match_sort_key(match, prefer_higher_reliability=False)
            key_within = _match_sort_key(
                match,
                prefer_higher_reliability=prefer_higher_reliability_in_limit,
            )

            if best_any is None or key_any < _match_sort_key(best_any, prefer_higher_reliability=False):
                best_any = match

            if max_reliability_range is None or match["reliability_range"] <= max_reliability_range:
                if (
                    best_within_limit is None
                    or key_within < _match_sort_key(
                        best_within_limit,
                        prefer_higher_reliability=prefer_higher_reliability_in_limit,
                    )
                ):
                    best_within_limit = match

    if best_any is None:
        return None

    if max_reliability_range is not None and best_within_limit is None:
        print(
            f"  no combination within the requested reliability range "
            f"<= {max_reliability_range:.2f}%; best available range is "
            f"{best_any['reliability_range']:.2f}%"
        )
        if not force_interval:
            return None
        best = best_any
    else:
        best = best_within_limit if best_within_limit is not None else best_any

    print(
        f"  selected common window = "
        f"{best['min_reliability']:.2f}--{best['max_reliability']:.2f}% | "
        f"range = {best['reliability_range']:.2f}% | "
        f"mean reliability = {best['mean_reliability']:.2f}%"
    )
    return best


def _print_selected_match(match: Dict[str, object]) -> None:
    """
    Print the selected reliability, matching difference, and lambda by algorithm.
    """
    print("Selected reliability by algorithm:")
    reliabilities = match["reliabilities"]
    diffs = match["diffs"]
    lambdas = match["lambdas"]

    for algo_label in ALGO_ORDER:
        reliability = reliabilities[algo_label]
        diff = diffs[algo_label]
        lambda_value = lambdas[algo_label]
        print(
            f"  {algo_label}: {reliability:.2f}% "
            f"(diff from mean = {diff:.2f}%, lambda = {lambda_value:.2f})"
        )


def _fallback_nearest_common_reliability(
    raw_candidate_map: Dict[str, List[Tuple[Path, float, float]]],
) -> Dict[str, Path]:
    """
    Select nearest candidates around the median reliability as a fallback.
    """
    all_reliabilities = [
        item[2]
        for algo_label in ALGO_ORDER
        for item in raw_candidate_map[algo_label]
    ]
    if not all_reliabilities:
        raise ValueError("No evaluation candidates are available.")

    reference = float(np.nanmedian(all_reliabilities))
    selected: Dict[str, Path] = {}

    print(
        "No full segmented common-window match was found. "
        f"Using nearest candidates around the median reliability ({reference:.2f}%)."
    )

    for algo_label in ALGO_ORDER:
        candidates = raw_candidate_map[algo_label]
        nearest = min(candidates, key=lambda item: (abs(item[2] - reference), item[1], item[0].name.lower()))
        selected[algo_label] = nearest[0]
        print(
            f"  {algo_label}: {nearest[0].name} | "
            f"reliability = {nearest[2]:.2f}% | lambda = {nearest[1]:.2f}"
        )

    return selected


def select_best_eval_file_combination(
    outputs_dir: Path,
    firm_power: float,
    floor_rules: List[Tuple[float, float]] | None = None,
) -> Dict[str, Path]:
    """
    Select one evaluation CSV per algorithm using high-priority common-window matching.
    """
    if floor_rules is None:
        floor_rules = RELIABILITY_FLOOR_RULES

    raw_candidate_map: Dict[str, List[Tuple[Path, float, float]]] = {}

    for algo_label in ALGO_ORDER:
        algo_dir_name = ALGO_DIR_MAP[algo_label]
        eval_dir = outputs_dir / "evaluation" / algo_dir_name
        raw_candidate_map[algo_label] = collect_eval_candidates(
            algo_label,
            eval_dir,
            firm_power=firm_power,
        )

    print("Reliability coverage diagnostics:")
    floor_has_candidate_for_all: Dict[float, bool] = {}

    for floor_pct, min_ratio in floor_rules:
        has_candidate_for_all = True
        print(
            f"  Floor >= {floor_pct:.2f}% | "
            f"candidate ratio shown for reference only; "
            f"minimum-ratio gate disabled"
        )

        for algo_label in ALGO_ORDER:
            candidates = raw_candidate_map[algo_label]
            n_total = len(candidates)
            n_valid = sum(1 for item in candidates if item[2] >= floor_pct)
            ratio = n_valid / n_total if n_total > 0 else 0.0
            print(f"    {algo_label}: {n_valid}/{n_total} candidates ({ratio * 100:.1f}%)")

            if n_valid == 0:
                has_candidate_for_all = False

        floor_has_candidate_for_all[floor_pct] = has_candidate_for_all

    target_range = 6.0

    # Hard priority: if every algorithm has at least one candidate in [80, 100]%,
    # the selected combination must come from this interval. The common-window
    # search then minimizes cross-algorithm reliability spread directly, so it
    # will not miss valid high-reliability combinations simply because one local
    # reference-point tolerance was too small.
    if floor_has_candidate_for_all.get(80.0, False):
        print(
            "High-reliability interval [80, 100]% will be searched first "
            "and kept as the final interval whenever all algorithms have candidates."
        )
        high_match = _select_best_interval_common_window(
            raw_candidate_map=raw_candidate_map,
            lower=80.0,
            upper=100.0,
            max_reliability_range=target_range,
            force_interval=True,
            prefer_higher_reliability_in_limit=True,
        )

        if high_match is not None and len(high_match["selected"]) == len(ALGO_ORDER):
            _print_selected_match(high_match)
            return high_match["selected"]

    else:
        print(
            "High-reliability interval [80, 100]% skipped because "
            "at least one algorithm has no candidate in this interval."
        )

    # Lower reliability is considered only when the high-reliability interval is
    # not available for all algorithms.
    low_match = _select_best_interval_common_window(
        raw_candidate_map=raw_candidate_map,
        lower=60.0,
        upper=80.0,
        max_reliability_range=target_range,
        force_interval=False,
        prefer_higher_reliability_in_limit=False,
    )

    if low_match is not None and len(low_match["selected"]) == len(ALGO_ORDER):
        _print_selected_match(low_match)
        return low_match["selected"]

    return _fallback_nearest_common_reliability(raw_candidate_map)


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

def build_reference_inflow_long(reference_inflow_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the evaluation inflow table to a strict year-month long format.
    """
    month_cols = [col for col in MONTH_COLUMNS if col in reference_inflow_df.columns]

    if len(month_cols) != 12:
        raise ValueError(
            "Reference inflow table must contain 12 monthly columns: "
            f"{MONTH_COLUMNS}. Found: {month_cols}"
        )

    inflow_long = reference_inflow_df.melt(
        id_vars="year",
        value_vars=month_cols,
        var_name="month_label",
        value_name="inflow",
    )

    inflow_long["month"] = inflow_long["month_label"].map(MONTH_TO_INDEX).astype(int)
    inflow_long["inflow"] = pd.to_numeric(inflow_long["inflow"], errors="coerce")

    return (
        inflow_long
        .dropna(subset=["inflow"])
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

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
    inflow_long = build_reference_inflow_long(reference_inflow_df)

    annual = (
        inflow_long
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


def plot_energy_inflow_level(all_df: pd.DataFrame, output_dir: Path, reservoir: SharedReservoirModel,
) -> None:
    """
    Save the scenario energy-inflow-level bubble plot as a 2 x 2 algorithm panel.

    The four algorithms are plotted in separate subplots with identical x/y ranges
    and tick locations. Bubble size and bubble color jointly represent the mean
    head class rather than the algorithm identity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "scenario_energy_inflow_level_bubble.pdf"
    png_path = output_dir / "scenario_energy_inflow_level_bubble.png"

    df = all_df.copy()

    power = pd.to_numeric(df["power"], errors="coerce").to_numpy(dtype=float)
    turbine = pd.to_numeric(df["turbine"], errors="coerce").to_numpy(dtype=float)
    denom = reservoir.power_coefficient * turbine

    df["head"] = np.where(
        (denom > 1e-9) & np.isfinite(power),
        power / denom,
        np.nan,
    )

    annual = (
        df.groupby(["algo", "year"], as_index=False)
        .agg(
            inflow_mean=("inflow", "mean"),
            energy_sum=("energy", "sum"),
            head_mean=("head", "mean"),
        )
    )

    # Average repeated annual inflow cases within each algorithm.
    annual["inflow_key"] = pd.to_numeric(annual["inflow_mean"], errors="coerce").round(6)
    annual_plot = (
        annual.groupby(["algo", "inflow_key"], as_index=False)
        .agg(
            inflow_mean=("inflow_mean", "mean"),
            energy_sum=("energy_sum", "mean"),
            head_mean=("head_mean", "mean"),
        )
        .sort_values(["algo", "inflow_mean"])
        .reset_index(drop=True)
    )

    annual_plot["energy_twh"] = annual_plot["energy_sum"] / 1000.0  # GWh/yr -> TWh/yr

    x_all = pd.to_numeric(annual_plot["inflow_mean"], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(annual_plot["energy_twh"], errors="coerce").to_numpy(dtype=float)

    x_all = x_all[np.isfinite(x_all)]
    y_all = y_all[np.isfinite(y_all)]

    if len(x_all) == 0 or len(y_all) == 0:
        raise ValueError("No valid bubble-plot data available.")

    def compact_limits(values, pad_ratio=0.05, round_to=None):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        vrange = vmax - vmin

        if np.isclose(vrange, 0.0):
            vrange = max(abs(vmin), 1.0) * 0.1

        pad = vrange * pad_ratio
        low = vmin - pad
        high = vmax + pad

        if round_to is not None and round_to > 0:
            low = np.floor(low / round_to) * round_to
            high = np.ceil(high / round_to) * round_to

        return low, high

    xlim = compact_limits(x_all, pad_ratio=0.03, round_to=None)
    ylim = compact_limits(y_all, pad_ratio=0.03, round_to=None)

    head_values = annual_plot["head_mean"].to_numpy(dtype=float)
    finite_heads = head_values[np.isfinite(head_values)]

    head_values = annual_plot["head_mean"].to_numpy(dtype=float)
    finite_heads = head_values[np.isfinite(head_values)]

    if len(finite_heads) == 0:
        h_min, h_max = 0.0, 1.0
    else:
        h_min, h_max = np.nanpercentile(finite_heads, [5, 95])
        if np.isclose(h_min, h_max):
            h_min = float(np.nanmin(finite_heads))
            h_max = float(np.nanmax(finite_heads))
        if np.isclose(h_min, h_max):
            h_max = h_min + 1.0

    # Bubble size/color classes: smallest to largest.
    bubble_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    bubble_diameters = np.array([5.0, 9.3, 13.6, 18.0], dtype=float)
    bubble_sizes = bubble_diameters ** 2
    head_bin_edges = np.linspace(h_min, h_max, 5)

    def head_to_class(value: float) -> int:
        """Map mean head to one of four ordered bubble classes."""
        if not np.isfinite(value):
            return 0
        clipped = float(np.clip(value, h_min, h_max))
        class_idx = np.digitize(clipped, head_bin_edges[1:-1], right=False)
        return int(np.clip(class_idx, 0, 3))

    annual_plot["_size_class"] = annual_plot["head_mean"].apply(head_to_class)
    annual_plot["_size"] = annual_plot["_size_class"].map(lambda idx: bubble_sizes[int(idx)])
    annual_plot["_bubble_color"] = annual_plot["_size_class"].map(lambda idx: bubble_colors[int(idx)])

    # Use a single global plotting window so that all four panels are directly comparable.
    x_values = pd.to_numeric(annual_plot["inflow_mean"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(annual_plot["energy_twh"], errors="coerce").to_numpy(dtype=float)
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]

    if len(finite_x) == 0 or len(finite_y) == 0:
        raise ValueError("No finite annual inflow or energy values are available for bubble plotting.")

    x_min, x_max = float(np.nanmin(finite_x)), float(np.nanmax(finite_x))
    y_min, y_max = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
    x_pad = (x_max - x_min) * 0.045 if not np.isclose(x_min, x_max) else max(abs(x_min) * 0.045, 1.0)
    y_pad = (y_max - y_min) * 0.075 if not np.isclose(y_min, y_max) else max(abs(y_min) * 0.075, 0.1)
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)

    panel_items = [
        ("(a) DDP", "DDP"),
        ("(b) SDP", "SDP"),
        ("(c) ISO-SURF", "ISO-SURF"),
        ("(d) DQN", "DQN"),
    ]

    legend_labels = [
        f"{head_bin_edges[i]:.1f}–{head_bin_edges[i + 1]:.1f} m"
        for i in range(4)
    ]

    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.5, 8.2),
            dpi=600,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.ravel()

        for idx, (panel_label, algo) in enumerate(panel_items):
            ax = axes_flat[idx]
            subset = annual_plot[annual_plot["algo"] == algo]

            if not subset.empty:
                ax.scatter(
                    subset["inflow_mean"],
                    subset["energy_twh"],
                    s=subset["_size"],
                    c=subset["_bubble_color"],
                    alpha=0.68,
                    edgecolors="black",
                    linewidths=0.45,
                )

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.grid(True, linestyle="--", alpha=0.32, linewidth=0.55)
            ax.tick_params(axis="both", labelsize=TICK_FT)
            ax.text(
                0.015,
                0.985,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
            )

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.35")

        # Keep y-axis information only for the left-column panels.
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean Annual Electricity (TWh/yr)", fontsize=LABEL_FT)
        for ax in axes[:, 1]:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Keep x-axis information only for the bottom-row panels.
        for ax in axes[0, :]:
            ax.set_xlabel("")
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for ax in axes[1, :]:
            ax.set_xlabel("Mean Annual Inflow (m³/s)", fontsize=LABEL_FT)
            ax.tick_params(axis="x", bottom=True, labelbottom=True)

        # Place the only bubble-size/color legend at the figure top to avoid covering data points.
        legend_ax = axes[0, 1]
        size_handles = [
            legend_ax.scatter(
                [],
                [],
                s=bubble_sizes[i],
                color=bubble_colors[i],
                edgecolors="black",
                linewidths=0.45,
                alpha=0.68,
            )
            for i in range(4)
        ]
        fig.legend(
            handles=size_handles,
            labels=legend_labels,
            title="Mean head class",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            frameon=False,
            fontsize=LEGEND_FT - 1,
            title_fontsize=LEGEND_FT - 1,
            labelspacing=0.55,
            handletextpad=0.8,
            columnspacing=1.25,
        )

        fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.935], w_pad=1.1, h_pad=1.15)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    print(f"Saved bubble plot: {pdf_path.resolve()}")


def _prepare_unique_xy(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort x-y pairs, remove non-finite values, and collapse duplicate x values.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)

    grouped = (
        pd.DataFrame({"x": x[mask], "y": y[mask]})
        .groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
        .reset_index(drop=True)
    )
    return grouped["x"].to_numpy(dtype=float), grouped["y"].to_numpy(dtype=float)


def _evaluate_smooth_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a smooth, shape-preserving curve on a target x grid.
    """
    x, y = _prepare_unique_xy(x_values, y_values)
    if len(x) == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    if len(x) == 1:
        return np.full_like(x_grid, y[0], dtype=float)

    if PchipInterpolator is not None and len(x) >= 3:
        interpolator = PchipInterpolator(x, y, extrapolate=False)
        y_grid = interpolator(x_grid)
        edge_mask = ~np.isfinite(y_grid)
        if np.any(edge_mask):
            y_grid[edge_mask] = np.interp(x_grid[edge_mask], x, y)
        return np.asarray(y_grid, dtype=float)

    return np.interp(x_grid, x, y)


def _build_energy_inflow_envelope(
    annual_df: pd.DataFrame,
    algo: str,
    n_bins: int = 32,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Build a binned annual inflow-energy envelope for one algorithm.

    Quantile bins are used rather than equal-width bins so that each local
    segment contains a comparable number of annual samples. This stabilizes the
    envelope when annual inflow values are unevenly distributed.
    """
    subset = annual_df[annual_df["algo"] == algo].copy()
    subset["inflow_mean"] = pd.to_numeric(subset["inflow_mean"], errors="coerce")
    subset["energy_twh"] = pd.to_numeric(subset["energy_twh"], errors="coerce")
    subset = subset.dropna(subset=["inflow_mean", "energy_twh"]).sort_values("inflow_mean")

    if len(subset) < 6:
        return pd.DataFrame(columns=["inflow_mean", "energy_mid", "energy_low", "energy_high", "n"])

    # Keep bins broad enough to form a stable envelope but numerous enough to
    # retain the nonlinear inflow-generation relationship.
    bin_count = min(n_bins, max(6, len(subset) // 8))
    bin_count = min(bin_count, len(subset))
    subset = subset.reset_index(drop=True)
    subset["_rank_bin"] = pd.qcut(
        np.arange(len(subset), dtype=float),
        q=bin_count,
        duplicates="drop",
    )

    envelope = (
        subset.groupby("_rank_bin", observed=True)
        .agg(
            inflow_mean=("inflow_mean", "median"),
            energy_mid=("energy_twh", "median"),
            energy_low=("energy_twh", lambda values: values.quantile(lower_quantile)),
            energy_high=("energy_twh", lambda values: values.quantile(upper_quantile)),
            n=("energy_twh", "size"),
        )
        .dropna(subset=["inflow_mean", "energy_mid", "energy_low", "energy_high"])
        .sort_values("inflow_mean")
        .reset_index(drop=True)
    )

    return envelope


def plot_energy_inflow_envelope_comparison(all_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save a wide envelope plot comparing annual inflow-generation relationships.

    Each algorithm is represented by a smooth median annual generation curve and
    a semi-transparent 10%--90% quantile envelope. This plot complements the
    four-panel bubble figure by placing all algorithms in one common view.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "scenario_energy_inflow_envelope_comparison.pdf"
    png_path = output_dir / "scenario_energy_inflow_envelope_comparison.png"

    annual_df = (
        all_df.groupby(["algo", "year"], as_index=False)
        .agg(
            inflow_mean=("inflow", "mean"),
            energy_sum=("energy", "sum"),
        )
        .sort_values(["algo", "inflow_mean"])
        .reset_index(drop=True)
    )
    annual_df["energy_twh"] = annual_df["energy_sum"] / 1000.0  # GWh/yr -> TWh/yr
    annual_df = annual_df.dropna(subset=["inflow_mean", "energy_twh"])

    algos = get_algos_present(annual_df)
    if not algos:
        raise ValueError("No algorithm data available for envelope plotting.")

    x_values = pd.to_numeric(annual_df["inflow_mean"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(annual_df["energy_twh"], errors="coerce").to_numpy(dtype=float)
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]

    if len(finite_x) == 0 or len(finite_y) == 0:
        raise ValueError("No finite annual inflow or energy values are available for envelope plotting.")

    x_min, x_max = float(np.nanmin(finite_x)), float(np.nanmax(finite_x))
    y_min, y_max = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
    x_pad = (x_max - x_min) * 0.045 if not np.isclose(x_min, x_max) else max(abs(x_min) * 0.045, 1.0)
    y_pad = (y_max - y_min) * 0.100 if not np.isclose(y_min, y_max) else max(abs(y_min) * 0.100, 0.1)

    with plt.rc_context(FONT_RC):
        fig, ax = plt.subplots(figsize=(13.2, 4.4), dpi=600)

        for algo in algos:
            envelope = _build_energy_inflow_envelope(annual_df, algo=algo)
            if envelope.empty or len(envelope) < 2:
                continue

            x_env = envelope["inflow_mean"].to_numpy(dtype=float)
            x_grid = np.linspace(float(np.nanmin(x_env)), float(np.nanmax(x_env)), 360)

            y_mid = _evaluate_smooth_curve(
                x_env,
                envelope["energy_mid"].to_numpy(dtype=float),
                x_grid,
            )
            y_low = _evaluate_smooth_curve(
                x_env,
                envelope["energy_low"].to_numpy(dtype=float),
                x_grid,
            )
            y_high = _evaluate_smooth_curve(
                x_env,
                envelope["energy_high"].to_numpy(dtype=float),
                x_grid,
            )

            y_lower = np.minimum(y_low, y_high)
            y_upper = np.maximum(y_low, y_high)
            y_mid = np.clip(y_mid, y_lower, y_upper)

            color = algo_color(algo)
            ax.fill_between(
                x_grid,
                y_lower,
                y_upper,
                color=color,
                alpha=0.16,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(x_grid, y_lower, color=color, linewidth=0.8, alpha=0.28, zorder=2)
            ax.plot(x_grid, y_upper, color=color, linewidth=0.8, alpha=0.28, zorder=2)
            ax.plot(x_grid, y_mid, color=color, linewidth=2.2, alpha=0.96, label=algo, zorder=3)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Mean Annual Inflow (m³/s)", fontsize=LABEL_FT)
        ax.set_ylabel("Annual Electricity Generation (TWh/yr)", fontsize=LABEL_FT)
        ax.tick_params(axis="both", labelsize=TICK_FT)
        ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.55)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.35")

        legend_handles = [
            Line2D([0], [0], color=algo_color(algo), lw=2.2, label=algo)
            for algo in algos
        ]
        fig.legend(
            handles=legend_handles,
            labels=algos,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(algos),
            frameon=False,
            fontsize=LEGEND_FT,
            columnspacing=1.25,
            handlelength=2.2,
        )

        fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.90])
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    print(f"Saved energy-inflow envelope PDF: {pdf_path.resolve()}")
    print(f"Saved energy-inflow envelope PNG: {png_path.resolve()}")


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
    ax.tick_params(axis="y", labelsize=TICK_FT)

    if show_xlabel:
        ax.set_xticklabels(months_abbr, fontsize=TICK_FT)
        ax.tick_params(axis="x", labelsize=TICK_FT, bottom=True, labelbottom=True)
        ax.set_xlabel("Month", fontsize=LABEL_FT)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_xlabel("")

    ax.set_ylabel(ylabel, fontsize=LABEL_FT)

    if title:
        ax.set_title(title, fontsize=LABEL_FT, pad=6)

    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.5)

    ax.set_xlim(0.45, 12.55)
    ax.set_xticks(base_pos)
    ax.tick_params(axis="y", labelsize=TICK_FT)

    if show_xlabel:
        ax.set_xticklabels(months_abbr, fontsize=TICK_FT)
        ax.tick_params(axis="x", labelsize=TICK_FT, bottom=True, labelbottom=True)
        ax.set_xlabel("Month", fontsize=LABEL_FT)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_xlabel("")

    ax.set_ylabel(ylabel, fontsize=LABEL_FT)

    if title:
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
    Save the monthly comparison figure as vertically stacked boxplots.
    """
    pdf_path = output_dir / "scenario_monthly_comparison_boxplots.pdf"
    png_path = output_dir / "scenario_monthly_comparison_boxplots.png"

    algos = get_algos_present(all_df)
    if not algos:
        raise ValueError("No algorithm data available for plotting.")

    plot_df = all_df.copy()
    plot_df["power_gw"] = pd.to_numeric(plot_df["power"], errors="coerce") / 1e6  #  kW -> GW

    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11.5, 10.2),
            dpi=600,
            sharex=True,
        )

        power_hlines = [
            (reservoir.max_power / 1e6, "Installed capacity", "#d62728", "--"),
            (reservoir.firm_power / 1e6, "Firm-power requirement", "#2ca02c", "--"),
        ]
        draw_grouped_box_panel(
            ax=axes[0],
            df=plot_df,
            value_col="power_gw",
            ylabel="Power (GW)",
            title="",
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
            (reservoir.dead_level, "Dead water level", "#2ca02c", "--"),
        ]
        draw_grouped_box_panel(
            ax=axes[1],
            df=all_df,
            value_col="level",
            ylabel="Water level (m)",
            title="",
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
            value_col="spill",
            ylabel=r"Spill discharge (m$^3$/s)",
            title="",
            panel_label="(c)",
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
            fontsize=LEGEND_FT,
        )

        fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.962])
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

    inflow_long = build_reference_inflow_long(reference_inflow_df)

    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), dpi=600, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, (tag, year) in enumerate(year_items):
            ax = axes[i]
            sub = all_df[all_df["year"] == year].copy()
            if sub.empty:
                continue

            inflow_ref = (
                inflow_long[inflow_long["year"] == year]
                .groupby("month", as_index=False)["inflow"]
                .mean()
                .sort_values("month")
            )

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
            ax.set_title(f"{panel} {year_type}", fontsize=LABEL_FT, loc="left", pad=6)

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

        fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.958])
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
        floor_rules=RELIABILITY_FLOOR_RULES,
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
    plot_energy_inflow_level(all_df, visualization_dir, reservoir)
    plot_energy_inflow_envelope_comparison(all_df, visualization_dir)
    plot_representative_years_timeseries(all_df, reference_inflow_df, visualization_dir, reservoir)
    plot_monthly_boxplots_suite(all_df, visualization_dir, reservoir)

    print("\nScenario visualization completed.")
    print(f"visualization directory: {visualization_dir.resolve()}")


if __name__ == "__main__":
    main()
