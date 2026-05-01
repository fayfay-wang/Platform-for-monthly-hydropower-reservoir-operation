"""
Shared reservoir characteristics module for the unified reservoir operation framework.

This module provides the common physical relationships and shared operating
parameters used by all four algorithms (DP, SDP, ISO-SURF, and DQN), including:
1) water level-to-storage conversion,
2) storage-to-water level conversion,
3) tailwater level estimation from discharge,
4) hydropower output calculation, and
5) unified access to reservoir constraints and basic operating parameters.

The purpose of this module is to ensure that all algorithms are evaluated under the
same reservoir environment, physical assumptions, constraints, and parameter
settings, so that their differences arise from the control strategy rather than
from inconsistencies in the underlying model.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


class SharedReservoirModel:
    """Shared reservoir parameters, constraints, and physical relationships for all algorithms."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = (
            Path(data_dir).resolve()
            if data_dir is not None
            else Path(__file__).resolve().parents[3] / "data"
        )

        # Load reservoir parameters
        self.params = self._load_reservoir_parameters()

        # Expose shared reservoir parameters and operating constraints as attributes
        self.dead_level = float(self._get_param("dead_level"))
        self.normal_level = float(self._get_param("normal_level"))
        self.initial_level = float(self._get_param("initial_level"))
        self.max_turbine_discharge = float(self._get_param("max_turbine_discharge"))
        self.max_power = float(self._get_param("max_power"))
        self.firm_power = float(self._get_param("firm_power"))
        self.power_coefficient = float(self._get_param("power_coefficient"))

        # Optional shared temporal parameter
        self.delta_t_days = float(self._get_param("delta_t_days", default=30.4))
        self.delta_t_hours = self.delta_t_days * 24.0
        self.delta_t_seconds = self.delta_t_hours * 3600.0

        # Load the original water level-storage table
        self._levels, self._storages = self._load_level_storage_data()

        # Prepare for interpolation: sort by water level
        level_sort_idx = np.argsort(self._levels)
        self._levels_h = self._levels[level_sort_idx].astype(float)
        self._storages_h = self._storages[level_sort_idx].astype(float)

        # Prepare for interpolation: sort by storage
        storage_sort_idx = np.argsort(self._storages)
        self._storages_s = self._storages[storage_sort_idx].astype(float)
        self._levels_s = self._levels[storage_sort_idx].astype(float)

        # Boundary limits
        self._curve_level_min = float(self._levels_h[0])
        self._curve_level_max = float(self._levels_h[-1])
        self._curve_storage_min = float(self._storages_s[0])
        self._curve_storage_max = float(self._storages_s[-1])

        # Fit the tailwater rating curve
        self.tailwater_curve_coefficients = self._fit_discharge_tailwater_curve()

    def _get_param(self, *keys, default=None):
        for key in keys:
            if key in self.params:
                return self.params[key]
        if default is not None:
            return default
        raise KeyError(f"Missing required parameter. Expected one of: {keys}")

    def _load_reservoir_parameters(self):
        file_path = self.data_dir / "reservoir_settings.txt"

        if not file_path.exists():
            raise FileNotFoundError(f"Cannot find file: {file_path}")

        text = file_path.read_text(encoding="utf-8")
        pattern = r'"([^"]+)"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        matches = re.findall(pattern, text)
        return {key: float(value) for key, value in matches}

    def _load_level_storage_data(self):
        """Load the original water level-storage curve data from the CSV file."""
        file_path = self.data_dir / "level_storage_curve.csv"
        df = pd.read_csv(file_path)
        water_levels = df["water_level_m"].values.astype(float)
        storage = df["storage_1e9_m3"].values.astype(float)
        return water_levels, storage

    def _load_tailwater_discharge_data(self):
        """Load the tailwater level-discharge curve data from the CSV file."""
        file_path = self.data_dir / "tailwater_discharge_curve.csv"
        df = pd.read_csv(file_path)
        tailwater_levels = df["tail_water_level_m"].values.astype(float)
        discharge = df["discharge_m3s"].values.astype(float)
        return tailwater_levels, discharge

    def _fit_discharge_tailwater_curve(self):
        """Fit a fourth-order polynomial tailwater rating curve as a function of discharge."""
        tailwater_levels, discharge = self._load_tailwater_discharge_data()
        return np.polyfit(discharge, tailwater_levels, 4)

    def calculate_storage(self, level: float) -> float:
        """Convert water level to storage (1e9 m³) using linear interpolation."""
        level = float(np.clip(float(level), self._curve_level_min, self._curve_level_max))
        return float(np.interp(level, self._levels_h, self._storages_h))

    def calculate_level(self, storage: float) -> float:
        """Convert storage (1e9 m³) to water level using linear interpolation."""
        storage = float(np.clip(float(storage), self._curve_storage_min, self._curve_storage_max))
        return float(np.interp(storage, self._storages_s, self._levels_s))

    def calculate_tailwater(self, discharge: float) -> float:
        """Calculate the tailwater level from the total discharge."""
        poly = np.poly1d(self.tailwater_curve_coefficients)
        return float(poly(float(discharge)))

    def calculate_power(
        self,
        turbine_discharge: float,
        total_discharge: float,
        current_level: float,
        next_level: float,
    ) -> float:
        """Calculate hydropower output from turbine discharge and average hydraulic head."""
        tailwater_level = self.calculate_tailwater(total_discharge)
        average_head = (current_level + next_level) / 2 - tailwater_level
        return self.power_coefficient * turbine_discharge * average_head