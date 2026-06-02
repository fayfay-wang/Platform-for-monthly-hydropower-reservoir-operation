"""
Deterministic dynamic programming solver module for the unified reservoir
operation framework.

This module implements the core deterministic dynamic programming (DDP)
optimization logic for single-reservoir operation, including:
1) one-step reservoir transition calculation,
2) hydraulic head and hydropower computation under shared reservoir physics,
3) soft penalty treatment for firm-power deficit,
4) backward Bellman recursion over discretized water levels, and
5) forward simulation of the optimal trajectory.

The purpose of this module is to isolate the algorithmic core of DDP from
project-specific path handling, file input/output, and script-level execution,
so that the solver remains focused on optimization logic and can be reused
cleanly within the unified framework.
"""

from __future__ import annotations

import numpy as np


class ReservoirOperation:
    """
    Deterministic dynamic programming (DDP) for single-reservoir operation.

    This solver only contains the DDP logic itself.
    It does not handle project paths, file reading, or output writing.
    """

    def __init__(self, reservoir, inflows: np.ndarray):
        self.reservoir = reservoir
        self.inflows = inflows  # shape: [years, 12]
        self.total_years = len(self.inflows)

        # Shared parameters from SharedReservoirModel
        self.dead_level = self.reservoir.dead_level
        self.normal_level = self.reservoir.normal_level
        self.initial_level = self.reservoir.initial_level
        self.max_turbine_discharge = self.reservoir.max_turbine_discharge
        self.max_power = self.reservoir.max_power
        self.firm_power = self.reservoir.firm_power
        self.power_coefficient = self.reservoir.power_coefficient
        self.delta_t_days = self.reservoir.delta_t_days

        # Time settings
        self.months_per_year = 12

        # State discretization
        self.levels = np.arange(self.dead_level, self.normal_level + 1, 1)
        self.volumes = np.array([self.reservoir.calculate_storage(h) for h in self.levels])

    def _calc_penalized_energy_gwh(
        self,
        power_kw: float,
        firm_power_penalty_lambda: float,
    ) -> float:
        """
        Compute monthly electricity (GWh) with an optional soft penalty
        for firm-power deficit.
        """
        hours = self.delta_t_days * 24.0
        energy_gwh = power_kw * hours / 1e6

        if self.firm_power > 0.0 and firm_power_penalty_lambda > 0.0:
            deficit_kw = max(0.0, self.firm_power - power_kw)
            if deficit_kw > 0.0:
                deficit_energy_gwh = deficit_kw * hours / 1e6
                energy_gwh -= firm_power_penalty_lambda * deficit_energy_gwh

        return energy_gwh

    def _calculate_average_head(
        self,
        start_level: float,
        end_level: float,
        total_discharge: float,
    ) -> float:
        """
        Compute average hydraulic head (m).
        """
        average_upstream_level = (start_level + end_level) / 2.0
        tailwater_level = self.reservoir.calculate_tailwater(total_discharge)
        return max(average_upstream_level - tailwater_level, 0.0)

    def calculate_operation(self, current_level: float, target_level: float, inflow: float):
        """
        Compute one-step reservoir operation.

        Returns:
            (
                power_kw,
                total_discharge_m3s,
                turbine_discharge_m3s,
                spill_discharge_m3s,
                final_level_m,
                feasible,
                current_storage_1e9_m3,
            )
        """
        # Convert the model time step from days to seconds so that storage-volume
        # change and discharge can be linked through the water-balance equation.
        delta_t_seconds = self.delta_t_days * 24.0 * 60.0 * 60.0

        # Compute storage corresponding to the current and target water levels.
        # Unit: 1e9 m3.
        current_storage = self.reservoir.calculate_storage(current_level)
        target_storage = self.reservoir.calculate_storage(target_level)

        # Reject any transition whose target level is outside the admissible
        # reservoir operating range.
        if target_level < self.dead_level or target_level > self.normal_level:
            return 0, 0, 0, 0, target_level, False, current_storage

        # Water balance:
        #   storage_change = target_storage - current_storage
        #   total_discharge = inflow - storage_change / delta_t
        #
        # If target storage is higher than current storage, some inflow is retained,
        # so discharge decreases. If target storage is lower, extra water must be
        # released, so discharge increases.
        storage_change = target_storage - current_storage  # 1e9 m3
        total_discharge = inflow - storage_change * 1e9 / delta_t_seconds  # m3/s

        # Negative discharge is physically infeasible, meaning the requested target
        # level cannot be reached under the available inflow in this time step.
        if total_discharge < 0:
            return 0, 0, 0, 0, target_level, False, current_storage

        # Start from the ideal assumption that all release passes through the
        # turbine. Spill is introduced only if turbine-related constraints bind.
        turbine_discharge = total_discharge
        spill_discharge = 0.0

        # Compute unconstrained hydropower output under the current transition.
        power_kw = self.reservoir.calculate_power(
            turbine_discharge,
            total_discharge,
            current_level,
            target_level,
        )

        # Installed capacity constraint:
        # if computed power exceeds max_power, back-calculate the admissible turbine
        # discharge from the hydropower equation P = K * H * Q, and route the
        # remaining release to spill.
        if power_kw > self.max_power:
            average_head = self._calculate_average_head(
                current_level,
                target_level,
                total_discharge,
            )
            turbine_discharge = self.max_power / (self.power_coefficient * average_head)
            spill_discharge = total_discharge - turbine_discharge
            power_kw = self.reservoir.calculate_power(
                turbine_discharge,
                total_discharge,
                current_level,
                target_level,
            )

        # Maximum turbine discharge constraint:
        # even if installed capacity is satisfied, turbine flow still cannot exceed
        # the physical turbine-flow limit. Any excess release becomes spill.
        if turbine_discharge >= self.max_turbine_discharge:
            turbine_discharge = self.max_turbine_discharge
            spill_discharge = total_discharge - turbine_discharge
            power_kw = self.reservoir.calculate_power(
                turbine_discharge,
                total_discharge,
                current_level,
                target_level,
            )

        # Return one-step operation results for DDP recursion or forward simulation.
        return (
            power_kw,
            total_discharge,
            turbine_discharge,
            spill_discharge,
            target_level,
            True,
            current_storage,
        )

    def optimize(self, firm_power_penalty_lambda: float):
        """
        Run deterministic dynamic programming.

        Returns:
            (
                value_function,
                decisions,
                optimal_levels,
                optimal_storages,
                optimal_powers,
                optimal_electricity,
                optimal_discharges,
                optimal_turbine_flows,
                optimal_spill_flows,
                inflows,
            )
        """
        # Total number of monthly stages over the full deterministic horizon.
        months_per_year = self.total_years * self.months_per_year
        n_states = len(self.levels)

        # Flatten the inflow matrix [years, 12] into one deterministic monthly
        # inflow sequence so the DDP can be solved over a single finite horizon.
        inflows = self.inflows.flatten()

        # value_function[t, state] stores the maximum cumulative value from stage t
        # to the end of the horizon, starting from the given water-level state.
        #
        # Initialize all stages as very negative values to represent infeasible or
        # not-yet-updated states, and set the terminal value to zero.
        value_function = np.full((months_per_year + 1, n_states), -1e10)
        value_function[months_per_year, :] = 0.0

        # decisions[t, state] stores the optimal next water level selected at stage t
        # when the current state is the given water-level state.
        decisions = np.zeros((months_per_year, n_states))

        # Backward DDP recursion:
        # evaluate stages from the end of the horizon back to the beginning.
        for t in reversed(range(months_per_year)):
            for state in range(n_states):
                current_level = self.levels[state]
                max_value = -1e10
                best_next_level = current_level

                # Enumerate all candidate next water levels.
                for next_state in range(n_states):
                    next_level = self.levels[next_state]

                    (
                        power_kw,
                        total_discharge,
                        turbine_discharge,
                        spill_discharge,
                        final_level,
                        feasible,
                        _,
                    ) = self.calculate_operation(current_level, next_level, inflows[t])

                    # Skip infeasible transitions immediately.
                    if not feasible:
                        continue

                    # Immediate stage value is monthly electricity, optionally reduced
                    # by the soft penalty for firm-power deficit.
                    current_value = self._calc_penalized_energy_gwh(
                        power_kw,
                        firm_power_penalty_lambda,
                    )

                    # Add the future value from the next stage.
                    if t + 1 < months_per_year:
                        # Because water level is discretized at 1 m resolution, the
                        # final level can be mapped directly to a state index.
                        final_state = int(final_level - self.dead_level)
                        future_value = value_function[t + 1, final_state]

                        # Only accumulate future value if the next-stage state has a
                        # valid value-function entry.
                        if future_value > -1e9:
                            current_value += future_value
                        else:
                            continue

                    # Keep the best transition among all candidate next states.
                    if current_value > max_value:
                        max_value = current_value
                        best_next_level = next_level

                # If at least one feasible transition was found, record the optimal
                # value and decision; otherwise keep the state at the current level.
                if max_value > -1e9:
                    value_function[t, state] = max_value
                    decisions[t, state] = best_next_level
                else:
                    decisions[t, state] = current_level

        # Forward simulation of the optimal trajectory:
        # starting from the initial level, follow the stored optimal decisions
        # stage by stage and record the resulting operation sequence.
        optimal_levels = []
        optimal_storages = []
        optimal_powers = []
        optimal_electricity = []
        optimal_discharges = []
        optimal_turbine_flows = []
        optimal_spill_flows = []

        # Find the discrete state closest to the prescribed initial water level.
        current_state = np.argmin(np.abs(self.levels - self.initial_level))
        current_level = self.levels[current_state]

        hours = self.delta_t_days * 24.0

        for t in range(months_per_year):
            next_level = decisions[t, current_state]

            (
                power_kw,
                total_discharge,
                turbine_discharge,
                spill_discharge,
                final_level,
                feasible,
                storage_current,
            ) = self.calculate_operation(current_level, next_level, inflows[t])

            # Save end-of-period state for standardized outputs.
            storage_final = self.reservoir.calculate_storage(final_level)

            optimal_levels.append(final_level)
            optimal_storages.append(storage_final)

            optimal_powers.append(power_kw)
            optimal_electricity.append(power_kw * hours / 1e6)
            optimal_discharges.append(total_discharge)
            optimal_turbine_flows.append(turbine_discharge)
            optimal_spill_flows.append(spill_discharge)

            current_level = final_level
            current_state = np.argmin(np.abs(self.levels - current_level))

        # Return both the DDP solution objects and the forward-simulated optimal
        # trajectory for downstream saving, evaluation, or plotting.
        return (
            value_function,
            decisions,
            optimal_levels,
            optimal_storages,
            optimal_powers,
            optimal_electricity,
            optimal_discharges,
            optimal_turbine_flows,
            optimal_spill_flows,
            inflows,
        )