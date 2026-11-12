from __future__ import annotations

"""
Stochastic dynamic programming solver module for the unified reservoir
operation framework.

This module implements the core SDP optimization logic for single-reservoir
operation, including:
1) one-step stage-benefit calculation,
2) backward SDP recursion over water-level and inflow states,
3) convergence checking across annual iterations, and
4) generation of the final SDP decision table in memory.

The purpose of this module is to isolate the SDP algorithmic core from
path handling, input/output operations, and script-level execution, so that
the solver remains focused on optimization logic within the unified framework.
"""

import numpy as np


class ReservoirOperation:
    """
    Stochastic dynamic programming (SDP) for single-reservoir operation.
    """

    def __init__(self, reservoir, inflows, flow_probs=None, flow_intervals=None):
        self.reservoir = reservoir
        self.inflows = inflows
        self.flow_probs = flow_probs
        self.flow_intervals = flow_intervals
        self.n_flows = inflows.shape[1]

        # Shared parameters from SharedReservoirModel
        self.dead_level = self.reservoir.dead_level
        self.normal_level = self.reservoir.normal_level
        self.initial_level = self.reservoir.initial_level
        self.max_turbine_discharge = self.reservoir.max_turbine_discharge
        self.max_power = self.reservoir.max_power
        self.firm_power = self.reservoir.firm_power
        self.power_coefficient = self.reservoir.power_coefficient
        self.delta_t_days = self.reservoir.delta_t_days

        # SDP-specific configuration
        self.config = {
            "threshold": 0.01,
            "firm_power_penalty_lambda": 0.0,
        }

        # Time settings
        self.months_per_year = 12

        # Water-level discretization
        self.levels = np.arange(self.dead_level, self.normal_level + 1, 1)
        self.volumes = np.array([self.reservoir.calculate_storage(h) for h in self.levels])

        # History container
        self.history = {
            "expected_total_energy": {},
            "decisions": {},
            "energy_matrix": {},
            "annual_energy": [],
            "prev_decisions": None,
        }

    def _calc_penalized_energy_gwh(self, power_kw: float) -> float:
        """
        Compute monthly electricity (GWh) with an optional soft penalty
        for firm-power deficit.
        """
        hours = self.delta_t_days * 24.0
        energy_gwh = power_kw * hours / 1e6

        lam = self.config.get("firm_power_penalty_lambda", 0.0)

        if self.firm_power > 0.0 and lam > 0.0:
            deficit_kw = max(0.0, self.firm_power - power_kw)
            if deficit_kw > 0.0:
                deficit_energy_gwh = deficit_kw * hours / 1e6
                energy_gwh -= lam * deficit_energy_gwh

        return energy_gwh

    def _calculate_average_head(
        self,
        start_level: float,
        end_level: float,
        discharge: float,
    ) -> float:
        """
        Compute average hydraulic head (m).
        """
        average_upstream_level = (start_level + end_level) / 2.0
        tailwater = self.reservoir.calculate_tailwater(discharge)
        return max(average_upstream_level - tailwater, 0.0)

    def _calculate_immediate_benefit(
        self,
        current_level: float,
        target_level: float,
        inflow: float,
    ):
        """
        Compute the immediate stage benefit under a given transition.

        Returns:
            (immediate_benefit_kwh, discharge, turbine_discharge, spill_discharge)
        """
        current_storage = self.reservoir.calculate_storage(current_level)
        next_storage = self.reservoir.calculate_storage(target_level)

        # Mass balance:
        #   discharge = inflow + (current_storage - next_storage) / delta_t
        #
        # If the target storage is larger than the current storage, water is
        # retained and discharge decreases. If the target storage is smaller,
        # more water is released and discharge increases.
        discharge = (
            (current_storage - next_storage) * 1e9
            / (self.delta_t_days * 24.0 * 60.0 * 60.0)
            + inflow
        )

        # A negative discharge is physically infeasible, so this transition is
        # rejected by returning negative infinity as the stage benefit.
        if discharge < 0:
            return -np.inf, 0.0, 0.0, 0.0

        # Start from the ideal assumption that all release passes through the
        # turbine. Spill is only introduced later if physical limits are hit.
        turbine_discharge = discharge
        spill_discharge = 0.0

        # Compute the unconstrained power output for the current transition.
        power_kw = self.reservoir.calculate_power(
            turbine_discharge,
            discharge,
            current_level,
            target_level,
        )

        # Installed capacity constraint:
        # if the unconstrained power exceeds max_power, solve the hydropower
        # equation backward to find the admissible turbine discharge and route
        # the remaining release to spill.
        if power_kw > self.max_power:
            average_head = self._calculate_average_head(current_level, target_level, discharge)
            turbine_discharge = self.max_power / (self.power_coefficient * average_head)
            spill_discharge = discharge - turbine_discharge
            power_kw = self.reservoir.calculate_power(
                turbine_discharge,
                discharge,
                current_level,
                target_level,
            )

        # Maximum turbine discharge constraint:
        # even if installed capacity is respected, turbine flow cannot exceed
        # the physical turbine-flow limit.
        if turbine_discharge >= self.max_turbine_discharge:
            turbine_discharge = self.max_turbine_discharge
            spill_discharge = discharge - turbine_discharge
            power_kw = self.reservoir.calculate_power(
                turbine_discharge,
                discharge,
                current_level,
                target_level,
            )

        # Convert the stage power into penalized monthly electricity, then into
        # kWh-equivalent stage value so it is consistent with the stored value
        # function used in the recursion.
        benefit_gwh = self._calc_penalized_energy_gwh(power_kw)
        energy_kwh = benefit_gwh * 1e6

        return energy_kwh, discharge, turbine_discharge, spill_discharge

    def _check_convergence(self, current_decisions: np.ndarray):
        """
        Check whether the full annual decision matrix has converged.
    
        The convergence criterion is based on the maximum absolute difference
        between the current and previous decision matrices across all months,
        water-level states, and inflow classes.
    
        Returns:
            (converged, max_difference)
        """
        # No convergence check is possible on the first annual iteration
        # because there is no previous decision table yet.
        if self.history["prev_decisions"] is None:
            return False, np.inf
    
        prev_decisions = self.history["prev_decisions"]
    
        # Check the whole annual policy:
        # dimensions = [month, water-level state, inflow class]
        max_diff = np.max(np.abs(current_decisions - prev_decisions))
    
        converged = max_diff <= self.config["threshold"]
        return converged, max_diff

    def _stage_benefits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute rewards and feasible ending states for all realized inflow classes."""
        storage_per_discharge = self.reservoir.delta_t_seconds / 1e9
        available_storage = (
            self.volumes[None, :, None]
            + np.asarray(self.inflows)[:, None, :] * storage_per_discharge
        )
        release_volume = available_storage[:, :, None, :] - self.volumes[None, None, :, None]
        feasible = release_volume >= 0.0
        discharge = np.maximum(release_volume, 0.0) / storage_per_discharge
        average_level = (self.levels[:, None] + self.levels[None, :]) / 2.0
        tailwater = np.polyval(self.reservoir.tailwater_curve_coefficients, discharge)
        head = average_level[None, :, :, None] - tailwater
        turbine_discharge = np.minimum(discharge, self.max_turbine_discharge)
        power = np.minimum(self.power_coefficient * turbine_discharge * head, self.max_power)
        lam = self.config["firm_power_penalty_lambda"]
        if self.firm_power > 0.0 and lam > 0.0:
            power -= lam * np.maximum(self.firm_power - power, 0.0)
        # If the reservoir cannot reach the target, keep the inflow with zero release.
        # Map the resulting storage to the lower grid point used by the policy lookup.
        available_states = np.searchsorted(self.volumes, available_storage, side="right") - 1
        available_states = np.clip(available_states, 0, len(self.levels) - 1)
        return power * self.reservoir.delta_t_hours, feasible, available_states

    def optimize_with_forecast(
        self,
        inflow_probs: np.ndarray,
        forecast_probs: np.ndarray,
    ) -> np.ndarray:
        """Optimize each target using actual inflow probabilities given the current forecast."""
        stage_benefits, feasible, available_states = self._stage_benefits()
        values = np.zeros((self.months_per_year, len(self.levels), forecast_probs.shape[1]))
        decisions = np.empty_like(values)
        year = 0

        while True:
            year += 1
            for month in range(self.months_per_year - 1, -1, -1):
                next_month = (month + 1) % self.months_per_year
                # Future inflows follow the monthly distributions, so average
                # over the forecast that will be available next month.
                next_values = values[next_month] @ forecast_probs[next_month]
                future_values = np.where(
                    feasible[month],
                    next_values[None, :, None],
                    next_values[available_states[month]][:, None, :],
                )
                action_values = (stage_benefits[month] + future_values) @ inflow_probs[month].T
                best_targets = np.argmax(action_values, axis=1)
                decisions[month] = self.levels[best_targets]
                values[month] = np.take_along_axis(action_values, best_targets[:, None, :], axis=1)[:, 0]

            # Center the relative value function once per annual sweep.
            values -= values[0, 0, 0]
            converged, max_difference = self._check_convergence(decisions)
            print(f"SDP iteration year {year}, maximum decision change: {max_difference:.6f} m")
            if converged:
                return decisions
            self.history["prev_decisions"] = decisions.copy()

    def optimize(self):
        """
        Run SDP optimization.

        State dimensions:
            [month, water-level state, inflow state]
        """
        n_states = len(self.levels)
        n_flows = self.n_flows

        # Value function:
        #   expected_total_energy[month, state, inflow_class]
        #
        # This stores the maximum expected total future value from a given
        # monthly stage, water-level state, and inflow class.
        expected_total_energy = np.zeros((self.months_per_year, n_states, n_flows))

        # Track the January starting level associated with each inflow class
        # across annual SDP iterations.
        current_january_levels = np.full(n_flows, self.initial_level)

        year = 0
        converged = False

        # The annual policy is iterated until January decisions converge.
        # The "or year < 2" condition ensures that at least two full annual
        # iterations are completed before convergence is checked.
        while not converged or year < 2:
            year += 1
            print(f"Starting SDP iteration year {year}...")

            # decisions[month, state, inflow_class] stores the optimal next
            # water level for the current iteration.
            decisions = np.zeros((self.months_per_year, n_states, n_flows))

            # energy_matrix stores the immediate stage value corresponding to
            # the selected optimal transition.
            energy_matrix = np.zeros((self.months_per_year, n_states, n_flows))

            # Use the previous annual value function as the future-value
            # reference when building the current annual policy.
            prev_expected_total_energy = expected_total_energy.copy()

            # Backward dynamic programming over months:
            # December -> ... -> January.
            for month in range(self.months_per_year - 1, -1, -1):
                for state in range(n_states):
                    current_level = self.levels[state]

                    for current_flow_idx in range(n_flows):
                        current_inflow = self.inflows.loc[month + 1, current_flow_idx]

                        # Initialize the best transition for the current
                        # (month, state, inflow_class).
                        best_next_level = current_level
                        max_benefit = -np.inf
                        best_energy = 0.0

                        # Enumerate all candidate next water levels.
                        for next_state in range(n_states):
                            next_level = self.levels[next_state]

                            immediate_benefit, _, _, _ = self._calculate_immediate_benefit(
                                current_level,
                                next_level,
                                current_inflow,
                            )
                            if immediate_benefit == -np.inf:
                                continue

                            # Compute the expected future value by integrating
                            # over all possible next-period inflow classes.
                            expected_future_value = 0.0
                            for next_flow_idx in range(n_flows):
                                # For December, the next stage is January of the
                                # next annual iteration, so the recursion wraps
                                # around to month 0.
                                if month == self.months_per_year - 1:
                                    future_value = prev_expected_total_energy[0, next_state, next_flow_idx]
                                else:
                                    future_value = prev_expected_total_energy[month + 1, next_state, next_flow_idx]

                                expected_future_value += future_value * self.flow_probs[next_flow_idx]

                            total_benefit = immediate_benefit + expected_future_value

                            # Keep the transition with the highest total value.
                            if total_benefit > max_benefit:
                                max_benefit = total_benefit
                                best_next_level = next_level
                                best_energy = immediate_benefit

                        expected_total_energy[month, state, current_flow_idx] = max_benefit
                        decisions[month, state, current_flow_idx] = best_next_level
                        energy_matrix[month, state, current_flow_idx] = best_energy

            # Save the full annual results of this iteration.
            self.history["expected_total_energy"][year] = expected_total_energy.copy()
            self.history["decisions"][year] = decisions.copy()
            self.history["energy_matrix"][year] = energy_matrix.copy()

            # Update January starting levels by applying the December decision
            # under each inflow class. This is part of the cyclic annual policy
            # iteration used to obtain a stationary monthly rule.
            next_january_levels = np.zeros(n_flows)
            for flow_idx in range(n_flows):
                level_idx = np.argmin(np.abs(self.levels - current_january_levels[flow_idx]))
                next_january_levels[flow_idx] = decisions[self.months_per_year - 1, level_idx, flow_idx]

            # Compute the expected annual electricity under the current policy
            # by simulating one annual cycle across all inflow classes and
            # averaging with their probabilities.
            annual_expected_energy = 0.0
            for flow_idx in range(n_flows):
                scenario_annual_energy = 0.0
                current_level = current_january_levels[flow_idx]

                for month in range(self.months_per_year):
                    level_idx = np.argmin(np.abs(self.levels - current_level))
                    scenario_annual_energy += energy_matrix[month, level_idx, flow_idx]
                    current_level = decisions[month, level_idx, flow_idx]

                annual_expected_energy += scenario_annual_energy * self.flow_probs[flow_idx]

            self.history["annual_energy"].append(annual_expected_energy)

            # Starting from the second annual iteration, check whether the
            # January decision table has stabilized.
            if year >= 2:
                converged, max_diff = self._check_convergence(decisions)
                if converged:
                    print(
                        f"Year {year} converged. "
                        f"Maximum January decision difference across all states: {max_diff:.6f} m"
                    )
                else:
                    print(
                        f"Year {year} not converged yet. "
                        f"Maximum January decision difference: {max_diff:.6f} m"
                    )
            else:
                print(f"Year {year} completed.")

            print(f"Expected annual generation in year {year}: {annual_expected_energy / 1e6:.2f} GWh")

            # Store the current decision table for the next convergence check
            # and update the January initial levels for the next annual loop.
            self.history["prev_decisions"] = decisions.copy()
            current_january_levels = next_january_levels.copy()

        print(f"\nSDP converged in year {year}.")
        print("Final January decisions are stable across all states.")
        print(f"Cumulative total value: {np.max(expected_total_energy) / 1e6:.2f} GWh")

        return (
            expected_total_energy,
            decisions,
            self.history["annual_energy"],
            self.history,
        )
