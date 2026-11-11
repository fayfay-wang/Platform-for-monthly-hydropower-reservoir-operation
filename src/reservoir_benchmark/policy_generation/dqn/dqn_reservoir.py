"""
Shared DQN reservoir environment module for the unified reservoir operation
framework.

This module provides the common environment and state-transition logic used by
the DQN-based reservoir operation workflow, including:
1) storage-level conversion through the shared reservoir model,
2) tailwater calculation and hydraulic head evaluation,
3) month-by-month closed-loop reservoir simulation,
4) reward calculation based on electricity generation and firm-power penalty, and
5) a Gymnasium-compatible environment for DQN training and evaluation.

The purpose of this module is to keep the DQN environment consistent with the
shared reservoir physics, parameter definitions, and operating constraints used
by the other algorithms in the unified framework.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box as GBox, Discrete as GDiscrete

from reservoir_benchmark.shared.inflow_forecast import ForecastScenario
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


class ReservoirUtils:
    """
    Utility wrapper around the shared reservoir model.
    """

    def __init__(self, reservoir_model: SharedReservoirModel):
        self.reservoir = reservoir_model

    def level_to_storage(self, level: float) -> float:
        """
        Convert water level (m) to storage (1e9 m3).
        """
        return float(self.reservoir.calculate_storage(level))

    def storage_to_level(self, storage: float) -> float:
        """
        Convert storage (1e9 m3) to water level (m).
        """
        return float(self.reservoir.calculate_level(storage))

    def calculate_tailwater(self, discharge: float) -> float:
        """
        Calculate tailwater level (m) from discharge (m3/s).
        """
        return float(self.reservoir.calculate_tailwater(discharge))


class ReservoirEnv(gym.Env):
    """
    Gymnasium environment for single-reservoir DQN operation.

    State definition:
        [storage (1e9 m3), inflow (m3/s), month]
        Perfect and noisy scenarios append the observed next-month inflow (m3/s).

    Action definition:
        Discrete target end-of-month storage candidates.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        basic_info,
        debug: bool = False,
        *,
        forecast: ForecastScenario | None = None,
        is_training: bool = True,
    ):
        super().__init__()

        self.debug = debug
        self.basic_info = basic_info
        self.max_steps = int(self.basic_info.config.train_config["max_steps"])

        self.reservoir_model = self.basic_info.reservoir_model
        self.utils = ReservoirUtils(self.reservoir_model)

        # Shared physical parameters
        self.dead_level = float(self.basic_info.dead_level)
        self.normal_level = float(self.basic_info.normal_level)
        self.initial_level = float(self.basic_info.initial_level)
        self.max_turbine_discharge = float(self.basic_info.max_turbine_discharge)
        self.max_power = float(self.basic_info.max_power)
        self.power_coefficient = float(self.basic_info.power_coefficient)
        self.firm_power = float(self.basic_info.firm_power)
        self.firm_penalty_lambda = float(self.basic_info.firm_penalty_lambda)
        self.inflow_upper_bound = float(self.basic_info.inflow_upper_bound)

        # Unified time-step parameters
        self.delta_t_days = float(self.basic_info.delta_t_days)
        self.delta_t_hours = float(self.basic_info.delta_t_hours)
        self.delta_t_seconds = float(self.basic_info.delta_t_seconds)
        self.step_hours = self.delta_t_hours

        # Action space: discrete target end-of-month storage candidates
        self.action_space = GDiscrete(len(self.basic_info.state_present_storage))

        # Storage bounds
        self.dead_storage = float(self.basic_info.dead_storage)
        self.normal_storage = float(self.basic_info.normal_storage)

        # Inflow data
        self.train_data = self.basic_info.train_inflow
        self.test_data = self.basic_info.test_inflow
        self.is_training = is_training
        self.forecast = forecast
        self.forecast_inflows = None
        if forecast is not None and forecast.mode != "none":
            data = self.train_data if is_training else self.test_data
            seed = forecast.training_seed if is_training else forecast.evaluation_seed
            self.forecast_inflows = forecast.next_month(data.to_numpy(), seed)

        # Observation space: [storage, inflow, month, optional next inflow]
        low = [self.dead_storage, 0.0, 1.0]
        high = [self.normal_storage, self.inflow_upper_bound, 12.0]
        if self.forecast_inflows is not None:
            low.append(0.0)
            high.append(np.inf)
        self.observation_space = GBox(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

        # Monthly inflow columns in SynMonthlyInflow_train.xlsx and SynMonthlyInflow_evaluation.xlsx
        self.month_columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        # Running state
        self.current_step = 0
        self.current_year_index = 0
        self.current_month = 1
        self.state = None

        # Annual tracking
        self.yearly_electricity = 0.0
        self.yearly_electricity_history = []
        self.monthly_electricity_in_year = []
        self.year_counter = 0

        # Reward normalization scale: one month of full-capacity generation
        self.reward_scale_gwh = (self.max_power * self.step_hours) / 1e6
        if self.reward_scale_gwh <= 0:
            self.reward_scale_gwh = 1.0

    def reset(self, seed=None, options=None, debug: bool = False):
        """
        Reset the environment.
        """
        super().reset(seed=seed)

        self.debug = debug
        data = self.train_data if self.is_training else self.test_data
        max_years = len(data)
        self.current_year_index = max(0, min(self.current_year_index, max_years - 1))
        if (
            self.forecast is not None
            and self.is_training
            and self.current_year_index == max_years - 1
            and self.current_month == 12
        ):
            self.current_year_index = 0

        initial_inflow = float(data.iloc[self.current_year_index][self.month_columns[0]])
        initial_storage = self.normal_storage

        self.current_month = 1
        self.state = self._make_observation(initial_storage, initial_inflow)
        self.current_step = 0

        self.yearly_electricity = 0.0
        self.monthly_electricity_in_year = []
        self.year_counter = 0

        return self.state, {}

    def step(self, action):
        """
        Advance the environment by one monthly step.
        """
        action = int(action)
        action = np.clip(action, 0, len(self.basic_info.state_present_storage) - 1)

        current_storage, current_inflow = self.state[:2]
        current_storage = float(current_storage)
        current_inflow = float(current_inflow)

        next_inflow = self._get_next_inflow()
        target_storage = float(self.basic_info.state_present_storage[action])

        (
            next_storage,
            reward,
            total_discharge,
            turbine_discharge,
            spill_discharge,
            power_output,
            electricity_kwh,
            head,
            tailwater_level,
            start_level,
            end_level,
            water_balance_error_m3,
        ) = self._calculate_operation_reward(
            current_storage=current_storage,
            current_inflow=current_inflow,
            target_storage=target_storage,
        )

        self.yearly_electricity += electricity_kwh
        is_year_end = self.current_month == 12
        if is_year_end:
            self.yearly_electricity_history.append(self.yearly_electricity)

        info = {
            "month": int(self.current_month),
            "year": self.current_year_index + 1,
            "start_water_level": start_level,
            "water_level": end_level,
            "start_storage": current_storage,
            "end_storage": next_storage,
            "inflow": current_inflow,
            "total_discharge": total_discharge,
            "turbine_discharge": turbine_discharge,
            "spill_discharge": spill_discharge,
            "power_output": power_output,
            "electricity_kwh": electricity_kwh,
            "target_storage": target_storage,
            "tailwater_level": tailwater_level,
            "head": head,
            "water_balance_error_m3": water_balance_error_m3,
            "yearly_electricity_so_far": self.yearly_electricity,
            "is_year_end": is_year_end,
            "year_index": self.year_counter,
        }

        # Advance the calendar by one month. When December is finished, move to
        # January and update the year index accordingly.
        next_month = 1 if self.current_month == 12 else self.current_month + 1

        if is_year_end:
            data = self.train_data if self.is_training else self.test_data
            if self.is_training:
                self.current_year_index = (self.current_year_index + 1) % len(data)
            else:
                self.current_year_index = self.current_year_index + 1
            self.year_counter += 1
            self.yearly_electricity = 0.0
            self.monthly_electricity_in_year = []

        self.current_month = next_month
        self.state = self._make_observation(next_storage, next_inflow)

        self.current_step += 1
        # The final month has no following inflow within the training record.
        end_of_training_data = (
            self.forecast is not None
            and self.is_training
            and self.current_year_index == len(self.train_data) - 1
            and self.current_month == 12
        )
        reached_step_limit = self.current_step >= self.max_steps
        terminated = end_of_training_data or (self.forecast is None and reached_step_limit)
        truncated = reached_step_limit and not terminated

        return self.state, reward, terminated, truncated, info

    def _make_observation(self, storage: float, inflow: float) -> np.ndarray:
        observation = [storage, inflow, self.current_month]
        if self.forecast_inflows is not None:
            year = self.current_year_index
            next_inflow = (
                self.forecast_inflows[year, self.current_month - 1]
                if year < len(self.forecast_inflows)
                else 0.0
            )
            observation.append(next_inflow)
        return np.array(observation, dtype=np.float32)

    def _get_next_inflow(self) -> float:
        """
        Return the next-month inflow used in the next state.
        """
        data = self.train_data if self.is_training else self.test_data
        current_year = self.current_year_index
        current_month = int(self.current_month)
        next_month = 1 if current_month == 12 else current_month + 1

        if self.is_training:
            next_year = (current_year + 1) % len(data) if current_month == 12 else current_year
        else:
            if current_month == 12:
                if current_year >= len(data) - 1:
                    return float(data.iloc[current_year][self.month_columns[11]])
                next_year = current_year + 1
            else:
                next_year = current_year

        month_col = self.month_columns[next_month - 1]
        return float(data.iloc[next_year][month_col])

    def _calculate_operation_reward(
        self,
        current_storage: float,
        current_inflow: float,
        target_storage: float,
    ):
        """
        Compute one-step operation, reward, and diagnostic variables using
        storage-based state transition and water balance.

        Returns:
            (
                next_storage,
                reward,
                total_discharge,
                turbine_discharge,
                spill_discharge,
                power_output,
                electricity_kwh,
                head,
                tailwater_level,
                start_level,
                end_level,
                water_balance_error_m3,
            )
        """
        dead_storage = float(self.dead_storage)
        normal_storage = float(self.normal_storage)
        eps = 1e-6

        if target_storage < dead_storage - eps or target_storage > normal_storage + eps:
            raise ValueError(
                f"Target storage out of bounds: {target_storage:.10f} "
                f"not in [{dead_storage:.10f}, {normal_storage:.10f}]"
            )

        # Mass balance:
        #   storage_change = target_storage - current_storage
        #   total_discharge = inflow - storage_change / delta_t
        #
        # If the target storage is higher than the current storage, part of the
        # inflow must be retained in the reservoir, so discharge decreases.
        # If the target storage is lower, extra water must be released, so
        # discharge increases.
        storage_change = target_storage - current_storage
        total_discharge = current_inflow - storage_change * 1e9 / self.delta_t_seconds

        start_level = float(self.utils.storage_to_level(current_storage))
        end_level = float(self.utils.storage_to_level(target_storage))

        if total_discharge < 0:
            # A negative total discharge means the requested target storage would
            # require retaining more water than the inflow can physically provide
            # within the current month. In that case, the requested transition is
            # not directly feasible, so the environment switches to a pure-storage
            # update with zero release and assigns a negative reward.

            # The maximum feasible storage increase under zero release is the
            # monthly inflow volume itself. Therefore, the reservoir stores all
            # incoming water, sets discharge to zero, and updates the end-of-
            # month storage accordingly.
            available_storage_increase = current_inflow * self.delta_t_seconds / 1e9
            next_storage = min(current_storage + available_storage_increase, normal_storage)

            end_level = float(self.utils.storage_to_level(next_storage))

            reward = -3.0
            total_discharge = 0.0
            turbine_discharge = 0.0
            spill_discharge = 0.0
            power_output = 0.0
            electricity_kwh = 0.0
            tailwater_level = float(self.utils.calculate_tailwater(total_discharge))
            head = max(0.0, (start_level + end_level) / 2.0 - tailwater_level)

            water_balance_error_m3 = (
                (current_inflow - total_discharge) * self.delta_t_seconds
                - (next_storage - current_storage) * 1e9
            )

            return (
                float(next_storage),
                float(reward),
                float(total_discharge),
                float(turbine_discharge),
                float(spill_discharge),
                float(power_output),
                float(electricity_kwh),
                float(head),
                float(tailwater_level),
                float(start_level),
                float(end_level),
                float(water_balance_error_m3),
            )

        # Start from the ideal assumption that all release goes through the
        # turbine. Spill will only be introduced later if the hydropower unit
        # is limited by installed capacity or maximum turbine discharge.
        turbine_discharge = float(total_discharge)
        spill_discharge = 0.0

        tailwater_level = float(self.utils.calculate_tailwater(total_discharge))
        head = max(0.0, (start_level + end_level) / 2.0 - tailwater_level)

        if head <= 0.0 or turbine_discharge <= 0.0:
            turbine_discharge = 0.0
            spill_discharge = total_discharge
            power_output = 0.0
        else:
            power_output = self.power_coefficient * head * turbine_discharge

            # Hydropower equation:
            #   P = K * H * Q
            # where:
            #   P = power output,
            #   K = power_coefficient,
            #   H = hydraulic head,
            #   Q = turbine discharge.
            #
            # When installed capacity is exceeded, we need to solve the same
            # equation for Q under the condition P = max_power:
            #   Q = max_power / (K * H)
            #
            # Therefore, the denominator below is:
            #   denom = K * H
            #
            # The small threshold 1e-12 is a numerical safety guard to avoid
            # division by zero or an unstable discharge estimate when head is
            # extremely close to zero.
            denom = self.power_coefficient * head

            # Only back-calculate turbine discharge when the hydraulic term is
            # numerically valid and the current power output actually exceeds
            # the installed-capacity limit.
            if denom > 1e-12 and power_output > self.max_power:
                turbine_discharge = self.max_power / denom
                spill_discharge = total_discharge - turbine_discharge
                power_output = self.power_coefficient * head * turbine_discharge

            if turbine_discharge > self.max_turbine_discharge:
                turbine_discharge = self.max_turbine_discharge
                spill_discharge = total_discharge - turbine_discharge
                power_output = self.power_coefficient * head * turbine_discharge

        electricity_kwh = power_output * self.step_hours
        electricity_gwh = electricity_kwh / 1e6

        # Soft penalty for firm-power deficit:
        # if monthly power output is below the firm-power threshold, the deficit
        # is converted into an equivalent electricity shortfall and penalized
        # by firm_penalty_lambda.
        deficit_gwh = 0.0
        if self.firm_power > 0:
            deficit_kw = max(0.0, self.firm_power - power_output)
            deficit_gwh = (deficit_kw * self.step_hours) / 1e6

        penalty = self.firm_penalty_lambda * deficit_gwh

        # Reward is defined as normalized net electricity benefit:
        #   reward = (electricity - firm-power penalty) / normalization_scale
        #
        # This keeps the reward dimensionless and helps stabilize DQN training.
        reward = (electricity_gwh - penalty) / self.reward_scale_gwh

        reward = float(np.clip(reward, -10.0, 10.0))

        next_storage = target_storage

        # This diagnostic term checks the residual of the mass-balance equation.
        # It should remain close to zero apart from floating-point error.
        water_balance_error_m3 = (
            (current_inflow - total_discharge) * self.delta_t_seconds
            - (next_storage - current_storage) * 1e9
        )

        return (
            float(next_storage),
            float(reward),
            float(total_discharge),
            float(turbine_discharge),
            float(spill_discharge),
            float(power_output),
            float(electricity_kwh),
            float(head),
            float(tailwater_level),
            float(start_level),
            float(end_level),
            float(water_balance_error_m3),
        )
