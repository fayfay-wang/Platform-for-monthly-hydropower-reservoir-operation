"""
DQN agent module for the unified reservoir operation framework.

This module provides the DQN-side shared data container, replay buffer,
network definition, and agent wrapper, including:
1) unified DQN configuration,
2) shared reservoir and inflow data loading,
3) action discretization and storage candidate generation,
4) replay buffer implementation,
5) DQN network definition, and
6) DQN agent training utilities.

The purpose of this module is to isolate model- and agent-related logic from
environment simulation, training orchestration, and evaluation workflows.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp

from reservoir_benchmark.shared.io_utils import load_monthly_inflows_from_excel
from reservoir_benchmark.shared.project_paths import get_data_dir
from reservoir_benchmark.shared.reservoir_model import SharedReservoirModel


def set_global_seed(seed: int = 42) -> None:
    """
    Set Python / NumPy / PyTorch random seeds consistently.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    """
    Configuration container for DQN training and evaluation.
    """

    def __init__(self):
        self.agent_config = {
            "learning_rate": 0.0008,
            "reward_decay": 0.97,
            "epsilon_start": 0.95,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.990,
            "replace_target_iter": 400,
            "memory_size": 2000,
            "batch_size": 128,
            "firm_power": 1_854_000,
            "firm_penalty_lambda": 20.0,
            "power_coefficient": 9.81,
        }

        self.network_config = {
            "hidden_size": 768,
        }

        self.train_config = {
            "study_iteration": 10000,
            "max_steps": 240,
            "num_envs": 1,
        }

        self.display_config = {
            "show_progress": True,
            "progress_update_freq": 10,
        }

        self.seed = 42

    def get_seed(self):
        return getattr(self, "seed", 42)

    def get_agent_params(self):
        return self.agent_config

    def get_network_params(self):
        return self.network_config

    def get_train_params(self):
        return self.train_config


class BasicInfo:
    """
    Basic data and shared reservoir information for DQN.
    """

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = get_data_dir()
        self.reservoir_model = SharedReservoirModel(data_dir=self.data_dir)

        # Shared reservoir parameters (new unified names only)
        self.normal_level = self.reservoir_model.normal_level
        self.dead_level = self.reservoir_model.dead_level
        self.initial_level = self.reservoir_model.initial_level
        self.max_turbine_discharge = self.reservoir_model.max_turbine_discharge
        self.max_power = self.reservoir_model.max_power
        self.power_coefficient = float(
            self.config.agent_config.get(
                "power_coefficient",
                self.reservoir_model.power_coefficient,
            )
        )

        self.delta_t_days = self.reservoir_model.delta_t_days
        self.delta_t_hours = self.reservoir_model.delta_t_hours
        self.delta_t_seconds = self.reservoir_model.delta_t_seconds

        self.firm_power = float(
            self.config.agent_config.get("firm_power", self.reservoir_model.firm_power)
        )
        self.firm_penalty_lambda = float(
            self.config.agent_config.get("firm_penalty_lambda", 0.0)
        )

        self.months = list(range(1, 13))

        self.init_arrays()

    def init_arrays(self):
        """
        Load action storage candidates and inflow series.
        """
        # Action discretization: dead_level to normal_level at 1 m resolution
        action_levels = np.arange(
            int(self.dead_level),
            int(self.normal_level) + 1,
            1,
            dtype=np.int32,
        )

        self.state_present_storage = np.array(
            [float(self.reservoir_model.calculate_storage(float(z))) for z in action_levels],
            dtype=np.float64,
        )

        self.dead_storage = float(
            self.reservoir_model.calculate_storage(float(self.dead_level))
        )
        self.normal_storage = float(
            self.reservoir_model.calculate_storage(float(self.normal_level))
        )

        # Inflow series
        train_path = self.data_dir / "SynMonthlyInflow_train.xlsx"
        test_path = self.data_dir / "SynMonthlyInflow_evaluation.xlsx"

        if not train_path.exists():
            raise FileNotFoundError(f"Training inflow file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Evaluation inflow file not found: {test_path}")

        self.train_inflow = load_monthly_inflows_from_excel(train_path)
        self.test_inflow = load_monthly_inflows_from_excel(test_path)


class ReplayBuffer:
    """
    High-performance circular replay buffer.
    """

    def __init__(self, capacity, n_features):
        self.capacity = capacity
        self.n_features = n_features
        self.ptr = 0
        self.size = 0
        self.s = np.empty((capacity, n_features), dtype=np.float32)
        self.a = np.empty((capacity, 1), dtype=np.int64)
        self.r = np.empty((capacity, 1), dtype=np.float32)
        self.s_next = np.empty((capacity, n_features), dtype=np.float32)
        self.done = np.empty((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s_, d):
        i = self.ptr
        self.s[i] = s
        self.a[i, 0] = a
        self.r[i, 0] = r
        self.s_next[i] = s_
        self.done[i, 0] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size, self.capacity - 1) + 1

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s_next[idx],
            self.done[idx],
        )

    def __len__(self):
        return self.size


class DQN(nn.Module):
    """
    Deep Q-network with LayerNorm.
    """

    def __init__(self, n_features, n_actions, config: Config):
        super().__init__()

        hidden_size = config.get_network_params()["hidden_size"]

        self.fc1 = nn.Linear(n_features, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, n_actions)

        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size // 2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.out(x)


class DQNAgent:
    """
    DQN agent with normalized states and AMP support.
    """

    def __init__(self, n_actions, n_features, config: Config, basic_info: BasicInfo, device=None):
        self.config = config
        self.n_actions = n_actions
        self.n_features = n_features
        self.basic_info = basic_info

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB"
            )

        agent_cfg = self.config.agent_config
        self.learning_rate = agent_cfg["learning_rate"]
        self.gamma = agent_cfg["reward_decay"]
        self.replace_target_iter = agent_cfg["replace_target_iter"]
        self.memory_size = agent_cfg["memory_size"]
        self.batch_size = agent_cfg["batch_size"]

        self.epsilon_start = agent_cfg["epsilon_start"]
        self.epsilon_end = agent_cfg["epsilon_end"]
        self.epsilon_decay = agent_cfg["epsilon_decay"]
        self.epsilon = self.epsilon_start

        self.memory = ReplayBuffer(min(self.memory_size, 10000), n_features)

        self.eval_net = DQN(n_features, n_actions, config).to(self.device)
        self.target_net = DQN(n_features, n_actions, config).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.SmoothL1Loss()

        self.scaler = amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.learn_step_counter = 0

        self.state_low = np.array(
            [basic_info.dead_storage, 0.0, 1.0],
            dtype=np.float32,
        )
        self.state_high = np.array(
            [basic_info.normal_storage, 5000.0, 12.0],
            dtype=np.float32,
        )
        self._norm_scale = 1.0 / (self.state_high - self.state_low + 1e-8)

        self.debug = False
        self.loss_history = []
        self.epsilon_history = []

    def normalize_state(self, state):
        """
        Normalize state to [0, 1].
        """
        normalized = (state - self.state_low) * self._norm_scale
        return np.clip(normalized, 0, 1)

    def choose_action(self, observation, env=None, evaluate=False):
        normalized_obs = self.normalize_state(observation)
        obs_tensor = torch.as_tensor(
            normalized_obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        greedy = evaluate or np.random.uniform() > self.epsilon
        if greedy:
            with torch.no_grad():
                q_values = self.eval_net(obs_tensor).squeeze(0)
                action = int(torch.argmax(q_values).item())
        else:
            action = np.random.randint(self.n_actions)

        return action

    def store_transition(self, s, a, r, s_, done=False):
        s_norm = self.normalize_state(s)
        s_next_norm = self.normalize_state(s_)
        self.memory.push(s_norm, a, r, s_next_norm, done)

    def learn(self):
        """
        Learn from replay buffer using Double DQN.
        """
        if len(self.memory) < self.batch_size:
            return None

        b_s, b_a, b_r, b_s_, b_done = self.memory.sample(self.batch_size)

        b_s = torch.as_tensor(b_s, dtype=torch.float32, device=self.device)
        b_a = torch.as_tensor(b_a, dtype=torch.long, device=self.device).squeeze(1)
        b_r = torch.as_tensor(b_r, dtype=torch.float32, device=self.device).squeeze(1)
        b_s_ = torch.as_tensor(b_s_, dtype=torch.float32, device=self.device)
        b_done = torch.as_tensor(b_done, dtype=torch.float32, device=self.device).squeeze(1)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
            q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1))
            with torch.no_grad():
                next_actions = self.eval_net(b_s_).argmax(1, keepdim=True)
                q_next = self.target_net(b_s_).gather(1, next_actions).squeeze(1)
                q_target = (b_r + self.gamma * q_next * (1.0 - b_done)).unsqueeze(1)
            loss = self.loss_func(q_eval, q_target)

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.epsilon_history.append(self.epsilon)
        self.learn_step_counter += 1

        return loss_value

    def save_model(self, path):
        """
        Save model state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "eval_net_state_dict": self.eval_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
            "state_low": self.state_low,
            "state_high": self.state_high,
            "device": str(self.device),
            "seed": getattr(self.config, "seed", 42),
        }

        torch.save(checkpoint, path)
        return True

    def load_model(self, path):
        """
        Load model state.
        """
        path = Path(path)
        if not path.exists():
            print(f"Model file not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.eval_net.load_state_dict(checkpoint["eval_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
            self.learn_step_counter = checkpoint.get("learn_step_counter", 0)

            if "state_low" in checkpoint:
                self.state_low = checkpoint["state_low"]
            if "state_high" in checkpoint:
                self.state_high = checkpoint["state_high"]

            print(f"Model loaded successfully (device: {self.device})")
            return True

        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False
