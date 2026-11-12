"""
DQN training runner for the unified reservoir operation framework.

This module orchestrates DQN policy generation, including:
1) loading lambda candidates from penalty_settings.yaml,
2) constructing the shared DQN configuration and environment,
3) running training for each lambda candidate, and
4) saving only the best-performing model to the standardized output path.

The purpose of this module is to keep training orchestration separate from
environment simulation, network definition, and evaluation.
"""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
from collections import deque

import numpy as np
import torch
import torch._dynamo

try:
    from gymnasium.vector import AsyncVectorEnv
except Exception:
    AsyncVectorEnv = None

from reservoir_benchmark.policy_generation.dqn.dqn_reservoir import ReservoirEnv
from reservoir_benchmark.policy_generation.dqn.dqn_agent import (
    Config,
    BasicInfo,
    DQNAgent,
    set_global_seed,
)
from reservoir_benchmark.shared.inflow_forecast import (
    ForecastScenario,
    load_forecast_penalty_candidates,
)
from reservoir_benchmark.shared.io_utils import load_penalty_candidates
from reservoir_benchmark.shared.project_paths import (
    format_lambda_for_filename,
    get_data_dir,
    get_policy_generation_dir,
)

torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

torch._dynamo.config.suppress_errors = True


def run_training_sweep_lambdas(forecast: ForecastScenario | None = None):
    """
    Train one DQN model for each lambda defined in penalty_settings.yaml.
    """
    if forecast is None:
        penalty_settings_path = get_data_dir() / "penalty_settings.yaml"
        lam_list = load_penalty_candidates(penalty_settings_path, "dqn")
    else:
        lam_list = load_forecast_penalty_candidates("dqn")

    base_config = Config()

    for lam in lam_list:
        print("\n" + "=" * 80)
        print(f"[SWEEP] Start training with firm_penalty_lambda = {lam}")
        print("=" * 80)
        for seed in base_config.get_seed_list():
            run_training(firm_penalty_lambda=float(lam), seed=int(seed), forecast=forecast)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_training(
    firm_penalty_lambda: float,
    seed: int,
    forecast: ForecastScenario | None = None,
) -> None:
    """
    Train one DQN model for a single lambda value.
    """
    # Select the computation device once at the beginning of training.
    # If CUDA is available, use the GPU; otherwise, fall back to CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"Current device: {device} "
            f"(GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
        )
    else:
        device = torch.device("cpu")
        print("GPU unavailable. Using CPU.")

    # Build a fresh DQN configuration for the current training run and inject
    # the current firm-power penalty coefficient into the agent settings.

    config = Config()
    config.set_seed(seed)
    config.agent_config["firm_penalty_lambda"] = float(firm_penalty_lambda)

    # Fix all random seeds before environment and network creation so that
    # each lambda run is reproducible.
    set_global_seed(config.get_seed())

    # Format the current lambda value into a filename-safe string and define
    # the standardized output path for the best model of this run.
    lam_val = float(firm_penalty_lambda)
    lam_str = format_lambda_for_filename(lam_val)

    model_output_dir = get_policy_generation_dir("dqn") / "policy"
    if forecast is not None:
        model_output_dir = model_output_dir / "current_month" / forecast.name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_output_dir / f"dqn_train_lambda_{lam_str}_seed_{seed}.pth"

    print(f"Model output path: {model_path}")

    # Construct the shared DQN data container, the reservoir environment, and
    # the DQN agent. The environment defines the state/action spaces, while
    # the agent contains the replay buffer, networks, and optimizer.
    basic_info = BasicInfo(config)
    env = ReservoirEnv(basic_info, debug=True, forecast=forecast)

    agent = DQNAgent(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        config=config,
        basic_info=basic_info,
        device=device,
    )

    # Read the number of vectorized environments from the training config.
    # If num_envs > 1 and AsyncVectorEnv is available, training will use the
    # asynchronous multi-environment branch below.
    num_envs = config.train_config.get("num_envs", 1)

    # episode_rewards stores one scalar reward per episode and is used to
    # compute moving averages for training logs.
    # recent_losses stores recent optimization losses for reporting only.
    # best_reward tracks the best episode reward seen so far; the model is
    # saved only when a new best reward is achieved.
    episode_rewards = []
    recent_losses = deque(maxlen=10000)
    best_reward = float("-inf")

    # Logging frequency is controlled centrally through the config.
    show_progress = config.display_config.get("show_progress", True)
    log_interval = max(1, int(config.display_config.get("progress_update_freq", 10)))

    # Multi-environment training branch:
    # this is used only when asynchronous vector environments are available
    # and the configuration explicitly requests more than one environment.
    if (AsyncVectorEnv is not None) and (num_envs > 1):

        def make_env():
            return ReservoirEnv(basic_info, debug=False, forecast=forecast)

        envs = AsyncVectorEnv([make_env for _ in range(num_envs)])
        seed = config.get_seed()

        total_episodes = config.train_config["study_iteration"]
        episode_reward_buffer = np.zeros(num_envs, dtype=np.float64)

        for episode in range(total_episodes):
            # Reset all environments with deterministic but different seeds so
            # that the parallel environments do not produce identical rollouts.
            obs, _ = envs.reset(seed=[seed + i + episode * 1000 for i in range(num_envs)])

            while True:
                # Normalize observations before feeding them into the Q-network.
                if forecast is None:
                    norm = (obs - agent.state_low) * agent._norm_scale
                else:
                    norm = agent.normalize_state(obs)
                obs_tensor = torch.as_tensor(norm, dtype=torch.float32, device=agent.device)

                # Greedy actions from the current online network.
                with torch.no_grad():
                    q = agent.eval_net(obs_tensor)
                    greedy_actions = q.argmax(dim=1).detach().cpu().numpy()

                # Epsilon-greedy exploration: replace part of the greedy actions
                # with random actions according to the current epsilon value.
                explore_mask = np.random.rand(num_envs) <= agent.epsilon
                random_actions = np.random.randint(0, agent.n_actions, size=num_envs)
                actions = np.where(explore_mask, random_actions, greedy_actions)

                next_obs, rewards, terminated, truncated, infos = envs.step(actions)
                dones = np.logical_or(terminated, truncated)
                terminal_transitions = dones if forecast is None else terminated

                # Store one transition per environment into the replay buffer.
                for i in range(num_envs):
                    agent.store_transition(
                        obs[i],
                        int(actions[i]),
                        float(rewards[i]),
                        next_obs[i],
                        bool(terminal_transitions[i]),
                    )

                episode_reward_buffer += rewards

                # Start learning only after the replay buffer contains more
                # samples than one minibatch.
                if len(agent.memory) > agent.batch_size:
                    loss = agent.learn()
                    if loss is not None:
                        recent_losses.append(loss)

                obs = next_obs

                # In the vectorized branch, the episode is considered complete
                # only when all environments in the batch are done.
                if np.all(dones):
                    episode_reward = float(episode_reward_buffer.mean())
                    episode_rewards.append(episode_reward)

                    avg_reward = (
                        np.mean(episode_rewards[-100:])
                        if len(episode_rewards) >= 100
                        else np.mean(episode_rewards)
                    )

                    # Save only the best-performing model for the current
                    # lambda run, rather than saving checkpoints every episode.
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        agent.save_model(model_path)
                        print(
                            f"Episode {episode}: saved best model, "
                            f"reward = {episode_reward:.2f}"
                        )

                    # Periodic progress logging.
                    if show_progress and (
                        (episode + 1) % log_interval == 0
                        or episode == 0
                        or episode == total_episodes - 1
                    ):
                        if recent_losses:
                            last_losses = list(recent_losses)[-10:]
                            recent_loss = float(np.mean(last_losses))
                        else:
                            recent_loss = 0.0

                        print(
                            f"Episode {episode}: reward = {episode_reward:.2f}, "
                            f"avg_reward = {avg_reward:.2f}, "
                            f"epsilon = {agent.epsilon:.3f}, "
                            f"loss = {recent_loss:.4f}"
                        )

                    # Reset the reward accumulator for the next vectorized episode.
                    episode_reward_buffer[:] = 0.0
                    break

    # Single-environment training branch:
    # this is the default branch used when num_envs == 1.
    else:
        total_episodes = config.train_config["study_iteration"]

        for episode in range(total_episodes):

            # Periodically clear Python and CUDA caches to reduce long-run
            # memory accumulation during large training jobs.
            if episode > 0 and episode % 1000 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Keep only the recent tail of the logging histories so that
                # they do not grow without bound.
                if len(agent.loss_history) > 10000:
                    agent.loss_history = agent.loss_history[-5000:]
                if len(agent.epsilon_history) > 10000:
                    agent.epsilon_history = agent.epsilon_history[-5000:]

            # Reset the environment at the start of each episode.
            observation, _ = env.reset(debug=True)
            episode_reward = 0.0
            step_counter = 0
            episode_yearly_electricity = []

            while True:
                # Choose one action using the agent's epsilon-greedy policy.
                action = agent.choose_action(observation, env)
                next_observation, reward, terminated, truncated, info = env.step(action)

                # Record year-end electricity as a diagnostic summary for
                # progress logging. This does not affect learning.
                if info.get("is_year_end", False):
                    yearly_electricity_kwh = info.get("yearly_electricity_so_far", 0.0)
                    episode_yearly_electricity.append(yearly_electricity_kwh)

                step_counter += 1

                # Store the transition and, if enough data are available,
                # perform one learning step.
                agent.store_transition(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done=forecast is not None and terminated,
                )

                if len(agent.memory) > agent.batch_size:
                    loss = agent.learn()
                    if loss is not None:
                        recent_losses.append(loss)

                episode_reward += reward
                observation = next_observation

                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    avg_reward = (
                        np.mean(episode_rewards[-100:])
                        if len(episode_rewards) >= 100
                        else np.mean(episode_rewards)
                    )

                    # Save only the best model encountered so far.
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        agent.save_model(model_path)
                        print(
                            f"Episode {episode}: saved best model, "
                            f"reward = {episode_reward:.2f}"
                        )

                    # Periodic progress logging. If year-end electricity has
                    # been recorded in this episode, print it together with the
                    # standard reward / epsilon / loss diagnostics.
                    if show_progress and (
                        (episode + 1) % log_interval == 0
                        or episode == 0
                        or episode == total_episodes - 1
                    ):
                        if episode_yearly_electricity:
                            avg_yearly_electricity = np.mean(episode_yearly_electricity)
                            if recent_losses:
                                last_losses = list(recent_losses)[-10:]
                                recent_loss = float(np.mean(last_losses))
                            else:
                                recent_loss = 0.0

                            print(
                                f"Episode {episode}: reward = {episode_reward:.2f}, "
                                f"avg_reward = {avg_reward:.2f}, "
                                f"epsilon = {agent.epsilon:.3f}, "
                                f"loss = {recent_loss:.4f}, "
                                f"yearly_electricity = {avg_yearly_electricity / 1e6:.2f} GWh, "
                                f"steps = {step_counter}"
                            )
                        else:
                            print(
                                f"Episode {episode}: reward = {episode_reward:.2f}, "
                                f"avg_reward = {avg_reward:.2f}, "
                                f"epsilon = {agent.epsilon:.3f}"
                            )

                    break

    # Summarize the training result for the current lambda run.
    print(f"Training completed. Best model path: {model_path}")


def main_dqn(forecast: ForecastScenario | None = None) -> None:
    run_training_sweep_lambdas(forecast)


if __name__ == "__main__":
    main_dqn()
