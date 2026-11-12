# Reservoir Benchmark

Reservoir Benchmark is a unified benchmark framework for monthly single-reservoir hydropower operation. It compares four policy families under the same reservoir physics, inflow data, penalty settings, and evaluation protocol:

- **DDP**: deterministic dynamic programming
- **SDP**: stochastic dynamic programming
- **ISO-SURF**: fitted monthly rule surfaces based on upstream DDP trajectories
- **DQN**: deep Q-network reinforcement learning

The project is intended for reproducible comparison of reservoir operation policies, sensitivity analysis under different inflow conditions, and adaptation to new reservoirs by replacing input data and reservoir-specific parameters.

## Features

- Shared reservoir model for all algorithms, including level-storage conversion, tailwater estimation, and hydropower calculation.
- Standardized input layout for training inflows, evaluation inflows, reservoir settings, level-storage curves, tailwater curves, penalty coefficients, and algorithm experiment settings.
- Policy generation workflows for SDP, ISO-SURF, and DQN.
- Direct deterministic evaluation with DDP.
- Standardized evaluation outputs for cross-method comparison.
- Scenario-level and overall visualization scripts for generating figures and summary tables.
- Sphinx documentation under `docs/`.

## Repository Layout

```text
data/
  SynMonthlyInflow_train.xlsx
  SynMonthlyInflow_evaluation.xlsx
  reservoir_settings.txt
  level_storage_curve.csv
  tailwater_discharge_curve.csv
  penalty_settings.yaml
  experiment_settings.yaml
scripts/
  run_policy_generation.py
  run_evaluation.py
  run_visualization.py
src/reservoir_benchmark/
  shared/
  policy_generation/
    ddp/
    sdp/
    iso_surf/
    dqn/
  evaluation/
    ddp/
    sdp/
    iso_surf/
    dqn/
  visualization/
docs/
  source/
```

Generated artifacts are written under `outputs/`.

## Installation

Python 3.11 is recommended.

```bash
conda create -n reservoir_benchmark python=3.11 -y
conda activate reservoir_benchmark
python -m pip install -e .
```

To run DQN experiments, install the optional DQN dependencies:

```bash
python -m pip install -e ".[dqn]"
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

## Input Data

The benchmark reads inputs from the repository-level `data/` directory.

### Inflow Files

Training and evaluation inflows use the same Excel format:

- `data/SynMonthlyInflow_train.xlsx`
- `data/SynMonthlyInflow_evaluation.xlsx`

The loader skips the first column and treats the next 12 columns as monthly inflows from January to December. The recommended columns are:

```text
Year, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
```

Requirements:

- The 12 monthly columns must be numeric monthly mean inflows in `m3/s`.
- Missing values are not allowed.
- Negative values are not allowed.
- After the first column is skipped, exactly 12 monthly columns must remain.

### Reservoir Parameters

Reservoir parameters are stored in `data/reservoir_settings.txt`.

Required fields include:

- `dead_level`
- `normal_level`
- `initial_level`
- `max_turbine_discharge`
- `max_power`
- `firm_power`
- `power_coefficient`
- `delta_t_days`
- `inflow_upper_bound`

`inflow_upper_bound` is the single source of truth for the DQN inflow upper bound. It is read by the DQN environment and agent for the inflow component of the observation space and state normalization. Update this value when adapting the benchmark to a reservoir whose inflow range exceeds the configured bound.

### Curves

The level-storage curve is read from `data/level_storage_curve.csv` and must contain:

```text
water_level_m,storage_1e9_m3
```

The tailwater-discharge curve is read from `data/tailwater_discharge_curve.csv` and must contain:

```text
discharge_m3s,tail_water_level_m
```

### Penalty Settings

Penalty candidates are defined in `data/penalty_settings.yaml`, for example:

```yaml
ddp:
  candidates: [0, 1, 2]
sdp:
  candidates: [0, 1, 2]
iso_surf:
  candidates: [0, 1, 2]
dqn:
  candidates: [0, 5, 10]
```

### Experiment Settings

Algorithm-specific experiment parameters are defined in `data/experiment_settings.yaml`. This file keeps reusable engineering settings out of the runners and separates algorithm hyperparameters from reservoir physical parameters. It includes:

- SDP inflow discretization settings such as `sdp.n_inflow_classes` and `sdp.threshold`.
- ISO-SURF fitting settings such as `iso_surf.smoothing_factor`, `iso_surf.spline_order`, `iso_surf.grid_size`, `iso_surf.max_factor`, and `iso_surf.min_samples_spline`.
- DQN agent, network, training, seed, and progress-display settings under `dqn.agent`, `dqn.network`, `dqn.train`, `dqn.experiment`, and `dqn.display`.

## Quick Start

### 1. Generate Policies

```bash
python scripts/run_policy_generation.py
```

Note: the current `scripts/run_policy_generation.py` enables SDP policy generation by default. The ISO-SURF and DQN calls may need to be enabled in the script if a complete retraining run is required.

### 2. Run Evaluation

```bash
python scripts/run_evaluation.py
```

The evaluation script runs DDP, SDP, ISO-SURF, and DQN in sequence. DDP solves directly on the evaluation inflow. SDP, ISO-SURF, and DQN load previously generated policy artifacts.

### 3. Generate Figures

```bash
python scripts/run_visualization.py
```

Figures and summary files are written to `outputs/visualization/`.

### Forecast Sensitivity

Run the forecast experiment with the same training and evaluation inflow files used by the benchmark.

```bash
python scripts/run_policy_generation.py --forecast-sensitivity
python scripts/run_evaluation.py --forecast-sensitivity
python scripts/run_visualization.py --forecast-sensitivity
```

At the start of the operating month, all three methods observe the calendar month and the starting reservoir level or storage. The perfect-forecast case receives the realized inflow over the operating month as a zero-error forecast. The noisy cases multiply that inflow by a mean-one lognormal factor with a relative standard deviation of 10%, 25%, or 50%. The no-forecast case uses only the calendar month and initial reservoir state. After the controller chooses a target, the simulator uses the realized inflow to update storage and calculate generation.

SDP uses monthly training inflow distributions and conditions them on the forecast when available. It averages operating rewards and next reservoir states over the resulting distribution of realized inflow. ISO-SURF fits each rule against the same full-horizon DDP trajectory computed on the training inflow. Each training example aligns the start-of-month level and the available forecast with the DDP end-of-month target. ISO-SURF fits one bivariate surface per month with a forecast and one univariate curve per month without it. DQN is trained separately with each observation structure.

Evaluation compares policies trained for the supplied forecast with perfect-forecast policies evaluated under the same noisy signals. Noise draws are paired across methods and error levels using the evaluation seeds. All months in the evaluation record are used. DQN reliability is computed separately for each trained model before averaging.

The fixed penalty coefficients, error levels, and noise seeds are in `forecast_sensitivity` in `data/experiment_settings.yaml`. Policy and evaluation files are saved under the `current_month` subdirectory of each algorithm's respective output directory. The sensitivity figure and summary and per-seed tables are written to `outputs/visualization/forecast_sensitivity/current_month/`.

## Workflow Overview

```text
Input data
   |
   +-- Training inflow + reservoir parameters
   |      + experiment settings
   |      -> SDP / ISO-SURF / DQN policy generation
   |
   +-- Evaluation inflow + reservoir parameters + generated policies
   |      -> DDP / SDP / ISO-SURF / DQN evaluation
   |
   +-- Evaluation CSV files
          -> visualization and summary reports
```

## Outputs

Policy artifacts:

```text
outputs/policy_generation/
  sdp/
    preprocessed/sdp_inflow_intervals_detailed.csv
    policy/sdp_train_lambda_*.xlsx
  iso_surf/
    preprocessed/iso_upstream_dp_train_lambda_*.csv
    policy/iso_surf_train_lambda_*.pkl
  dqn/
    policy/dqn_train_lambda_*.pth
```

Evaluation results:

```text
outputs/evaluation/
  ddp/ddp_eval_lambda_*.csv
  sdp/sdp_eval_lambda_*.csv
  iso_surf/iso_surf_eval_lambda_*.csv
  dqn/dqn_eval_lambda_*.csv
```

Visualization outputs:

```text
outputs/visualization/
```

## Adapting the Benchmark to a New Reservoir

To apply the repository to a new reservoir, update at least:

- `data/reservoir_settings.txt`
- `data/level_storage_curve.csv`
- `data/tailwater_discharge_curve.csv`
- `data/SynMonthlyInflow_train.xlsx`
- `data/SynMonthlyInflow_evaluation.xlsx`
- `data/penalty_settings.yaml`
- `data/experiment_settings.yaml`

Before rerunning the benchmark, check that:

- `dead_level <= initial_level <= normal_level`
- The level-storage curve covers the full operating range.
- Storage generally increases monotonically with water level.
- The tailwater curve covers the normal total-discharge range.
- Inflow units are `m3/s`.
- `inflow_upper_bound` in `data/reservoir_settings.txt` covers the expected inflow range used by the DQN observation space and normalization.
- `max_power` and `firm_power` are expressed in kW.

After changing the training inflow or reservoir physical parameters, regenerate SDP, ISO-SURF, and DQN policy artifacts. DDP does not require a pre-trained policy, but evaluation must be rerun.

## Documentation

The full documentation is in `docs/source/` and can be built with Sphinx:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs/source docs/_build/html
```

Open the built documentation at:

```text
docs/_build/html/index.html
```

For the local browser build used in this workspace, the same site may also be generated under:

```text
docs_build/html/index.html
```

## Notes and Known Assumptions

- Input paths currently point to the repository-level `data/` directory.
- DDP, SDP, and ISO-SURF use 1 m water-level discretization by default.
- SDP inflow discretization is configured by `sdp.n_inflow_classes` in `data/experiment_settings.yaml`.
- The DQN observation space and normalization use `inflow_upper_bound` from `data/reservoir_settings.txt`; this replaces the former hard-coded `5000.0 m3/s` inflow upper-bound assumption.
- Trained DQN models can be large and are ignored by default through `*.pth` and `*.pt` patterns.

## License

This project is released under the MIT License. See `LICENSE` for details.
