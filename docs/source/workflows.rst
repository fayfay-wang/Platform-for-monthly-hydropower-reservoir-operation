Workflows
=========

Overall Workflow
----------------

.. code-block:: text

   Input data
      |
      +-- Training inflow + reservoir parameters + experiment settings -> SDP / ISO-SURF / DQN policy generation
      |
      +-- Evaluation inflow + reservoir parameters + generated policies -> DDP / SDP / ISO-SURF / DQN evaluation
      |
      +-- Evaluation CSV files -> visualization and summary reports

DDP
---

During evaluation, DDP uses ``data/SynMonthlyInflow_evaluation.xlsx`` directly. It solves the deterministic dynamic programming problem over the full evaluation inflow sequence and writes ``outputs/evaluation/ddp/ddp_eval_lambda_*.csv``.

DDP does not require a pre-saved policy file. However, evaluation should be rerun whenever the evaluation inflow, reservoir parameters, or penalty coefficients change.

SDP
---

The SDP training workflow is:

1. Read the training inflow from ``data/SynMonthlyInflow_train.xlsx``.
2. Discretize each month's inflow into ``sdp.n_inflow_classes`` equal-probability classes from ``data/experiment_settings.yaml``.
3. Generate ``outputs/policy_generation/sdp/preprocessed/sdp_inflow_intervals_detailed.csv``.
4. Train a rule table for each lambda value in ``sdp.candidates`` from ``penalty_settings.yaml``.
5. Write ``outputs/policy_generation/sdp/policy/sdp_train_lambda_*.xlsx``.

SDP evaluation loads the corresponding rule table and runs a continuous multi-year simulation on ``data/SynMonthlyInflow_evaluation.xlsx``.

ISO-SURF
--------

The ISO-SURF training workflow is:

1. Run the upstream DDP workflow on the training inflow.
2. Save ``outputs/policy_generation/iso_surf/preprocessed/iso_upstream_dp_train_lambda_*.csv``.
3. Fit monthly rule surfaces of the form ``(S_t, I_t) -> S_{t+1}`` from the DDP trajectories, using the ISO-SURF fitting settings in ``data/experiment_settings.yaml``.
4. Write ``outputs/policy_generation/iso_surf/policy/iso_surf_train_lambda_*.pkl``.

ISO-SURF evaluation loads the saved ``.pkl`` policy and runs a closed-loop simulation on the evaluation inflow.

DQN
---

The DQN training workflow is:

1. Read the training inflow from ``data/SynMonthlyInflow_train.xlsx``.
2. Build a Gymnasium environment from ``reservoir_settings.txt``, the level-storage curve, and the tailwater curve. The environment uses ``inflow_upper_bound`` from ``reservoir_settings.txt`` for the inflow component of the observation space and normalization.
3. Train one model for each lambda value in ``dqn.candidates``.
4. Write ``outputs/policy_generation/dqn/policy/dqn_train_lambda_*.pth``.

DQN evaluation loads ``data/SynMonthlyInflow_evaluation.xlsx`` and the corresponding ``.pth`` model.

The DQN training length, seed list, network width, replay settings, and progress display are configured under ``dqn`` in ``data/experiment_settings.yaml``. The default ``dqn.train.study_iteration`` value is 10000.
