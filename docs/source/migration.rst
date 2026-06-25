Migrating to a New Reservoir
============================

Minimum Replacement Checklist
-----------------------------

To use this repository for a new reservoir, update or replace at least the following files:

.. list-table::
   :header-rows: 1

   * - File
     - What to update
   * - ``data/reservoir_settings.txt``
     - Dead water level, normal pool level, initial level, turbine capacity, firm power, power coefficient, time-step length, and DQN inflow upper bound
   * - ``data/level_storage_curve.csv``
     - The new reservoir's level-storage curve. Column names must stay unchanged.
   * - ``data/tailwater_discharge_curve.csv``
     - The new reservoir's discharge-tailwater curve. Column names must stay unchanged.
   * - ``data/SynMonthlyInflow_train.xlsx``
     - Training inflow used by SDP, ISO-SURF, and DQN policy generation
   * - ``data/SynMonthlyInflow_evaluation.xlsx``
     - Evaluation inflow used by the unified benchmark
   * - ``data/penalty_settings.yaml``
     - Lambda candidates scanned by each algorithm
   * - ``data/experiment_settings.yaml``
     - SDP inflow discretization, ISO-SURF fitting settings, and DQN training hyperparameters

Pre-Run Checks
--------------

Before running a new-reservoir case, check that:

1. ``dead_level <= initial_level <= normal_level``.
2. The level-storage curve covers the full range from ``dead_level`` to ``normal_level``.
3. Storage generally increases monotonically with water level.
4. The tailwater curve covers the normal total-discharge range.
5. The inflow files contain exactly 12 monthly columns after the first column is skipped, with units in ``m3/s``.
6. ``inflow_upper_bound`` in ``data/reservoir_settings.txt`` covers the expected inflow range used by DQN.
7. ``max_power`` and ``firm_power`` are both expressed in kW.
8. Old policy files have either been cleared or are expected to be overwritten.

Artifacts That Must Be Regenerated
----------------------------------

After changing the training inflow or reservoir physical parameters, regenerate:

* SDP rule tables.
* ISO-SURF upstream DDP trajectories and ``.pkl`` policies.
* DQN ``.pth`` models.

DDP does not depend on a pre-trained policy, but it is solved again during evaluation.

When Only the Evaluation Inflow Changes
---------------------------------------

If only ``SynMonthlyInflow_evaluation.xlsx`` changes, while the reservoir parameters and training inflow remain the same:

* DDP must be evaluated again.
* SDP, ISO-SURF, and DQN can reuse existing policy files, but evaluation still needs to be rerun.

Hard-Coded Assumptions to Check
-------------------------------

Several default settings are sensitive to the new reservoir:

* DDP, SDP, and ISO-SURF use 1 m water-level discretization. A large operating range increases computational cost; a finer operating policy requires changing the discretization step in the algorithms.
* The SDP inflow discretization is configured by ``sdp.n_inflow_classes`` in ``data/experiment_settings.yaml``.
* The DQN observation space and inflow normalization use ``inflow_upper_bound`` from ``data/reservoir_settings.txt``. If the new reservoir has higher inflows, update that setting instead of editing DQN source code.
* The policy-generation entry point currently runs SDP by default. Enable the ISO-SURF and DQN calls if a full retraining run is required.
* Input paths currently point to the repository-level ``data`` directory and are not command-line parameters.
