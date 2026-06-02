API and Module Map
==================

Common Entry Points
-------------------

.. list-table::
   :header-rows: 1

   * - Goal
     - Entry point
   * - Generate policies
     - ``scripts/run_policy_generation.py``
   * - Run evaluation
     - ``scripts/run_evaluation.py``
   * - Generate figures
     - ``scripts/run_visualization.py``

Key Classes and Functions
-------------------------

.. list-table::
   :header-rows: 1

   * - Module
     - Main object
     - Purpose
   * - ``reservoir_benchmark.shared.reservoir_model``
     - ``SharedReservoirModel``
     - Read reservoir parameters, interpolate curves, fit tailwater levels, and calculate power output
   * - ``reservoir_benchmark.shared.io_utils``
     - ``load_monthly_inflows_from_excel``
     - Read and validate training and evaluation inflows
   * - ``reservoir_benchmark.policy_generation.ddp.ddp_solver``
     - ``ReservoirOperation``
     - Deterministic dynamic programming solver
   * - ``reservoir_benchmark.policy_generation.sdp.sdp_preprocessor``
     - ``SDPInflowDiscretizer``
     - Discretize monthly inflows into equal-probability classes
   * - ``reservoir_benchmark.policy_generation.sdp.sdp_solver``
     - ``ReservoirOperation``
     - SDP rule-table solver
   * - ``reservoir_benchmark.policy_generation.iso_surf.iso_surf_runner``
     - ``ISOSurfPolicyFitter``
     - Fit and save ISO-SURF rule surfaces
   * - ``reservoir_benchmark.policy_generation.dqn.dqn_reservoir``
     - ``ReservoirEnv``
     - Reservoir operation environment for DQN
   * - ``reservoir_benchmark.policy_generation.dqn.dqn_agent``
     - ``Config``, ``BasicInfo``, ``DQNAgent``
     - DQN configuration, data container, and agent

Extension Notes
---------------

If you later want to support different data directories from the command line, start by refactoring ``project_paths.py`` and the script entry points so that ``data_dir`` can be supplied through a command-line argument or an environment variable. That would allow new-reservoir cases to live in separate folders instead of repeatedly overwriting ``data``.
