Input Data Format
=================

Inflow Excel Files
------------------

Training and evaluation inflows use the same table format:

* ``data/SynMonthlyInflow_train.xlsx``
* ``data/SynMonthlyInflow_evaluation.xlsx``

The loader skips the first column and treats the next 12 columns as monthly inflows from January to December. The recommended header is:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
     - Unit
   * - ``Year``
     - Year or sample identifier. This column is skipped by the code.
     - -
   * - ``Jan`` to ``Dec``
     - Monthly mean reservoir inflow
     - ``m3/s``

Requirements:

* The inflow data must be a two-dimensional table.
* Missing values are not allowed.
* Negative values are not allowed.
* After the first column is skipped, exactly 12 monthly columns must remain.

Reservoir Parameters
--------------------

``data/reservoir_settings.txt`` uses a key-value style format:

.. code-block:: text

   "dead_level": 1166
   "normal_level": 1240
   "initial_level": 1240
   "max_turbine_discharge": 2340
   "max_power": 4200000
   "firm_power": 1854000
   "power_coefficient": 8.5
   "delta_t_days": 30.4
   "inflow_upper_bound": 5000.0

``inflow_upper_bound`` is the configured DQN inflow upper bound. It is read from this file by the DQN environment and agent, so the observation-space bound and inflow normalization stay consistent with the reservoir data instead of being maintained as code constants.

Field descriptions:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
     - Unit
   * - ``dead_level``
     - Dead water level
     - m
   * - ``normal_level``
     - Normal pool level and current operating upper bound
     - m
   * - ``initial_level``
     - Initial water level
     - m
   * - ``max_turbine_discharge``
     - Maximum turbine discharge
     - ``m3/s``
   * - ``max_power``
     - Installed capacity or maximum power output
     - kW
   * - ``firm_power``
     - Firm-power threshold
     - kW
   * - ``power_coefficient``
     - Power coefficient used by the current hydropower equation
     - As defined by the existing formula
   * - ``delta_t_days``
     - Time-step length expressed as days per month
     - day
   * - ``inflow_upper_bound``
     - Upper bound used by the DQN observation space and inflow normalization
     - ``m3/s``

Level-Storage Curve
-------------------

``data/level_storage_curve.csv`` must contain:

.. code-block:: text

   water_level_m,storage_1e9_m3

The code uses linear interpolation for both level-to-storage and storage-to-level conversion. The curve must cover ``dead_level``, ``normal_level``, and ``initial_level``. Storage is measured in ``1e9 m3``.

Tailwater Curve
---------------

``data/tailwater_discharge_curve.csv`` must contain:

.. code-block:: text

   discharge_m3s,tail_water_level_m

The code fits a fourth-order polynomial relationship from discharge to tailwater level. When switching to a new reservoir, check that the fitted curve behaves reasonably over the normal discharge range, especially at low flows and beyond the observed sample range.

Penalty Coefficients
--------------------

``data/penalty_settings.yaml`` controls which firm-power penalty coefficients are scanned for each algorithm:

.. code-block:: yaml

   ddp:
     candidates: [0, 1, 2]
   sdp:
     candidates: [0, 1, 2]
   iso_surf:
     candidates: [0, 1, 2]
   dqn:
     candidates: [0, 5, 10]

The lambda value is embedded in output file names, for example ``sdp_train_lambda_1.xlsx``.

Experiment Settings
-------------------

``data/experiment_settings.yaml`` stores algorithm-specific experiment parameters that are shared by the runners. Use this file for reusable algorithm behavior and hyperparameters; keep reservoir physical parameters, including ``inflow_upper_bound``, in ``data/reservoir_settings.txt``.

.. code-block:: yaml

   sdp:
     n_inflow_classes: 20
     threshold: 0.01

   iso_surf:
     smoothing_factor: 1.0
     spline_order: [3, 3]
     grid_size: 25
     max_factor: 10000
     min_samples_spline: 20

   dqn:
     agent:
       learning_rate: 0.0008
       reward_decay: 0.97
       epsilon_start: 0.95
       epsilon_end: 0.01
       epsilon_decay: 0.990
       replace_target_iter: 400
       memory_size: 1000
       batch_size: 128
       firm_penalty_lambda: 20
     network:
       hidden_size: 768
     train:
       study_iteration: 10000
       max_steps: 240
       num_envs: 1
     experiment:
       seed_list: [1, 10, 42, 100]
     display:
       show_progress: true
       progress_update_freq: 10

The penalty sweep values still belong in ``data/penalty_settings.yaml``. The experiment settings file controls reusable algorithm behavior such as SDP inflow discretization, ISO-SURF fitting, and DQN training hyperparameters.
