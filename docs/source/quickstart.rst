Quick Start
===========

Prepare the Data
----------------

Make sure the ``data`` directory contains at least the following files:

.. code-block:: text

   data/
     SynMonthlyInflow_train.xlsx
     SynMonthlyInflow_evaluation.xlsx
     reservoir_settings.txt
     level_storage_curve.csv
     tailwater_discharge_curve.csv
     penalty_settings.yaml
     experiment_settings.yaml

Generate Policies
-----------------

Run:

.. code-block:: powershell

   python scripts/run_policy_generation.py

Note: ``scripts/run_policy_generation.py`` currently enables SDP policy generation by default. The ISO-SURF and DQN calls are commented out in the script. To generate all three trained policy types in one run, enable the ``main_iso_surf()`` and ``main_dqn()`` calls, or run the corresponding modules separately.

Run Evaluation
--------------

.. code-block:: powershell

   python scripts/run_evaluation.py

The evaluation script runs DDP, SDP, ISO-SURF, and DQN in sequence. DDP solves directly on the evaluation inflow series. SDP, ISO-SURF, and DQN load previously trained policy files. If a required policy file is missing, the corresponding evaluation will fail or be skipped.

Generate Figures
----------------

.. code-block:: powershell

   python scripts/run_visualization.py

Figures and summary files are written to ``outputs/visualization``.
