Project Structure
=================

.. code-block:: text

   reservoir_benchmark/
     shared/
       reservoir_model.py
       io_utils.py
       project_paths.py
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

Shared Layer
------------

``shared/reservoir_model.py`` reads reservoir parameters, the level-storage curve, and the tailwater curve. It also provides level-storage conversion, tailwater calculation, and hydropower output calculation.

``shared/io_utils.py`` reads inflow Excel files, penalty lambda candidates, and algorithm experiment settings.

``shared/project_paths.py`` centralizes paths for ``data`` and ``outputs``.

Policy Generation Layer
-----------------------

``policy_generation`` contains the training workflows for SDP, ISO-SURF, and DQN. DDP is mainly used as a deterministic optimizer and is also reused by the ISO-SURF preprocessing step.

Evaluation Layer
----------------

``evaluation`` generates standardized CSV files for each algorithm under the same evaluation inflow conditions, making cross-method comparison easier.

Script Entry Points
-------------------

The ``scripts`` directory provides three common entry points:

* ``run_policy_generation.py``: policy generation.
* ``run_evaluation.py``: unified evaluation.
* ``run_visualization.py``: figures and summary outputs.
