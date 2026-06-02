Output Files
============

Policy Outputs
--------------

.. code-block:: text

   outputs/policy_generation/
     sdp/
       preprocessed/sdp_inflow_intervals_detailed.csv
       policy/sdp_train_lambda_*.xlsx
     iso_surf/
       preprocessed/iso_upstream_dp_train_lambda_*.csv
       policy/iso_surf_train_lambda_*.pkl
     dqn/
       policy/dqn_train_lambda_*.pth

Evaluation Outputs
------------------

.. code-block:: text

   outputs/evaluation/
     ddp/ddp_eval_lambda_*.csv
     sdp/sdp_eval_lambda_*.csv
     iso_surf/iso_surf_eval_lambda_*.csv
     dqn/dqn_eval_lambda_*.csv

Main columns in the evaluation CSV files:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
   * - ``Month``
     - Continuous monthly time index
   * - ``Water_Level_m``
     - End-of-month or recorded water level
   * - ``Power_kW``
     - Power output
   * - ``Electricity_GWh``
     - Monthly electricity generation
   * - ``Total_Discharge_m3s``
     - Total release
   * - ``Turbine_Discharge_m3s``
     - Turbine discharge
   * - ``Spill_Discharge_m3s``
     - Spill discharge
   * - ``Inflow_m3s``
     - Reservoir inflow

Figure Outputs
--------------

``scripts/run_visualization.py`` reads the evaluation results and generates scenario-level figures and overall trade-off plots under:

.. code-block:: text

   outputs/visualization/
