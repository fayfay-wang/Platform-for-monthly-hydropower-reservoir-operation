FAQ
===

Can I rerun the benchmark by replacing only the evaluation inflow?
------------------------------------------------------------------

Yes. The file name must remain ``data/SynMonthlyInflow_evaluation.xlsx``, and the format must keep the first column as a year or sample identifier followed by 12 monthly inflow columns. After replacing the file, rerun ``python scripts/run_evaluation.py``.

Can I retrain policies by replacing only the training inflow?
-------------------------------------------------------------

Yes, but the trained policies must be regenerated. SDP rediscretizes inflows from the new training dataset. ISO-SURF regenerates the upstream DDP trajectories and refits the rule surfaces. DQN retrains its models. The current policy-generation script enables only SDP by default, so enable ISO-SURF and DQN if you need a complete retraining run.

Is it enough to change only the three reservoir curves when switching reservoirs?
---------------------------------------------------------------------------------

No. In addition to the level-storage and tailwater curves, you also need to update the water-level constraints, turbine capacity, firm power, power coefficient, and other settings in ``reservoir_settings.txt``.

Why cannot old policies be used directly for a new reservoir?
-------------------------------------------------------------

The SDP rule tables, ISO-SURF rule surfaces, and DQN models all encode the state range, storage relationship, tailwater behavior, and power constraints of the original reservoir. Once those physical relationships change, the old policies are usually no longer valid.

Where are the evaluation results?
---------------------------------

CSV files are written under ``outputs/evaluation``. Figures are written under ``outputs/visualization``.

Where is the documentation website?
-----------------------------------

After building the docs, open ``docs/_build/html/index.html``.
