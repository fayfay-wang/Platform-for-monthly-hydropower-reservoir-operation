Reservoir Benchmark Documentation
=================================

Reservoir Benchmark is a single-reservoir operation benchmark. It runs DDP, SDP, ISO-SURF, and DQN under a shared reservoir model, inflow dataset, and evaluation protocol.

This documentation is organized as a practical project guide: start with installation and a quick start, then move through data requirements, training and evaluation workflows, migration to a new reservoir, output files, and documentation maintenance.

Scope
-----

This repository is designed to answer questions such as:

* How do different reservoir operation policies compare in power generation and firm-power reliability under the same reservoir and evaluation inflow conditions?
* After replacing the training inflow, which policies need to be regenerated?
* When adapting the benchmark to a new reservoir, which curves, constraints, and parameters need to be changed?

Reuse Summary
-------------

The repository is reusable, but the inputs must follow the expected file names and schemas. In particular:

* Evaluation inflow: replace ``data/SynMonthlyInflow_evaluation.xlsx``.
* Training inflow: replace ``data/SynMonthlyInflow_train.xlsx``, then regenerate the trained policies.
* Reservoir parameters: edit ``data/reservoir_settings.txt``.
* Level-storage curve: replace ``data/level_storage_curve.csv``.
* Tailwater-discharge curve: replace ``data/tailwater_discharge_curve.csv``.

All replacement data must satisfy the column-name, unit, and range requirements described in :doc:`data-format`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   data-format
   workflows
   migration
   outputs

.. toctree::
   :maxdepth: 2
   :caption: Development and Maintenance

   project-structure
   api-map
   making-docs
   faq
