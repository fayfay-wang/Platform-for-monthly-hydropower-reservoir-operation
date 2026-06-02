Installation
============

Requirements
------------

Python 3.11 is recommended. The base dependencies include ``numpy``, ``pandas``, ``scipy``, ``matplotlib``, ``openpyxl``, and ``PyYAML``. To run DQN experiments, install the optional ``torch`` and ``gymnasium`` dependencies as well.

Basic Installation
------------------

Run the following commands from the repository root:

.. code-block:: powershell

   conda create -n reservoir_benchmark python=3.11 -y
   conda activate reservoir_benchmark
   python -m pip install -e .

DQN Optional Dependencies
-------------------------

.. code-block:: powershell

   python -m pip install -e ".[dqn]"

Documentation Dependencies
--------------------------

To build this documentation locally:

.. code-block:: powershell

   python -m pip install -r docs/requirements.txt

If ``sphinx_rtd_theme`` is not installed, the documentation can still be built with the default Sphinx theme, but the appearance will differ slightly from the ReadTheDocs style.
