Building and Publishing This Documentation
==========================================

Documentation Stack
-------------------

This documentation is built with Sphinx and published as static HTML. The configuration lives in ``docs/source/conf.py``, and the page sources are the ``docs/source/*.rst`` files. Installing ``sphinx-rtd-theme`` gives the site a ReadTheDocs-like sidebar layout.

Build Locally
-------------

From the repository root, run:

.. code-block:: powershell

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html docs/source docs/_build/html

After the build completes, open:

.. code-block:: text

   docs/_build/html/index.html

On Windows, you can also use:

.. code-block:: powershell

   cd docs
   .\make.bat html

On Linux or macOS:

.. code-block:: bash

   cd docs
   make html

Add a New Page
--------------

1. Create a new ``.rst`` file under ``docs/source``.
2. Add the file name, without the extension, to the ``toctree`` in ``docs/source/index.rst``.
3. Rebuild the documentation.

Publish to GitHub Pages
-----------------------

There are two common approaches.

The first option is to manually publish ``docs/_build/html`` to a GitHub Pages branch.

The second option is to build and deploy automatically with GitHub Actions. For example, add ``.github/workflows/docs.yml``:

.. code-block:: yaml

   name: docs

   on:
     push:
       branches: [main]

   permissions:
     contents: read
     pages: write
     id-token: write

   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.11"
         - run: python -m pip install -r docs/requirements.txt
         - run: python -m sphinx -b html docs/source docs/_build/html
         - uses: actions/upload-pages-artifact@v3
           with:
             path: docs/_build/html
     deploy:
       needs: build
       runs-on: ubuntu-latest
       environment:
         name: github-pages
         url: ${{ steps.deployment.outputs.page_url }}
       steps:
         - id: deployment
           uses: actions/deploy-pages@v4

In the repository settings, enable GitHub Pages and select GitHub Actions as the source.
