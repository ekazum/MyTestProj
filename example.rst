.. MyTestProj Documentation documentation master file, created by
   sphinx-quickstart on Tue Dec 31 14:00:00 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MyTestProj's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Project Overview
================

This is a sample project to demonstrate the capabilities of the PyHealth library for healthcare AI applications.

Features
--------

*   **Data Loading**: Utilizes `pyhealth.datasets.MIMIC4Dataset` to load data.
*   **Task Definition**: Implements in-hospital mortality prediction.
*   **Data Splitting**: Splits data by patient for robust evaluation.


Example Code
------------

Here is a sample of how to load the MIMIC-IV dataset:

.. code-block:: python

   from pyhealth.datasets import MIMIC4Dataset

   mimic4_dataset = MIMIC4Dataset(
       ehr_root="/path/to/your/mimiciv/data",
       ehr_tables=["patients", "admissions", "diagnoses_icd"],
       dev=True
   )

   print(f"Dataset loaded with {len(mimic4_dataset.patients)} patients.")

This is a basic example to get you started with reStructuredText. You can expand this file to build comprehensive documentation for your project.
