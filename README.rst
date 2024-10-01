pyOMA
=====

.. image:: https://raw.githubusercontent.com/simonmarwitz/pyOMA/refs/heads/master/doc/_static/logo.png
  :width: 110
  :height: 110
  :align: left 

pyOMA is an open-source toolbox for Operational Modal Analysis (OMA) developed 
by Simon Marwitz, Volkmar Zabel et al. at the Institute of Structural Mechanics (ISM) 
of the Bauhaus-Universität Weimar. Operational Modal Analysis is a methodogy for
the identification of structural modal properties from ambient (output-only) 
vibration measurements. It is written in python 3.xx.


 * **Documentation:** https://py-oma.readthedocs.io
 * **Source Code:** https://github.com/simonmarwitz/pyOMA


About Operational Modal Analysis
================================

In a broader sense OMA consists of a series of processes:

.. image:: https://raw.githubusercontent.com/simonmarwitz/pyOMA/refs/heads/master/doc/_static/concept_map.png
  :width: 800
  :alt: blockdiagram


.. list-table::

      * - Ambient Vibration Testing
        - Acquiring vibration signals (acceleration, velocity, ...) from mechanical structures under ambient excitation (wind, traffic, microtremors, ...)
      * - Signal Processing
        - Filters, Windows, Decimation, Spectral Estimation, Correlation Function Estimation
      * - System Identification
        - Various time-domain and frequency-domain methods for identifiying mathematical models from acquired vibration signals.
      * - Modal Analysis
        - Estimation of modal parameters (frequencies, damping ratios, mode shapes) from identified systems. Manually, using stabilization diagrams or automatically using multi-stage clustering methods.
      * - Post Processing
        - E.g. plotting of mode shapes, merging of multiple result datasets (setups), statistical analyses, SHM


Applications of pyOMA
=====================

The toolbox is currently used on a daily basis to analyze the continuously 
acquired vibration measurements of a structural health monitoring system (since 2015). 
Further uses include various academic and commercial measreument campaigns 
on civil engineering structures including bridges, towers/masts, widespanned floors, etc.


Install
=======

Requirements
------------

- python https://www.python.org/ or https://www.anaconda.com/download
- matplotlib http://matplotlib.org/
- numpy http://www.numpy.org/
- scipy https://scipy.org/

Optional libraries:

- ipywidgets https://github.com/jupyter-widgets/ipywidgets
- ipympl https://matplotlib.org/ipympl/
- JupyterLab https://jupyter.org/

Install latest release version via git
--------------------------------------

.. code-block:: bash

   $ git clone https://github.com/simonmarwitz/pyOMA.git /dir/to/pyOMA/
   $ pip install -r /dir/to/pyOMA/requirements.txt


Get started with a project
==========================

 #. Setup a project directory ``/dir/to/project/`` containing measurement and result files 
 #. Copy the script ``scripts/single_setup_analysis.ipynb`` to your project directory. An example JuPyter notebook can be found on the left.
 #. Startup JupyterLab or JupyterNotebook and open the script ``/dir/to/project/single_setup_analysis.ipynb``
 #. Modify the paths in the second cell and run the script

Getting help
============

 #. In case of errors check that:
 
  * input files are formatted correctly
  
  * arguments are of the right type and order
  
  * search the internet for similar errors
  
 #. Open an issue at https://github.com/simonmarwitz/pyOMA/issues

Toolbox Structure
=================

::

    pyOMA
    ├── pyOMA
    │   ├── core
    │   │  ├── PreProcessingTools.py
    │   │  ├── ModalBase.py
    │   │  ├── PLSCF.py
    │   │  ├── PRCE.py
    │   │  ├── SSICovRef.py
    │   │  ├── SSIData.py
    │   │  ├── VarSSIRef.py
    │   │  ├── StabilDiagram.py
    │   │  ├── PlotMSH.py
    │   │  ├── PostProcessingTools.py
    │   │  └── ...
    │   ├── GUI
    │   │  ├── PlotMSHGUI.py
    │   │  ├── StabilGUI.py
    │   │  ├── Helpers.py
    │   │  └── ...
    
Additionally some further files are provided with it:

::

    ├── doc
    ├── input_files
    ├── scripts
    ├── tests
    │   ├── basic_tests.py
    │   └── files
    │       └── ...
    ├── LICENSE
    ├── README.rst
    ├── requirements.txt
    └── setup.py
 

Current development is focused on the ``core`` package which contains all the algorithms.

The ``input_files`` packages provides templates for input files for automated and structured analysis of a dataset consisting of multiple measurements.

The ``scripts`` package shall contain templates for certain recurring tasks, as well as commonly used functions, derived from the core and GUI packages.

The ``tests`` package contains common use cases and files, which could be run to test if any changes in the modules result in breaking existing functionality.

The documentation is generated from the git repository by `Sphinx <https://www.sphinx-doc.org/>`_  automatically and available on `<https://py-oma.readthedocs.io/>`_



Contributing
============

For beginners:

 * Learn about documenting code with Sphinx (`Tutorial <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ , `Cheatsheet <https://matplotlib.org/sampledoc/cheatsheet.html>`_). The code can be built by navigating to the doc folder in a CLI and run ``make clean && make html`` to mitigate any errors from wrongly formatted documentation syntax.
 * Learn about version control systems with Git: `<https://www.youtube.com/watch?v=8JJ101D3knE>`_
 * Fork the project on GitHub and start development
 * Open a Pull Request to get your changes merged into the project

