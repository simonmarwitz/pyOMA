Welcome to pyOMA's documentation!
=================================

A toolbox for Operational Modal Analysis (OMA) developed by Simon Marwitz, 
Volkmar Zabel and Andrei Udrea at the Institute of Structural Mechanics (ISM) 
of the Bauhaus-Universität Weimar. Operational Modal Analysis is 
the process of identifying the modal properties of a civil structure 
from ambient (output-only) vibration measurements.

In a broader sense it consists of a series of processes:

.. blockdiag::
   :desctable:

    blockdiag admin {
      "Ambient Vibration Testing" -> "Signal Processing" -> "System Identification" -> "Modal Analysis" -> "Post Processing";
      "Ambient Vibration Testing" [description = "Acquiring vibration signals (acceleration, velocity, ...) from mechanical structures under ambient excitation (wind, traffic, microtremors, ...)"];
      "Signal Processing" [description = "Filters, Windows, Decimation, Spectral Estimation, Correlation Function Estimation"];
      "System Identification" [description = "Various time-domain and frequency-domain methods for identifiying mathematical models from acquired vibration signals."];
      "Modal Analysis" [description = "Estimation of modal parameters (frequencies, damping ratios, mode shapes) from identified systems. Manually, using stabilization diagrams or automatically using multi-stage clustering methods."];
      "Post Processing" [description = "E.g. plotting of mode shapes, merging of multiple result datasets (setups), statistical analyses."];
    }


The toolbox is currently used on a daily basis to analyze the continuously 
acquired vibration measurements of a structural health monitoring system (since 2015). 
Further uses include various academic and commercial measreument campaigns 
on civil engineering structures including bridges, towers/masts, widespanned floors, etc.

It is written in python 3.xx and runs on any operating system that runs 
python and the required python packages.

Getting started
===============

 * Install Python with scientific Python packages e.g. https://www.anaconda.com/download
 * Download or clone the project in to some directory  ``git clone https://github.com/simonmarwitz/pyOMA.git /dir/to/pyOMA/``
 * Install required packages from ``/dir/to/pyOMA/requirements.txt`` using the Anaconda GUI or with pip ``pip install -r /dir/to/pyOMA/requirements.txt`` 
 * Setup a project directory with measurement and result files ``/dir/to/project/``
 * Copy the script ``scripts/single_setup_analysis.ipynb`` to your project directory. An example JuPyter notebook can be found on the left.
 * Startup JupyterLab or JupyterNotebook and open the script ``/dir/to/project/single_setup_analysis.ipynb``
 * Modify the paths and etc. in the script

For developers:

 * Learn about documenting code with Sphinx (`Tutorial <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ , `Cheatsheet <https://matplotlib.org/sampledoc/cheatsheet.html>`_). The code can be built by navigating to the doc folder in a CLI and run ``make clean && make html`` to mitigate any errors from wrongly formatted documentation syntax.
 * Learn about version control systems with Git: `<https://www.youtube.com/watch?v=8JJ101D3knE>`_

Toolbox Structure
=================

The pyOMA package consists of two modules  :

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

Major rework has to be done for the ``GUI``, 
 * Re-design the GUIs with QtDesigner and then only add functionality in the respective classes.
 * Supplying Jupyter Notebooks with JupyterWidgets for interactive Analysis 

The ``input_files`` packages provides templates for input files for automated and structured analysis of a dataset consisting of multiple measurements.

The ``scripts`` package shall contain templates for certain recurring tasks, as well as commonly used functions, derived from the core and GUI packages.

The ``tests`` package contains common use cases and files, which could be run to test if any changes in the modules result in breaking existing functionality.

The documentation is generated from the git repository by `Sphinx <https://www.sphinx-doc.org/>`_  automatically and available on `<https://py-oma.readthedocs.io/>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   preprocessing
   oma
   postprocessing
   _collections/single_setup_analysis

   

.. TODO::
   
    * Beginner :
        * Creation and simplifaction of scripts on the basis of exemplary measurment campaigns
        * Creating missing GUI parts for the PreProcessing and OMA modules
        * Setup jupyter notebooks for interactive analyses
        * Improvement of the documentation, where needed
    * Intermediate :
        * re-factoring of the StabilizationGUI class into the new GUI package using QtDesigner and proper python code
        * re-factoring of the PlotMSH GUI class into the new GUI package using QtDesigner and proper python code
        * Implementing support for various measurement file formats 
        * Improvement of the documentation, where needed
    * Advanced :
        * Automatic unit tests
        * Creating a new Modeshape plot class based on pyvista or mayavi
        * Profiling and performance improvement of frequently executed code pieces
        * Implementation of variance estimation for PLSCF, PRCE?
        * Improvement of the documentation, where needed
        * Implement MTN=Modal Contribution in Spectral Densities
        * Correct Uncertainty Estimation for SSI-Data based on IOMAC Paper
        * Implement PreGER with Uncertainty Bounds


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
