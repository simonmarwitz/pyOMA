Welcome to pyOMA's documentation!
=================================

A toolbox for Operational Modal Analysis developed by Simon Marwitz, 
Volkmar Zabel and Andrei Udrea at the Institute of Structural Mechanics (ISM) 
of the Bauhaus-Universität Weimar. Operational Modal Analysis (OMA) is 
the process of identifying the modal properties of a civil structure 
from ambient (output-only) vibration measurements....

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
acquired vibration measurements of the Geyer tower monitoring system (since 2015). 
Further uses include various academic and commercial measreument campaigns 
on civil engineering structures including bridges, towers/masts, widespanned floors, etc.

It is written in python 3.xx and runs on any operating system that runs 
python and the required python packages.

Getting started
===============

 * Learn python 3.xx with packages: ``numpy``, ``scipy``, ``matplotlib``, (``PyQt5``/``PySide``, ``jupyter-notebook``)
 * Install python with scientific Python packages → `Intelpython <https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html#python>`_
 * Learn about documenting code with Sphinx (`Tutorial <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ , `Cheatsheet <https://matplotlib.org/sampledoc/cheatsheet.html>`_)
 * Learn about version control systems with Git: `<https://www.youtube.com/watch?v=8JJ101D3knE>`_
 * (Learn GUI programming with PyQt5)
 * QT-Designer for GUI development
 * Install Eclipse with PyDev and Git, or any other development environment
 * Clone the code with git ``git clone ssh://ism_username1234@stratos.bauing.uni-weimar.de/vegas/projects/pyOMA/pyOMA.git/``

All of the above is available on the `VEGAS-cluster@ISM <https://santafe.bauing.uni-weimar.de/dokuwiki/doku.php?id=ism_it:comp_publ:digitallab:cluster:start>`_ (for working remotely)

Toolbox Structure
=================

The toolbox consists of four packages :

::

    pyOMA
    ├── classes
    │   ├── PreprocessingTools.py
    │   ├── ModalBase.py
    │   ├── PLSCF.py
    │   ├── PRCE.py
    │   ├── SSICovRef.py
    │   ├── SSIData.py
    │   ├── VarSSIRef.py
    │   ├── StabilDiagram.py
    │   ├── PlotMSH.py
    │   ├── PostProcessingTools.py
    │   └── ...
    ├── doc
    │   └── ...
    ├── GUI
    │   ├── PlotMSHGUI.py
    │   ├── StabilGUI.py
    │   ├── Helpers.py
    │   └── ...
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
 

Current development is focused on the ``classes`` package which contains all the algorithms.

Major rework has to be done for the ``GUI``, using QTDesigner to design the GUIs and then only add functionality in the respective classes.

The ``input_files`` packages provides templates for input files for automated and structured analysis of a dataset consisting of multiple measurements.

The ``scripts`` package shall contain templates for certain recurring tasks, as well as commonly used functions, derived from the classes and GUI packages.

The ``tests`` package contains common use cases and files, which could be run to test if any changes in the classes result in breaking existing functionality.

The documentation is generated from the git repository by `Sphinx <https://www.sphinx-doc.org/>`_  automatically and available on `<https://santafe.bauing.uni-weimar.de/pyOMA>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    
   preprocessing
   oma
   postprocessing

.. TODO::
   
    * Beginner :
        * Creation and simplifaction of scripts on the basis of exemplary measurment campaigns
        * Creating missing GUI parts for the PreProcessing and OMA classes
        * Setup jupyter notebooks for interactive analyses
        * Improvement of the documentation, where needed
    * Intermediate :
        * re-factoring of the StabilizationGUI class into the new GUI package using QtDesigner and proper python code
        * re-factoring of the PlotMSH GUI class into the new GUI package using QtDesigner and proper python code
        * Implementing support for various measurement file formats 
        * Improvement of the documentation, where needed
    * Advanced :
        * Automatic unit tests
        * Test out possibilities for an interactive jupyter StabilizationDiagram `<https://towardsdatascience.com/interactive-visualizations-in-jupyter-notebook-3be02ab2b8cd>`_
        * Creating a new Modeshape plot class based on pyvista or mayavi
        * Profiling and performance improvement of frequently executed code pieces
        * Implementation of variance estimation for PLSCF, PRCE?
        * Improvement of the documentation, where needed


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
