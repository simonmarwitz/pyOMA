Welcome to pyOMA's documentation!
=================================

A toolbox for Operational Modal Analysis developed by Simon Marwitz, 
Volkmar Zabel and Andrei Udrea at the Institute of Structural Mechanics (ISM) 
of the Bauhaus-Universität Weimar. Operational Modal Analysis (OMA) is 
the process of identifying the modal properties of a civil structure 
from ambient (output-only) vibration measurements....

In a broader sense it consists of a series of processes:


The toolbox is currently used on a daily basis to analyze the continuously acquired vibration measurements of the Geyer tower monitoring system (since 2015). Further uses include various academic and commercial measreument campaigns on civil engineering structures including bridges, towers/masts, widespanned floors, etc.

It is written in python 3.7 and hosted on a git repository on /vegas/projects/modal_analysis_python/ on the institutes fileserver (binion.bauing.uni-weimar.de). It runs on any operating system that runs python and the required python packages.

Getting started
===============

 * Learn python 3 with packages: numpy, scipy, matplotlib, (PyQt5, jupyter-notebook)
 * Install python with scientific Python packages → `Intelpython <https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html#python>`_
 * Learn about documenting code with Sphinx (`Tutorial <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ , `Cheatsheet <https://matplotlib.org/sampledoc/cheatsheet.html>`_)
 * Learn about versioning with GIT: `<https://www.youtube.com/watch?v=8JJ101D3knE>`_
 * (Learn GUI programming with PyQt5)
 * QT-Designer for GUI development
 * Install Eclipse with PyDev and Git, or any other development environment
 * Clone the code with git from ssh://ism_username1234@stratos.bauing.uni-weimar.de/vegas/projects/pyOMA/repository.git/

All of the above is available on the VEGAS cluster @ ISM (for working remotely)

Toolbox Structure
=================

The toolbox consists of four packages: 
 * classses
 * GUI
 * input_files
 * scripts

Current development is focused on the classes package which contains all the algorithms:


Currently all the GUIs are included with the classes, which should be dissentangled in the furture into the GUI packages.
The input_files packages provides templates for input files for automated and structured analysis of a dataset consisting of multiple measurements
The scripts package shall contain templates for certain recurring tasks, as well as commonly used functions, derived from the classes and GUI packages.

The documentation is generated from the git repository by Sphinx automatically and available on https://santafe.bauing.uni-weimar.de/pyOMA

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    
   preprocessing
   oma
   postprocessing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
