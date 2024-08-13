import sys
# repo_path = '/usr/wrk/people9/sima9999/git/pyOMA'
# repo_path = '/ismhome/staff/womo1998/git/pyOMA'
# sys.path.append(repo_path)
import os
from pathlib import Path

import numpy as np
from pyOMA.core.PreProcessingTools import PreProcessSignals, GeometryProcessor
from pyOMA.core.PLSCF import PLSCF
from pyOMA.core.PRCE import PRCE
from pyOMA.core.SSICovRef import BRSSICovRef
from pyOMA.core.SSIData import SSIData, SSIDataMC
from pyOMA.core.VarSSIRef import VarSSIRef
from pyOMA.core.StabilDiagram import StabilCalc, StabilPlot
from pyOMA.core.PlotMSH import ModeShapePlot

from pyOMA.GUI.PlotMSHGUI import start_msh_gui
from pyOMA.GUI.StabilGUI import start_stabil_gui


# Define a function that loads the provided measurement file(s)
PreProcessSignals.load_measurement_file = np.load

working_dir = Path(f'/home/sima9999/git/pyOMA/tests/files/')
result_folder = Path(f'{working_dir}/measurement_1/')
meas_name = os.path.basename(result_folder)
setup_info=result_folder / 'setup_info.txt'
meas_file=result_folder / (meas_name + '.npy')
chan_dofs_file=result_folder / "channel_dofs.txt"

# Select OMA Method, one of: PLSCF PRCE BRSSICovRef PogerSSICovRef SSIData SSIDataMC VarSSIRef
method=BRSSICovRef
conf_file=working_dir / 'varssi_config.txt'

# define script switches
skip_existing=False
save_results=False
interactive=True


geometry_data = GeometryProcessor.load_geometry(
    nodes_file=working_dir / 'grid.txt',
    lines_file=working_dir / 'lines.txt')

if not os.path.exists(result_folder / 'prep_data.npz') or not skip_existing:
    prep_data = PreProcessSignals.init_from_config(
        conf_file=setup_info,
        meas_file=meas_file,
        chan_dofs_file=chan_dofs_file)

    if save_results:
        prep_data.save_state(result_folder / 'prep_data.npz')
else:
    prep_data = PreProcessSignals.load_state(result_folder / 'prep_data.npz')

if not os.path.exists(
        result_folder /
        'modal_data.npz') or not skip_existing:

    modal_data = method.init_from_config(conf_file, prep_data)
    
    if save_results:
        modal_data.save_state(result_folder / 'modal_data.npz')
else:
    modal_data = method.load_state(
        result_folder / 'modal_data.npz', prep_data)

if os.path.exists(result_folder / 'stabil_data.npz') and skip_existing:
    stabil_calc = StabilCalc.load_state(
        result_folder / 'stabil_data.npz', modal_data, prep_data)
else:
    stabil_calc = StabilCalc(modal_data, prep_data)
stabil_calc.export_results('/usr/scratch4/sima9999/test.txt')

if interactive:
    stabil_plot = StabilPlot(stabil_calc)
    start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_data)

    if save_results:
        stabil_calc.save_state(result_folder / 'stabil_data.npz')

if interactive:

    mode_shape_plot = ModeShapePlot(
        prep_data=prep_data,
        stabil_calc=stabil_calc,
        geometry_data=geometry_data,
        modal_data=modal_data)
    start_msh_gui(mode_shape_plot)
