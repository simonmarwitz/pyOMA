from PreprocessingTools import PreprocessData, GeometryProcessor
from SSICovRef import BRSSICovRef, PogerSSICovRef
from StabilDiagram import StabilCalc, StabilPlot, StabilGUI, start_stabil_gui
from PlotMSH import ModeShapePlot, start_msh_gui
import os
import matplotlib.pyplot as plot
import pandas as pd


files=['/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_15_asc_Dateien/Measurement_1.asc',
       '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_15_asc_Dateien/Measurement_2.asc',
       '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_15_asc_Dateien/Measurement_3.asc',
       '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_15_asc_Dateien/Measurement_4.asc',
       '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_22_asc_Dateien/Measurement_1.asc',
       '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_22_asc_Dateien/Measurement_2.asc',
       '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_22_asc_Dateien/Measurement_3.asc',
       '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/2018_05_22_asc_Dateien/Measurement_4.asc',]
files=['/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_1_ambient.asc',
'/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_2_ambient.asc',
'/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_3_ambient.asc',
'/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_4_ambient.asc',
'/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_5_ambient_2.asc',
'/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_6_ambient.asc',
'/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_7_ambient.asc',
'/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Ambient_Tests/Messdaten/2018_06_12_asc_Dateien/meas_8_ambient.asc']
tables = [pd.read_table(file) for file in files]
nodes_file = '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/MACEC/Grid.asc'
beams_file = '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/MACEC/Beams_4.asc'
master_slave_file = '/ismhome/staff/womo1998/Projects/2018_ExperimentalStructuralDynamics/Messdaten/MACEC/Slave.asc'
modal_data = PogerSSICovRef()

geometry_data = GeometryProcessor.load_geometry(nodes_file, beams_file, master_slave_file)

for j,table in enumerate(tables[:]):
    prep_data = PreprocessData(table.iloc[:,1:11].values,2000,ref_channels=[0,1,2,3])
    prep_data.decimate_data(decimate_factor=8, highpass=None)
    prep_data.decimate_data(decimate_factor=4, highpass=None)
    file = files[j]
    #chan_dofs_file = os.path.dirname(file)+'/'+os.path.splitext(os.path.basename(file))[0]+'_channel_dofs.txt'
    #chan_dofs = prep_data.load_chan_dofs(chan_dofs_file)
    #prep_data.add_chan_dofs(chan_dofs)
    prep_data.add_chan_dofs([[0,    0,0,0],[1,    0,90,0],
                             [2,    1,0,0],[3,    1,90,0],
                             [4,j*3+2,0,0],[5,j*3+2,90,0],
                             [6,j*3+3,0,0],[7,j*3+3,90,0],
                             [8,j*3+4,0,0],[9,j*3+4,90,0],
                             ])
    #print(prep_data.chan_dofs)
    #prep_data.get_fft()
    #prep_data.plot_data()
    
    prep_data.compute_correlation_matrices(200, num_blocks=False)
    #prep_data.plot_svd_spectrum()
    modal_data.add_setup(prep_data)

modal_data.pair_channels()
modal_data.build_merged_subspace_matrix(100)
modal_data.compute_state_matrices(100)
modal_data.compute_modal_params()
    
# modal_data = BRSSICovRef(prep_data)
# modal_data.build_toeplitz_cov(100)
# modal_data.compute_state_matrices(100)
# modal_data.compute_modal_params()

stabil_data = StabilCalc(modal_data, prep_data)
stabil_plot = StabilPlot(stabil_data)
start_stabil_gui(stabil_plot, modal_data)#, geometry_data)



