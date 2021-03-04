'''
Created on 04.03.2021

@author: womo1998
'''
import numpy as np

from classes.PreprocessingTools  import *

from classes.PLSCF  import *
from classes.PRCE  import *
from classes.SSICovRef import *
from classes.SSIData import *
from classes.VarSSIRef import *
#from classes.ERA import *

from classes.StabilDiagram import *
from classes.PostProcessingTools  import *
from classes.PlotMSH  import *


    
    
def analysis_chain(tmpdir):
    #Generate some random measurement data
    num_channels = 10
    num_timesteps = 25*20*60 # 25 Hz for 20 minutes
    measurement = np.random.rand(num_timesteps, num_channels)
    
    #Initialize Geometry and PreProcessData
    geometry = GeometryProcessor.load_geometry(nodes_file='../input_files/grid', lines_file='../input_files/lines', master_slaves_file='../input_files/master_slaves')
    PreprocessData.load_measurement_file = lambda measurement : measurement
    prep_data = PreprocessData.init_from_config(conf_file='../input_files/meas_1/setup_info.txt', meas_file=measurement, chan_dofs_file='../input_files/meas_1/channel_dofs')
    
    #test all functions of PreProcessData
    prep_data.correct_offset()
    prep_data.filter_data(lowpass=12)
    prep_data.decimate_data(2)
    prep_data.psd_welch()
    prep_data.corr_welch(400)
    prep_data.psd_blackman_tukey()
    prep_data.welch(400)
    prep_data.get_s_vals_psd()
    prep_data.compute_correlation_matrices(tau_max=400)
    prep_data.get_fft(svd=True)
    
    # for each OMA method 
    for method, config in list(zip([PLSCF,PRCE,BRSSICovRef,SSIData,SSIDataMC, VarSSIRef],
                              ['../input_files/meas_1/plscf_config.txt',
                               '../input_files/meas_1/prce_config.txt',
                               '../input_files/meas_1/ssi_config.txt',
                               '../input_files/meas_1/ssi_config.txt',
                               '../input_files/meas_1/ssi_config.txt',
                               '../input_files/meas_1/varssi_config.txt']))[:]:
        modal_obj=method.init_from_config(config,prep_data)
        # run the full analysis from the sample config file
        
        # run the full analysis manually
        # test save and load functions
        modal_obj.save_state(tmpdir+'test.npz')
        modal_obj = method.load_state(tmpdir+'test.npz', prep_data)
        # pass the loaded object to StabilDiagram and PlotMSH
        
    # test all the functionality of StabilCalc
    # test the functionality of ModeShapePlot
    # test MergePoser
    

if __name__ =='__main__':
    analysis_chain(tmpdir='/dev/shm/womo1998/')

