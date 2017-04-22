'''
Created on Apr 20, 2017

@author: womo1998

Multi-Setup Merging PoSER

for each setup provide:

prep_data -> PreprocessData: chan_dofs, ref_channels, roving_channels
modal_data -> modal_frequencies, modal_damping, mode_shapes
stabil_data -> select_modes

changed/new variables:
    - chan_dofs
    - modal_frequencies
    - modal_damping
    - std_frequencies
    - std_damping
    - mode_shapes
    - select_modes -> actually a dummy
    - ref_channels, roving_channels

in PoGer/PreGer merging
modal_data ->  modal_frequencies, modal_damping, mode_shapes, chan_dofs, ref_channels, roving_channels
stabil_data -> select_modes

PlotMSH (or other postprocessing routines) have to distinguish these three cases:
single-setup (prep_data, modal_data, stabil_data)
poger/preger multi-setup (modal_data, stabil_data)
poser multi-setup (merged_data)
'''

import numpy as np
import datetime
from PreprocessingTools import PreprocessData
from SSICovRef import BRSSICovRef
from PLSCF import PLSCF
from PRCE import PRCE
from SSIData import SSIData, SSIDataMEC
from VarSSIRef import VarSSIRef
from StabilDiagram import StabilCalc


class MergePoSER(object):
    '''
    classdocs
    '''

    def __init__(self,):
        '''
        Constructor
        '''
        self.setups = []
        
        self.merged_chan_dofs = []
        self.merged_num_channels = None
        
        self.merged_ref_channels = None
        self.merged_roving_channels = None
        
        self.mean_frequencies = None
        self.mean_damping = None
        self.merged_mode_shapes = None
        
        self.std_frequencies = None
        self.std_damping = None
        
        self.setup_name = ''
        self.start_time = datetime.datetime.now()
        
    
    def add_setup(self, prep_data, modal_data, stabil_data, override_ref_channels = None):
        assert isinstance(prep_data, PreprocessData)
        assert isinstance(modal_data, (BRSSICovRef, PLSCF, PRCE, SSIData, SSIDataMEC))
        assert isinstance(stabil_data, StabilCalc)
        
        # assure objects belong to the same setup
        assert prep_data.setup_name == modal_data.setup_name
        assert modal_data.setup_name == stabil_data.setup_name
        
        # assure chan_dofs were assigned
        assert prep_data.chan_dofs
        
        # assure modes were selected
        assert stabil_data.select_modes
        
        # assure all setups were analyzed with the same method
        #if self.setups:
        #    if type(self.setups[0]) != type(modal_data):
        #        print(type(self.setups[0]), type(modal_data))
        #        raise RuntimeWarning('All setups should be analyzed with the same method to assure consistent results!')
        
        # extract needed information and store them in a dictionary
        self.setups.append({'setup_name': prep_data.setup_name,
                            'chan_dofs': prep_data.chan_dofs,
                            'num_channels': prep_data.num_analised_channels,
                            'ref_channels': prep_data.ref_channels,
                            'roving_channels': prep_data.roving_channels,
                            'modal_frequencies': [modal_data.modal_frequencies[index] for index in stabil_data.select_modes],
                            'modal_damping': [modal_data.modal_damping[index] for index in stabil_data.select_modes],
                            'mode_shapes': [modal_data.mode_shapes[:,index[1],index[0]] for index in stabil_data.select_modes]
                            })
        print('Added setup "{}" with {} channels and {} selected modes.'.format(prep_data.setup_name, prep_data.num_analised_channels, len(stabil_data.select_modes)))
        
        
        
    def merge(self, base_setup_num = 0, ):
        # generate new_chan_dofs
        # assign modes from each setup
        # for each mode:
        #     for each setup:
        #         rescale
        #         merge
        
        # TODO: rescale w.r.t to the average solution from all setups rather than specifying a base setup
        # compute scaling factors for each setup with each setup and average them for each setup before rescaling
        # corresponding standard deviations can be used to asses the quality of fit
        
        def pair_modes(frequencies_1, frequencies_2):
            delta_matrix=np.ma.array(np.zeros((len(frequencies_1),len(frequencies_2))))
            for index,frequency in enumerate(frequencies_1):
                delta_matrix[index,:]=np.abs((frequencies_2-frequency)/frequency)
            mode_pairs=[]
            while True:
                row, col = np.unravel_index(np.argmin(delta_matrix), delta_matrix.shape)
                for col_ind in range(delta_matrix.shape[1]):
                    if col_ind == col: continue
                    if np.argmin(delta_matrix[:,col_ind])==row:
                        del_col = False
                else: del_col = True
                for row_ind in range(delta_matrix.shape[0]):
                    if row_ind == row: continue
                    if np.argmin(delta_matrix[row_ind,:])== col:
                        del_row = False
                else: del_row = True
                
                if del_col and del_row:
                    delta_matrix[row,:]=np.ma.masked
                    delta_matrix[:,col]=np.ma.masked
                    mode_pairs.append((row,col))
                if len(mode_pairs) == len(frequencies_1):
                    break
                if len(mode_pairs) == len(frequencies_2):
                    break
            return mode_pairs

        setups = self.setups
        
        # get values from base instance   

        chan_dofs_base = setups[base_setup_num]['chan_dofs']
        num_channels_base = setups[base_setup_num]['num_channels']
        mode_shapes_base = setups[base_setup_num]['mode_shapes']
        frequencies_base=setups[base_setup_num]['modal_frequencies']
        damping_base=setups[base_setup_num]['modal_damping']
        
        del setups[base_setup_num]
        # pair channels and modes of each instance with base instance
        
        channel_pairing = []
        mode_pairing = []
        total_dofs = 0
        total_dofs += num_channels_base    
        for setup in setups:
            # calculate the common reference dofs, which may be different channels
            # furthermore reference channels for covariances need not be the reference channels for mode merging
            # channel dof assignments have to be present in each of the instances
            
            chan_dofs_this = setup['chan_dofs']
            num_channels_this = setup['num_channels']
            
            these_pairs=[]
            for chan_dof_base in chan_dofs_base:
                chan_base, node_base, az_base, elev_base = chan_dof_base[0:4]
                for chan_dof_this in chan_dofs_this:
                    chan_this, node_this, az_this, elev_this = chan_dof_this[0:4]
                    if node_this == node_base and az_this == az_base and elev_this == elev_base:
                        these_pairs.append((chan_base, chan_this))
                        
            channel_pairing.append(these_pairs)
            
            total_dofs += num_channels_this-len(these_pairs)
            
            # calculate the mode pairing by minimal frequency difference
            # check that number of modes is equal in all instances (not necessarily)
            # assert len(self.selected_modes_indices) == len(instance.selected_modes_indices)
                
            frequencies_this=setup['modal_frequencies']
                
            mode_pairs = pair_modes(frequencies_base, frequencies_this)
            mode_pairing.append(mode_pairs)
        
        # delete modes not common to all instance from mode pairing
        for mode_num in range(len(frequencies_base)-1,-1,-1):
            in_all = True
            for mode_pairs in mode_pairing:
                for mode_pair in mode_pairs:
                    if mode_pair[0]==mode_num:
                        break
                else:
                    in_all = False
                    break
            if in_all:
                continue
            for mode_pairs in mode_pairing:
                while True:
                    for index, mode_pair in enumerate(mode_pairs):
                        if mode_pair[0]==mode_num:
                            del mode_pairs[index]
                            break
                    else:
                        break

        lengths=[len(mode_pairs) for mode_pairs in mode_pairing]
            
        common_modes = min(lengths)  
        
        new_mode_nums = [mode_num[0] for mode_num in mode_pairing[0]]
        
        # allocate output objects
        mode_shapes = np.zeros((total_dofs, 1,common_modes),dtype=complex)
        f_list=np.zeros((len(setups)+1,common_modes))
        d_list=np.zeros((len(setups)+1,common_modes))
        scale_factors = np.zeros((len(setups),common_modes), dtype=complex)
        
        start_dof=0
        
        # copy modal values from base instance first
        #for mode_num_base,mode_num_this in mode_pairing[0]:  
        for mode_num_base in range(common_modes):
            
            mode_base = mode_shapes_base[mode_num_base]
                                   
            mode_shapes[start_dof:start_dof+num_channels_base,0, mode_num_base, ] = mode_base
            f_list[0, mode_num_base] = frequencies_base[mode_num_base]
            d_list[0, mode_num_base] = damping_base[mode_num_base]
            
        start_dof += num_channels_base
        
        # iterate over instances and assemble output objects (mode_shapes, chan_dofs)
        for setup_num,setup in enumerate(setups):
            

            chan_dofs_this = setup['chan_dofs']
            num_channels_this = setup['num_channels']
            mode_shapes_this = setup['mode_shapes']
                
            these_pairs = channel_pairing[setup_num]            
            num_ref_channels = len(these_pairs)
            num_remain_channels = num_channels_this-num_ref_channels
            ref_channels_base=[pair[0] for pair in these_pairs]
            ref_channels_this=[pair[1] for pair in these_pairs]
            print('Next Instance', ref_channels_base, ref_channels_this)
            
            # create 0,1 matrices to extract and reorder channels from base instance and this instance
            split_mat_refs_base=np.zeros((num_ref_channels, num_channels_base))
            split_mat_refs_this=np.zeros((num_ref_channels, num_channels_this))
            split_mat_rovs_this=np.zeros((num_remain_channels, num_channels_this))

            row_ref=0
            for channel in range(num_channels_base):
                if channel in ref_channels_base:
                    split_mat_refs_base[row_ref,channel]=1
                    row_ref+=1   
                    
            row_ref=0
            row_rov=0
            #print(instance)
            for channel in range(num_channels_this):
                if channel in ref_channels_this:       
                    split_mat_refs_this[row_ref,channel]=1
                    
                    row_ref+=1
                else:
                    split_mat_rovs_this[row_rov,channel]=1    
                    for chan_dof_this in chan_dofs_this:
                        chan, node, az, elev = chan_dof_this[0:4]
                        if chan==channel:             
                            chan = int(start_dof + row_rov)
                            chan_dofs_base.append([chan,node,az,elev])                    
                            row_rov+=1
           
            # loop over modes and rescale them and merge with the other instances
            for mode_num_base, mode_num_this in mode_pairing[setup_num]:
                mode_index = new_mode_nums.index(mode_num_base)
                
                #order_base, index_base = modes_indices_base[mode_num_base]
                
                mode_base = mode_shapes_base[mode_num_base]
                     
                mode_refs_base = np.dot(split_mat_refs_base, mode_base)
                
                #order_this, index_this = modes_indices_this[mode_num_this]
                
                mode_this = mode_shapes_this[mode_num_this]
                 
                mode_refs_this = np.dot(split_mat_refs_this, mode_this)
                mode_rovs_this = np.dot(split_mat_rovs_this, mode_this)
                
                numer = np.dot(np.transpose(np.conjugate(mode_refs_this)), mode_refs_base)
                denom = np.dot(np.transpose(np.conjugate(mode_refs_this)), mode_refs_this)
                
                scale_fact=numer/denom
                scale_factors[setup_num,mode_index]=(scale_fact)
                mode_shapes[start_dof:start_dof+num_remain_channels,0, mode_index] = scale_fact*mode_rovs_this
                    
                f_list[setup_num+1, mode_index]=setup['modal_frequencies'][mode_num_this]
                d_list[setup_num+1, mode_index]=setup['modal_damping'][mode_num_this]
                
            start_dof += num_remain_channels
            


        
        mean_frequencies = np.zeros((common_modes,))   
        std_frequencies = np.zeros((common_modes,))        
        mean_damping = np.zeros((common_modes,))
        std_damping = np.zeros((common_modes,))
        
        for mode_num_base, mode_num_this in mode_pairing[0]:
            mode_index = new_mode_nums.index(mode_num_base)
            #order_base, index_base = modes_indices_base[mode_num_base]
            
        #for mode_num,(order_base, index_base) in enumerate(modes_indices_base):
            
            # rescaling of mode shape 
            mode_tmp = mode_shapes[:, 0,mode_index]  
            abs_mode_tmp = np.abs(mode_tmp)
            index_max = np.argmax(abs_mode_tmp)
            this_max = mode_tmp[index_max]
            mode_tmp = mode_tmp / this_max      
            #mpcs[0,index] = StabilCalc.calculateMPC(mode_tmp)
            #mpds[0,index], mps[0,index] = StabilCalc.calculateMPD(mode_tmp)
            mode_shapes[:, 0,mode_index] = mode_tmp  
            mean_frequencies[mode_index,] = np.mean(f_list[:,mode_index],axis=0)
            std_frequencies[mode_index,] = np.std(f_list[:,mode_index],axis=0)

            mean_damping[mode_index,] = np.mean(d_list[:,mode_index], axis=0)
            std_damping[mode_index,] = np.std(d_list[:,mode_index], axis=0)
            
        self.merged_chan_dofs = chan_dofs_base
        self.merged_num_channels = total_dofs
        
        self.merged_mode_shapes = mode_shapes
        self.mean_frequencies = np.expand_dims(mean_frequencies, axis=1)
        self.std_frequencies = np.expand_dims(std_frequencies, axis=1)
        self.mean_damping = np.expand_dims(mean_damping, axis=1)
        self.std_damping = np.expand_dims(std_damping, axis=1)
        
        #self.select_modes = list(range(common_modes))
    
    def save_state(self):
        pass
    
    @staticmethod
    def load_state():
        pass
    
    def export_results(self, fname):
        pass

def main():
    from PreprocessingTools import PreprocessData, GeometryProcessor
    from SSICovRef import BRSSICovRef
    from StabilDiagram import StabilCalc
    from PlotMSH import ModeShapePlot, start_msh_gui
    
    working_dir = '/home/womo1998/Projects/2017_modal_merging_test_files/'  
    interactive = False
    
    merger = MergePoSER()
    geometry_data = GeometryProcessor.load_geometry(nodes_file=working_dir+'macec/grid_full.asc',
                                                    lines_file=working_dir+'macec/beam_full.asc')
    
    setups = ['meas_1', 'meas_2']
    for setup in setups:
        result_folder = working_dir + setup
        
        prep_data =PreprocessData.load_state(result_folder+'prep_data.npz')
        modal_data = BRSSICovRef.load_state(result_folder+'modal_data.npz')
        stabil_data = StabilCalc.load_state(result_folder+'stabi_data.npz')
    
        merger.add_setup(prep_data, modal_data, stabil_data)
    
    merger.merge()
    
    if interactive:
        mode_shape_plot= ModeShapePlot(merger, geometry_data)
        start_msh_gui(mode_shape_plot)
        
    

if __name__ == '__main__':
    main()