# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''

import numpy as np
import sys
import os
import json
#comme nt123
import multiprocessing as mp
import ctypes as c
from collections import OrderedDict, deque
from copy import deepcopy

from PreprocessingTools import *
from StabilDiagram import main_stabil, StabilPlot, nearly_equal

'''
TODO:
- change channels numbers such, that user input channels start at 1 while internally they start at 0
    affects: ref_channels, roving_channels and channel-dof-assignments
- generally define unit tests to check functionality after changes
- extensive testing of merging and orthogonalize methods

'''
def dummy_object():
    measurement = np.zeros((2,1))
    sampling_rate = 1.0
    return SSICovRef(measurement, sampling_rate)
    
class SSICovRef(object):
    
    def __init__(self,measurement, sampling_rate, total_time_steps=None, num_channels=None,
                ref_channels=None, roving_channels=None, quantities=None):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        
        #             0         1           2           3          4            5            6             7
        #self.state= [Toeplitz, State Mat., Modal Par., Select M., Merge PoSER, Merge PoGer, Merge Preger, Ortho]
        self.state  =[False,    False,      False,      False,     False,       False,       False,        False ]
        
        assert isinstance(measurement, np.ndarray)
        assert measurement.shape[0] > measurement.shape[1]
        self.measurement = measurement
        
        assert isinstance(sampling_rate, (int,float))
        self.sampling_rate = sampling_rate
        
        if total_time_steps is None:
            total_time_steps = measurement.shape[0]
        assert  measurement.shape[0] >= total_time_steps
        self.total_time_steps = total_time_steps
        
        if ref_channels is None:
            ref_channels = list(range(measurement.shape[1]))
        self.ref_channels = ref_channels
        if roving_channels is None:
            roving_channels = []
        self.roving_channels = roving_channels
        
        self.num_ref_channels = len(self.ref_channels)
        self.num_roving_channels = len(self.roving_channels)
        self.num_analised_channels = self.num_ref_channels + self.num_roving_channels
        if num_channels is None:
            num_channels = self.num_analised_channels
            
        self.merged_num_channels = None
        self.merged_num_channels_multiref = None
        self.ortho_num_channels = None
        
        if ((self.num_ref_channels + self.num_roving_channels) > num_channels):
                sys.exit('The sum of reference and roving channels is greater than the number of all channels!')
                
        for ref_channel in self.ref_channels:
            if (ref_channel < 0):
                sys.exit('A reference channel number cannot be negative!')
            if (ref_channel > (num_channels - 1)):
                sys.exit('A reference channel number cannot be greater than the number of all channels!')
            for rov_channel in self.roving_channels:
                if (rov_channel < 0):
                    sys.exit('A roving channel number cannot be negative!')
                if (rov_channel > (num_channels - 1)):
                    sys.exit('A roving channel number cannot be greater than the number of all channels!')
                if (ref_channel == rov_channel):
                    sys.exit('Any channel can be either a reference OR a roving channel. Check your definitions!')
                    
        #sort channels       
        #self.chan_switch_mat=np.zeros((self.num_analised_channels,self.num_analised_channels))
        #for col, channel in enumerate(self.ref_channels+self.roving_channels):
        #    self.chan_switch_mat[channel,col]=1
            
        #self.measurement = np.dot(self.measurement,self.chan_switch_mat)
        self.num_block_columns = None
        self.num_block_rows = None
        self.toeplitz_matrix = None
        self.merged_toeplitz_matrix = None
        
        self.max_model_order = None
        self.state_matrix = None
        self.output_matrix = None
        
        self.modal_damping = None
        self.modal_frequencies = None
        self.mean_f = None
        self.std_f = None
        self.mean_d = None
        self.std_d = None
        self.mode_shapes = None
        self.merged_mode_shapes = None
        self.ortho_mode_shapes = None
        
        if quantities is None:
            quantities = ['a' for i in range(self.num_analised_channels)]
        for quantity in quantities:
            assert quantity in ['a','v','d']      
        assert len(quantities) == self.num_analised_channels
        
        self.quantities=quantities    
        self.merged_quantities = None
        
        self.selected_modes_indices = []
        self.merged_modes_indices = []
        self.ortho_modes_indices = []
        
        self.chan_dofs=[]        
        self.merged_chan_dofs = None 
        self.ortho_chan_dofs = None
        
        self.rescale_ref_channels = None
        self.rescale_rov_channels = None
        
    def add_chan_dof_assignment(self, chan_dof_assignment):
        '''
             chan_dofs[i] = (chan, node, az, elev)
                          = (int, int, float, float)
        azimuth angle starting from x axis towards y axis
        elevation defined from x-y plane up
        x: 0.0, 0.0
        y: 90.0, 0.0
        z: 0.0, 90.0
        '''
        if self.state[4] or self.state[5] or self.state[6]:
            existing_chan_dofs = self.merged_chan_dofs
        elif self.state[7]:
            existing_chan_dofs = self.ortho_chan_dofs
        else:
            existing_chan_dofs = self.chan_dofs
        
        for assignment in chan_dof_assignment:
            assert len(assignment)==4
            chan, node, az, elev = assignment
            assert chan in self.ref_channels or chan in self.roving_channels
            for index,existing in enumerate(existing_chan_dofs):
                if chan == existing[0]:
                    existing_chan_dofs[index]=(int(chan),int(node),float(az),float(elev))
                    break
            else:
                existing_chan_dofs.append((int(chan),int(node),float(az),float(elev)))
        
        if self.state[4] or self.state[5] or self.state[6]:
            self.merged_chan_dofs = existing_chan_dofs
        elif self.state[7]:
            self.ortho_chan_dofs = existing_chan_dofs
        else:
            self.chan_dofs = existing_chan_dofs
            
    def clear_chan_dof_assignments(self):
        
        self.merged_chan_dofs = []
        
        self.ortho_chan_dofs = []
        
        self.chan_dofs = []
        
    def save_chan_dof_assignment(self, result_folder, which = None):
        
        if which is not None:
            assert which in ['merged', 'ortho'] 
            
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            
        with open(result_folder + 'chan_dofs.txt', 'wt') as file:
            if which == 'merged':
                json.dump(self.merged_chan_dofs, file)
            elif which == 'ortho':
                json.dump(self.ortho_chan_dofs, file)
            else:
                json.dump(self.chan_dofs, file)

    def build_toeplitz_cov(self, num_block_columns, num_block_rows, multiprocess=True):
        '''
        Builds a Block-Toeplitz Matrix of Covariances with varying time lags
        
            |    R_i    R_i-1    ...    R_0    |
            |    R_i+1  R_i      ...    R_1    |
            |    ...    ...      ...    ...    |
            |    R_2i-1 ...      ...    R_i    |
        '''
        assert isinstance(num_block_columns, int)
        assert isinstance(num_block_rows, int)
        
        self.num_block_columns=num_block_columns
        self.num_block_rows=num_block_rows
        
        # Extract reference time series for covariances 
        extract_length = self.total_time_steps - (self.num_block_columns + self.num_block_rows) + 1
        
        ref_channels = sorted(self.ref_channels)
        all_channels = self.ref_channels + self.roving_channels
        all_channels.sort()   
        
        refs = (self.measurement[0:extract_length,ref_channels]).T 
        
        ### Create Toeplitz matrix and fill it with covariances 
        #    |    R_i    R_i-1    ...    R_0    |                                     #
        #    |    R_i+1  R_i      ...    R_0    |                                     #
        #    |    ...    ...      ...    ...    |                                     #
        #    |    R_2i-1 ...      ...    R_i    |                                     #
        
        print('Computing covariances...')
        n, m = self.num_analised_channels*self.num_block_rows, self.num_ref_channels*self.num_block_columns
 
        if multiprocess:
            toeplitz_memory = mp.Array(c.c_double, np.zeros(n*m)) # shared memory, can be used by multiple processes
            toeplitz_shape = (n,m)
            measurement_shape=self.measurement.shape
            measurement_memory = mp.Array(c.c_double, self.measurement.reshape(self.measurement.size, 1))
            
            refs_memory = mp.Array(c.c_double, refs.reshape(refs.size,1 ))
            refs_shape=refs.shape
            
            pool=mp.Pool(initializer=self.init_child_process, initargs=(refs_memory,  measurement_memory, toeplitz_memory, ))
        
            
            for i in range(0,self.num_block_rows):
                if i == 0:
                    # in multiprocessing, errors in the single processes are sometimes not reported to console, in this case 
                    # the resulting block toeplitzmatrix will be filled with zeros
                    # try to change pool.apply_async to pool.apply and run in debug mode
                    for ii in range(0,self.num_block_columns):#fill the first block row
                        pool.apply_async(self.compute_covariance , args=(i, 
                                                            ii, 
                                                            self.num_block_columns, 
                                                            extract_length, 
                                                            ref_channels, 
                                                            all_channels, 
                                                            refs_shape, 
                                                            measurement_shape,
                                                            toeplitz_shape))
                else: #fill the first block column
                    ii = 0
                    pool.apply_async(self.compute_covariance , args=(i, 
                                                            ii, 
                                                            self.num_block_columns, 
                                                            extract_length, 
                                                            ref_channels, 
                                                            all_channels, 
                                                            refs_shape, 
                                                            measurement_shape,
                                                            toeplitz_shape))
                        
            pool.close()
            pool.join()               
            
            del measurement_memory
            del refs_memory
            del refs  

      
            Toeplitz_matrix = np.frombuffer(toeplitz_memory.get_obj()).reshape((n,m)) 

            
            for i in range(1,self.num_block_rows): #finish assembling block toeplitz matrix
                # copys and shifts contents from previous block row to next block row
                # shifts by one block column to the right
                previous_Toeplitz_row = (i-1)*self.num_analised_channels
                this_block = Toeplitz_matrix[previous_Toeplitz_row:(previous_Toeplitz_row+self.num_analised_channels),
                                             0:(self.num_ref_channels * self.num_block_columns - self.num_ref_channels)]
                begin_Toeplitz_row = i*self.num_analised_channels
                Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+self.num_analised_channels),
                                self.num_ref_channels:(self.num_ref_channels * self.num_block_columns)] = this_block
    
        else: # old single threaded way
            
            Toeplitz_matrix = np.zeros((n,m))

            for i in range(0,self.num_block_rows):
                if i == 0:
                    for ii in range(0,self.num_block_columns):
                        
                        begin_extract = self.num_block_columns + i - (ii)
                        current_signals = self.measurement[begin_extract : (begin_extract+extract_length), all_channels].T
                        covariances = np.cov(refs,current_signals)
                        this_block = covariances[self.num_ref_channels:(self.num_ref_channels + self.num_analised_channels),:self.num_ref_channels]
                        
                        begin_Toeplitz_row = i*self.num_analised_channels
                        
                        Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+self.num_analised_channels),
                                        ii*self.num_ref_channels:(ii*self.num_ref_channels+self.num_ref_channels)] = this_block
                else: 
                    previous_Toeplitz_row = (i-1)*self.num_analised_channels
                    this_block = Toeplitz_matrix[previous_Toeplitz_row:(previous_Toeplitz_row+self.num_analised_channels),
                                                  0:self.num_ref_channels * (self.num_block_columns-1)]
                    begin_Toeplitz_row = i*self.num_analised_channels
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+self.num_analised_channels),
                                     self.num_ref_channels:(self.num_ref_channels * self.num_block_columns)] = this_block
                     
                    begin_extract = self.num_block_columns + i
                    current_signals = (self.measurement[begin_extract:(begin_extract + extract_length),all_channels]).T
                    covariances = np.cov(refs,current_signals)
                    this_block = covariances[self.num_ref_channels:(self.num_ref_channels + self.num_analised_channels),:self.num_ref_channels]
                     
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+self.num_analised_channels),
                                     0:self.num_ref_channels] = this_block
        self.toeplitz_matrix = Toeplitz_matrix              
        self.state[0]=True
        
    def init_child_process(self, refs_memory_, measurement_memory_, toeplitz_memory_):
        #make the  memory arrays available to the child processes
        global refs_memory
        refs_memory = refs_memory_
        
        global measurement_memory
        measurement_memory = measurement_memory_   
        
        global toeplitz_memory
        toeplitz_memory = toeplitz_memory_
        
    def compute_covariance(self, i,ii, num_block_columns, extract_length, ref_channels, all_channels, refs_shape, measurement_shape, toeplitz_shape):
        
        num_analised_channels = len(all_channels)
        num_ref_channels = len(ref_channels)
        
        measurement = np.frombuffer(measurement_memory.get_obj()).reshape(measurement_shape)
        refs = np.frombuffer(refs_memory.get_obj()).reshape(refs_shape)
        toeplitz = np.frombuffer(toeplitz_memory.get_obj()).reshape(toeplitz_shape)
            
        begin_extract = num_block_columns + i - (ii)

        current_signals = (measurement[begin_extract:(begin_extract + extract_length), all_channels]).T

        #covariances = np.cov(refs,current_signals)
        #this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
        this_block = (np.dot(refs, current_signals.T.conj()) / refs_shape[1]).T
        
        
        begin_toeplitz_row = i*num_analised_channels 
        end_toeplitz_row = begin_toeplitz_row+num_analised_channels 
        begin_toeplitz_col = ii*num_ref_channels
        end_toeplitz_col = begin_toeplitz_col + num_ref_channels
        
        with toeplitz_memory.get_lock():
    
            toeplitz[begin_toeplitz_row: end_toeplitz_row,
                            begin_toeplitz_col:end_toeplitz_col] = this_block
   
    def extract_part_toeplitz_cov(self, row_channels=None, col_channels=None, recompute = False, auto_only = False):
        '''
        returns a reduced block-toeplitz-matrix of covariances between 
        channels (block-rows) and the previously used 
        reference channels (self.ref_channels, block-columns)
        
        extract and reorder channels block-row-wise
        '''
        all_channels = self.ref_channels + self.roving_channels
        all_channels.sort()
        ref_channels = self.ref_channels
        ref_channels.sort()
        
        if auto_only:
            row_channels = col_channels
            
        row_indices = []        
        if row_channels is not None:
            for channel in row_channels:
                if channel in all_channels:
                    row_indices.append(all_channels.index(channel))
                else:
                    recompute = True
        else:
            row_channels = all_channels
            row_indices = list(range(len(all_channels)))
            
        col_indices = []
        if col_channels is not None:
            for channel in col_channels:
                if channel in ref_channels:
                    col_indices.append(ref_channels.index(channel))
                else:
                    recompute = True
        else:
            col_channels = ref_channels
            col_indices = list(range(len(ref_channels)))
        
        col_channels.sort()
        row_channels.sort()
        
        num_ref_channels = len(col_channels)
        num_rov_channels = len(row_channels)  
              
        if recompute:
            
            num_analised_channels = num_rov_channels
            
            extract_length = self.total_time_steps - (self.num_block_columns + self.num_block_rows) + 1
    
            refs = (self.measurement[0:extract_length,col_channels]).T 

            n, m = num_analised_channels * self.num_block_rows, num_ref_channels * self.num_block_columns

            toeplitz_memory = mp.Array(c.c_double, np.zeros(n * m)) # shared memory, can be used by multiple processes
            toeplitz_shape = (n, m)
            measurement_shape=self.measurement.shape
            measurement_memory = mp.Array(c.c_double, self.measurement.reshape(self.measurement.size, 1))
            
            refs_memory = mp.Array(c.c_double, refs.reshape(refs.size,1 ))
            refs_shape=refs.shape
            
            pool=mp.Pool(initializer=self.init_child_process, initargs=(refs_memory,  measurement_memory, toeplitz_memory, ))
        
            for i in range(0,self.num_block_rows):
                if i == 0:
                    
                    for ii in range(0,self.num_block_columns):#fill the first block row
                        pool.apply_async(self.compute_covariance , args=(i, 
                                                            ii, 
                                                            self.num_block_columns, 
                                                            extract_length, 
                                                            col_channels, 
                                                            row_channels, 
                                                            refs_shape, 
                                                            measurement_shape,
                                                            toeplitz_shape))
                else: #fill the first block column
                    ii = 0
                    pool.apply_async(self.compute_covariance , args=(i, 
                                                            ii, 
                                                            self.num_block_columns, 
                                                            extract_length, 
                                                            col_channels,
                                                            row_channels, 
                                                            refs_shape, 
                                                            measurement_shape,
                                                            toeplitz_shape))
                            
            pool.close()
            pool.join() 

            del measurement_memory
            del refs_memory
            del refs 
            
            red_toeplitz = np.frombuffer(toeplitz_memory.get_obj()).reshape((n,m)) 
            
            if auto_only:
                
                for i in range(0,self.num_block_rows):
                    if i == 0:                    
                        for ii in range(0,self.num_block_columns):
                            this_block = red_toeplitz[num_ref_channels * i:(num_ref_channels + 1) * i, (num_ref_channels) * ii:(num_ref_channels + 1) * ii]
                            diag = list(range(num_ref_channels))
                            red_toeplitz[num_ref_channels * i:(num_ref_channels + 1) * i, (num_ref_channels) * ii:(num_ref_channels + 1) * ii] =  np.diag(this_block[diag,diag])
                    ii = 0
                    this_block = red_toeplitz[num_ref_channels * i:(num_ref_channels + 1) * i, (num_ref_channels) * ii:(num_ref_channels + 1) * ii]
                    diag = list(range(num_ref_channels))
                    red_toeplitz[num_ref_channels * i:(num_ref_channels + 1) * i, (num_ref_channels) * ii:(num_ref_channels + 1) * ii] =  np.diag(this_block[diag,diag])
            for i in range(1,self.num_block_rows): #finish assembling block toeplitz matrix
                # copys and shifts contents from previous block row to next block row
                # shifts by one block column to the right
                
                previous_Toeplitz_row = (i-1) * num_analised_channels
                this_block = red_toeplitz[previous_Toeplitz_row:previous_Toeplitz_row + num_analised_channels,
                                             0:num_ref_channels * (self.num_block_columns - 1)]
                begin_Toeplitz_row = i * num_analised_channels
                red_toeplitz[begin_Toeplitz_row:begin_Toeplitz_row + num_analised_channels,
                                num_ref_channels:num_ref_channels * self.num_block_columns] = this_block
            
        else:
            
            row_indices = np.array(sorted(row_indices))
            col_indices = np.array(sorted(col_indices))  
            rows = [row_indices + (self.num_analised_channels * block_row) for block_row in range(self.num_block_rows)]
            rows = np.array(rows).flatten()
            
            cols = [col_indices + (self.num_ref_channels * block_column) for block_column in range(self.num_block_columns)]
            cols = np.array(cols).flatten()
            #print(rows, cols)
            red_toeplitz = self.toeplitz_matrix[rows,:][:,cols]
            if auto_only:                
                for i in range(0,self.num_block_rows):
                    if i == 0:                    
                        for ii in range(0,self.num_block_columns):
                            this_block = red_toeplitz[num_ref_channels * i:num_ref_channels  * (i + 1), num_ref_channels * ii:num_ref_channels  * (ii + 1)]
                            diag = list(range(num_ref_channels))
                            #print(diag, this_block.shape)
                            red_toeplitz[num_ref_channels * i:num_ref_channels  * (i + 1), num_ref_channels * ii:num_ref_channels  * (ii + 1)] =  np.diag(this_block[diag,diag])
                    ii = 0
                    this_block = red_toeplitz[num_ref_channels * i:num_ref_channels  * (i + 1), num_ref_channels * ii:num_ref_channels  * (ii + 1)]
                    diag = list(range(num_ref_channels))
                    red_toeplitz[num_ref_channels * i:num_ref_channels  * (i + 1), num_ref_channels * ii:num_ref_channels  * (ii + 1)] =  np.diag(this_block[diag,diag])

        return red_toeplitz
       
    def compute_state_matrices(self, max_model_order):
        '''
        computes the state and output matrix of the state-space-model
        by applying a singular value decomposition to the block-toeplitz-matrix of covariances
        the state space model matrices are obtained by appropriate truncation 
        of the svd matrices at max_model_order
        the decision whether to take merged covariances is taken automatically
        '''
        
        assert isinstance(max_model_order, int)
        self.max_model_order=max_model_order
        
        assert self.state[0]
        
        if self.state[6]:
            toeplitz_matrix = self.merged_toeplitz_matrix
            num_channels = self.merged_num_channels
        elif self.state[5]:
            toeplitz_matrix = self.merged_toeplitz_matrix
            num_channels = self.merged_num_channels_multiref
        else:
            toeplitz_matrix = self.toeplitz_matrix
            num_channels = self.num_analised_channels
        
        print('Computing state matrices...')
        
        [U,S,V_T] = np.linalg.svd(toeplitz_matrix,0)
        S_2 = np.diag(np.power(S[:self.max_model_order], 0.5))
        U = U[:,:self.max_model_order]
        Oi = np.dot(U, S_2)
        C = Oi[:num_channels,:]   
        
        A = np.dot(np.linalg.pinv(Oi[:(num_channels * (self.num_block_columns - 1)),:]),
                   Oi[num_channels:(num_channels * self.num_block_columns),:])
       
        self.state_matrix = A
        self.output_matrix = C
        self.state[1]=True
        self.state[2] = False # previous modal params are invalid now
        self.state[3] = False # previous selection are invalid now
        self.state[7] = False # previous orthogonalizations are invalid now
    def compute_modal_params(self, multiprocessing=True): 
        
        assert self.state[1]
        
        print('Computing modal parameters...')
        
        self.modal_frequencies = np.zeros((self.max_model_order, self.max_model_order))        
        self.modal_damping = np.zeros((self.max_model_order, self.max_model_order))
        if self.state[5] or self.state[6]: # poger or preger
            mode_shapes = np.zeros((self.merged_num_channels, self.max_model_order, self.max_model_order), dtype = complex)
            if self.state[5]:
                self.rescale_ref_channels = [np.array(ref_channel_pairs) for ref_channel_pairs in self.rescale_ref_channels]
                self.rescale_rov_channels = [np.array(rov_channel_pairs) for rov_channel_pairs in self.rescale_rov_channels]
        else:
            mode_shapes = np.zeros((self.num_analised_channels, self.max_model_order, self.max_model_order),dtype=complex)
            
        if multiprocessing:
            manager=mp.Manager()        
            pool = mp.Pool()       
            return_dict=manager.dict()
            
            #balanced allocation of work (numpy.linalg.eig takes approx. n^3 operations)
            work_slice_size = sum([n**3 for n in range(self.max_model_order)])/self.max_model_order
            current_size = 0
            current_orders = []
            for order in range(0,self.max_model_order,1):
                current_orders.append(order)
                current_size += order**3
                if current_size >= work_slice_size:
                    pool.apply_async(self.multiprocess_evd , args=(self.state_matrix, current_orders, return_dict))
                    current_orders = []
                    current_size = 0
            pool.close()
            pool.join()

        for order in range(0,self.max_model_order,1):    
            
            if multiprocessing:
                eigenvalues_single, eigenvectors_single = return_dict[order]
            else:
                eigenvalues_paired, eigenvectors_paired = np.linalg.eig(self.state_matrix[0:order+1, 0:order+1])
    
                eigenvectors_single,eigenvalues_single = \
                    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
                
            for index,k in enumerate(eigenvalues_single): 
                lambda_k = np.log(complex(k)) * self.sampling_rate
                freq_j = np.abs(lambda_k) / (2*np.pi)        
                damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100)  
                mode_shapes_j = np.dot(self.output_matrix[:, 0:order + 1], eigenvectors_single[:,index])
               
                if self.state[5] : # poger
                    # rescale mode shapes obtained form merged covariances (PoGer)
                    mode_shapes_j = self.rescale_by_references(mode_shapes_j)
                    # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                    mode_shapes_j = self.integrate_quantities(mode_shapes_j, self.merged_quantities, np.abs(lambda_k))
                elif self.state[6]: #preger
                    # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                    mode_shapes_j = self.integrate_quantities(mode_shapes_j, self.merged_quantities, np.abs(lambda_k))
                else: 
                    # restore channel order
                    #mode_shapes_j = np.dot(self.chan_switch_mat, mode_shapes_j)
                    # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                    mode_shapes_j = self.integrate_quantities(mode_shapes_j, self.quantities, np.abs(lambda_k))                
                        
                self.modal_frequencies[order,index]=freq_j
                self.modal_damping[order,index]=damping_j
                mode_shapes[:,index,order]=mode_shapes_j
                
        if self.state[5] or self.state[6]:
            self.merged_mode_shapes = mode_shapes
        else:
            self.mode_shapes = mode_shapes
            
        self.state[2]=True
        self.state[3] = False # previous selections are invalid now
        self.state[7] = False # previous orthogonalizations are invalid now
        
    def multiprocess_evd(self, a, truncation_orders, return_dict):
        for truncation_order in truncation_orders:
            eigenvalues_paired, eigenvectors_paired = np.linalg.eig(a[0:truncation_order+1, 0:truncation_order+1])
    
            eigenvectors_single,eigenvalues_single = \
                    SSICovRef.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            return_dict[truncation_order] = (eigenvalues_single, eigenvectors_single)
        
        return
                    
    @staticmethod
    def remove_conjugates_new (vectors, values):
        '''
        removes conjugates and marks the vectors which appear in pairs
        
        vectors.shape = [order+1, order+1]
        values.shape = [order+1,1]
        '''
        num_val=vectors.shape[1]
        conj_indices=deque()
        
        for i in range(num_val):
            this_vec=vectors[:,i]
            this_conj_vec = np.conj(this_vec)
            this_val=values[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val: #remove real eigenvalues
                continue
            for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
                if vectors[0,j] == this_conj_vec[0] and \
                   vectors[-1,j] == this_conj_vec[-1] and \
                   values[j] == this_conj_val:
                    # saves computation time this function gets called many times and 
                    #numpy's np.all() function causes a lot of computation time
                    conj_indices.append(i)
                    break
        conj_indices=list(conj_indices)
        vector = vectors[:,conj_indices]
        value = values[conj_indices]

        return vector,value
    
    def rescale_by_references(self, mode_shape):
        '''
        This is PoGer Rescaling
        not implemented yet, only a dummy method
        self.rescale_references 
            = [channel_pairs_1, channel_pairs_2,...]
                 1,2,... num_instance
             channel_pairs = [(ref_chan_base, ref_chan_this),..., (None, rov_chan_this),...]
        channel_pairs have been determined by comparison of channel-dof-assignments
        note: reference channels for SSI need not necessarily be reference channels for rescaling and vice versa
        '''
        
        new_mode_shape = np.zeros((self.merged_num_channels), dtype=complex)
        
        end_row  = 0
        
        start_row = end_row
        end_row += self.num_analised_channels
        new_mode_shape[start_row:end_row] = mode_shape[start_row:end_row]
        
        for ref_channel_pairs, rov_channel_pairs in zip(self.rescale_ref_channels, self.rescale_rov_channels):
            '''
            in self.pair_modes:
            rescale_pairs=[(chan_base, this_total_channel), ...] # ref_channel_pairs
            rov_pairs = [(this_total_channel, chan_this), ...] # rov_channel_pairs
            '''
            #base_refs_ = [pair[0] for pair in ref_channel_pairs]
            #this_refs_ = [pair[1] for pair in ref_channel_pairs]
            #this_rovs_ = [pair[0] for pair in rov_channel_pairs]
            
            #ref_channel_pairs = np.array(ref_channel_pairs)
            base_refs = ref_channel_pairs[:,0]# [pair[0] for pair in ref_channel_pairs]
            this_refs = ref_channel_pairs[:,1]#[pair[1] for pair in ref_channel_pairs]
            this_rovs = rov_channel_pairs[:,0] #[pair[0] for pair in rov_channel_pairs]
            
            #print(this_refs == this_refs_)
            #print(base_refs == base_refs_)
            #print(this_rovs == this_rovs_)
            
            mode_refs_base = mode_shape[base_refs]
            mode_refs_this = mode_shape[this_refs]
            mode_refs_this_conj = mode_refs_this.conj()
            mode_rovs_this = mode_shape[this_rovs]
            
            #numer = np.dot( np.transpose( np.conjugate( mode_refs_this ) ), mode_refs_base )
            #denom = np.dot( np.transpose( np.conjugate( mode_refs_this ) ), mode_refs_this )
            #scale_fact_ = numer/denom
            numer = np.inner(mode_refs_this_conj, mode_refs_base )
            denom = np.inner(mode_refs_this_conj, mode_refs_this )
            scale_fact=numer/denom    
            #print(scale_fact == scale_fact_)   
                
            start_row = end_row
            end_row += this_rovs.shape[0]
            #print(len(this_rovs)==this_rovs.shape[0])
            
            new_mode_shape[start_row:end_row] = scale_fact * mode_rovs_this
            
        return new_mode_shape
    
    @staticmethod
    def integrate_quantities(vector, quantities, omega):
        # input quantities = [a, v, d]
        # output quantities = [d, d, d]
        # converts amplitude and phase
        # 
        
        #mathematical derivation
        '''
        from sympy import *
        init_printing()
        
        # input quantity = v, output quantity = d
        # s_d = int(s_v, dt)
        
        v  = symbols('v')
        f = symbols('f',positive=True)
        t=symbols('t')
        
        # convert to polar form  
        r_v = abs(v)
        phi_v = arg(v)
        
        # create a function for the 'sine of a'
        s_v = r_v*cos(f*t-phi_v)
        
        # integrate 'sine of v' once to obtain the 'sine of d'
        s_d = integrate(s_v, t)
                    
        # extract amplitude and phase (polar form)
        r_d = r_v/f
        phi_d = pi/2 + phi_v
        
        assert s_d == r_d*cos(f*t-phi_d)
        
        # cartesian form of complex displacement
        d = r_d * exp(I*phi_d)
        
        
        # input quantity = a, output quantity = d
        # s_d = int(int(s_a, dt),dt)
        
        a  = symbols('a')
        f = symbols('f',positive=True)
        t=symbols('t')
        
        # convert to polar form  
        r_a = abs(a)
        phi_a = arg(a)
        
        # create a function for the 'sine of a'
        s_a = r_a*cos(f*t-phi_a)
        
        # integrate 'sine of a' once to obtain the 'sine of v'
        s_v =integrate(s_a,t)
                    
        # extract amplitude and phase (polar form)
        r_v = r_a/f
        phi_v =pi/2 + phi_a
        
        assert s_v == r_v*cos(f*t-phi_v)
        
        # integrate 'sine of v' again to obtain the 'sine of d'
        s_d = integrate(s_v, t)
                    
        # extract amplitude and phase (polar form)
        r_d = r_v/f
        phi_d = pi/2 + phi_v
        
        assert s_d == r_d*cos(f*t-phi_d)
        
        # cartesian form of complex displacement
        d = r_d * exp(I*phi_d)
        '''
        
        #reference implementation, slow and less precise!
        '''
        new_vector = np.empty(vector.shape, dtype=complex)
        for index, quantity in enumerate(quantities):
            num=vector[index]
            if quantity == 'd' or quantity is None:
                new_vector[index]=num
                continue
            if quantity == 'a':
                r_a = np.absolute(num)
                phi_a = np.angle(num)
                r_v = r_a/omega
                phi_v = phi_a + np.pi/2
            elif quantity == 'v':
                r_v = np.absolute(num)
                phi_v = np.angle(num) 
            r_d = r_v/omega
            phi_d = phi_v + np.pi/2
            new_vector[index]=r_d * np.exp(1j*phi_d)
        return new_vector
        '''
        quantities = np.array(quantities)   
        a_indices = quantities == 'a'
        v_indices = quantities == 'v'
        #                     phase + 180; magn / omega^2
        vector[a_indices] *= -1       / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[v_indices] *=  1j        / omega
        
        return vector   
    
    
    def select_modes(self, stab_frequency, stab_damping, stab_MAC, result_folder):
        '''
        use a stability diagram to select the modes from the eigenstructure
        '''
        #             0         1           2           3          4     5        6
        #self.state= [Toeplitz, State Mat., Modal Par., Select M., Merge, Rescale, Ortho]
        assert self.state[0]
        assert self.state[1]
        assert self.state[2]
        
        #print(self.state)
        
        if self.state[7]:
            mode_shapes = self.ortho_mode_shapes
            num_channels = self.ortho_num_channels
            mode_indices = self.ortho_modes_indices
        elif self.state[4] or self.state[5] or self.state[6]:
            mode_shapes = self.merged_mode_shapes
            num_channels = self.merged_num_channels
            mode_indices = self.merged_modes_indices
        else:
            mode_shapes = self.mode_shapes
            num_channels = self.num_analised_channels
            mode_indices = self.selected_modes_indices
            
        #print(mode_indices)
        
        selected_indices = main_stabil(self.modal_frequencies, 
                            self.modal_damping,
                            mode_shapes,
                            num_channels, # could be obtained from mode_shapes array's shape
                            self.max_model_order, # could be obtained from mode shapes array's shape
                            stab_frequency, 
                            stab_damping, 
                            stab_MAC,
                            self.measurement,# only used for fft 
                            self.sampling_rate, # only used for fft 
                            result_folder, # only used if results are to be saved 
                            self.num_block_columns, # only used for result output 
                            self.num_block_rows, # only used for result output 
                            mode_indices)
        self.state[3]=True
        
        if self.state[7]:
            self.ortho_modes_indices = selected_indices
        elif self.state[4] or self.state[5] or self.state[6]:# merged
            self.merged_modes_indices = selected_indices
        else:
            self.selected_modes_indices = selected_indices

        
           
    def merge_poser(self, instances, use_ortho = True):
        
        '''
        usage: self.merge_poser([instances of SSICovRef])
        the systemidentification and eigenstructure analysis has to be 
        completed in every instance, as well as the selection of modes 
        in a stability diagram will have to be present in select_modes variable   
        '''
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
        print('Merging results (PoSER)...')
        # automatically determine if all instances have orthogonalized 
        # mode shapes and if yes, merge these
        use_ortho = use_ortho and self.state[7]
        for instance in instances:
            #check datatype of instance
            assert isinstance(instance, SSICovRef)            
            use_ortho = use_ortho and bool(instance.state[6])
        
        # get values from base instance   
        if use_ortho:
            modes_indices_base = self.ortho_modes_indices
            chan_dofs_base = deepcopy(self.ortho_chan_dofs)
            num_channels_base = self.ortho_num_channels
            mode_shapes_base = self.ortho_mode_shapes
        elif self.state[5] or self.state[6]: #poger or preger
            modes_indices_base = self.merged_modes_indices
            chan_dofs_base = deepcopy(self.merged_chan_dofs)
            num_channels_base = self.merged_num_channels
            mode_shapes_base = self.merged_mode_shapes           
        else:
            modes_indices_base = self.selected_modes_indices
            chan_dofs_base = deepcopy(self.chan_dofs)
            num_channels_base = self.num_analised_channels
            mode_shapes_base = self.mode_shapes
        
        # pair channels and modes of each instance with base instance
        frequencies_base=[]
        for mode_index in modes_indices_base:
            frequencies_base.append(self.modal_frequencies[tuple(mode_index)])
        channel_pairing = []
        mode_pairing = []
        total_dofs = 0
        total_dofs += num_channels_base    
        for instance in instances:
            # calculate the common reference dofs, which may be different channels
            # furthermore reference channels for covariances need not be the reference channels for mode merging
            # channel dof assignments have to be present in each of the instances
            
            if use_ortho:
                chan_dofs_this = instance.ortho_chan_dofs
                modes_indices_this = instance.ortho_modes_indices
                num_channels_this = instance.ortho_num_channels
            elif instance.state[5] or instance.state[6]:#poger or preger
                chan_dofs_this = instance.merged_chan_dofs
                modes_indices_this = instance.merged_modes_indices
                num_channels_this = instance.merged_num_channels
            else:
                assert instance.chan_dofs #return false if list is empty
                chan_dofs_this = instance.chan_dofs
                modes_indices_this = instance.selected_modes_indices
                num_channels_this = instance.num_analised_channels
            
            these_pairs=[]
            for chan_base, node_base, az_base, elev_base in chan_dofs_base:
                for chan_this, node_this, az_this, elev_this in chan_dofs_this:
                    if node_this == node_base and az_this == az_base and elev_this == elev_base:
                        these_pairs.append((chan_base, chan_this))                        
            channel_pairing.append(these_pairs)
            
            total_dofs += num_channels_this-len(these_pairs)
            
            # calculate the mode pairing by minimal frequency difference
            # check that number of modes is equal in all instances (not necessarily)
            # assert len(self.selected_modes_indices) == len(instance.selected_modes_indices)
                
            frequencies_this=[]
            for mode_index in modes_indices_this:
                frequencies_this.append(instance.modal_frequencies[tuple(mode_index)])
                
            mode_pairs = pair_modes(frequencies_base, frequencies_this)
            mode_pairing.append(mode_pairs)
        
        # delete modes not common to all instance from mode pairing
        for mode_num in range(len(modes_indices_base)-1,-1,-1):
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
        if self.merged_mode_shapes is None:
            self.merged_mode_shapes = np.zeros((total_dofs, self.max_model_order, self.max_model_order),dtype=complex)
        elif self.merged_mode_shapes.shape[0] != total_dofs:
            print(RuntimeWarning('The total number of DOFs has changed since the last merging! Resetting all merged mode shapes!'))
            self.merged_mode_shapes = np.zeros((total_dofs, self.max_model_order, self.max_model_order),dtype=complex)
        f_list=np.zeros((len(instances)+1,common_modes))
        d_list=np.zeros((len(instances)+1,common_modes))
        scale_factors = np.zeros((len(instances),common_modes), dtype=complex)
        start_dof=0
        
        # copy modal values from base instance first
        #for mode_num_base,mode_num_this in mode_pairing[0]:  
        for mode_num_base, _ in mode_pairing[0]:
            mode_index = new_mode_nums.index(mode_num_base)
                          
            order_base, index_base = modes_indices_base[mode_num_base]
            
            mode_base = mode_shapes_base[:, index_base, order_base]
                                   
            self.merged_mode_shapes[start_dof:start_dof+num_channels_base, index_base, order_base] = mode_base
            f_list[0, mode_index] = self.modal_frequencies[(order_base, index_base)]
            d_list[0, mode_index] = self.modal_damping[(order_base, index_base)]
        start_dof += num_channels_base
        
        # iterate over instances and assemble output objects (mode_shapes, chan_dofs)
        for inst_num,instance in enumerate(instances):
            
            if use_ortho:
                chan_dofs_this = instance.ortho_chan_dofs
                modes_indices_this = instance.ortho_modes_indices
                num_channels_this = instance.ortho_num_channels
                mode_shapes_this = instance.ortho_mode_shapes
            elif instance.state[5] or instance.state[6]:                
                chan_dofs_this = instance.merged_chan_dofs
                modes_indices_this = instance.merged_modes_indices
                num_channels_this = instance.merged_num_channels
                mode_shapes_this = instance.merged_mode_shapes
            else:
                chan_dofs_this = instance.chan_dofs
                modes_indices_this = instance.selected_modes_indices
                num_channels_this = instance.num_analised_channels
                mode_shapes_this = instance.mode_shapes
                
            these_pairs = channel_pairing[inst_num]            
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
                    for chan, node, az, elev in chan_dofs_this:
                        if chan==channel:             
                            chan = int(start_dof + row_rov)
                            chan_dofs_base.append([chan,node,az,elev])                    
                            row_rov+=1
           
            # loop over modes and rescale them and merge with the other instances
            for mode_num_base, mode_num_this in mode_pairing[inst_num]:
                mode_index = new_mode_nums.index(mode_num_base)
            #for mode_num_base, mode_num_this in mode_pairing[inst_num]:
                
                #index = new_mode_nums.index(mode_num_base)
                
                order_base, index_base = modes_indices_base[mode_num_base]
                
                mode_base = mode_shapes_base[:, index_base, order_base]
                     
                mode_refs_base = np.dot(split_mat_refs_base, mode_base)
                
                order_this, index_this = modes_indices_this[mode_num_this]
                
                mode_this = mode_shapes_this[:, index_this, order_this]
                 
                mode_refs_this = np.dot(split_mat_refs_this, mode_this)
                mode_rovs_this = np.dot(split_mat_rovs_this, mode_this)
                
                numer = np.dot(np.transpose(np.conjugate(mode_refs_this)), mode_refs_base)
                denom = np.dot(np.transpose(np.conjugate(mode_refs_this)), mode_refs_this)
                
                scale_fact=numer/denom
                scale_factors[inst_num,mode_index]=(scale_fact)
                self.merged_mode_shapes[start_dof:start_dof+num_remain_channels, index_base, order_base] = scale_fact*mode_rovs_this
                    
                f_list[inst_num+1, mode_index]=instance.modal_frequencies[(order_this, index_this)]
                d_list[inst_num+1, mode_index]=instance.modal_damping[(order_this, index_this)]
                
            start_dof += num_remain_channels
            
        self.merged_chan_dofs = chan_dofs_base
        self.merged_num_channels = total_dofs
        self.merged_modes_indices = [modes_indices_base[mode_num_base] for mode_num_base, _ in mode_pairing[0]]
        
        mpcs = np.zeros((1,common_modes))
        mps = np.zeros((1,common_modes))
        mpds = np.zeros((1,common_modes))
        
        self.mean_frequencies = np.zeros((self.max_model_order, self.max_model_order))   
        self.std_frequencies = np.zeros((self.max_model_order, self.max_model_order))        
        self.mean_damping = np.zeros((self.max_model_order, self.max_model_order))
        self.std_damping = np.zeros((self.max_model_order, self.max_model_order))
        
        for mode_num_base, mode_num_this in mode_pairing[0]:
            mode_index = new_mode_nums.index(mode_num_base)
            order_base, index_base = modes_indices_base[mode_num_base]
            
        #for mode_num,(order_base, index_base) in enumerate(modes_indices_base):
            
            # rescaling of mode shape 
            mode_tmp = self.merged_mode_shapes[:, index_base, order_base]  
            abs_mode_tmp = np.abs(mode_tmp)
            index_max = np.argmax(abs_mode_tmp)
            this_max = mode_tmp[index_max]
            mode_tmp = mode_tmp / this_max      
            #mpcs[0,index] = StabilPlot.calculateMPC(mode_tmp)
            #mpds[0,index], mps[0,index] = StabilPlot.calculateMPD(mode_tmp)
            self.merged_mode_shapes[:, index_base, order_base] = mode_tmp  
            self.mean_frequencies[(order_base, index_base)] = np.mean(f_list[:,mode_index],axis=0)
            self.std_frequencies[(order_base, index_base)] = np.std(f_list[:,mode_index],axis=0)

            self.mean_damping[(order_base, index_base)] = np.mean(d_list[:,mode_index], axis=0)
            self.std_damping[(order_base, index_base)] = np.std(d_list[:,mode_index], axis=0)
        
        self.state[4]=True
        
                
    def merge_poger(self, instances, ):
        '''
        usage: self.merge_poger([instances of SSICovRef])
        merging the covariances before systemidentification
        toeplitz_cov matrices have to be built beforehand in every instance
        rescaling of modeshapes is done automatically at computation of modal parameters
        '''
        print('Merging covariances (PoGER)...')
        
        # toeplitz covariances have been built
        assert self.state[0]
        
        total_channels = deepcopy(self.num_analised_channels)
        #total_ref_channels = deepcopy(self.num_ref_channels)
        
        for instance in instances:
            #check datatype of instance
            assert isinstance(instance, SSICovRef)
            # toeplitz covariances have been built
            assert instance.state[0]
            
            assert self.num_block_columns == instance.num_block_columns
            assert self.num_block_rows == instance.num_block_rows
            assert self.num_ref_channels >= instance.num_ref_channels
            assert instance.chan_dofs #return false if list is empty
            
            total_channels += instance.num_analised_channels
            #total_ref_channels += instance.num_ref_channels
        
        ssi_ref_channels, rescale_ref_channels, rescale_rov_channels = self.pair_channels(instances, total_channels = True)
        
        if not ssi_ref_channels[0]:
            raise RuntimeError('Could not find common SSI-Reference Channels, maybe your Channel-Dof-Assignments are wrong!')
        if not rescale_ref_channels[0]:
            raise RuntimeError('Could not find common Rescale-Reference Channels, maybe your Channel-Dof-Assignments are wrong!')
        
        #for ssi_ref_channel, rescale_ref_channel, rescale_rov_channel in zip(ssi_ref_channels, rescale_ref_channels, rescale_rov_channels):
        #    print('Next Instance')
        #    print(ssi_ref_channel, rescale_ref_channel, rescale_rov_channel)
            
        num_ssi_ref_channels = len(ssi_ref_channels[0])
        total_ref_channels = num_ssi_ref_channels * (len(instances) + 1)
        
        merged_quantities = deepcopy(self.quantities) 
        this_rescale_channel = deepcopy(self.num_analised_channels)-1
        chan_dofs = deepcopy(self.chan_dofs)
        
        for instance, roving_channels_this in zip(instances, rescale_rov_channels):
            chan_dofs_this = instance.chan_dofs
            
            for rov_chan_this in roving_channels_this:
                this_rescale_channel += 1
                # find channel-dof-assignments for current channel
                for chan_this, node_this, az_this, elev_this in chan_dofs_this:
                    if chan_this == rov_chan_this[1]:
                        chan_dofs.append([this_rescale_channel, node_this, az_this, elev_this])
                        
                        merged_quantities.append(instance.quantities[int(chan_this)])
                        assert merged_quantities[this_rescale_channel] == instance.quantities[int(chan_this)]
                        break
                else:
                    print( RuntimeWarning('Channel {} could not be found in Channel-Dof-Assignments of instance {}'.format(chan_this, instances.index(instance))))
                    merged_quantities.append(None)
        
        assert len(merged_quantities) == total_channels - total_ref_channels + num_ssi_ref_channels
        assert this_rescale_channel == total_channels - total_ref_channels + num_ssi_ref_channels - 1
        
        self.rescale_ref_channels = rescale_ref_channels
        self.rescale_rov_channels = rescale_rov_channels
        self.merged_chan_dofs = chan_dofs
        self.merged_quantities = merged_quantities
        self.merged_num_channels = total_channels - total_ref_channels + num_ssi_ref_channels
        self.merged_num_channels_multiref = total_channels
        
        print('self.merged_num_channels ', self.merged_num_channels)
        print('self.merged_num_channels_multiref ', self.merged_num_channels_multiref)
        
        n, m = total_channels*self.num_block_rows, num_ssi_ref_channels*self.num_block_columns  
        Toeplitz_matrix = np.zeros((n,m))
        
        base_ref_channels = [pair[0] for pair in ssi_ref_channels[0]]
        #print(base_ref_channels)
        base_toeplitz_matrix = self.extract_part_toeplitz_cov(col_channels = base_ref_channels)
        
        toeplitz_matrices = []
        for ssi_ref_channels_this, instance in zip(ssi_ref_channels, instances):
            ssi_refs = [c[1] for c in ssi_ref_channels_this]
            this_toeplitz = instance.extract_part_toeplitz_cov(col_channels = ssi_refs)
            toeplitz_matrices.append(this_toeplitz)
        
        total_end_row = 0
        for block_row in range(self.num_block_rows):
            num_chans = self.num_analised_channels
            this_start_row = block_row * num_chans
            this_end_row = (block_row + 1) * num_chans
            total_start_row = total_end_row
            total_end_row += num_chans
            
            this_covs = base_toeplitz_matrix[this_start_row:this_end_row,:]
            
            Toeplitz_matrix[total_start_row:total_end_row,:] = this_covs
            
            for this_toeplitz, instance in zip(toeplitz_matrices, instances):
                num_chans = instance.num_analised_channels
                this_start_row = block_row * num_chans
                this_end_row = (block_row + 1) * num_chans
                total_start_row = total_end_row
                total_end_row += num_chans
                
                this_covs = this_toeplitz[this_start_row:this_end_row,:]
                
                Toeplitz_matrix[total_start_row:total_end_row,:] = this_covs
                
            assert total_end_row == (block_row + 1) * total_channels
        
        self.merged_toeplitz_matrix = Toeplitz_matrix
        self.state[1] = False # state matrices
        self.state[2] = False # modal params
        self.state[3] = False # select modes
        self.state[7] = False # orthogonalize modes
        self.state[5] = True # merged PoGer
        
    def merge_preger(self, instances, auto_cov=False):
        '''
        usage: self.merge_preger([instances of SSICovRef])
        toeplitz_cov matrices have to be built beforehand in every instance
        merging and rescaling covariances before systemidentification
        selecting the modes has to be done afterwards
        '''
        print('Merging covariances (PreGER)...')
        
        assert self.state[0]
        
        total_channels = deepcopy(self.num_analised_channels)
        print('primary checks')
        
        for instance in instances:
            #check datatype of instance
            assert isinstance(instance, SSICovRef)
            # toeplitz covariances have been built
            assert instance.state[0]
            
            assert self.num_block_columns >= instance.num_block_columns
            assert self.num_block_rows >= instance.num_block_rows
            assert self.num_ref_channels >= instance.num_ref_channels
            assert instance.chan_dofs #return false if list is empty
            
            total_channels += instance.num_analised_channels
        
        num_block_rows = self.num_block_rows
        num_block_columns = self.num_block_columns
        
        ssi_ref_channels = self.pair_channels(instances)[0]
        num_ref_channels = len(ssi_ref_channels[0])
        total_ref_channels = num_ref_channels * (len(instances) + 1)
        
        merged_quantities = deepcopy(self.quantities) 
        chan_dofs = deepcopy(self.chan_dofs)
        
        this_rescale_channel = int(max(self.ref_channels+self.roving_channels))
        
        # generate merged channel-dof-assignments and merged quantities
        for channel_pairs, instance in zip(ssi_ref_channels, instances):         
            this_ref_channels = [channel_pair[1] for channel_pair in channel_pairs]
            # loop over channels in ascending order
            all_channels = instance.ref_channels + instance.roving_channels
            all_channels.sort()
            for chan_this in all_channels:
                # channel is a reference channel and will be used for rescaling only
                if chan_this in this_ref_channels:
                    continue
                this_rescale_channel += 1
                # find channel-dof-assignments for current channel
                for chan, node_this, az_this, elev_this in instance.chan_dofs:
                    if chan_this == chan:
                        chan_dofs.append((this_rescale_channel,node_this,az_this,elev_this))
                    
                if len(merged_quantities) <= this_rescale_channel:
                    merged_quantities.append('')
                merged_quantities[this_rescale_channel] = instance.quantities[chan_this]            
            
        self.merged_chan_dofs = chan_dofs
        self.merged_quantities = merged_quantities
        self.merged_num_channels = total_channels - total_ref_channels + num_ref_channels

        print('separation of toeplitz matrices')
        
        base_ref_channels = [channel_pair[0] for channel_pair in ssi_ref_channels[0]]
        print('reference channels: {}'.format(base_ref_channels))
        all_channels = self.ref_channels+self.roving_channels
        ref_toeplitz_matrix_base = self.extract_part_toeplitz_cov(row_channels = base_ref_channels, col_channels = base_ref_channels, auto_only= auto_cov)
        #ref_toeplitz_matrix_base_ = self.extract_part_toeplitz_cov(row_channels = base_ref_channels, col_channels = base_ref_channels, recompute = True)
        #print((ref_toeplitz_matrix_base==ref_toeplitz_matrix_base_).all())
        rov_toeplitz_matrix_base = self.extract_part_toeplitz_cov(row_channels = all_channels, col_channels = base_ref_channels)
        #rov_toeplitz_matrix_base = self.extract_part_toeplitz_cov(row_channels = all_channels, col_channels = base_ref_channels, recompute = True)
        #print((rov_toeplitz_matrix_base==rov_toeplitz_matrix_base_).all())
        
        ref_toeplitz_matrices = []
        rov_toeplitz_matrices = []
        for channel_pairs, instance in zip(ssi_ref_channels,instances):
            
            ref_list = [channel_pair[1] for channel_pair in channel_pairs]
            rov_list = [channel for channel in instance.ref_channels + instance.roving_channels if channel not in ref_list]
            
            #print('####')
            print('instance: {}, roving_channels: {}'.format(instances.index(instance), rov_list))
           
            rov_toeplitz_matrices.append(instance.extract_part_toeplitz_cov(row_channels = rov_list, col_channels = ref_list))
            
            print('instance: {}, ref_channels: {}'.format(instances.index(instance), ref_list))
            
            ref_toeplitz_matrices.append(instance.extract_part_toeplitz_cov(row_channels = ref_list, col_channels = ref_list, auto_only = auto_cov))
            
        n, m = num_ref_channels * num_block_rows, total_ref_channels * num_block_columns 
        #print(n,m) 
        ref_toeplitz_matrix = np.zeros((n,m))
        print('block-column wise interleaving of references')
        
        
        total_end_column = 0
        for block_column in range(num_block_columns):
            total_start_column = total_end_column
            total_end_column += num_ref_channels
            
            this_start_column = block_column * num_ref_channels
            this_end_column = (block_column + 1) * num_ref_channels
            
            this_block = ref_toeplitz_matrix_base[:,this_start_column:this_end_column]         
            
            ref_toeplitz_matrix[:,total_start_column:total_end_column] = this_block
        
            for ref_toeplitz_matrix_this, instance in zip(ref_toeplitz_matrices,instances):
                
                this_start_column = block_column * num_ref_channels
                this_end_column = (block_column + 1) * num_ref_channels
                
                total_start_column = total_end_column
                total_end_column += num_ref_channels
                
                this_block = ref_toeplitz_matrix_this[:,this_start_column:this_end_column]
                #print(this_start_column, this_end_column)        
                #print(this_block.shape)
                #print(total_start_column, total_end_column)
                #print(ref_toeplitz_matrix[:,total_start_column:total_end_column].shape)
                ref_toeplitz_matrix[:,total_start_column:total_end_column] = this_block
        assert total_end_column == num_block_columns * total_ref_channels
        #print(ref_toeplitz_matrix[-6:,-6:])
        print('SVD')
        [U,S,V_T] = np.linalg.svd(ref_toeplitz_matrix,0);
        
        S_2 = np.diag(np.power(S, 0.5)) 
        #S_2 = np.zeros(( S.shape[0], V_T.shape[0]), dtype=complex)
        #S_2[:S.shape[0], :S.shape[0]] = np.diag(np.power(S, 0.5)) 
        Gamma_i = np.dot(S_2, V_T)
        
        print('Gammas')
        base_gamma = np.zeros((num_block_rows * num_ref_channels, num_block_columns * num_ref_channels ))
        controllability_matrices = []
        for instance in instances:
            this_gamma = np.zeros((num_block_rows * num_ref_channels, num_block_columns * num_ref_channels ))
            controllability_matrices.append(this_gamma)
            
        total_end_column = 0
        
        #print(controllability_matrices[0][:4,:4])
        for block_column in range(num_block_columns):
            total_start_column = total_end_column
            total_end_column += num_ref_channels
            
            this_start_column = block_column * num_ref_channels
            this_end_column = (block_column + 1) * num_ref_channels
                
            base_gamma[:,this_start_column:this_end_column] = Gamma_i[:,total_start_column:total_end_column]
            
            for this_gamma, instance in zip(controllability_matrices, instances):
                total_start_column = total_end_column
                total_end_column += num_ref_channels
            
                this_start_column = block_column * num_ref_channels
                this_end_column = (block_column + 1) * num_ref_channels
                
                this_gamma[:,this_start_column:this_end_column] = Gamma_i[:,total_start_column:total_end_column]
                
        assert this_end_column == num_block_columns * num_ref_channels
        assert total_end_column == num_block_columns * total_ref_channels
        #print(controllability_matrices[0][:4,:4])
        
        print('rescale roving toeplitz')
        #print(rov_toeplitz_matrices[0][:4,:4])
        
#         cov_plots_base = [[] for i in range(num_ref_channels)]
#         for block_column in range(num_block_columns):
#             
#             this_start_column = block_column * num_ref_channels
#             this_end_column = (block_column + 1) * num_ref_channels
#             
#             this_block_ = ref_toeplitz_matrix_base[:,this_start_column:this_end_column]
#             
#             for i in range(num_ref_channels):
#                 cov_plots_base[i].append(this_block_[i,i])        
#         
#         cov_plots_this = [[[] for j in range(num_ref_channels)] for instance in instances]
#         cov_plots_this_r = [[[] for j in range(num_ref_channels)] for instance in instances]
        
        for this_gamma, rov_toeplitz_matrix in zip(controllability_matrices, rov_toeplitz_matrices):
            #this_gamma_inv = np.linalg.inv(np.dot(this_gamma, this_gamma.T))
            #scale_mat = np.dot(this_gamma.T, np.dot(this_gamma_inv, base_gamma))
            scale_mat = np.dot(np.linalg.pinv(this_gamma), base_gamma)
            rov_toeplitz_matrix[:,:] = np.dot(rov_toeplitz_matrix, scale_mat)
            
            
#             ref_toeplitz_matrix_r = np.dot(ref_toeplitz_matrix_, scale_mat)
#             inst_num = ref_toeplitz_matrices.index(ref_toeplitz_matrix_)
#             
#             for block_column in range(num_block_columns):
#                 
#                 this_start_column = block_column * num_ref_channels
#                 this_end_column = (block_column + 1) * num_ref_channels
#                 
#                 this_block_ = ref_toeplitz_matrix_[:,this_start_column:this_end_column]
#                 this_block_r = ref_toeplitz_matrix_r[:,this_start_column:this_end_column]
#                 for i in range(num_ref_channels):
#                     cov_plots_this[inst_num][i].append(this_block_[i,i])
#                     cov_plots_this_r[inst_num][i].append(this_block_r[i,i])     
#             
        #print(rov_toeplitz_matrices[0][:4,:4])
        
        n,m = self.merged_num_channels * num_block_rows, num_ref_channels * num_block_columns
        Toeplitz_matrix = np.zeros((n,m))
                

        print('assemble full toeplitz')
        total_end_row = 0
        for block_row in range(num_block_rows):
            this_start_row = block_row * self.num_analised_channels
            this_end_row = (block_row + 1) * self.num_analised_channels
           
            this_covs = rov_toeplitz_matrix_base[this_start_row:this_end_row, 0:num_ref_channels * num_block_columns]
            

            total_start_row = total_end_row
            total_end_row += self.num_analised_channels
            
            Toeplitz_matrix[total_start_row:total_end_row,:] = this_covs
            
            for rov_toeplitz_matrix_this, instance in zip(rov_toeplitz_matrices,instances):
                num_roving_channels = instance.num_analised_channels - num_ref_channels
                
                this_start_row = block_row * num_roving_channels
                this_end_row = (block_row + 1) * num_roving_channels
                
                this_block = rov_toeplitz_matrix_this[this_start_row:this_end_row,:]
                
                inst_num = instances.index(instance)
                
                    
                total_start_row = total_end_row
                total_end_row += num_roving_channels
                
                Toeplitz_matrix[total_start_row:total_end_row,:] = this_block
                
            assert total_end_row == (block_row + 1) * self.merged_num_channels
#         fig = plot.figure()
#         for i in range(num_ref_channels):
#             plot.plot(cov_plots_base[i], label='base channel {}'.format(i))
#         plot.legend()
#         #plot.savefig(os.getcwd()+'/' + str(auto_cov) + str(num_block_rows) + 'base.png')
#         #plot.close(fig)
#         
#         for j in range(len(instances)):
#             fig = plot.figure()
#             for i in range(num_ref_channels):
#                 plot.plot(cov_plots_this[j][i], label='inst {}, channel {}'.format(j, i))
#             plot.legend()
#             #plot.savefig(os.getcwd()+'/' + str(auto_cov) + str(num_block_rows) + str(j)+'.png')
#             #plot.close(fig)
#             
#         for j in range(len(instances)):
#             fig = plot.figure()
#             for i in range(num_ref_channels):
#                 plot.plot(cov_plots_this_r[j][i], label='inst {}, channel {} r'.format(j, i))
#             plot.legend()
#             #plot.savefig(os.getcwd()+'/' + str(auto_cov) + str(num_block_rows) + str(j)+'rs.png')
#             #plot.close(fig)
#         plot.show()
        
        self.merged_toeplitz_matrix = Toeplitz_matrix
        
        self.state[1] = False # state matrices
        self.state[2] = False # modal params
        self.state[3] = False # select modes
        self.state[7] = False # orthogonalize modes
        self.state[6] = True# merged PreGer
    

    def pair_channels(self, instances, total_channels = False):
        '''
        pairs channels from all given instances for the poger and preger merging methods
        
        in poger and preger ssi_ref_channels must be equal in all instances
        in preger rescale ref channels must be equal in all instances
        in poger they can differ
        
        loops over instances and pairs channels with common dofs
        identifies reference channels that are common to all instances 
        and uses them as ssi-ref-channels
        
        channels for rescaling are mainly used by poger merging and 
        should be created with total_channels flag set True
        '''
        
        print('pairing channels and dofs')
        
        # find common channels in all instances, by comparing dofs
        # reference channels must be in ref_channels list i.e. be a ssi reference channel
        # and must have equal dofs i.e. must be a rescaling reference channel
        # else the channels will be regarded as roving channels
        this_total_channel = deepcopy(self.num_analised_channels)-1
        #channel_pairings = []    
        ssi_ref_channels = []
        rescale_ref_channels = []
        roving_channels = []
        #print(self.chan_dofs, self.ref_channels)
        for instance in instances:    
            #print(instance.chan_dofs, instance.ref_channels)
            ssi_pairs = []
            rescale_pairs = []
            rov_pairs = []
            
            # loop over channels in ascending order
            channels = instance.ref_channels + instance.roving_channels
            channels.sort()
            
            for chan_this in channels:
                
                this_total_channel += 1
                # find channel-dof-assignments for current channel
                for chan, node_this, az_this, elev_this in instance.chan_dofs:
                    if chan_this == chan: #channel is assigned to a dof
                        # determine if channel has equal dofs in base instance, 
                        # i.e. can be used for rescaling, i.e. is reference channel
                        #print(node_this,az_this, elev_this)
                        
                        for chan_base, node_base, az_base, elev_base in self.chan_dofs:
                            if node_this == node_base and az_this == az_base and elev_this == elev_base \
                                and chan_this in instance.ref_channels and chan_base in self.ref_channels:
                                
                                rescale_pairs.append((chan_base, this_total_channel))
                                ssi_pairs.append((chan_base, chan_this))
                                
                                break
                        else:# channel is a roving channel
                            rov_pairs.append((this_total_channel, chan_this))
                        break
                else:# channel is not assigned to dof and assumed a roving channel
                    rov_pairs.append((this_total_channel, chan_this))
                    
            ssi_ref_channels.append(ssi_pairs)
            rescale_ref_channels.append(rescale_pairs)
            roving_channels.append(rov_pairs)
        
        # find base channels common to all instances
        for channel in self.ref_channels:
            in_all = True
            for channel_pairs in ssi_ref_channels:
                for channel_pair in channel_pairs:
                    if channel_pair[0] == channel:
                        break
                else:
                    in_all = False
                    break
            if in_all:
                continue
            # channel is not common to all instances, delete it
            for channel_pairs in ssi_ref_channels:
                for index, channel_pair in enumerate(channel_pairs):
                    if channel_pair[0] == channel:
                        del channel_pairs[index]
                        break

        # make sure once again, that the number of reference channels is equal
        # as the block columns of the block toeplitz matrix of covariances will be built from it
        for i in range(1,len(instances)):
            assert len(ssi_ref_channels[i-1]) == len(ssi_ref_channels[i])
        
        return ssi_ref_channels, rescale_ref_channels, roving_channels
    
    def orthogonalize_dofs(self, modes_indices=None, method='complex', num_phi=3600, restrain_nodes = None):
        '''
        
        mode_indices = [ (order_index, mode_index), ...]
            the indices of the modes to be converted, defaults to the indices of the selected modes
            
        method = 'real', 'complex' or 'complex_polar'
            the orthogonalization can be done in the real or complex domain
            a more descriptive, but less accurate way in the complex domain consists of looping over 
            the full phase circle, thereby determining the maximum displacements and phase angles
            in each direction from the resultant circle in 3D-Space, iterates over num_phi points
            i.e. simulate a full phase cycle
            
        num_phi integer -> see description of method above
        
        restrain_nodes = {node:[restrain_dir, ...]}; restrain_dir = (az, elev)
            for each node several restraining conditions can be set, by providing a dictionary as outlined above
            if there are three or more channeld assigned to node, due to the computation of mean values the restraining
            condition might not be met
            each restraining condition adds a channel to the converted mode shape 
            
        Computes common orthogonal components of displacements given in a skewed coordinate system
        the skewed coordinate system (e.g the directions of the vectors) for each node are taken from the previously 
        provided channel-dof-assignments
        currently only nodes with two or three assigned displacements can be converted (eg. plane or spatial orthogonalization)
        '''
        print('Orthogonalize DOFs...')
        def calc_aer(x,y,z):
            xy = x**2 + y**2
            r = np.sqrt(xy + z**2)
            #elev = np.arctan2(np.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
            elev = np.arctan2(z, np.sqrt(xy)) # for elevation angle defined from XY-plane up
            az = np.arctan2(y, x)
            return az, elev, r
        
        def calc_xyz(az, elev, r=1):
            az = az * np.pi / 180
            elev = elev * np.pi / 180
            x=r*np.cos(elev)*np.cos(az) # for elevation angle defined from XY-plane up
            #x=r*np.sin(elev)*np.cos(az) # for elevation angle defined from Z-axis down
            y=r*np.cos(elev)*np.sin(az) # for elevation angle defined from XY-plane up
            #y=r*np.sin(elev)*np.sin(az)# for elevation angle defined from Z-axis down
            z=r*np.sin(elev)# for elevation angle defined from XY-plane up
            #z=r*np.cos(elev)# for elevation angle defined from Z-axis down
            return x,y,z
        
        def pair_node_channels_angles(chan_dofs, num_total_channels=None):
                
            finished_nodes=[]
            node_channels = OrderedDict()
            node_angles = OrderedDict()
            all_channels = set()
            
            while True:
                current_angles = []
                current_chans=[]
                this_node=None
                for chan, node, az, elev in chan_dofs:
                    if node in finished_nodes:
                        continue
                    elif this_node is None:
                        this_node=node
                    if node==this_node:
                        all_channels.add(chan)
                        current_chans.append(chan)                        
                        current_angles.append([az,elev])
                if this_node is None:
                    break                
                finished_nodes.append(this_node)    
                node_channels[this_node] = current_chans
                node_angles[this_node] = current_angles
                
            if num_total_channels is None:
                num_total_channels = len(all_channels)
                
            remain_channels = set(range(num_total_channels)).difference(all_channels)
            #print(all_channels, list(range(num_total_channels)))
            print(node_angles)
            return node_channels, node_angles, remain_channels         
        
        def transform_chan_dofs(node_channels, node_angles, remain_channels, restrain_nodes):
            
            next_chan = 0
            ortho_chan_dofs = []
            for node in node_channels.keys():
                current_chans = node_channels[node]
                skew_angles = node_angles[node] 
                
                if node in restrain_nodes:
                    for angle in restrain_nodes[node]:
                        skew_angles.append(angle)
                        current_chans.append(None)
                        
                if len(current_chans) == 1:
                    ortho_angles=skew_angles        
                elif len(current_chans) == 2:
                    for i in range(2):
                        if skew_angles[0][i] == skew_angles[1][i]:
                            i_ = int(not(i))                        
                            ortho_angles = [0.0, 90.0]
                            for j in range(2):
                                angle = skew_angles[j]
                                angle[i_]=ortho_angles[j]
                                ortho_angles[j]=angle 
                    else:
                        ortho_angles=skew_angles
                else:  
                    ortho_angles = [[0.0, 0.0], [90.0,0.0], [0.0,90.0]]
                
                for az,elev in ortho_angles:
                    while next_chan in remain_channels:
                        next_chan += 1                                              
                    ortho_chan_dofs.append([next_chan, node, az, elev])
                    next_chan += 1
                                    
            return ortho_chan_dofs        
        
        def transform_mode_shape(skew_mode_shape, node_channels, node_angles, restrain_nodes = {}, remain_channels = [], method='complex', num_phi=3600):
            
            if method=='real':
                skew_mode_shape = np.real(skew_mode_shape)
            elif method=='complex_polar':
                skew_mode_shape = np.array(list(zip(np.abs(skew_mode_shape),np.angle(skew_mode_shape))))
            
            
            ortho_mode_shape = np.empty(num_channels, dtype=complex)
            
            next_chan = 0
            
            for node in node_channels.keys():
                current_chans = node_channels[node]
                num_current_chans = len(current_chans)
                skew_angles = node_angles[node] 
                skew_msh_part = skew_mode_shape[current_chans]
                #print('Node {}'.format(node))
                
                if node in restrain_nodes:
                    for angle in restrain_nodes[node]:
                        num_current_chans += 1
                        skew_msh_part.append(0) # in that direction no displacement
                        skew_angles.append(angle)
                if num_current_chans > 6:
                    RuntimeWarning('The number of channels for node {} is greater than 6. Computation of each channel combination and averaging might take very long!'.format(node))
                
                if num_current_chans == 1:
                    ortho_msh_part=skew_msh_part 
                elif num_current_chans == 2:
                    for i in range(2):
                        if skew_angles[0][i] == skew_angles[1][i]:
                            i_ = int(not(i))                            
                            base_vects=[(np.sin(skew_angles[j][i_]), np.cos(skew_angles[j][i_])) for j in range(2)]
                            
                            if method != 'complex_polar':
                                ortho_msh_part = get_line_intersection(skew_msh_part, base_vects)
                            else:
                                ortho_msh_part = intersect_by_polar_iteration(skew_msh_part, base_vects, num_phi)   
                    else:
                        ortho_msh_part=skew_msh_part
                else:
                    ortho_msh_part = None
                    range_2 = list(range(num_current_chans))
                    for i_1 in range(num_current_chans):
                        range_2.remove(i_1)
                        range_3 = deepcopy(range_2)
                        for i_2 in range_2:
                            range_3.remove(i_2)
                            for i_3 in range_3:
                                base_vects = []
                                this_skew_msh = np.empty((0,1))
                                for i_n in [i_1,i_2,i_3]: 
                                    base_vects.append(calc_xyz(*skew_angles[i_n]))
                                    this_skew_msh = np.append(this_skew_msh,skew_msh_part[i_n])
                                if method != 'complex_polar': 
                                    this_ortho_msh = get_plane_intersection(this_skew_msh, base_vects)
                                else:
                                    this_ortho_msh = intersect_by_polar_iteration(this_skew_msh, base_vects, num_phi)
                                if ortho_msh_part is None:
                                    ortho_msh_part = this_ortho_msh[:,np.newaxis]
                                else:
                                    ortho_msh_part = np.concatenate((ortho_msh_part, this_ortho_msh[:,np.newaxis]), axis=1)   
                    if ortho_msh_part.shape[1]>1:
                        ortho_msh_part = np.mean(ortho_msh_part, axis=1)

                for i in range(min(num_current_chans,3)):
                    while next_chan in remain_channels:
                        ortho_mode_shape[next_chan]=skew_mode_shape[next_chan]
                        next_chan += 1                                         
                    ortho_mode_shape[next_chan]=ortho_msh_part[i].squeeze()
                    next_chan += 1                    
                    
            return ortho_mode_shape        
        
        def get_plane_intersection(mode_shape, base_vects):
            
            #plane coefficients A,B,C,D
            A=np.empty([3,1], dtype=np.complex128)
            B=np.empty([3,1], dtype=np.complex128)
            C=np.empty([3,1], dtype=np.complex128)
            D=np.empty([3,1], dtype=np.complex128)
            #a=np.empty([3,3], dtype=np.complex128)
            
            for i in range(3):
                disp = mode_shape[i]
                x,y,z = base_vects[i]
                x,y,z=[coord*disp for coord in (x,y,z)]
                #a[i,0]=x
                #a[i,1]=y
                #a[i,2]=z

                # calculate coefficients of plane perpendicular to vector (x,y,z) and through point (x,y,z)
                A[i]=x
                B[i]=y
                C[i]=z
                D[i]=-x**2-y**2-z**2
    
            d=np.linalg.det(np.concatenate((A,B,C), axis=1))
            dx=np.linalg.det(np.concatenate((D,B,C), axis=1))
            dy=np.linalg.det(np.concatenate((A,D,C), axis=1))
            dz=np.linalg.det(np.concatenate((A,B,D), axis=1))
            
            if d == 0:
                print('Planes do not intersect in a common point')
                x=0
                y=0
                z=0
            else:
                x=-dx/d
                y=-dy/d
                z=-dz/d
            #max_ = max(np.abs(x),np.abs(y),np.abs(z))
            #print('x = {:2.3f}, y = {:2.3f}, z = {:2.3f}'.format(np.abs(x)/max_,np.abs(y)/max_, np.abs(z)/max_))
            return np.array([x,y,z])
        
        def get_line_intersection(mode_shape, base_vects):
            #line coefficients A,B,C
            
            A=np.empty([2,1], dtype=np.complex128)
            B=np.empty([2,1], dtype=np.complex128)
            C=np.empty([2,1], dtype=np.complex128)
            
            #a=np.empty([3,3], dtype=np.complex128)
            
            for i, ((x,y), disp) in enumerate(zip(base_vects, mode_shape)):
                x,y=[coord*disp for coord in (x,y)]
                #a[i,0]=x
                #a[i,1]=y

                # calculate coefficients of line perpendicular to vector (x,y) and through point (x,y)
                A[i]=x
                B[i]=y
                C[i]=-x**2-y**2
    
            d=np.linalg.det(np.concatenate((A,B), axis=1))
            dx=np.linalg.det(np.concatenate((B,C), axis=1))
            dy=np.linalg.det(np.concatenate((A,C), axis=1))
            
            if d == 0:
                raise RuntimeError('Lines do not intersect in a common point')
    
            x=-dx/d
            y=-dy/d
            
            return x,y
       
        def intersect_by_polar_iteration(mode_shape, base_vects, num_phi):                
            x_phi=np.empty([num_phi,1])
            y_phi=np.empty([num_phi,1])
            z_phi=np.empty([num_phi,1])   
            phi_j=np.empty([num_phi,1])
            
            for j in range(num_phi):
                phi=j/(0.5*num_phi)*np.pi-np.pi # [-pi, ..., pi]
    
                this_mode_shape=[]
                for r,beta in mode_shape:
                    this_mode_shape.append(r*np.cos(phi+beta))
                if len(base_vects)==2:
                    x,y=get_line_intersection(base_vects, this_mode_shape)
                elif len(base_vects)==3:
                    x,y,z=get_plane_intersection(base_vects, this_mode_shape)
                
                x_phi[j]=x.real
                y_phi[j]=y.real
                if len(base_vects)==3:
                    z_phi[j]=z.real
                phi_j[j]=phi
                
            x=x_phi.max() * np.exp(1j*phi_j[x_phi.argmax()])[0]
            y=y_phi.max() * np.exp(1j*phi_j[y_phi.argmax()])[0]
            if len(base_vects)==3:
                z=z_phi.max() * np.exp(1j*phi_j[z_phi.argmax()])[0]
            
            return (x,y,z)           

        '1'
        #check userprovided inputs
        if modes_indices is None:
            if self.state[4]:# Poser merged (= previously selected)
                modes_indices = self.merged_modes_indices
                mode_shapes = self.merged_mode_shapes
                chan_dofs = self.merged_chan_dofs
                num_channels = self.merged_num_channels
            elif self.state[3] and (self.state[5] or self.state[6]): # PoGer or Preger and selected
                modes_indices = self.merged_modes_indices
                mode_shapes = self.merged_mode_shapes
                chan_dofs = self.merged_chan_dofs
                num_channels = self.merged_num_channels
            elif self.state[3]: # not merged but selected
                modes_indices = self.selected_modes_indices
                mode_shapes = self.mode_shapes
                chan_dofs = self.chan_dofs
                num_channels = self.num_analised_channels
            else:
                print(RuntimeWarning('Now orthogonalizing all mode shapes: {}'.format(self.max_model_order**2)))
                modes_indices = []
                for order in range(self.max_model_order):
                    for mode in range(order):
                        modes_indices.append((order,mode))
                if self.state[5] or self.state[6]:
                    mode_shapes = self.merged_mode_shapes
                    chan_dofs = self.merged_chan_dofs
                    num_channels = self.merged_num_channels
                else:
                    mode_shapes = self.mode_shapes    
                    chan_dofs = self.chan_dofs     
                    num_channels = self.num_analised_channels  
        else:
            assert isinstance(modes_indices, (list,tuple))
            for index in modes_indices:
                assert len(index)== 2
                assert index[0] <= self.max_model_order
                assert index[1] <= self.max_model_order
            mode_shapes = self.mode_shapes
            chan_dofs = self.chan_dofs     
            num_channels = self.num_analised_channels  
        
        assert method in ['real', 'complex', 'complex_polar']
        
        assert isinstance(num_phi, int)
        if num_phi < 360:
            RuntimeWarning('num_phi should be much greater than 360° to assure adequate precision of results!')
            
        if restrain_nodes is not None:
            for restrain_dir in restrain_nodes.values():
                assert len(restrain_dir) == 2
                for dir in restrain_dir:
                    assert isinstance(dir, (float, int))
        else:
            restrain_nodes = {}
            
        node_channels, node_angles, remain_channels = pair_node_channels_angles(chan_dofs, num_channels)
        print(remain_channels, num_channels)
        all_channels = []
        for node,channels in node_channels.items():
            for channel in channels:
                all_channels.append(channel)
            if node in restrain_nodes:
                all_channels.append(None)
        for channel in remain_channels:
            all_channels.append(channel)
        
        num_channels_ = len(all_channels) # might be higher than num_total_channels
        
        if self.ortho_chan_dofs is None:
            self.ortho_chan_dofs = transform_chan_dofs(node_channels, node_angles, remain_channels, restrain_nodes)
            self.ortho_mode_shapes = np.zeros((num_channels_, self.max_model_order, self.max_model_order), dtype=complex)
            self.ortho_modes_indices = []
            self.ortho_num_channels = num_channels_
        else:
            new_chan_dofs = transform_chan_dofs(node_channels, node_angles, remain_channels, restrain_nodes)
            equal = len(new_chan_dofs) == len(self.ortho_chan_dofs)
            if equal:
                for i in range(len(new_chan_dofs)):
                    equal = equal and new_chan_dofs[i]==self.ortho_chan_dofs[i]
            if not equal:
                RuntimeWarning('Channel-DOF-Definitions have changed! Deleting previously transformed modeshapes!')               
                self.ortho_chan_dofs = new_chan_dofs
                self.ortho_mode_shapes = np.zeros((num_channels_, self.max_model_order, self.max_model_order),dtype=complex)
                self.ortho_modes_indices = []
                self.ortho_num_channels = num_channels_
        
        for mode_num, (order, mode_ind) in enumerate(modes_indices):
            print('Mode: {} Hz'.format(self.modal_frequencies[order, mode_ind]))
            self.ortho_modes_indices.append((order,mode_ind))
            skew_mode_shape = mode_shapes[:, mode_ind, order]
            ortho_mode_shape = transform_mode_shape(skew_mode_shape, node_channels, node_angles, restrain_nodes, remain_channels, method, num_phi)
            self.ortho_mode_shapes[:, mode_ind, order] = ortho_mode_shape
        self.state[7]=True
        
    def save_state(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        #             0         1           2           3          4            5            6             7
        #self.state= [Toeplitz, State Mat., Modal Par., Select M., Merge PoSER, Merge PoGer, Merge Preger, Ortho]
        out_dict={'self.state':self.state}
        out_dict['self.measurement'] = self.measurement
        out_dict['self.sampling_rate'] = self.sampling_rate
        out_dict['self.total_time_steps'] = self.total_time_steps
        out_dict['self.ref_channels'] = self.ref_channels
        out_dict['self.roving_channels'] = self.roving_channels
        out_dict['self.num_ref_channels'] = self.num_ref_channels
        out_dict['self.num_roving_channels'] = self.num_roving_channels
        out_dict['self.num_analised_channels'] = self.num_analised_channels
        out_dict['self.max_model_order'] = self.max_model_order
        out_dict['self.quantities'] = self.quantities
        out_dict['self.chan_dofs'] = self.chan_dofs
        if self.state[0]:# covariances
            out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[5] or self.state[6]: #merged poger,preger
            out_dict['self.merged_toeplitz_matrix'] = self.merged_toeplitz_matrix
            out_dict['self.merged_chan_dofs'] = self.merged_chan_dofs
            out_dict['self.merged_num_channels'] = self.merged_num_channels
            out_dict['self.merged_quantities'] = self.merged_quantities
        if self.state[5]:
            out_dict['self.merged_num_channels_multiref'] = self.merged_num_channels_multiref
            out_dict['self.rescale_ref_channels'] = self.rescale_ref_channels
            out_dict['self.rescale_rov_channels'] = self.rescale_rov_channels
        if self.state[1]:# state models
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
        if self.state[2]:# modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
        if self.state[2] and (self.state[5] or self.state[6]): # merged poger,preger and modal params
            out_dict['self.merged_mode_shapes'] = self.merged_mode_shapes
        if self.state[3] and not (self.state[5] or self.state[6]):# selected but not merged (poger or preger)
            out_dict['self.selected_modes_indices'] = self.selected_modes_indices
        if self.state[4]: # merged poser
            out_dict['self.merged_mode_shapes'] = self.merged_mode_shapes
            out_dict['self.merged_modes_indices'] = self.merged_modes_indices
            out_dict['self.merged_num_channels'] = self.merged_num_channels
            out_dict['self.merged_chan_dofs'] = self.merged_chan_dofs
            out_dict['self.mean_frequencies'] = self.mean_frequencies # inconsistent data layout model -> see save_results for explanation
            out_dict['self.std_frequencies'] = self.std_frequencies # as above
            out_dict['self.mean_damping'] = self.mean_damping # as above
            out_dict['self.std_damping'] = self.std_damping # as above
        if self.state[3] and (self.state[5] or self.state[6]): #selected and merged poger, preger  
            out_dict['self.merged_mode_shapes'] = self.merged_mode_shapes
            out_dict['self.merged_modes_indices'] = self.merged_modes_indices    
        if self.state[7]: # ortho
            out_dict['self.ortho_mode_shapes'] = self.ortho_mode_shapes
            out_dict['self.ortho_chan_dofs'] = self.ortho_chan_dofs
            out_dict['self.ortho_modes_indices'] = self.ortho_modes_indices
            out_dict['self.ortho_num_channels'] = self.ortho_num_channels
        
        np.savez(folder+'ssi_state.npz', **out_dict)
        
    def load_state(self, folder):
        print('Now loading previous results from  {}'.format(folder))
        
        in_dict=np.load(folder+'ssi_state.npz')    
        #             0         1           2           3          4            5            6             7
        #self.state= [Toeplitz, State Mat., Modal Par., Select M., Merge PoSER, Merge PoGer, Merge Preger, Ortho]
        if 'self.state' in in_dict:
            self.state= list(in_dict['self.state'])
        else:
            return
        
        for state, state_string in zip(self.state, ['Covariance Matrices Built',
                                                    'State Matrices Computed',
                                                    'Modal Parameters Computed',
                                                    'Modes Selected', 
                                                    'Modes Merged (PoSER)', 
                                                    'Covariances Merged (PoGer)', 
                                                    'Covariances Merged (Preger)', 
                                                    'Modes Orthogonalized']):
            if state: print(state_string)
        
        self.measurement = in_dict['self.measurement']
        self.sampling_rate = float(in_dict['self.sampling_rate'])
        self.total_time_steps = int(in_dict['self.total_time_steps'])
        self.ref_channels = list(in_dict['self.ref_channels'])
        self.roving_channels = list(in_dict['self.roving_channels'])
        self.num_ref_channels = len(self.ref_channels)
        self.num_roving_channels = len(self.roving_channels)
        self.num_analised_channels = self.num_ref_channels + self.num_roving_channels
        self.quantities = list(in_dict['self.quantities'])
        self.chan_dofs = [[int(chan_dof[0]), int(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3])] for chan_dof in in_dict['self.chan_dofs']]
        if self.state[0]:# covariances
            self.toeplitz_matrix = in_dict['self.toeplitz_matrix']
            self.num_block_columns = int(in_dict['self.num_block_columns'])
            self.num_block_rows = int(in_dict['self.num_block_rows'])
        if self.state[5] or self.state[6]: #merged poger,preger
            self.merged_toeplitz_matrix = in_dict['self.merged_toeplitz_matrix']
            self.merged_chan_dofs = [[int(chan_dof[0]), int(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3])] for chan_dof in in_dict['self.merged_chan_dofs']]
            self.merged_num_channels = in_dict['self.merged_num_channels']
            self.merged_quantities = list(in_dict['self.merged_quantities'])
        if self.state[5]:
            self.merged_num_channels_multiref = in_dict['self.merged_num_channels_multiref']
            if 'self.rescale_references' in in_dict:
                rescale_references = list(in_dict['self.rescale_references'])
                self.rescale_ref_channels = []
                self.rescale_rov_channels = []
                for channel_pairs in rescale_references:
                    self.rescale_ref_channels.append([pair for pair in rescale_references if pair[0] is not None])
                    self.rescale_rov_channels.append([pair for pair in rescale_references if pair[0] is None])
            else:
                self.rescale_ref_channels = list(in_dict['self.rescale_ref_channels'].astype(np.int))
                self.rescale_rov_channels = list(in_dict['self.rescale_rov_channels'].astype(np.int))
        if self.state[1]:# state models
            self.max_model_order = int(in_dict['self.max_model_order'])
            self.state_matrix= in_dict['self.state_matrix']
            self.output_matrix = in_dict['self.output_matrix']
        if self.state[2]:# modal params
            self.modal_frequencies = in_dict['self.modal_frequencies']
            self.modal_damping = in_dict['self.modal_damping']
            self.mode_shapes = in_dict['self.mode_shapes']
        if self.state[2] and (self.state[5] or self.state[6]): # merged poger,preger and modal params
            self.merged_mode_shapes = in_dict['self.merged_mode_shapes']
        if self.state[3] and not (self.state[5] or self.state[6]):# selected but not merged (poger or preger)
            self.selected_modes_indices = list(in_dict['self.selected_modes_indices'])
        if self.state[4]: #merged (poser)
            self.merged_mode_shapes = in_dict['self.merged_mode_shapes']
            self.merged_modes_indices = in_dict['self.merged_modes_indices']
            self.merged_num_channels = in_dict['self.merged_num_channels']
            self.merged_chan_dofs = [[int(chan_dof[0]), int(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3])] for chan_dof in in_dict['self.merged_chan_dofs']]
            self.mean_frequencies = in_dict['self.mean_frequencies']
            self.std_frequencies = in_dict['self.std_frequencies']
            self.mean_damping = in_dict['self.mean_damping']
            self.std_damping = in_dict['self.std_damping']
        if self.state[3] and (self.state[5] or self.state[6]): #selected and merged poger, preger  
            self.merged_mode_shapes = in_dict['self.merged_mode_shapes']
            self.merged_modes_indices  = list(in_dict['self.merged_modes_indices'])   
        if self.state[7]: # ortho
            self.ortho_mode_shapes = in_dict['self.ortho_mode_shapes']
            self.ortho_chan_dofs = [[int(chan_dof[0]), int(chan_dof[1]), float(chan_dof[2]), float(chan_dof[3])] for chan_dof in in_dict['self.ortho_chan_dofs']]
            self.ortho_modes_indices = in_dict['self.ortho_modes_indices']
            self.ortho_num_channels = in_dict['self.ortho_num_channels']
        
        
    def save_results(self, result_folder, which = 'all'):
        assert which in ['all', 'selected', 'merged', 'ortho']
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)
            
        frequency_file = result_folder + 'frequencies.npy'
        damping_file = result_folder + 'damping.npy'
        mode_shape_file = result_folder + 'mode_shapes.npy'
        
        if which == 'all':
            orders, modes = [], []
            for order in range(self.max_model_order):
                for mode in range(self.max_model_order):
                    orders.append(order)
                    modes.append(mode)
            mode_shapes = self.mode_shapes[:, modes, orders]
        elif which == 'selected':
            orders = [i[0] for i in self.selected_modes_indices]
            modes = [i[1] for i in self.selected_modes_indices]
            mode_shapes = self.mode_shapes[:, modes, orders]
        elif which == 'merged':
            orders = [i[0] for i in self.merged_modes_indices]
            modes = [i[1] for i in self.merged_modes_indices]
            mode_shapes = self.merged_mode_shapes[:, modes, orders]
        elif which == 'ortho':
            orders = [i[0] for i in self.ortho_modes_indices]
            modes = [i[1] for i in self.ortho_modes_indices]
            mode_shapes = self.ortho_mode_shapes[:, modes, orders]
        if which in ['merged', 'ortho'] and self.state[4]: 
            frequencies = self.mean_frequencies[orders, modes]
            damping = self.mean_damping[orders, modes]       
        else:
            frequencies = self.modal_frequencies[orders, modes]
            damping = self.modal_damping[orders, modes]                       
        
        assert mode_shapes.shape[1] == len(modes)
        
        mpcs = []
        mps = []
        mpds = []
        for col in range(mode_shapes.shape[1]):
            mode_shape = mode_shapes[:,col]
            mode_shapes[:,col] = self.rescale_mode_shape(mode_shape)
            mpcs.append(StabilPlot.calculateMPC_sing(mode_shapes[:,col]))
            [mpd], [mp] = StabilPlot.calculateMPD(mode_shapes[:,col, np.newaxis])
            
            mps.append(mp)
            mpds.append(mpd)
        mpcs = np.array(mpcs)
        mps = np.array(mps)
        mpds = np.array(mpds)
        orders = np.array(orders)
        
        #reorder everything by ascending frequency
        inds = frequencies.argsort()
        frequencies = frequencies[inds]
        damping = damping[inds]
        orders = orders[inds]
        mpcs= mpcs[inds]
        mps=mps[inds]
        mpds=mpds[inds]
        mode_shapes = mode_shapes[:,inds]
        
        freq_str, damp_str, ord_str, msh_str, mpc_str, mp_str, mpd_str = '', '', '', '', '', '', ''
        
        for col in range(len(orders)):
            freq_str += '{:3.3f} \t\t'.format(frequencies[col])
            damp_str += '{:3.3f} \t\t'.format(damping[col])
            ord_str += '{:3d} \t\t'.format(orders[col])
            mpc_str += '{:3.3f}\t \t'.format(mpcs[col])
            mp_str += '{:3.2f} \t\t'.format(mps[col])
            mpd_str += '{:3.2f} \t\t'.format(mpds[col])
            
        for row in range(mode_shapes.shape[0]):
            msh_str+='\n           \t\t'
            for col in range(mode_shapes.shape[1]):
                msh_str+='{:+3.4f} \t'.format(mode_shapes[row,col])       
        
        export_modes = 'MANUAL MODAL ANALYSIS\n'\
                      + '=======================\n'\
                      + 'Frequencies [Hz]:\t'         + freq_str       + '\n'\
                      + 'Damping [%]:\t\t'            + damp_str       + '\n'\
                      + 'Mode shapes:\t\t'            + msh_str        + '\n'\
                      + 'Model order:\t\t'            + ord_str        + '\n'\
                      + 'MPC [-]:\t\t'                + mpc_str        + '\n'\
                      + 'MP  [\u00b0]:\t\t'           + mp_str         + '\n'\
                      + 'MPD [-]:\t\t'                + mpd_str        + '\n\n'\
                      + 'SSI parameters\n'\
                      + '=======================\n'\
                      + 'Maximum order :\t\t'     + str(self.max_model_order) + '\n'\
                      + 'Block rows :\t\t'        + str(self.num_block_rows)     + '\n'\
                      + 'Block columns :\t\t'     + str(self.num_block_columns)  + '\n'
        #              + 'Decimation :\t\t'        + str(dec_fact)       + '\n'\
        #              + 'Filtering :\t\t'         + str(filt_w)
        
        f = open(result_folder + 'modal_info.txt', 'w')          
        f.write(export_modes)
        f.close()
        
        np.save(frequency_file, frequencies)
        np.save(damping_file, damping)
        np.save(mode_shape_file, mode_shapes)
        
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape
    
def get_measurement_names(series_num):
    # main function was confusing with all these lists in the beginning
    
    measurement_names=[]
    measurement_names.append(['measurement_1'])
    measurement_names.append(['measurement_2',
                              'measurement_3',
                              'measurement_4',
                              'measurement_5',
                              'measurement_6',
                              'measurement_7',
                              'measurement_8',
                              'measurement_9',
                              'measurement_10',
                              'measurement_11',
                              'measurement_12',
                              'measurement_13',
                              'measurement_14',
                              'measurement_15',
                              'measurement_16'
                              ])
    measurement_names.append(['measurement_17',
                              'measurement_18',
                              'measurement_19',
                              'measurement_20',
                              'measurement_21',
                              'measurement_22',
                              'measurement_23',
                              'measurement_24',
                              'measurement_25',
                              'measurement_26',
                              'measurement_27',
                              'measurement_28',
                              'measurement_29',
                              'measurement_30',
                              'measurement_31',
                              ])    
    measurement_names.append(['measurement_32',
                              'measurement_33',
                              'measurement_34',
                              'measurement_35',
                              'measurement_36',
                              'measurement_37',
                              'measurement_38',
                              'measurement_39', 
                              'measurement_40', 
                              'measurement_41', 
                              'measurement_42', 
                              'measurement_43', 
                              'measurement_44', 
                              'measurement_45', 
                              'measurement_46',
                              ])
    measurement_names.append(['measurement_47', 
                              'measurement_48', 
                              'measurement_49', 
                              'measurement_50', 
                              'measurement_51', 
                              'measurement_52', 
                              'measurement_53', 
                              'measurement_54', 
                              'measurement_55', 
                              'measurement_56', 
                              'measurement_57', 
                              'measurement_58', 
                              'measurement_59', 
                              'measurement_60', 
                              'measurement_61'
                              ])
    measurement_names.append(['measurement_62',
                              'measurement_63',
                              'measurement_64',
                              'measurement_65',
                              'measurement_66',
                              'measurement_67', 
                              'measurement_68', 
                              'measurement_69',
                              'measurement_70',
                              'measurement_71', 
                              'measurement_72', 
                              'measurement_73', 
                              'measurement_74', 
                              'measurement_75', 
                              'measurement_76', 
                              'measurement_77', 
                              'measurement_78', 
                              'measurement_79', 
                              'measurement_80', 
                              'measurement_81',
                              'measurement_82',
                              'measurement_83'
                              ])
    return measurement_names[series_num]
    
def main(series_num, sim=True):
    
    project_path = os.path.expanduser('~/Dropbox/simon_master/masterarbeit/') 
    
    meas=not sim
    if sim:
        measurement_series=['sim_i_full/', # 0
                            'sim_ii_pre-conv-(singl)-refacc/', # 1
                            'sim_iii_post-conv-singl-refacc/', # 2
                            'sim_iv_post-conv-distr-refacc/', # 3
                            'sim_v_post-conv-distr-refvib/', # 4
                            'sim_vi_post-conv-distr-refself/'] # 5
    if meas:
        measurement_series=['meas_i_full/', # 0 does not exist
                            'meas_ii_pre-conv-(singl)-refacc/', # 1
                            'meas_iii_post-conv-singl-refacc/', # 2
                            'meas_iv_post-conv-distr-refacc/', # 3
                            'meas_v_post-conv-distr-refvib/', # 4
                            'meas_vi_post-conv-distr-refself/'] # 5
                            
    
    measurement_path = project_path + 'messungen_simulationen/' + measurement_series[series_num]
    
    chan_dofs = json.load(open(measurement_path+'channel_dof_assignment.txt'))
    ssi_objects = []
    
    measurement_names = get_measurement_names(series_num)
    
    for measurement_name in measurement_names:

        print(measurement_name)
        
        result_folder = measurement_path + measurement_name  + '/'
        
        
        if os.path.exists(result_folder+'ssi_state.npz'):
            print('Loading previous results...')
            ssi_object = dummy_object()
            ssi_object.load_state(result_folder)
            ssi_object.clear_chan_dof_assignments()
            ssi_object.add_chan_dof_assignment(chan_dofs[measurement_name])        
            ssi_object.save_state(result_folder)
            ssi_objects.append(ssi_object)
            continue
        
        measurement_file = measurement_path + measurement_name + '.asc'
        
        sampling_rate = 512     # initial sampling rate in [Hz] 
        
        decimate_factor =2     # decimation factor, zero for no decimation
        from time import sleep

        measurement = np.loadtxt(measurement_file)#, usecols=[39,40,41, 42,43,44, 45,46,47])
        if meas:
            if series_num <= 3:
                measurement = np.loadtxt(measurement_file, usecols=[0,1,2,3,4])
            if series_num == 4:
                measurement = np.loadtxt(measurement_file, usecols=[0,1,2,5])
            if series_num == 5:    
                measurement = np.loadtxt(measurement_file, usecols=[0,1,2])
            if series_num == 6:    
                measurement = np.loadtxt(measurement_file, usecols=[3,4,5])
        assert measurement.shape [0] > measurement.shape [1]
        num_channels = measurement.shape[1]
        
        if decimate_factor > 1:
            measurement = decimate_data(measurement, decimate_factor)
            sampling_rate = sampling_rate / decimate_factor

        total_time_steps = measurement.shape[0]
        
        measurement = correct_offset(measurement)    

        if series_num==0:#'sim_i_full'
            ref_channels = [45, 46,47]   
            roving_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                39, 40, 41, 42, 43, 44] 
            units = ['v', 'v', 'v','v', 'v', 'v','v', 'v', 'v','v', 'v', 'v',
                     'v', 'v', 'v','v', 'v', 'v','v', 'v', 'v','v', 'v', 'v',
                     'v', 'v', 'v','v', 'v', 'v','v', 'v', 'v','v', 'v', 'v',
                     'v', 'v', 'v','v', 'v', 'v','v', 'v', 'v', 'a', 'a', 'a']
        elif series_num == 1:#'sim_ii_pre-conv-(singl)-refacc'
            roving_channels = [0, 1, 2]
            if sim:
                ref_channels = [3, 4, 5]
                units = ['v', 'v', 'v', 'a', 'a', 'a']
            if meas: 
                ref_channels = [3, 4]#, 5]
                units = ['v', 'v', 'v', 'a', 'a']#, 'a']
        elif series_num == 2:#'sim_iii_post-conv-singl-refacc'
            roving_channels = [0, 1, 2]
            if sim:
                ref_channels = [3, 4, 5]
                units = ['v', 'v', 'v', 'a', 'a', 'a']
            if meas:
                ref_channels = [3, 4]
                units = ['v', 'v', 'v', 'a', 'a']
        elif series_num == 3:#'sim_iv_post-conv-distr-refacc'
            roving_channels = [0, 1, 2]
            if sim:
                ref_channels = [3, 4, 5]
                units = ['v', 'v', 'v', 'a', 'a', 'a']
            if meas:
                ref_channels = [3, 4]
                units = ['v', 'v', 'v', 'a', 'a']
        elif series_num == 4:#'sim_v_post-conv-distr-refvib'
            roving_channels = [0, 1, 2]
            ref_channels = [3]
            units = ['v', 'v', 'v', 'v']
        elif series_num == 5:# 'sim_vi_post-conv-distr-refself'
            refs={
              'measurement_62':[0,1,2],
              'measurement_63':[2],
              'measurement_64':[0],
              'measurement_65':[1],
              'measurement_66':[2],
              'measurement_67':[0], 
              'measurement_68':[1], 
              'measurement_69':[2],
              'measurement_70':[0],
              'measurement_71':[1], 
              'measurement_72':[2], 
              'measurement_73':[0], 
              'measurement_74':[1], 
              'measurement_75':[2], 
              'measurement_76':[0], 
              'measurement_77':[1], 
              'measurement_78':[2], 
              'measurement_79':[0], 
              'measurement_80':[1], 
              'measurement_81':[2],
              'measurement_82':[0],
              'measurement_83':[1]}
            ref_channels = list(refs[measurement_name])
            roving_channels = [i for i in range(3) if i not in ref_channels]
            
            units = ['v', 'v', 'v']

        ssi_object = SSICovRef(measurement, sampling_rate, total_time_steps, num_channels, 
                ref_channels, roving_channels, units)
        
        num_block_columns = 200
        num_block_rows = num_block_columns
        
        ssi_object.build_toeplitz_cov(num_block_columns, num_block_rows, multiprocess=True)
        
        ssi_object.add_chan_dof_assignment(chan_dofs[measurement_name])        
        
        max_model_order = 200
    
        ssi_object.compute_state_matrices(max_model_order)
        
        ssi_object.compute_modal_params()
        
        ssi_object.save_state(result_folder)
        
            
        ssi_objects.append(ssi_object)
    if  series_num==0:
        return
    base_object = ssi_objects[0]
    
    if series_num !=5:
        
        base_object.save_state(measurement_path + 'merged_preger'  + '/')
        
    
        result_folder = measurement_path + 'merged_poger'  + '/'  
        
        base_object.merge_poger(ssi_objects[1:])
    
        base_object.compute_state_matrices(200)
    
        base_object.compute_modal_params()
        
        base_object.save_state(result_folder)
        
        
        result_folder = measurement_path + 'merged_preger'  + '/'     
        base_object = dummy_object()
        
        base_object.load_state(result_folder)
        
        base_object.merge_preger(ssi_objects[1:])
    
        base_object.compute_state_matrices(200)
    
        base_object.compute_modal_params()
        
        base_object.save_state(result_folder)
        
    elif series_num == 5:
    
        base_object.save_state(measurement_path + 'merged_preger'  + '/')
        base_object.save_state(measurement_path + 'merged_poger'  + '/')
        result_folder = measurement_path + 'merged_poger'  + '/'
                 
        for i,b in enumerate([[1,4,7,10,13,16,19],#meas: r ; sim: t
                              [2,5,8,11,14,17,20],#meas: t ; sim: r
                              [3,6,9,12,15,18,21]]):#meas: l ; sim: l
             
            object_list = [ssi_objects[i] for i in b]
            i_result_folder = measurement_path + 'merged_poger_{}'.format(i)  + '/'
            
            base_object.merge_poger(object_list)
     
            base_object.compute_state_matrices(200)
     
            base_object.compute_modal_params()
     
            base_object.save_state(i_result_folder)
             
            base_object = dummy_object()
            base_object.load_state(result_folder) 
        
        result_folder = measurement_path + 'merged_preger'  + '/'    
        
        base_object = dummy_object()
        base_object.load_state(result_folder)  
        
        for i,b in enumerate([[1,4,7,10,13,16,19],
                              [2,5,8,11,14,17,20],
                              [3,6,9,12,15,18,21]]):     
            
            object_list = [ssi_objects[i] for i in b]       
            i_result_folder = measurement_path + 'merged_preger_{}'.format(i)  + '/'  
            
            base_object.merge_preger(object_list)
            base_object.compute_state_matrices(200)
            base_object.compute_modal_params()      
            base_object.save_state(i_result_folder)
            
            base_object = dummy_object()
            base_object.load_state(result_folder) 


def main_results(series_num, sim=True):
    project_path = os.path.expanduser('~/Documents/Uni/masterarbeit/') 
    
    meas=not sim
    
    if sim:
        measurement_series=['sim_i_full/',
                        'sim_ii_pre-conv-(singl)-refacc/',
                        'sim_iii_post-conv-singl-refacc/',
                        'sim_iv_post-conv-distr-refacc/',
                        'sim_v_post-conv-distr-refvib/',
                        'sim_vi_post-conv-distr-refself/']
    if meas:
        measurement_series=['meas_i_full/', # 0 does not exist
                        'meas_ii_pre-conv-(singl)-refacc/', # 1
                        'meas_iii_post-conv-singl-refacc/', # 2
                        'meas_iv_post-conv-distr-refacc/', # 3
                        'meas_v_post-conv-distr-refvib/', # 4
                        'meas_vi_post-conv-distr-refself/'] # 5
    
    measurement_path = project_path + 'messungen_simulationen/' + measurement_series[series_num]
    
    ssi_objects=[]
    stab_frequency = 0.01*100
    stab_damping = 0.05*100            
    stab_MAC = 0.02*100
    
    measurement_names = get_measurement_names(series_num)
    chan_dofs = json.load(open(measurement_path+'channel_dof_assignment.txt'))
    
    for measurement_name in measurement_names:
        
        print(measurement_name)
         
        result_folder = measurement_path + measurement_name  + '/'
        ssi_object = dummy_object()     
        if not os.path.exists(result_folder+'ssi_state.npz'): continue
        ssi_object.load_state(result_folder)

        if not ssi_object.state[3]:  
            ssi_object.select_modes(stab_frequency, stab_damping, stab_MAC, result_folder)
            ssi_object.save_results(result_folder, 'selected')
            ssi_object.save_state(result_folder)
             
        ssi_object.save_results(result_folder, 'selected')

        ssi_objects.append(ssi_object) 
    if series_num == 0:
        return
     
    base_number = np.argmax(
                        np.array(
                            [len(ssi_object.selected_modes_indices) for 
                                           ssi_object in ssi_objects]))
    if series_num == 5: base_number=0
       
    base_object = ssi_objects.pop(base_number)
       
    result_folder = measurement_path + 'merged_poser'  + '/'     
    base_object.merge_poser(ssi_objects)    
    which = 'merged'    
    base_object.save_results(result_folder, which = which)
    base_object.save_chan_dof_assignment(result_folder, which = which)    
    base_object.save_state(result_folder)
        
    result_folder = measurement_path + 'merged_poser_ortho'  + '/'    
    base_object.orthogonalize_dofs()
    which = 'ortho'    
    base_object.save_results(result_folder, which = which)
    base_object.save_chan_dof_assignment(result_folder, which = which)    
    base_object.save_state(result_folder)
    

    if series_num != 5:
        
        result_folder = measurement_path + 'merged_poger'  + '/'   
        
        base_object = dummy_object()
        base_object.load_state(result_folder)
        
        if not base_object.state[3]:
            
            base_object.select_modes(stab_frequency, stab_damping, stab_MAC, result_folder)            
            which = 'merged'    
            base_object.save_results(result_folder, which = which)
            base_object.save_chan_dof_assignment(result_folder, which = which)    
            base_object.save_state(result_folder)
            
        result_folder = measurement_path + 'merged_poger_ortho'  + '/'   
        base_object.orthogonalize_dofs()
        which = 'ortho'    
        base_object.save_results(result_folder, which = which)
        base_object.save_chan_dof_assignment(result_folder, which = which)    
        base_object.save_state(result_folder)
                
        result_folder = measurement_path + 'merged_preger'  + '/'  
         
        base_object = dummy_object()
        base_object.load_state(result_folder)
        
        if not base_object.state[3]:
            base_object.select_modes(stab_frequency, stab_damping, stab_MAC, result_folder)
            which = 'merged'    
            base_object.save_results(result_folder, which = which)
            base_object.save_chan_dof_assignment(result_folder, which = which)    
            base_object.save_state(result_folder)
         
        result_folder = measurement_path + 'merged_preger_ortho'  + '/'   
        base_object.orthogonalize_dofs()
        which = 'ortho'    
        base_object.save_results(result_folder, which = which)
        base_object.save_chan_dof_assignment(result_folder, which = which)    
        base_object.save_state(result_folder)
        
    else:
        
        poger_objects = []
         
        for i in range(3):
            i_result_folder = measurement_path + 'merged_poger_{}'.format(i)  + '/'  
            base_object = dummy_object()
            base_object.load_state(i_result_folder) 
            
            if not base_object.state[3]:
                base_object.select_modes(stab_frequency, stab_damping, stab_MAC, i_result_folder)            
                base_object.save_state(i_result_folder)
                base_object.save_chan_dof_assignment(i_result_folder, which = 'merged')
                base_object.save_results(i_result_folder, which='merged')
                
            poger_objects.append(deepcopy(base_object))
          
        base_object = poger_objects.pop(0)
        base_object.merge_poser(poger_objects)
        result_folder = measurement_path + 'merged_poger_poser'  + '/'     
          
        which = 'merged'
        base_object.save_results(result_folder, which = which)
        base_object.save_chan_dof_assignment(result_folder, which = which)
        base_object.save_state(result_folder)
          
        result_folder = measurement_path + 'merged_poger_poser_ortho'  + '/'
        base_object.orthogonalize_dofs()   
        which = 'ortho'
        base_object.save_chan_dof_assignment(result_folder, which = which)
        base_object.save_results(result_folder, which = which)
        base_object.save_state(result_folder)
        
        preger_objects = []
        
        for i in range(3):
            i_result_folder = measurement_path + 'merged_preger_{}'.format(i)  + '/'  
            base_object = dummy_object()
            base_object.load_state(i_result_folder) 
            
            if not base_object.state[3]:
                base_object.select_modes(stab_frequency, stab_damping, stab_MAC, i_result_folder)            
                base_object.save_state(i_result_folder)
                base_object.save_chan_dof_assignment(i_result_folder, which = 'merged')
                base_object.save_results(i_result_folder, which='merged')
                
            preger_objects.append(deepcopy(base_object))
         
        base_object = preger_objects.pop(0)
        base_object.merge_poser(preger_objects)
        result_folder = measurement_path + 'merged_preger_poser'  + '/'     
         
        which = 'merged'
        base_object.save_results(result_folder, which = which)
        base_object.save_chan_dof_assignment(result_folder, which = which)
        base_object.save_state(result_folder)
         
        result_folder = measurement_path + 'merged_preger_poser_ortho'  + '/' 
        base_object.orthogonalize_dofs()  
        which = 'ortho'
        base_object.save_chan_dof_assignment(result_folder, which = which)
        base_object.save_results(result_folder, which = which)
        base_object.save_state(result_folder)

          
if __name__ =='__main__':
    i=5
    sim=True
    #main(i, sim)
    main_results(i, sim)          
