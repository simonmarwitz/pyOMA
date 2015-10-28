# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''

import numpy as np
import sys
import os
import json

import multiprocessing as mp
import ctypes as c
from collections import OrderedDict, deque
from copy import deepcopy

from PreprocessingTools import PreprocessData
from StabilDiagram import main_stabil, StabilPlot, nearly_equal
from numpy.testing.utils import measure

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
    
    def __init__(self,prep_data):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data =prep_data
        #             0         1           2         
        #self.state= [Toeplitz, State Mat., Modal Par.
        self.state  =[False,    False,      False]
        
        self.num_block_columns = None
        self.num_block_rows = None
        self.toeplitz_matrix = None
        
        self.max_model_order = None
        self.state_matrix = None
        self.output_matrix = None
        
        self.modal_damping = None
        self.modal_frequencies = None
        self.mode_shapes = None
            
    @classmethod
    def init_from_config(cls,conf_file, prep_data):
        assert os.path.exists(conf_file)
        assert isinstance(prep_data, PreprocessData)
        
        with open(conf_file, 'r') as f:
            
            assert f.__next__().strip('\n').strip(' ') == 'Number of Block-Columns:'
            num_block_columns = int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Maximum Model Order:'
            max_model_order= int(f. __next__().strip('\n'))
            assert f.__next__().strip('\n').strip(' ')== 'Use Multiprocessing:'
            multiprocessing= f.__next__().strip('\n').strip(' ')=='yes'
            
        
        ssi_object = cls(prep_data)
        ssi_object.build_toeplitz_cov(num_block_columns, multiprocessing)
        ssi_object.compute_state_matrices(max_model_order)
        ssi_object.compute_modal_params(multiprocessing)

    def build_toeplitz_cov(self, num_block_columns, num_block_rows=None, multiprocess=True):
        '''
        Builds a Block-Toeplitz Matrix of Covariances with varying time lags
        
            |    R_i    R_i-1    ...    R_0    |
            |    R_i+1  R_i      ...    R_1    |
            |    ...    ...      ...    ...    |
            |    R_2i-1 ...      ...    R_i    |
        '''
        assert isinstance(num_block_columns, int)
        if num_block_rows is None:
            num_block_rows=num_block_columns
        assert isinstance(num_block_rows, int)
        
        self.num_block_columns=num_block_columns
        self.num_block_rows=num_block_rows
        total_time_steps = self.prep_data.total_time_steps
        ref_channels = sorted(self.prep_data.ref_channels)
        roving_channels = self.prep_data.roving_channels
        measurement = self.prep_data.measurement
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels 
        # Extract reference time series for covariances 
        extract_length = total_time_steps - (self.num_block_columns + self.num_block_rows) + 1
        
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        
        refs = (measurement[0:extract_length,ref_channels]).T
        
        ### Create Toeplitz matrix and fill it with covariances 
        #    |    R_i    R_i-1    ...    R_0    |                                     #
        #    |    R_i+1  R_i      ...    R_0    |                                     #
        #    |    ...    ...      ...    ...    |                                     #
        #    |    R_2i-1 ...      ...    R_i    |                                     #
        
        print('Computing covariances...')
        n, m = num_analised_channels*self.num_block_rows, num_ref_channels*self.num_block_columns
 
        if multiprocess:
            toeplitz_memory = mp.Array(c.c_double, np.zeros(n*m)) # shared memory, can be used by multiple processes @UndefinedVariable
            toeplitz_shape = (n,m)
            measurement_shape=measurement.shape
            measurement_memory = mp.Array(c.c_double, measurement.reshape(measurement.size, 1))
            
            refs_memory = mp.Array(c.c_double, refs.reshape(refs.size,1 ))
            refs_shape=refs.shape
            
            pool=mp.Pool(initializer=self.init_child_process, initargs=(refs_memory,  measurement_memory, toeplitz_memory, ))
        
            
            for i in range(0,num_block_rows):
                if i == 0:
                    # in multiprocessing, errors in the single processes are sometimes not reported to console, in this case 
                    # the resulting block toeplitzmatrix will be filled with zeros
                    # try to change pool.apply_async to pool.apply and run in debug mode
                    for ii in range(0,self.num_block_columns):#fill the first block row
                        pool.apply_async(self.compute_covariance , args=(i, 
                                                            ii, 
                                                            num_block_columns, 
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
                                                            num_block_columns, 
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

            
            for i in range(1,num_block_rows): #finish assembling block toeplitz matrix
                # copys and shifts contents from previous block row to next block row
                # shifts by one block column to the right
                previous_Toeplitz_row = (i-1)*num_analised_channels
                this_block = Toeplitz_matrix[previous_Toeplitz_row:(previous_Toeplitz_row+num_analised_channels),
                                             0:(num_ref_channels * num_block_columns - num_ref_channels)]
                begin_Toeplitz_row = i*num_analised_channels
                Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                num_ref_channels:(num_ref_channels * num_block_columns)] = this_block
    
        else: # old single threaded way
            
            Toeplitz_matrix = np.zeros((n,m))

            for i in range(0,num_block_rows):
                if i == 0:
                    for ii in range(0,num_block_columns):
                        
                        begin_extract = num_block_columns + i - (ii)
                        current_signals = measurement[begin_extract : (begin_extract+extract_length), all_channels].T
                        covariances = np.cov(refs,current_signals)
                        this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                        
                        begin_Toeplitz_row = i*num_analised_channels
                        
                        Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                        ii*num_ref_channels:(ii*num_ref_channels+num_ref_channels)] = this_block
                else: 
                    previous_Toeplitz_row = (i-1)*num_analised_channels
                    this_block = Toeplitz_matrix[previous_Toeplitz_row:(previous_Toeplitz_row+num_analised_channels),
                                                  0:num_ref_channels * (num_block_columns-1)]
                    begin_Toeplitz_row = i*num_analised_channels
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                     num_ref_channels:(num_ref_channels * num_block_columns)] = this_block
                     
                    begin_extract = num_block_columns + i
                    current_signals = (measurement[begin_extract:(begin_extract + extract_length),all_channels]).T
                    covariances = np.cov(refs,current_signals)
                    this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                     
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                     0:num_ref_channels] = this_block
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
    
    @staticmethod    
    def compute_covariance(i,ii, num_block_columns, extract_length, ref_channels, all_channels, refs_shape, measurement_shape, toeplitz_shape):
        
        num_analised_channels = len(all_channels)
        num_ref_channels = len(ref_channels)
        
        measurement = np.frombuffer(measurement_memory.get_obj()).reshape(measurement_shape)
        refs = np.frombuffer(refs_memory.get_obj()).reshape(refs_shape)
        toeplitz = np.frombuffer(toeplitz_memory.get_obj()).reshape(toeplitz_shape)
            
        begin_extract = num_block_columns + i - (ii)

        current_signals = (measurement[begin_extract:(begin_extract + extract_length), all_channels]).T

        this_block = (np.dot(refs, current_signals.T.conj()) / refs_shape[1]).T
        
        
        begin_toeplitz_row = i*num_analised_channels 
        end_toeplitz_row = begin_toeplitz_row+num_analised_channels 
        begin_toeplitz_col = ii*num_ref_channels
        end_toeplitz_col = begin_toeplitz_col + num_ref_channels
        
        with toeplitz_memory.get_lock():
    
            toeplitz[begin_toeplitz_row: end_toeplitz_row,
                            begin_toeplitz_col:end_toeplitz_col] = this_block
          
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
        

        toeplitz_matrix = self.toeplitz_matrix
        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        print('Computing state matrices...')
        
        [U,S,V_T] = np.linalg.svd(toeplitz_matrix,0)
        S_2 = np.diag(np.power(S[:self.max_model_order], 0.5))
        U = U[:,:max_model_order]
        Oi = np.dot(U, S_2)
        C = Oi[:num_channels,:]   
        
        A = np.dot(np.linalg.pinv(Oi[:(num_channels * (num_block_columns - 1)),:]),
                   Oi[num_channels:(num_channels * num_block_columns),:])
       
        self.state_matrix = A
        self.output_matrix = C
        
        self.state[1]=True
        self.state[2] = False # previous modal params are invalid now

    def compute_modal_params(self, multiprocessing=True): 
        
        assert self.state[1]
        
        print('Computing modal parameters...')
        max_model_order = self.max_model_order
        num_analised_channels = self.prep_data.num_analised_channels
        state_matrix = self.state_matrix
        output_matrix = self.output_matrix
        sampling_rate = self.prep_data.sampling_rate
        accel_channels = self.prep_data.accel_channels
        velo_channels = self.prep_data.velo_channels
        
        modal_frequencies = np.zeros((max_model_order, max_model_order))        
        modal_damping = np.zeros((max_model_order, max_model_order))        
        mode_shapes = np.zeros((num_analised_channels, max_model_order, max_model_order),dtype=complex)
            
        if multiprocessing:
            manager=mp.Manager()        
            pool = mp.Pool()       
            return_dict=manager.dict()
            
            #balanced allocation of work (numpy.linalg.eig takes approx. n^3 operations)
            work_slice_size = sum([n**3 for n in range(max_model_order)])/max_model_order
            current_size = 0
            current_orders = []
            for order in range(0,max_model_order,1):
                current_orders.append(order)
                current_size += order**3
                if current_size >= work_slice_size:
                    pool.apply_async(self.multiprocess_evd , args=(state_matrix, current_orders, return_dict))
                    current_orders = []
                    current_size = 0
            pool.close()
            pool.join()

        for order in range(0,max_model_order,1):    
            
            if multiprocessing:
                eigenvalues_single, eigenvectors_single = return_dict[order]
            else:
                eigenvalues_paired, eigenvectors_paired = np.linalg.eig(state_matrix[0:order+1, 0:order+1])
    
                eigenvectors_single,eigenvalues_single = \
                    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
                
            for index,k in enumerate(eigenvalues_single): 
                lambda_k = np.log(complex(k)) * sampling_rate
                freq_j = np.abs(lambda_k) / (2*np.pi)        
                damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100)  
                mode_shapes_j = np.dot(output_matrix[:, 0:order + 1], eigenvectors_single[:,index])
            
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))                
                        
                modal_frequencies[order,index]=freq_j
                modal_damping[order,index]=damping_j
                mode_shapes[:,index,order]=mode_shapes_j
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
            
        self.state[2]=True
        
    @staticmethod
    def multiprocess_evd(a, truncation_orders, return_dict):
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
    

    
    @staticmethod
    def integrate_quantities(vector, accel_channels, velo_channels, omega):
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
        #                     phase + 180; magn / omega^2
        vector[accel_channels] *= -1       / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[velo_channels] *=  1j        / omega
        
        return vector   
    
    

        
    def save_state(self, folder):
        
        ######### continue here
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
        
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape
    
def main():
    pass

if __name__ =='__main__':
    main()