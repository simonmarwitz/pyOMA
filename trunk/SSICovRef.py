# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''

import numpy as np
#import sys
import os
#import json

import multiprocessing as mp
import ctypes as c
from collections import deque
#from copy import deepcopy

from PreprocessingTools import PreprocessData
#from StabilDiagram import main_stabil, StabilPlot, nearly_equal

#import pydevd
'''
TODO:
- change channels numbers such, that user input channels start at 1 while internally they start at 0
    affects: ref_channels, roving_channels and channel-dof-assignments
- generally define unit tests to check functionality after changes
- 

'''
    
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
        extract_length = total_time_steps - (num_block_columns + num_block_rows) + 1
        
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        
        refs = (measurement[0:extract_length,ref_channels]).T
        
        ### Create Toeplitz matrix and fill it with covariances 
        #    |    R_i    R_i-1    ...    R_0    |                                     #
        #    |    R_i+1  R_i      ...    R_0    |                                     #
        #    |    ...    ...      ...    ...    |                                     #
        #    |    R_2i-1 ...      ...    R_i    |                                     #
        
        print('Computing covariances...')
        n, m = num_analised_channels*num_block_rows, num_ref_channels*num_block_columns
 
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
    
   
    def compute_covariance(self, i,ii, num_block_columns, extract_length, ref_channels, all_channels, refs_shape, measurement_shape, toeplitz_shape):
        
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
        
        #b_accel_channels = np.array([False for i in range(num_analised_channels)])
        accel_channels=self.prep_data.accel_channels
        #b_velo_channels = np.array([False for i in range(num_analised_channels)])
        velo_channels=self.prep_data.velo_channels
        #accel_channels = self.prep_data.accel_channels
        #velo_channels = self.prep_data.velo_channels
        
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
    
    @staticmethod
    def integrate_quantities(vector, accel_channels, velo_channels, omega):
        # input quantities = [a, v, d]
        # output quantities = [d, d, d]
        # converts amplitude and phase
        #                     phase + 180; magn / omega^2
        
        vector[accel_channels] *= -1       / (omega ** 2)
        #                    phase + 90; magn / omega
        vector[velo_channels] *=  1j        / omega
        
        return vector   
    
    def save_state(self, fname):
        
        dirname, filename = os.path.split(fname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            
        #             0         1           2           
        #self.state= [Toeplitz, State Mat., Modal Par.]
        out_dict={'self.state':self.state}
        out_dict['self.prep_data']=self.prep_data
        if self.state[0]:# covariances
            out_dict['self.toeplitz_matrix'] = self.toeplitz_matrix
            out_dict['self.num_block_columns'] = self.num_block_columns
            out_dict['self.num_block_rows'] = self.num_block_rows
        if self.state[1]:# state models
            out_dict['self.max_model_order'] = self.max_model_order
            out_dict['self.state_matrix'] = self.state_matrix
            out_dict['self.output_matrix'] = self.output_matrix
        if self.state[2]:# modal params
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            
        np.savez(fname, **out_dict)
        
    @classmethod 
    def load_state(cls, fname):
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname)    
        #             0         1           2          
        #self.state= [Toeplitz, State Mat., Modal Par.]
        if 'self.state' in in_dict:
            state= list(in_dict['self.state'])
        else:
            return
        
        for this_state, state_string in zip(state, ['Covariance Matrices Built',
                                                    'State Matrices Computed',
                                                    'Modal Parameters Computed',
                                                    ]):
            if this_state: print(state_string)
        
        prep_data = in_dict['self.prep_data'].item()
        ssi_object = cls(prep_data)
        ssi_object.state = state
        if state[0]:# covariances
            ssi_object.toeplitz_matrix = in_dict['self.toeplitz_matrix']
            ssi_object.num_block_columns = int(in_dict['self.num_block_columns'])
            ssi_object.num_block_rows = int(in_dict['self.num_block_rows'])
        if state[1]:# state models
            ssi_object.max_model_order = int(in_dict['self.max_model_order'])
            ssi_object.state_matrix= in_dict['self.state_matrix']
            ssi_object.output_matrix = in_dict['self.output_matrix']
        if state[2]:# modal params
            ssi_object.modal_frequencies = in_dict['self.modal_frequencies']
            ssi_object.modal_damping = in_dict['self.modal_damping']
            ssi_object.mode_shapes = in_dict['self.mode_shapes']
        
        return ssi_object
    
    @staticmethod
    def rescale_mode_shape(modeshape):
        #scaling of mode shape
        modeshape = modeshape / modeshape[np.argmax(np.abs(modeshape))]
        return modeshape
    
def main():
    pass

if __name__ =='__main__':
    main()