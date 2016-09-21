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
import datetime
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
    
class BRSSICovRef(object):
    
    def __init__(self,prep_data):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data =prep_data
        self.setup_name = prep_data.setup_name
        self.start_time = prep_data.start_time
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
        ssi_object.build_toeplitz_cov(num_block_columns, multiprocess=multiprocessing)
        ssi_object.compute_state_matrices(max_model_order)
        ssi_object.compute_modal_params(multiprocessing)
        
        return ssi_object
        
    def build_toeplitz_cov(self, num_block_columns, num_block_rows=None, multiprocess=True):
        '''
        Builds a Block-Toeplitz Matrix of Covariances with varying time lags
            | <- num_block_columns*num_ref_channels-> |_
            |     R_i      R_i-1      ...      R_1    |^
            |     R_i+1    R_i        ...      R_2    |num_block_rows*num_analised_channels
            |     ...      ...        ...      ...    |v
            |     R_2i-1   ...        ...      R_i    |_
        '''
        #print(multiprocess)
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
        extract_length = int(total_time_steps - (num_block_columns + num_block_rows) + 1)
        
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        #print(all_channels)
        
        refs = (measurement[0:extract_length,ref_channels]).T
        
        ### Create Toeplitz matrix and fill it with covariances 
        #    |    R_i    R_i-1    ...    R_0    |                                     #
        #    |    R_i+1  R_i      ...    R_1    |                                     #
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
                        this_block = (np.dot(refs, current_signals.T.conj()) / refs.shape[1]).T/extract_length
                        #print(this_block)
                        #covariances = np.cov(refs,current_signals)
                        #this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                        
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
                    
                    this_block = (np.dot(refs, current_signals.T.conj()) / refs.shape[1]).T/extract_length
                    #print(this_block)
                    #covariances = np.cov(refs,current_signals)
                    #this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                     
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

        this_block = (np.dot(refs, current_signals.T.conj()) / refs_shape[1]).T/extract_length
        
        
        begin_toeplitz_row = i*num_analised_channels 
        end_toeplitz_row = begin_toeplitz_row+num_analised_channels 
        begin_toeplitz_col = ii*num_ref_channels
        end_toeplitz_col = begin_toeplitz_col + num_ref_channels
        
        with toeplitz_memory.get_lock():
    
            toeplitz[begin_toeplitz_row: end_toeplitz_row,
                            begin_toeplitz_col:end_toeplitz_col] = this_block
          
    def compute_state_matrices(self, max_model_order=None, max_modes=None):
        '''
        computes the state and output matrix of the state-space-model
        by applying a singular value decomposition to the block-toeplitz-matrix of covariances
        the state space model matrices are obtained by appropriate truncation 
        of the svd matrices at max_model_order
        the decision whether to take merged covariances is taken automatically
        max_modes is the number of expected modes for estimation of the state transition /
        system matrix A of the "crystal clear (TM)" algorithm of structural vibrations AS (Artemis)
        '''
        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
            
        assert self.state[0]
        

        toeplitz_matrix = self.toeplitz_matrix
        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        print('Computing state matrices...')
        
        [U,S,V_T] = np.linalg.svd(toeplitz_matrix,0)
        #print(U.shape, S.shape, V_T.shape)
    
        # choose highest possible model order
        if max_model_order is None:
            max_model_order=len(S)
        else:
            max_model_order = min(max_model_order,len(S))
    
        S_2 = np.diag(np.power(S[:max_model_order], 0.5))
        U = U[:,:max_model_order]
        Oi = np.dot(U, S_2)
        C = Oi[:num_channels,:]   
        
        Oi0=Oi[:(num_channels * (num_block_columns - 1)),:]
        Oi1=Oi[num_channels:(num_channels * num_block_columns),:]
        
        if max_modes is not None:
            [u,s,v_t]=np.linalg.svd(Oi0,0)
            s = 1./s[:max_modes]
            Oi0p= np.dot(np.transpose(v_t[:max_modes,:]), np.multiply(s[:, np.newaxis], np.transpose(u[:,:max_modes])))
        else:
            Oi0p = np.linalg.pinv(Oi0)
            
        A = np.dot(Oi0p,Oi1)
       
        self.state_matrix = A
        self.output_matrix = C
        self.max_model_order=max_model_order
        
        self.state[1]=True
        self.state[2] = False # previous modal params are invalid now
        

    def compute_modal_params(self, multiprocessing=True, max_model_order=None): 
        
        if max_model_order is not None:
            assert max_model_order<=self.max_model_order
            self.max_model_order=max_model_order
        
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
            
#         import matplotlib.pyplot as plot
#         fig=plot.figure()
#         ax1=fig.add_subplot(111)
#         fig=plot.figure()
#         ax2=fig.add_subplot(111)
        for order in range(0,max_model_order,1):    
            
            if multiprocessing:
                eigenvalues_single, eigenvectors_single = return_dict[order]
            else:
                eigenvalues_paired, eigenvectors_paired = np.linalg.eig(state_matrix[0:order+1, 0:order+1])
                
                eigenvectors_single,eigenvalues_single = \
                    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
#                 ax1.plot(eigenvalues_single.real,eigenvalues_single.imag, ls='', marker='o')
                
            lambdas=[]
            for index,k in enumerate(eigenvalues_single): 
                lambda_k = np.log(complex(k)) * sampling_rate
                lambdas.append(lambda_k)
                freq_j = np.abs(lambda_k) / (2*np.pi)        
                damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100)  
                mode_shapes_j = np.dot(output_matrix[:, 0:order + 1], eigenvectors_single[:,index])
            
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))                
                        
                modal_frequencies[order,index]=freq_j
                modal_damping[order,index]=damping_j
                mode_shapes[:,index,order]=mode_shapes_j
            lambdas = np.array(lambdas)
#             ax2.plot(lambdas.real,lambdas.imag, ls='', marker='o')
#         
#         ax1.axvline(x=0)
#         ax1.axhline(y=0)
#         ax2.axvline(x=0)
#         ax2.axhline(y=0)
#                 
#         xlim=np.max(np.abs(ax1.get_xlim()))
#         ax1.set_xlim((-xlim,xlim))
#         ylim=np.max(np.abs(ax1.get_ylim()))
#         ax1.set_ylim((-ylim,ylim))
#         
#         plot.show()
        
        self.modal_frequencies = modal_frequencies
        self.modal_damping = modal_damping
        self.mode_shapes = mode_shapes
            
        self.state[2]=True
        
    
    def multiprocess_evd(self, a, truncation_orders, return_dict):
        
        for truncation_order in truncation_orders:
            eigenvalues_paired, eigenvectors_paired = np.linalg.eig(a[0:truncation_order+1, 0:truncation_order+1])
    
            eigenvectors_single,eigenvalues_single = \
                    BRSSICovRef.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
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
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
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
            
        np.savez_compressed(fname, **out_dict)
        
    @classmethod 
    def load_state(cls, fname, prep_data):
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
        
        assert isinstance(prep_data, PreprocessData)
        setup_name= str(in_dict['self.setup_name'].item())
        start_time=in_dict['self.start_time'].item()
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time
        
        assert start_time == prep_data.start_time
        #prep_data = in_dict['self.prep_data'].item()
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



class CVASSICovRef(object):
    
    def __init__(self,prep_data):    
        '''
        channel definition: channels start at 0
        '''
        super().__init__()
        assert isinstance(prep_data, PreprocessData)
        self.prep_data =prep_data
        self.setup_name = prep_data.setup_name
        self.start_time = prep_data.start_time
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
        ssi_object.build_toeplitz_cov(num_block_columns, multiprocess=multiprocessing)
        ssi_object.compute_state_matrices(max_model_order)
        ssi_object.compute_modal_params(multiprocessing)
        
        return ssi_object
        
    def build_toeplitz_cov(self, num_block_columns, num_block_rows=None, multiprocess=True):
        '''
        Builds a Block-Toeplitz Matrix of Covariances with varying time lags
        
            |    R_i    R_i-1    ...    R_1    |
            |    R_i+1  R_i      ...    R_2    |
            |    ...    ...      ...    ...    |
            |    R_2i-1 ...      ...    R_i    |
        '''
        #print(multiprocess)
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
        
        refs = (measurement[0:extract_length,all_channels]).T
        
        ### Create Toeplitz matrix and fill it with covariances 
        #    |    R_i    R_i-1    ...    R_0    |                                     #
        #    |    R_i+1  R_i      ...    R_1    |                                     #
        #    |    ...    ...      ...    ...    |                                     #
        #    |    R_2i-1 ...      ...    R_i    |                                     #
        
        print('Computing covariances...')
        n, m = num_analised_channels*num_block_rows, num_analised_channels*num_block_columns
        
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
                                                            all_channels, 
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
                                                            all_channels, 
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
                                             0:(num_analised_channels * num_block_columns - num_analised_channels)]
                begin_Toeplitz_row = i*num_analised_channels
                Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                num_analised_channels:(num_analised_channels * num_block_columns)] = this_block
    
        else: # old single threaded way
            
            Toeplitz_matrix = np.zeros((n,m))

            for i in range(0,num_block_rows):
                if i == 0:
                    for ii in range(0,num_block_columns):
                        
                        begin_extract = num_block_columns + i - (ii)
                        current_signals = measurement[begin_extract : (begin_extract+extract_length), all_channels].T
                        this_block = (np.dot(refs, current_signals.T.conj()) / refs.shape[1]).T/extract_length
                        #print(this_block)
                        #covariances = np.cov(refs,current_signals)
                        #this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                        
                        begin_Toeplitz_row = i*num_analised_channels
                        
                        Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                        ii*num_analised_channels:(ii*num_analised_channels+num_analised_channels)] = this_block
                else: 
                    previous_Toeplitz_row = (i-1)*num_analised_channels
                    this_block = Toeplitz_matrix[previous_Toeplitz_row:(previous_Toeplitz_row+num_analised_channels),
                                                  0:num_analised_channels * (num_block_columns-1)]
                    begin_Toeplitz_row = i*num_analised_channels
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                     num_analised_channels:(num_analised_channels * num_block_columns)] = this_block
                     
                    begin_extract = num_block_columns + i
                    current_signals = (measurement[begin_extract:(begin_extract + extract_length),all_channels]).T
                    
                    this_block = (np.dot(refs, current_signals.T.conj()) / refs.shape[1]).T/extract_length
                    #print(this_block)
                    #covariances = np.cov(refs,current_signals)
                    #this_block = covariances[num_ref_channels:(num_ref_channels + num_analised_channels),:num_ref_channels]
                     
                    Toeplitz_matrix[begin_Toeplitz_row:(begin_Toeplitz_row+num_analised_channels),
                                     0:num_analised_channels] = this_block
        #print(Toeplitz_matrix.shape)
        Lplus,Lminus = self.compute_cva_weighting()
        Toeplitz_matrix = Lplus.solve_LD(Toeplitz_matrix.T)
        Toeplitz_matrix = Lminus.solve_DLt(Toeplitz_matrix)                             
        Toeplitz_matrix = Toeplitz_matrix.T    
        self.toeplitz_matrix = Toeplitz_matrix           
        #print(self.toeplitz_matrix.shape)   
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

        this_block = (np.dot(refs, current_signals.T.conj()) / refs_shape[1]).T/extract_length
        
        
        begin_toeplitz_row = i*num_analised_channels 
        end_toeplitz_row = begin_toeplitz_row+num_analised_channels 
        begin_toeplitz_col = ii*num_ref_channels
        end_toeplitz_col = begin_toeplitz_col + num_ref_channels
        
        with toeplitz_memory.get_lock():
    
            toeplitz[begin_toeplitz_row: end_toeplitz_row,
                            begin_toeplitz_col:end_toeplitz_col] = this_block

    def compute_cva_weighting(self):
        #try qr decomposition of yf
        # y0, y1, ..., yj-1
        # y1, y2, ..., yj
        # ...
        # yi, yi+1, ... yi+j-1
        # QR -> L = R.T       
        
        
        total_time_steps = self.prep_data.total_time_steps
        ref_channels = sorted(self.prep_data.ref_channels)
        roving_channels = self.prep_data.roving_channels
        measurement = self.prep_data.measurement
        num_analised_channels = self.prep_data.num_analised_channels
        num_ref_channels =self.prep_data.num_ref_channels
        
        extract_length = total_time_steps - (self.num_block_columns ) +1
        #print(extract_length, measurement.shape)        
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        #all_channels = [2,1,3,5,0]
        num_analised_channels=len(all_channels)
        
        extract_length = total_time_steps - 2*self.num_block_rows - self.num_block_columns
        Yf = np.zeros((self.num_block_rows/2*num_analised_channels,extract_length))
        Ypref = np.zeros((self.num_block_rows/2*num_ref_channels, extract_length))
        for i in range(self.num_block_rows/2):
            Ypref[i*num_ref_channels:(i+1)*num_ref_channels,:]=self.prep_data.measurement[i:extract_length+i, ref_channels].T
            Yf[i*num_analised_channels:(i+1)*num_analised_channels,:] = self.prep_data.measurement[i+self.num_block_rows/2:extract_length+i+self.num_block_rows/2].T   
        Yf/=np.sqrt(extract_length)
        Ypref/=np.sqrt(extract_length)
        
        U,S,V_T=np.linalg.svd(np.dot(Yf,Yf.T))
        W1=np.dot(U,np.diag(1/np.sqrt(S)))
        
        U,S,V_T=np.linalg.svd(np.dot(Ypref,Ypref.T))
        W1=np.dot(U,np.diag(1/np.sqrt(S)))
        
        #r=np.linalg.qr(Yf.T,'r')
        
        
        
        return r.T
    
        refs = (measurement[0:extract_length,all_channels])
        #refs -= np.mean(refs,axis=0)
        refs = refs.T
        #print(refs.shape)
        
        Tplus  = np.zeros([self.num_block_rows * num_analised_channels] * 2)
        Tminus = np.zeros([self.num_block_rows * num_analised_channels] * 2)
        

        
        for i in range(self.num_block_rows):
            
            current_signals = measurement[i : (i+extract_length), all_channels]
            #current_signals -= np.mean(current_signals,axis=0)
            #print(current_signals.shape)
            ri = (np.dot(refs, current_signals) 
                  / refs.shape[1]
                  /extract_length
                  )
            #print(ri)
            
            #rii = np.zeros_like(ri)
            #for k in range(ri.shape[0]):
            #    rii[k,k]=ri[k,k]
            #ri=rii
            
            assert ri.shape[0]==ri.shape[1]
            
            scol = i * num_analised_channels
            ecol = (i + 1) * num_analised_channels
            srow = 0
            erow = num_analised_channels
            
            #print('Putting at [{:04d}:{:04d},{:04d}:{:04d}]'.format(srow,erow,scol,ecol))
            #print((Tplus[srow:erow,scol:ecol] == 0).all())
            Tplus[srow:erow,scol:ecol] = ri.T
            Tminus[srow:erow,scol:ecol]= ri
            #print('Putting at [{:04d}:{:04d},{:04d}:{:04d}]'.format(scol,ecol,srow,erow))
            #print((Tplus[scol:ecol,srow:erow] == 0).all())
            Tplus[scol:ecol,srow:erow] = ri
            Tminus[scol:ecol,srow:erow]= ri.T
        
        
        for i in range(self.num_block_rows -1):
            
            scolo = 0
            ecolo = (self.num_block_rows -1) * num_analised_channels
            srowo = i * num_analised_channels
            erowo = (i + 1) * num_analised_channels
            
            scold = scolo + num_analised_channels
            ecold = ecolo + num_analised_channels
            srowd = srowo + num_analised_channels
            erowd = erowo + num_analised_channels
            #print('Shifting [{:04d}:{:04d},{:04d}:{:04d}] to [{:04d}:{:04d},{:04d}:{:04d}]'.format(srowo,erowo,scolo,ecolo,srowd,erowd,scold,ecold))
            #print((Tplus[srowd:erowd,scold:ecold] == 0).all())
            Tplus[srowd:erowd,scold:ecold] = Tplus[srowo:erowo,scolo:ecolo]
            Tminus[srowd:erowd,scold:ecold] = Tminus[srowo:erowo,scolo:ecolo]
        print(np.where(Tplus == 0))
        #import matplotlib.pyplot as plot  
#         plot.figure()
#         for channel in all_channels:
#             rowindices = [channel]*self.num_block_columns
#             colindices = [i*num_analised_channels+channel for i in range(self.num_block_columns)]
#             plot.plot(Tplus[(rowindices, colindices)])
#         plot.title('Tplus')
#         assert (Tplus.T == Tplus).all()
#         assert (Tminus.T == Tminus).all()
        #range_= Tplus.max()-Tplus.min()
        #Tplus /= range_
        #range_ = Tminus.max()-Tminus.min()
        #Tminus /= range_
#         print(Tplus.min(),Tplus.max(),np.linalg.cond(Tplus))
        #plot.matshow(Tplus)
        #plot.matshow(Tminus)
        #precondition matrix with the smallest possible value
        from sksparse.cholmod import cholesky   # @UnresolvedImport
        from scipy.sparse import csc_matrix
        try:
            #Lplus = np.linalg.cholesky(Tplus)#+1e-10*np.eye(*Tplus.shape))
            Tplus=csc_matrix(Tplus)
            Lplus = cholesky(Tplus)
            
            #Lplus = np.linalg.cholesky(Tplus)
        except np.linalg.linalg.LinAlgError:
            alpha = 1e-16
            while True:
                
                print('Now preconditioning weighting matrix 1 by {}*np.eye()'.format(alpha))
                try:
                    Lplus = np.linalg.cholesky(Tplus+alpha*np.eye(*Tplus.shape))
                except np.linalg.linalg.LinAlgError:
                    pass
                else:
                    break
                alpha *=10
#         nTplus = np.dot(Lplus,Lplus.T)
        #vmin = min(Tplus.min(),nTplus.min())
        #vmax=max(Tplus.max(),nTplus.max())

        #plot.matshow(nTplus,vmin=vmin, vmax=vmax)
#         print(np.allclose(Tplus ,nTplus))
        #print(Tplus.min(),Tplus.max(),nTplus.min(),nTplus.max())
        #print(Tplus)
        #print(nTplus)
        
#         plot.figure()
#         for channel in all_channels:
#             rowindices = [channel]*self.num_block_columns
#             colindices = [i*num_analised_channels+channel for i in range(self.num_block_columns)]
#             #print(rowindices,colindices)
#             plot.plot(nTplus[(rowindices, colindices)])
#         plot.title('nTplus')
        #precondition matrix with the smallest possible value
        try:
            Tminus=csc_matrix(Tminus)
            Lminus = cholesky(Tminus)
            #Lminus = np.linalg.cholesky(Tminus)
        except np.linalg.linalg.LinAlgError:
            alpha = 1e-16
            while True:
                
                print('Now preconditioning weighting matrix 2 by {}*np.eye()'.format(alpha))
                try:
                    Lminus = np.linalg.cholesky(Tminus+alpha*np.eye(*Tminus.shape))
                except np.linalg.linalg.LinAlgError:
                    pass
                else:
                    break
                alpha *=10
#         nTminus = np.dot(Lminus,Lminus.T)
#         #vmin = min(Tminus.min(),nTminus.min())
#         #vmax=max(Tminus.max(),nTminus.max())
#         #plot.matshow(Tminus,vmin=vmin, vmax=vmax)
#         #plot.matshow(nTminus,vmin=vmin, vmax=vmax)
#         print(np.allclose(Tminus ,nTminus))
#         #print(Tminus.min(),Tminus.max(),nTminus.min(),nTminus.max())
#         #print(Tminus)
#         #print(nTminus)
#         
#         plot.figure()
#         for channel in all_channels:
#             rowindices = [channel]*self.num_block_columns
#             colindices = [i*num_analised_channels+channel for i in range(self.num_block_columns)]
#             #print(rowindices,colindices)
#             plot.plot(nTminus[(rowindices, colindices)])
#         plot.title('nTminus')

        return Lplus, Lminus
         
        

          
    def compute_state_matrices(self, max_model_order=None):
        '''
        computes the state and output matrix of the state-space-model
        by applying a singular value decomposition to the block-toeplitz-matrix of covariances
        the state space model matrices are obtained by appropriate truncation 
        of the svd matrices at max_model_order
        the decision whether to take merged covariances is taken automatically
        '''
        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
            
        assert self.state[0]
        

        toeplitz_matrix = self.toeplitz_matrix
        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        print('Computing state matrices...')
        
        [U,S,V_T] = np.linalg.svd(toeplitz_matrix,0)
        #print(U.shape, S.shape, V_T.shape)
    
        # choose highest possible model order
        if max_model_order is None:
            max_model_order=len(S)
    
        S_2 = np.diag(np.power(S[:max_model_order], 0.5))
        U = U[:,:max_model_order]
        #print(S_2.shape, U.shape, W.shape)
        Oi = np.dot(U, S_2)
        #L = self.compute_cva_weighting()
        #Lplus,Lminus = self.compute_cva_weighting()
        #Oi = Lplus.solve_LD(Oi)
        #Oi = Lminus.solve_DLt(Oi)
        #print(L.shape, Oi.shape)
        #Oi = np.linalg.solve(L,Oi)
        C = Oi[:num_channels,:]   
        
        A = np.dot(np.linalg.pinv(Oi[:(num_channels * (num_block_columns - 1)),:]),
                   Oi[num_channels:(num_channels * num_block_columns),:])
       
        self.state_matrix = A
        self.output_matrix = C
        self.max_model_order=max_model_order
        
        self.state[1]=True
        self.state[2] = False # previous modal params are invalid now

    def compute_modal_params(self, multiprocessing=True, max_model_order=None): 
        
        if max_model_order is not None:
            assert max_model_order<=self.max_model_order
            self.max_model_order=max_model_order
        
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
        #print(self.modal_frequencies)
        self.state[2]=True
        
    
    def multiprocess_evd(self, a, truncation_orders, return_dict):
        
        for truncation_order in truncation_orders:
            eigenvalues_paired, eigenvectors_paired = np.linalg.eig(a[0:truncation_order+1, 0:truncation_order+1])
    
            eigenvectors_single,eigenvalues_single = \
                    CVASSICovRef.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            return_dict[truncation_order] = (eigenvalues_single, eigenvectors_single)
        
        return
                    
    @staticmethod
    def remove_conjugates_new (vectors, values):
        '''
        removes conjugates and marks the vectors which appear in pairs
        
        vectors.shape = [order+1, order+1]
        values.shape = [order+1,1]
        '''
        #return vectors, values
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
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
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
            
        np.savez_compressed(fname, **out_dict)
        
    @classmethod 
    def load_state(cls, fname, prep_data):
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
        
        assert isinstance(prep_data, PreprocessData)
        setup_name= str(in_dict['self.setup_name'].item())
        start_time=in_dict['self.start_time'].item()
        assert setup_name == prep_data.setup_name
        start_time = prep_data.start_time
        
        assert start_time == prep_data.start_time
        #prep_data = in_dict['self.prep_data'].item()
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