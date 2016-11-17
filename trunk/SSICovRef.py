# -*- coding: utf-8 -*-
'''
Based on previous works by Andrei Udrea 2014 and Volkmar Zabel 2015
Modified and Extended by Simon Marwitz 2015
'''

import numpy as np
import scipy.linalg 
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
            Oi0p = np.linalg.pinv(Oi0, rcond=1e-12)
            
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
        multiprocessing=False
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
        for order in range(1,max_model_order,1):    
            
            if multiprocessing:
                eigenvalues_single, eigenvectors_single = return_dict[order]
            else:
                eigenvalues_paired, eigenvectors_paired = np.linalg.eig(state_matrix[0:order, 0:order])
                
                eigenvalues_single,eigenvectors_single = \
                    self.remove_conjugates_new(eigenvalues_paired,eigenvectors_paired)
#                 ax1.plot(eigenvalues_single.real,eigenvalues_single.imag, ls='', marker='o')
                
            #lambdas=[]
            for index,k in enumerate(eigenvalues_single): 
                lambda_k = np.log(complex(k)) * sampling_rate
                #lambdas.append(lambda_k)
                freq_j = np.abs(lambda_k) / (2*np.pi)        
                damping_j = np.real(lambda_k)/np.abs(lambda_k) * (-100)  
                mode_shapes_j = np.dot(output_matrix[:, 0:order], eigenvectors_single[:,index])
            
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                mode_shapes_j = self.integrate_quantities(mode_shapes_j, accel_channels, velo_channels, np.abs(lambda_k))                
                        
                modal_frequencies[order,index]=freq_j
                modal_damping[order,index]=damping_j
                mode_shapes[:,index,order]=mode_shapes_j
            #lambdas = np.array(lambdas)
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
    
            eigenvalues_single,eigenvectors_single = \
                    BRSSICovRef.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            return_dict[truncation_order] = (eigenvalues_single, eigenvectors_single)
        
        return
    
    @staticmethod
    def remove_conjugates_new (eigval, eigvec_r, eigvec_l=None):
        '''
        removes conjugates
        
        eigvec_l.shape = [order+1, order+1]
        eigval.shape = [order+1,1]
        '''
        #return vectors, eigval
        num_val=len(eigval)
        conj_indices=deque()
        
        for i in range(num_val):
            this_val=eigval[i]
            this_conj_val = np.conj(this_val)
            if this_val == this_conj_val: #remove real eigvals
                conj_indices.append(i)
            for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
                if eigval[j] == this_conj_val:
        
                    #if not np.allclose(eigvec_l[j],eigvec_l[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_l is not')
                    #    continue
        
                    #if not np.allclose(eigvec_r[j],eigvec_r[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_r is not')
                    #    continue
                    
                    conj_indices.append(j)
                    break
                
        #print('indices of complex conjugate: {}'.format(conj_indices))
        conj_indices=list(set(range(num_val)).difference(conj_indices))
        #print('indices to keep and return: {}'.format(conj_indices))
        
        if eigvec_l is None:
            
            eigvec_r = eigvec_r[:,conj_indices]
            eigval = eigval[conj_indices]
    
            return eigval,eigvec_r      
        
        else:             
            eigvec_l = eigvec_l[:,conj_indices]
            eigvec_r = eigvec_r[:,conj_indices]
            eigval = eigval[conj_indices]
    
            return eigval,eigvec_l,eigvec_r      
                  
#     @staticmethod
#     def remove_conjugates_new (vectors, values):
#         '''
#         removes conjugates and marks the vectors which appear in pairs
#         
#         vectors.shape = [order+1, order+1]
#         values.shape = [order+1,1]
#         '''
#         num_val=vectors.shape[1]
#         conj_indices=deque()
#         
#         for i in range(num_val):
#             this_vec=vectors[:,i]
#             this_conj_vec = np.conj(this_vec)
#             this_val=values[i]
#             this_conj_val = np.conj(this_val)
#             if this_val == this_conj_val: #remove real eigenvalues
#                 continue
#             for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
#                 if vectors[0,j] == this_conj_vec[0] and \
#                    vectors[-1,j] == this_conj_vec[-1] and \
#                    values[j] == this_conj_val:
#                     # saves computation time this function gets called many times and 
#                     #numpy's np.all() function causes a lot of computation time
#                     conj_indices.append(i)
#                     break
#         conj_indices=list(conj_indices)
#         vector = vectors[:,conj_indices]
#         value = values[conj_indices]
# 
#         return vector,value
    
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

def vectorize(matrix):
    '''
        | 1  2  3 |
    A=  | 4  5  6 |
        | 7  8  9 |
        
    returns vertically stacked columns of matrix A
        
    | 1 |
    | 4 |
    | 7 |
    | 2 |
    | 5 |
    | 8 |
    | 3 |
    | 6 |
    | 9 |
    '''
    return np.reshape(matrix,(np.product(matrix.shape),1),'F')

def permutation(a,b):
    P = np.zeros((a*b, a*b))#zeros((a*b,a*b))     
    ind1=np.array(range(a*b))#range(a*b)
    ind2=np.mod(ind1*a,a*b-1) #mod(ind1*a,a*b-1)
    ind2[-1]=a*b-1 #a*b-1
    P[ind1,ind2]=1
    
    return P


class VarSSICovRef(object):
    
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
        #self.state= [Hankel, State Mat., Modal Par.
        self.state  =[False,    False,      False]
        
        self.num_block_columns = None
        self.num_block_rows = None
        self.hankel_matrix = None
        
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
        ssi_object.build_hankel_cov(num_block_columns, multiprocess=multiprocessing, num_blocks=1)
        ssi_object.compute_state_matrices(max_model_order)
        #ssi_object.compute_modal_params(multiprocessing)
        ssi_object.compute_modal_params_optim(multiprocessing)
        
        return ssi_object
        
    def build_hankel_cov(self, num_block_columns, num_block_rows=None, multiprocess=True, num_blocks=2):
        assert multiprocess
        self.num_blocks=num_blocks
        '''
        Builds a Block-Hankel Matrix of Covariances with varying time lags
        
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
        extract_length =int(np.floor((total_time_steps - (num_block_columns + num_block_rows) + 1)/num_blocks))
        print((num_block_columns+num_block_rows - 1) * num_analised_channels * num_ref_channels)
        #sigma_R = np.zeros(((num_block_columns+num_block_rows - 1) * num_analised_channels * num_ref_channels, (num_block_columns+num_block_rows - 1) * num_analised_channels * num_ref_channels))
        
        
        all_channels = ref_channels + roving_channels
        all_channels.sort()
        
        corr_matrices = []
        
        for n_block in range(num_blocks):
            this_measurement = measurement[(n_block)*extract_length:(n_block+1)*extract_length+num_block_columns+num_block_rows+1,:]
            refs = (this_measurement[:extract_length,ref_channels]).T
        
            print('Computing covariances for block {} of {} blocks...'.format(n_block, num_blocks))
            #n, m = num_analised_channels*num_block_rows, num_ref_channels*num_block_columns
        
            #hankel_memory = mp.Array(c.c_double, np.zeros(n*m)) # shared memory, can be used by multiple processes @UndefinedVariable
            #hankel_shape = (n,m)
            
            corr_mats_shape = ((num_block_columns+num_block_rows -1 ) * num_analised_channels, num_ref_channels)
            corr_memory = mp.Array(c.c_double, np.zeros((np.product(corr_mats_shape)))) # shared memory, can be used by multiple processes @UndefinedVariable
                        
            measurement_shape=this_measurement.shape
            measurement_memory = mp.Array(c.c_double, this_measurement.reshape(this_measurement.size, 1))
            
            refs_memory = mp.Array(c.c_double, refs.reshape(refs.size,1 ))
            refs_shape=refs.shape
            
            pool=mp.Pool(initializer=self.init_child_process, initargs=(refs_memory,  measurement_memory, corr_memory, ))
            
            for tau in range(1,num_block_columns+num_block_rows):
                pool.apply_async(self.compute_covariance , args=(tau,
                                                            extract_length, 
                                                            ref_channels, 
                                                            all_channels, 
                                                            refs_shape, 
                                                            measurement_shape,
                                                            corr_mats_shape))
            
            
            
            
#             for i in range(0,num_block_rows):
#                 if i == 0:
#                     # in multiprocessing, errors in the single processes are sometimes not reported to console, in this case 
#                     # the resulting block hankelmatrix will be filled with zeros
#                     # try to change pool.apply_async to pool.apply and run in debug mode
#                     for ii in range(0,self.num_block_columns):#fill the first block row
#                         pool.apply_async(self.compute_covariance , args=(i, 
#                                                             ii, 
#                                                             n_block,
#                                                             num_block_columns, 
#                                                             extract_length, 
#                                                             ref_channels, 
#                                                             all_channels, 
#                                                             refs_shape, 
#                                                             measurement_shape,
#                                                             hankel_shape))
#                 else: #fill the first block column
#                     ii = self.num_block_columns - 1
#                     pool.apply_async(self.compute_covariance , args=(i, 
#                                                             ii, 
#                                                             n_block,
#                                                             num_block_columns, 
#                                                             extract_length, 
#                                                             ref_channels, 
#                                                             all_channels, 
#                                                             refs_shape, 
#                                                             measurement_shape,
#                                                             hankel_shape))
#                         
            pool.close()
            pool.join()               
#             
#             del measurement_memory
#             del refs_memory
#             del refs  

            corr_mats = np.frombuffer(corr_memory.get_obj()).reshape(corr_mats_shape) 
#             Hankel_matrix = np.frombuffer(hankel_memory.get_obj()).reshape((n,m)) 

            
#             for i in range(1,num_block_rows): #finish assembling block hankel matrix
#                 # copys and shifts contents from previous block row to next block row
#                 # shifts by one block column to the right
#                 previous_Hankel_row = (i-1)*num_analised_channels
#                 this_block = Hankel_matrix[previous_Hankel_row:(previous_Hankel_row+num_analised_channels),
#                                              num_ref_channels:(num_ref_channels * num_block_columns)]
#                 begin_Hankel_row = i*num_analised_channels
#                 Hankel_matrix[begin_Hankel_row:(begin_Hankel_row+num_analised_channels),
#                                 0:num_ref_channels* (num_block_columns - 1)] = this_block
            
            corr_matrices.append(corr_mats)
            
        corr_mats_mean = np.mean(corr_matrices, axis=0)           
        
        self.hankel_matrix= np.zeros((num_block_rows*num_analised_channels, num_block_columns*num_ref_channels))
        for block_column in range(num_block_columns):
            this_block_column = corr_mats_mean[block_column*num_analised_channels:(num_block_rows+block_column)*num_analised_channels,:]
            self.hankel_matrix[:,block_column*num_ref_channels:(block_column+1)*num_ref_channels]=this_block_column        
        
        hankel_matrices = []
        for n_block in range(num_blocks):
            corr_matrix = corr_matrices[n_block]
            this_hankel_matrix= np.zeros((num_block_rows*num_analised_channels, num_block_columns*num_ref_channels))
            for block_column in range(num_block_columns):
                this_block_column = corr_matrix[block_column*num_analised_channels:(num_block_rows+block_column)*num_analised_channels,:]
                this_hankel_matrix[:,block_column*num_ref_channels:(block_column+1)*num_ref_channels]=this_block_column
            hankel_matrices.append(this_hankel_matrix)
        
        
#         import matplotlib.pyplot as plot
#  
#         for ref_channel in range(num_ref_channels):
#             #plot.errorbar(range(num_block_rows+num_block_rows-1), means[:,ref_channel], yerr=stds[:,ref_channel])
#             #plot.errorbar(range(num_block_rows+num_block_rows-1), means[:,ref_channel], yerr=np.std([raws1[:,ref_channel],raws2[:,ref_channel]],axis=0))
#             ref_channels = np.array(self.prep_data.ref_channels)
#             inds=np.hstack([i+ref_channel for i in range(0,(num_block_columns+num_block_rows-1)*num_analised_channels*num_ref_channels,num_ref_channels*num_analised_channels)])
#             means = np.mean(vec_R[inds,:],axis=1)
#             #print(means.shape, sigma_r[inds,inds].shape, len(inds))
#             plot.errorbar(range(num_block_rows+num_block_rows-1), means, yerr=np.sqrt(sigma_r[inds,inds]))
#             #print(np.sqrt(sigma_r[inds,inds]))
#              
#             plot.plot(vec_R[inds,0])
#             plot.plot(vec_R[inds,1])
#             
#         plot.show()
        #sigma_R = np.zeros(((num_block_columns+num_block_rows - 1) * num_analised_channels * num_ref_channels, (num_block_columns+num_block_rows - 1) * num_analised_channels * num_ref_channels))
        #sigma_R=None
        #for n_block in range(num_blocks):
        #    this_corr = vectorize(corr_matrices[n_block])-vectorize(corr_mats_mean)
        #    if sigma_R is None:
        #        sigma_R = np.dot(this_corr,this_corr.T)
        #    else:
        #        sigma_R += np.dot(this_corr,this_corr.T)
        #sigma_R /= (num_blocks*(num_blocks-1))
        #self.sigma_R = sigma_R
        
        T=np.zeros((num_block_rows*num_block_columns*num_analised_channels*num_ref_channels,num_blocks))
        for n_block in range(num_blocks):
            this_hankel = hankel_matrices[n_block]
            T[:,n_block:n_block+1]=vectorize(this_hankel)-vectorize(self.hankel_matrix)
        T/=np.sqrt(num_blocks*(num_blocks-1))        
        self.hankel_cov_matrix = T
        
        #import scipy.sparse 
        #S3=[]
        #for k in range(num_block_columns):
        #    S3.append(scipy.sparse.kron(scipy.sparse.identity(num_ref_channels),scipy.sparse.hstack([np.zeros((num_block_rows*num_analised_channels, (k)*num_analised_channels)),
        #                                                              scipy.sparse.identity(num_block_rows*num_analised_channels),
        #                                                              np.zeros((num_block_rows*num_analised_channels, (num_block_columns-k-1)*num_analised_channels))])).T)
        #S3=scipy.sparse.hstack(S3).T
        #self.S3 = S3
        
        #print(np.allclose(np.dot(S3,np.dot(sigma_R,S3.T)),np.dot(T,T.T)))
        #print(self.hankel_matrix.shape)   
        self.state[0]=True
         
        
    def init_child_process(self, refs_memory_, measurement_memory_, corr_memory_):
        #make the  memory arrays available to the child processes
        global refs_memory
        refs_memory = refs_memory_
        
        global measurement_memory
        measurement_memory = measurement_memory_   
        
        global corr_memory
        corr_memory = corr_memory_
    
   
    def compute_covariance(self, tau, extract_length, ref_channels, all_channels, refs_shape, measurement_shape, corr_mats_shape):
        # compute R_iii where iii = i+ii+1 equals the time lag
        
        num_analised_channels = len(all_channels)
        
        measurement = np.frombuffer(measurement_memory.get_obj()).reshape(measurement_shape)
        refs = np.frombuffer(refs_memory.get_obj()).reshape(refs_shape)
        corr_mats = np.frombuffer(corr_memory.get_obj()).reshape(corr_mats_shape)
        
        current_signals = (measurement[tau:(tau + extract_length), all_channels]).T
        this_block = (np.dot(refs, current_signals.T.conj()) / refs_shape[1]).T/extract_length
        
        with corr_memory.get_lock():
            corr_mats[(tau-1)*num_analised_channels:tau*num_analised_channels,:] = this_block
          
    def compute_state_matrices(self, max_model_order=None):
        '''
        computes the state and output matrix of the state-space-model
        by applying a singular value decomposition to the block-hankel-matrix of covariances
        the state space model matrices are obtained by appropriate truncation 
        of the svd matrices at max_model_order
        the decision whether to take merged covariances is taken automatically
        '''
        
        if max_model_order is not None:
            assert isinstance(max_model_order, int)
            
        assert self.state[0]
        

        hankel_matrix = self.hankel_matrix
        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        num_block_rows = self.num_block_rows # p
        print('Computing state matrices...')
        
        [U,S,V_T] = np.linalg.svd(hankel_matrix,0)
        #print(U.shape, S.shape, V_T.shape)
        
        # choose highest possible model order
        if max_model_order is None:
            max_model_order=len(S)        
        else:
            max_model_order = min(max_model_order,len(S))
        
    
        S_2 = np.diag(np.power(np.copy(S)[:max_model_order], 0.5))
        U = U[:,:max_model_order]
        V_T = V_T[:max_model_order,:]
        #print(S_2.shape, U.shape, W.shape)
        Oi = np.dot(U, S_2)
        #L = self.compute_cva_weighting()
        #Lplus,Lminus = self.compute_cva_weighting()
        #Oi = Lplus.solve_LD(Oi)
        #Oi = Lminus.solve_DLt(Oi)
        #print(L.shape, Oi.shape)
        #Oi = np.linalg.solve(L,Oi)
        C = Oi[:num_channels,:]   
        
        Oi_up = Oi[:num_channels * (num_block_rows-1),:]

        Oi_down = Oi[num_channels:num_channels * num_block_rows ,:]

        
        A = np.dot(np.linalg.pinv(Oi_up), Oi_down)
        
        self.Oi = Oi
        self.U = U
        self.S = S
        #self.S_2=np.diag(np.power(np.copy(S)[:max_model_order], -0.5))
        self.V_T = V_T
        
        self.state_matrix = A
        self.output_matrix = C
        self.max_model_order=max_model_order
        
        #num_channels = self.prep_data.num_analised_channels # r
        num_ref_channels = self.prep_data.num_ref_channels #r_o
        #num_block_columns = self.num_block_columns # q
        num_blocks = self.num_blocks
        

       # max_model_order = self.max_model_order
        #hankel_matrix = self.hankel_matrix
        T = self.hankel_cov_matrix
        
        #U = self.U
        #S = self.S
        #V_T = self.V_T
                
        # Computation of Q_1 ... Q_4 in (36): For i = 1...n_b compute B_i,1 in (29) T_i,1 , T_i,2 (I_O,H T)_i in Remark 9 and the i-th block line of Q_1 ... Q_4 in (37)
        

        
        # S_1 in 3.1
        S1 = np.hstack([np.identity((num_block_rows-1)*num_channels), np.zeros(((num_block_rows-1)*num_channels,num_channels))])
        S2 = np.hstack([ np.zeros(((num_block_rows-1)*num_channels,num_channels)), np.identity((num_block_rows-1)*num_channels)])
        #print(np.all(np.dot(S1,Oi)==Oi_up))
        #print(np.all(np.dot(S2,Oi)==Oi_down))
                
        Q1=np.zeros((max_model_order**2, num_blocks))
        Q2=np.zeros((max_model_order**2, num_blocks))
        Q3=np.zeros((max_model_order**2, num_blocks))
        Q4=np.zeros((max_model_order*num_channels, num_blocks))

        for i in range(1,max_model_order):     
            v_i_T =  V_T[i-1:i,:]
            u_i = U[:,i-1:i]
            s_i = S[i-1]
            # K_i, B_i,1; 
            K_i= (np.identity(num_block_columns*num_ref_channels)+
                  np.vstack([np.zeros((num_block_columns*num_ref_channels-1, num_block_columns*num_ref_channels)),
                             2*v_i_T])-
                  np.dot(hankel_matrix.T, hankel_matrix)/(s_i**2))
            
#             K_ii = np.linalg.inv(K_i)
#             
#             B_i1_o = np.hstack([np.identity(num_block_rows*num_channels),
#              np.dot(np.dot(np.dot(hankel_matrix,K_ii)/s_i,
#                            (hankel_matrix.T/s_i -
#                             np.vstack([np.zeros((num_block_columns*num_ref_channels-1,num_block_rows*num_channels)), 
#                                        u_i.T])
#                             )
#                            ),
#                     np.dot(hankel_matrix,K_ii)/s_i)
#              ])
            sol_hank_K_i=np.linalg.solve(K_i.T,hankel_matrix.T).T
            B_i1 = np.hstack([np.identity(num_block_rows*num_channels),
             np.dot(np.dot(sol_hank_K_i/s_i,
                           (hankel_matrix.T/s_i -
                            np.vstack([np.zeros((num_block_columns*num_ref_channels-1,num_block_rows*num_channels)), 
                                       u_i.T])
                            )
                           ),
                    sol_hank_K_i/s_i)
             ])
            
#             print(np.allclose(B_i1_o,B_i1, rtol=1e-04, atol=1e-06))
                             
                    
            #T_i,1; T_i,2
            
            T_i1 = np.dot(np.kron(np.identity(num_block_columns*num_ref_channels),u_i.T),T)
            T_i2 = np.dot(np.kron(v_i_T, np.identity(self.num_block_rows*num_channels)),T)
                        
            # (I_O,H T)_i
            
            I_OHT = (0.5*s_i**(-0.5)*np.dot(u_i,np.dot(v_i_T,T_i1))+
                         s_i**(0.5)*np.dot(B_i1,np.vstack([T_i2-np.dot(u_i,np.dot(u_i.T,T_i2)),
                                                           T_i1-np.dot(v_i_T.T,np.dot(v_i_T,T_i1))])))
            #print(np.dot(np.dot(Oi_up.T,S1),I_OHT).shape)
            beg,end=(i-1,i)
            #beg_end=(i,i+1)
            Q1[beg*max_model_order:end*max_model_order,:] = np.dot(np.hstack([Oi_up.T,np.zeros((max_model_order,num_channels))]),I_OHT)
            Q2[beg*max_model_order:end*max_model_order,:] = np.dot(np.hstack([Oi_down.T,np.zeros((max_model_order,num_channels))]),I_OHT)
            Q3[beg*max_model_order:end*max_model_order,:] = np.dot(np.hstack([np.zeros((max_model_order,num_channels)),Oi_up.T]),I_OHT)
            Q4[beg*   num_channels:end*num_channels,   :] = np.dot(np.hstack([np.identity(num_channels),np.zeros((num_channels,(num_block_rows-1)*num_channels))]),I_OHT)
            #print(np.all(np.dot(Oi_up.T,S1) ==np.hstack([Oi_up.T,np.zeros((max_model_order,num_channels))])))
            #print(np.all(np.dot(Oi_down.T,S1) ==np.hstack([Oi_down.T,np.zeros((max_model_order,num_channels))])))
            #print(np.all(np.dot(Oi_up.T,S2)==np.hstack([np.zeros((max_model_order,num_channels)),Oi_up.T])))
            #print(i*max_model_order,(i+1)*max_model_order)
        self.Q1 = Q1
        self.Q2 = Q2  
        self.Q3 = Q3   
        self.Q4 = Q4
        
        self.state[1]=True
        self.state[2] = False # previous modal params are invalid now
        
    def compute_modal_params_optim(self, multiprocessing=True, max_model_order=None): 
        if max_model_order is not None:
            assert max_model_order<=self.max_model_order
            self.max_model_order=max_model_order
        
        assert self.state[1]
        
        print('Computing modal parameters...')
        max_model_order = self.max_model_order
        state_matrix = self.state_matrix
        
        #print('Matrix is {} symmetric.'.format(['not',''][np.allclose(state_matrix.transpose(), state_matrix)]))
        #print('Matrix is {} Hermitian.'.format(['not',''][np.allclose(state_matrix.transpose().conj(), state_matrix)]))
        
        output_matrix = self.output_matrix
        sampling_rate = self.prep_data.sampling_rate
        
        Oi = self.Oi 
        Q1 = self.Q1
        Q2 = self.Q2
        Q3 = self.Q3
        Q4 = self.Q4
        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        
        #b_accel_channels = np.array([False for i in range(num_analised_channels)])
        accel_channels=self.prep_data.accel_channels
        #b_velo_channels = np.array([False for i in range(num_analised_channels)])
        velo_channels=self.prep_data.velo_channels
        #accel_channels = self.prep_data.accel_channels
        #velo_channels = self.prep_data.velo_channels
        
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        std_frequencies = np.zeros((max_model_order, max_model_order))
        
        modal_damping = np.zeros((max_model_order, max_model_order))  
        std_damping = np.zeros((max_model_order, max_model_order))
              
        mode_shapes = np.zeros((num_channels, max_model_order, max_model_order),dtype=complex)
        std_mode_shapes = np.zeros((num_channels, max_model_order, max_model_order),dtype=complex)
        
        
        
        for order in range(1,max_model_order,1):        
            print('Step up order: ',order)        
            if order>= 38: 
                pass
            
            eigval, eigvec_l, eigvec_r = scipy.linalg.eig(a=state_matrix[0:order, 0:order],b=None,left=True,right=True)
            eigval, eigvec_l, eigvec_r = self.remove_conjugates_new(eigval, eigvec_l, eigvec_r) 
            
            #eigenvectors_single,eigenvalues_single = \
            #    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            
            Oin = Oi[:,:order]
            
            S4 = np.kron(np.hstack([np.identity(order),np.zeros((order,max_model_order-order))]),
                         np.hstack([np.identity(order),np.zeros((order,max_model_order-order))]))
 
            Q1n = np.dot(S4,Q1)
            Q2n = np.dot(S4,Q2)
            Q3n = np.dot(S4,Q3)
            Q4n = np.dot(np.hstack([np.identity(num_channels*order),np.zeros((num_channels*order,num_channels*(max_model_order-order)))]),Q4)
            
            #Computation of (Oi_up Oi_up)^-1 , (P_nn + I_n2) Q1 and the sum P Q2 +Q3
            
            Oi_up = Oin[:num_channels * (num_block_columns -1),:]
            Oi_up2 = np.dot(Oi_up.T, Oi_up)
            #Oi_up2inv = np.linalg.inv(Oi_up2)

            Pnn = permutation(order,order)            
            
            PQ1 = np.dot(Pnn + np.identity(order**2) , Q1n)
            PQ23 = np.dot(Pnn, Q2n) + Q3n
            #print(eigval)
            
            for j,lambda_j in enumerate(eigval):
                print('Step up eigval: ',j)
                a_j=np.abs(np.arctan2(np.imag(lambda_j),np.real(lambda_j)))
                b_j=np.log(np.abs(lambda_j))
                freq_j = np.sqrt(a_j**2+b_j**2)*sampling_rate/2/np.pi
                damping_j = 100*np.abs(b_j)/np.sqrt(a_j**2+b_j**2)     
                #lambda_cj=np.log(complex(lambda_j))*sampling_rate
                #freq_j=np.abs(lambda_cj)/2/np.pi
                #damping_j=-100*np.real(lambda_cj)/np.abs(lambda_cj)
                
                mode_shape_j = np.dot(output_matrix[:, 0:order], eigvec_r[:,j])
                mode_shape_j = np.array(mode_shape_j, dtype=complex)
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                #mode_shape_j = self.integrate_quantities(mode_shape_j, accel_channels, velo_channels, complex(freq_j*2*np.pi))                
                
                #mode_shape_j*=self.prep_data.channel_factors
                
                #mode_shape_j = self.rescale_mode_shape(mode_shape_j)
                
                modal_frequencies[order,j]=freq_j
                modal_damping[order,j]=damping_j
                mode_shapes[:,j,order]=mode_shape_j
                
                #print(modal_frequencies[order,j], modal_damping[order,j], mode_shapes[:,j,order])
            
                Phi_j = eigvec_r[:,j:j+1]
                Chi_j = eigvec_l[:,j:j+1]
                
                #print('Eigenvector is complex conjugate transpose: {}'.format(np.allclose(Phi_j,Chi_j.T.conj())))
                
                #Compute Q_i in (44)
                Q_j = np.dot(np.kron(Phi_j.T , np.identity(order)) , (PQ23 - lambda_j*PQ1))
                
                #Compute J_fili , J_xili in Lemma 5
                tlambda_j = (b_j+1j*a_j)*sampling_rate
                
                J_fixiili=(sampling_rate/((np.abs(lambda_j)**2) * np.abs(tlambda_j))*
                 np.dot(np.dot(np.array([[1/(2*np.pi),  0                         ],
                                         [0,            100/(np.abs(tlambda_j)**2)]]),
                               np.array([[np.real(tlambda_j),       np.imag(tlambda_j)],
                                         [-np.imag(tlambda_j)**2,   np.real(tlambda_j)*np.imag(tlambda_j)]])),
                        np.array([[np.real(lambda_j),   np.imag(lambda_j)],
                                  [-np.imag(lambda_j),  np.real(lambda_j)]]))
                 )
                
                # compute J_liA J_AO J_OHT in (43)

                J_liAOHT = 1/np.dot(Chi_j.T.conj(),Phi_j)*np.dot(Chi_j.conj().T,np.linalg.solve(Oi_up2,Q_j))
                
                # Compute U_fixi in (42)
                U_fixi = np.dot(J_fixiili,np.vstack([np.real(J_liAOHT),np.imag(J_liAOHT)]))
                
                # Compute the covariance of fi and xi in (40)
                
                var_fixi=np.dot(U_fixi,U_fixi.T)
                
                #Compute J_phi,A J_A,O J_O,HT in (46)
                
#                 J3T_phii1 = np.dot(np.linalg.pinv(np.dot(lambda_j,np.identity(order)-state_matrix[0:order, 0:order])),
#                                   np.dot(np.identity(order)-np.dot(Phi_j, Chi_j.T.conj())/np.dot(Chi_j.T.conj(),Phi_j),
#                                          np.dot(Oi_up2inv,
#                                                 Q_j)))
                J_PhiiHT = np.dot(np.linalg.pinv(np.dot(lambda_j,np.identity(order)-state_matrix[0:order, 0:order])),
                                  np.dot(np.identity(order)-np.dot(Phi_j, Chi_j.T.conj())/np.dot(Chi_j.T.conj(),Phi_j),
                                         np.linalg.solve(Oi_up2,
                                                Q_j)))
               
                
                #Compute U_phi from (41) and (45)
                k = np.argmax(np.abs(mode_shape_j))
                J_mshiHT = (1/mode_shape_j[k]*
                np.dot(np.identity(num_channels, dtype=complex)-np.hstack([np.zeros((num_channels,k),dtype=complex),np.reshape(mode_shape_j,(num_channels,1)),np.zeros((num_channels,num_channels-(k+1)),dtype=complex)]),
                       np.dot(output_matrix[:, 0:order],J_PhiiHT)+np.dot(np.kron(Phi_j.T,np.identity(num_channels)),Q4n)))
                
                U_phii = np.vstack([np.real(J_mshiHT),np.imag(J_mshiHT)])
                #print(U_phii)
                #Compute the covariance of phi in (40)
                var_phii=np.dot(U_phii,U_phii.T)
                #np.cov()
                #print(var_fixi.shape)
                std_frequencies[order,j]=np.sqrt(var_fixi[0,0])
                std_damping[order, j]=np.sqrt(var_fixi[1,1])
                
                std_mode_shapes.real[:,j,order]=var_phii[range(num_channels),range(num_channels)]
                std_mode_shapes.imag[:,j,order]=var_phii[range(num_channels,2*num_channels),range(num_channels,2*num_channels)]
                
                #print('Frequency: {}, Std_Frequency: {}'.format(freq_j, std_frequencies[order,j]))
                #print('Damping: {}, Std_damping: {}'.format(damping_j, std_damping[order, j]))
                #print('Mode_Shape: {}, Std_Mode_Shape: {}'.format(mode_shape_j, std_mode_shapes[:,j,order]))
                
                #print(var_phii)
        self.modal_frequencies = modal_frequencies
        self.std_frequencies = std_frequencies
        
        self.modal_damping = modal_damping
        self.std_damping = std_damping
        
        self.mode_shapes = mode_shapes
        self.std_mode_shapes = std_mode_shapes
        
        #print(self.modal_frequencies)
        self.state[2]=True
        
    def compute_modal_params(self, multiprocessing=True, max_model_order=None): 
        
        if max_model_order is not None:
            assert max_model_order<=self.max_model_order
            self.max_model_order=max_model_order
        
        assert self.state[1]
        
        print('Computing modal parameters...')
        max_model_order = self.max_model_order
        state_matrix = self.state_matrix
        
        #print('Matrix is {} symmetric.'.format(['not',''][np.allclose(state_matrix.transpose(), state_matrix)]))
        #print('Matrix is {} Hermitian.'.format(['not',''][np.allclose(state_matrix.transpose().conj(), state_matrix)]))
        
        output_matrix = self.output_matrix
        sampling_rate = self.prep_data.sampling_rate
        
        Oi = self.Oi 

        num_channels = self.prep_data.num_analised_channels
        num_block_columns = self.num_block_columns
        
        #b_accel_channels = np.array([False for i in range(num_analised_channels)])
        accel_channels=self.prep_data.accel_channels
        #b_velo_channels = np.array([False for i in range(num_analised_channels)])
        velo_channels=self.prep_data.velo_channels
        #accel_channels = self.prep_data.accel_channels
        #velo_channels = self.prep_data.velo_channels
        
        modal_frequencies = np.zeros((max_model_order, max_model_order))
        std_frequencies = np.zeros((max_model_order, max_model_order))
        
        modal_damping = np.zeros((max_model_order, max_model_order))  
        std_damping = np.zeros((max_model_order, max_model_order))
              
        mode_shapes = np.zeros((num_channels, max_model_order, max_model_order),dtype=complex)
        std_mode_shapes = np.zeros((num_channels, max_model_order, max_model_order),dtype=complex)
        
        num_ref_channels = self.prep_data.num_ref_channels
        hankel_matrix = self.hankel_matrix
        num_block_rows = self.num_block_rows
        num_blocks = self.num_blocks

        

        S3 = self.S3
        S2 = np.hstack([np.zeros(((num_block_rows-1)*num_channels,num_channels)),np.identity((num_block_rows-1)*num_channels)])
        S1 = np.hstack([np.identity((num_block_rows-1)*num_channels),np.zeros(((num_block_rows-1)*num_channels,num_channels))])
        
        for order in range(1,max_model_order):
            print('Step up order: ',order)        
            #i=order
            eigval, eigvec_l, eigvec_r = scipy.linalg.eig(a=state_matrix[0:order, 0:order],b=None,left=True,right=True)
            #print(eigval)
            #print(eigvec_l.shape, eigvec_r.shape, eigval.shape)
            #print(np.allclose(eigvec_l,eigvec_r.T.conj()))
            eigval, eigvec_l, eigvec_r = self.remove_conjugates_new(eigval, eigvec_l, eigvec_r) 
            
            #eigenvectors_single,eigenvalues_single = \
            #    self.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            
            
            
            S4 = np.kron(np.hstack([np.identity(order),np.zeros((order,max_model_order-order))]),
                         np.hstack([np.identity(order),np.zeros((order,max_model_order-order))]))
            
            Oin = Oi[:,:order]
            Oi_up = Oin[:num_channels * (num_block_rows-1),:]
            Oi_down = Oin[num_channels:num_channels * num_block_rows ,:]

            # K_i, B_i,1; 
            BC=[]
            vu=[]
            for i in range(0,order):
                #print(i)
                v_i_T =  self.V_T[i:i+1,:]
                u_i = self.U[:,i:i+1]
                s_i = self.S[i]
                
                B_i=np.vstack([np.hstack([np.identity(num_block_rows*num_channels), -1/s_i*hankel_matrix]),
                              np.hstack([-1/s_i*hankel_matrix.T, np.identity(num_block_columns*num_ref_channels)])])
                
#                 P = np.zeros(((num_block_rows*num_channels)**2, (num_block_columns*num_ref_channels)**2))
#                 for k in range(num_block_rows*num_channels):
#                     for l in range(num_block_columns*num_ref_channels):
#                         Ekl=np.zeros((num_block_rows*num_channels, num_block_columns*num_ref_channels))
#                         Ekl[k,l]=1
#                         Elk=np.zeros((num_block_rows*num_channels, num_block_columns*num_ref_channels))
#                         Elk[l,k]=1
#                         pp=np.kron(Ekl,Elk)
#                         #print(pp)
#                         P+=pp
                
                P = permutation(num_block_rows*num_channels, num_block_columns*num_ref_channels)
                #print('P1',np.all(P==P2))

                
                C_i=1/s_i*np.vstack([np.kron(v_i_T, (np.identity(num_block_rows*num_channels))-np.dot(u_i,u_i.T)),
                                     np.dot(np.kron(u_i.T,(np.identity(num_block_columns*num_ref_channels)-np.dot(v_i_T.T,v_i_T))),P)])
                
                BC.append(np.dot(np.linalg.pinv(B_i),np.dot(C_i,S3)))
                vu.append(np.dot(np.kron(v_i_T.T,u_i).T,S3))
            BC=np.vstack(BC)
            vu=np.vstack(vu)     
                    
            S4=np.zeros((order**2,order))
            for k in range(order):
                S4[(k-1)*order+k,k]+=1
                
            I_OHS3 = (0.5*np.dot(np.dot(np.kron(np.identity(order), np.dot(self.U[:,:order], np.diag(np.power(np.copy(self.S)[:order], -0.5)))),S4),vu)+
                    np.dot(np.kron(np.diag(np.power(np.copy(self.S)[:order], 0.5)),
                                   np.hstack([np.identity(num_block_rows*num_channels),np.zeros((num_block_rows*num_channels, num_block_columns*num_ref_channels))]))
                           ,BC))
#             P = np.zeros(((num_block_rows*num_channels)*order, (num_block_rows*num_channels)*order))
#             for k in range(num_block_rows*num_channels):
#                 for l in range(order):
#                     Ekl=np.zeros((num_block_rows*num_channels, order))
#                     Ekl[k,l]=1
#                     Elk=np.zeros((order, num_block_rows*num_channels))
#                     Elk[l,k]=1
#                     pp=np.kron(Ekl,Elk)
#                     #print(pp.shape,(num_block_rows*num_channels)*order)
#                     P+=pp
            P = permutation(num_block_rows*num_channels, order)                    
            #print('P2',np.all(P==P2))

            I_AO=(np.kron(np.identity(order),np.dot(np.linalg.pinv(Oi_up[:,:order]),S2))-
                  np.kron(state_matrix[0:order, 0:order].T,np.dot(np.linalg.pinv(Oi_up[:,:order]),S1))+
                  np.dot(np.kron(np.dot(Oi_down[:,:order].T,S2)-
                                 np.dot(np.dot(state_matrix[0:order, 0:order],Oi_up[:,:order].T),S2),
                                 np.linalg.inv(np.dot(Oi_up[:,:order].T,Oi_up[:,:order])))
                  ,P))
            
            I_CO=np.kron(np.identity(order),np.hstack([np.identity(num_channels),np.zeros((num_channels,(num_block_rows-1)*num_channels))]))
            A=np.dot(np.vstack([I_AO,I_CO]), I_OHS3)
            
            sigma_AC = np.dot(np.dot(A,self.sigma_R),A.T)
            
                        
            for j,lambda_j in enumerate(eigval):
                print('Step up eigval: ',j)
                a_j=np.abs(np.arctan2(np.imag(lambda_j),np.real(lambda_j)))
                b_j=np.log(np.abs(lambda_j))
                freq_j = np.sqrt(a_j**2+b_j**2)*sampling_rate/2/np.pi
                damping_j = 100*np.abs(b_j)/np.sqrt(a_j**2+b_j**2)     
                #lambda_cj=np.log(complex(lambda_j))*sampling_rate
                #freq_j=np.abs(lambda_cj)/2/np.pi
                #damping_j=-100*np.real(lambda_cj)/np.abs(lambda_cj)
                
                mode_shape_j = np.dot(output_matrix[:, 0:order], eigvec_r[:,j])
                mode_shape_j = np.array(mode_shape_j, dtype=complex)
                # integrate acceleration and velocity channels to level out all channels in phase and amplitude
                mode_shape_j = self.integrate_quantities(mode_shape_j, accel_channels, velo_channels, complex(freq_j*2*np.pi))                
                
                mode_shape_j*=self.prep_data.channel_factors
                
                mode_shape_j = self.rescale_mode_shape(mode_shape_j)
                
                modal_frequencies[order,j]=freq_j
                modal_damping[order,j]=damping_j
                mode_shapes[:,j,order]=mode_shape_j
                
                #print(modal_frequencies[order,j], modal_damping[order,j], mode_shapes[:,j,order])
            
                Phi_j = eigvec_r[:,j:j+1]
                Chi_j = eigvec_l[:,j:j+1]
                
                #print('Eigenvector is complex conjugate transpose: {}'.format(np.allclose(Phi_j,Chi_j.T.conj())))
                
                J_liA = 1/np.dot(Chi_j.T.conj(),Phi_j)*np.kron(Phi_j.T,Chi_j.T.conj())
                J_PhiA= np.dot(np.linalg.pinv(lambda_j*np.identity(order)-state_matrix[:order,:order]),
                               np.kron(Phi_j.T,(np.identity(order)-np.dot(Phi_j,Chi_j.T.conj())/np.dot(Chi_j.T.conj(),Phi_j))))
                
                #Compute J_fili , J_xili in Lemma 5
                tlambda_j = (b_j+1j*a_j)*sampling_rate
                
                J_fixiili=(sampling_rate/((np.abs(lambda_j)**2) * np.abs(tlambda_j))*
                 np.dot(np.dot(np.array([[1/2/np.pi,    0                         ],
                                         [0,            100/(np.abs(tlambda_j)**2)]]),
                               np.array([[np.real(tlambda_j),       np.imag(tlambda_j)],
                                         [-np.imag(tlambda_j)**2,   np.real(tlambda_j)*np.imag(tlambda_j)]])),
                        np.array([[np.real(lambda_j),   np.imag(lambda_j)],
                                  [-np.imag(lambda_j),  np.real(lambda_j)]]))
                 )
                
                J_fiAxiA = np.dot(J_fixiili,np.vstack([np.real(J_liA),np.imag(J_liA)]))
                var_fixi = np.dot(np.hstack([J_fiAxiA, np.zeros((2,num_channels*order))]),np.dot(sigma_AC,np.hstack([J_fiAxiA, np.zeros((2,num_channels*order))]).T))
                
                k = np.argmax(np.abs(mode_shape_j))
                J_phiiAC = (1/mode_shape_j[k]*
                np.dot(np.identity(num_channels, dtype=complex)-np.hstack([np.zeros((num_channels,k),dtype=complex),np.reshape(mode_shape_j,(num_channels,1)),np.zeros((num_channels,num_channels-(k+1)),dtype=complex)]),
                       np.hstack([np.dot(output_matrix[:, 0:order],J_PhiA),np.kron(Phi_j.T,np.identity(num_channels))])))
                
                var_phii= np.dot(np.vstack([np.real(J_phiiAC),np.imag(J_phiiAC)]),np.dot(sigma_AC,np.vstack([np.real(J_phiiAC),np.imag(J_phiiAC)]).T))
                
                std_frequencies[order,j]=np.sqrt(var_fixi[0,0])
                std_damping[order, j]=np.sqrt(var_fixi[1,1])
                
                std_mode_shapes.real[:,j,order-1]=var_phii[range(num_channels),range(num_channels)]
                std_mode_shapes.imag[:,j,order-1]=var_phii[range(num_channels,2*num_channels),range(num_channels,2*num_channels)]
                
                #print('Frequency: {}, Std_Frequency: {}'.format(freq_j, std_frequencies[order,j]))
                #print('Damping: {}, Std_damping: {}'.format(damping_j, std_damping[order, j]))
                #print('Mode_Shape: {}, Std_Mode_Shape: {}'.format(mode_shape_j, std_mode_shapes[:,j,order]))
                
                #print(var_phii)
        self.modal_frequencies = modal_frequencies
        self.std_frequencies = std_frequencies
        
        self.modal_damping = modal_damping
        self.std_damping = std_damping
        
        self.mode_shapes = mode_shapes
        self.std_mode_shapes = std_mode_shapes
        
        #print(self.modal_frequencies)
        self.state[2]=True
        
    
    def multiprocess_evd(self, a, truncation_orders, return_dict):
        
        for truncation_order in truncation_orders:
            eigenvalues_paired, eigenvectors_paired = np.linalg.eig(a[0:truncation_order+1, 0:truncation_order+1])
    
            eigenvectors_single,eigenvalues_single = \
                    VarSSICovRef.remove_conjugates_new(eigenvectors_paired,eigenvalues_paired)
            return_dict[truncation_order] = (eigenvalues_single, eigenvectors_single)
        
        return
                    
    @staticmethod
    def remove_conjugates_new (eigval, eigvec_l, eigvec_r):
        '''
        removes conjugates
        
        eigvec_l.shape = [order+1, order+1]
        eigval.shape = [order+1,1]
        '''
        #return vectors, eigval
        num_val=len(eigval)
        conj_indices=deque()
        
        for i in range(num_val):
            this_val=eigval[i]
            this_conj_val = np.conj(this_val)
            #if this_val == this_conj_val: #remove real eigvals
            #    conj_indices.append(i)
            for j in range(i+1, num_val): #catches unordered conjugates but takes slightly longer
                if eigval[j] == this_conj_val:
        
                    #if not np.allclose(eigvec_l[j],eigvec_l[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_l is not')
                    #    continue
        
                    #if not np.allclose(eigvec_r[j],eigvec_r[i].conj()):
                    #    print('eigval is complex conjugate but eigvec_r is not')
                    #    continue
                    
                    conj_indices.append(j)
                    break
                
        #print('indices of complex conjugate: {}'.format(conj_indices))
        conj_indices=list(set(range(num_val)).difference(conj_indices))
        #print('indices to keep and return: {}'.format(conj_indices))
        
        
        eigvec_l = eigvec_l[:,conj_indices]
        eigvec_r = eigvec_r[:,conj_indices]
        eigval = eigval[conj_indices]

        return eigval,eigvec_l,eigvec_r
    
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
        #self.state= [Hankel, State Mat., Modal Par.]
        out_dict={'self.state':self.state}
        out_dict['self.setup_name'] = self.setup_name
        out_dict['self.start_time']=self.start_time
        #out_dict['self.prep_data']=self.prep_data
        if self.state[0]:# covariances
            out_dict['self.hankel_matrix'] = self.hankel_matrix
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
            out_dict['self.std_frequencies'] = self.std_frequencies
            out_dict['self.std_damping'] = self.std_damping
            out_dict['self.std_mode_shapes'] = self.std_mode_shapes
            
        np.savez_compressed(fname, **out_dict)
        
    @classmethod 
    def load_state(cls, fname, prep_data):
        print('Now loading previous results from  {}'.format(fname))
        
        in_dict=np.load(fname)    
        #             0         1           2          
        #self.state= [Hankel, State Mat., Modal Par.]
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
            ssi_object.hankel_matrix = in_dict['self.hankel_matrix']
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
            ssi_object.std_frequencies= in_dict['self.std_frequencies']
            ssi_object.std_damping= in_dict['self.std_damping']
            ssi_object.std_mode_shapes= in_dict['self.std_mode_shapes']
        
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